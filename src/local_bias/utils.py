import torch
import numpy as np
from torch.nn import functional as F
import scipy.stats
from sklearn.decomposition import PCA
import json
import scipy.stats
import torch.nn as nn

import torch.nn.functional as F
from transformers import top_k_top_p_filtering

import torch.nn.functional as F
from transformers import top_k_top_p_filtering
from sklearn.metrics.pairwise import cosine_similarity

"""
PCA（主成分分析）是一种常用的降维技术和数据分析方法，
用于从高维数据中提取出最重要的特征并进行数据压缩。
它通过线性变换将原始数据投影到新的低维空间上，其中投影轴是数据中方差最大的方向。
"""


def doPCA(pairs, num_components=10):
    matrix = []
    for a, b in pairs:
        center = (a + b) / 2
        norm_a = a - center
        norm_b = b - center
        norm_a, norm_b = norm_a.detach().numpy(), norm_b.detach().numpy()
        # norm_a, norm_b = norm_a/np.linalg.norm(norm_a), norm_b/np.linalg.norm(norm_b)
        matrix.append(norm_a)
        matrix.append(norm_b)
    matrix = np.array(matrix)
    pca = PCA(n_components=num_components, svd_solver="full")
    pca.fit(matrix)  # Produce different results each time...
    return pca


def dropspace(u, V):
    # u, V = u.detach().numpy(), V.detach().numpy()
    norm_sqrd = np.sum(V * V, axis=-1)
    vecs = np.divide(V @ u, norm_sqrd)[:, None] * V
    subspace = np.sum(vecs, axis=0)
    return u - subspace


def drop_bias(u, v):
    # return u - torch.ger(torch.matmul(u, v), v) / v.dot(v)
    projection = u.dot(v) * v / np.linalg.norm(v)
    return u - projection


def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)


def top_k_top_p_filtering(
        logits,  # (1, 50257)
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

# 它接收多个参数来计算与语言模型相关的一些指标。
def topk_kl_overlap(prompt_text, k, tokenizer, model, embedding, P, A, device):
    """
        :param prompt_text: a single prompt
        :param k: top k
        :param tokenizer: tokenizer
        :param model: gpt2 or other language model
        :param embedding: gpt2 word embedding
        :param P: nullspace matrix
        :param A: alpha list
        :param device: cpu or gpu
        """
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    # original gpt2 model
    input_ids = input_ids.to(device)
    outputs = model.transformer(input_ids=input_ids)[0][0][-1]  # (2, batch, len, dim)
    outputs = outputs.cpu().detach().numpy()
    logits = embedding.dot(outputs)

    old_rank = np.argsort(-logits).tolist()
    old_logits = np.sort(-logits).tolist()
    topk_raw = old_rank[:k]
    logits_raw = [-x for x in old_logits[:k]]

    # averaged hidden state debiased gpt2 model
    outputs_P = P.dot(outputs.T).T
    KL1 = [0 for ii in range(len(A))]
    KL2 = [0 for ii in range(len(A))]
    for a in range(len(A)):
        avg_outputs = A[a] * outputs + (1 - A[a]) * outputs_P
        avg_logits = embedding.dot(avg_outputs)

        logits_new = []
        for i, token in enumerate(topk_raw):
            logits_new.append(avg_logits[token])
        logits_new = np.array(logits_new)

        KL1[a] = scipy.stats.entropy(logits_raw, logits_new)
        KL2[a] = scipy.stats.entropy(logits_new, logits_raw)

    return KL1 + KL2

def topk_kl_overlap1(prompt_text, k, tokenizer, model, embedding, P, A, device):
    def normalize(x):
        norm = np.linalg.norm(x)
        return x if norm == 0 else x / norm

    def project(v, subspace):
        if subspace.shape[0] != v.shape[0]:
            subspace = subspace.T
        subspace = np.array([normalize(vec) for vec in subspace.T]).T
        projection_matrix = np.eye(v.shape[0]) - subspace @ subspace.T
        return projection_matrix @ v

    def debias_embeddings(embedding, subspace):
        debiased = []
        for vec in embedding:
            vec = normalize(vec)
            debiased_vec = project(vec, subspace)
            debiased.append(debiased_vec)
        return np.array(debiased)

    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.transformer(input_ids=input_ids)[0][0][-1].cpu().numpy()

    print("outputs shape:", outputs.shape)

    logits = embedding.dot(outputs)
    old_rank = np.argsort(-logits).tolist()
    old_logits = np.sort(-logits).tolist()
    topk_raw = old_rank[:k]
    logits_raw = [-x for x in old_logits[:k]]

    print("logits shape:", logits.shape)
    print("topk_raw:", topk_raw)
    print("logits_raw:", logits_raw)

    debiased_embedding = debias_embeddings(embedding, P)
    print("debiased_embedding shape:", debiased_embedding.shape)

    KL1 = []
    KL2 = []
    for alpha in A:
        avg_outputs = alpha * outputs + (1 - alpha) * (P @ outputs)
        avg_logits = debiased_embedding.dot(avg_outputs)

        logits_new = [avg_logits[token] for token in topk_raw]
        logits_new = np.array(logits_new)

        KL1.append(scipy.stats.entropy(logits_raw, logits_new))
        KL2.append(scipy.stats.entropy(logits_new, logits_raw))

    return (np.array(KL1) + np.array(KL2)) / 2


def topk_kl_overlap_subspace(prompt_text, k, tokenizer, model, embedding, mode, device):
    # 根据mode参数的值，确定使用的偏置子空间。如果mode[1]为"gender"，则使用与性别有关的偏置子空间；
    # 否则，使用与宗教有关的偏置子空间。这里根据具体的情况选择了两种不同的处理方式
    if mode[1] == "gender":
        if mode[0] == "direction":
            # 根据选择的偏置子空间，对词嵌入embedding进行处理，得到去偏版的词嵌入debiased_embedding。具体处理方式根据mode[0]的值进行选择。
            # 如果mode[0]为"direction"，则使用一个方向向量对词嵌入进行去偏；否则，使用一个子空间对词嵌入进行去偏。
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_direction.npy")
            debiased_embedding = np.array([drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        else:
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_subspace.npy")
            debiased_embedding = np.array(
                [dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
    else:
        religion_dir1 = np.load("../../data/bias_subspace/religion_direction1.npy")
        religion_dir2 = np.load("../../data/bias_subspace/religion_direction2.npy")
        religion_dir3 = np.load("../../data/bias_subspace/religion_direction3.npy")
        debiased_embedding = np.array([drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

    # 使用给定的tokenizer对prompt_text进行编码，生成输入模型的input_ids。
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    # original gpt2 model
    # 将input_ids移动到指定的device（CPU或GPU）上，并使用model的transformer部分对其进行前向传播。
    # 这将产生模型的输出outputs，它是最后一层隐藏状态的表示。
    input_ids = input_ids.to(device)
    outputs = model.transformer(input_ids=input_ids)[0][0][-1]  # (2, batch, len, dim)

    # 将outputs从GPU移动到CPU，并转换为NumPy数组。
    # 然后，通过将其与原始词嵌入embedding进行点积运算，得到对应于词汇表的logits（对数概率）
    outputs = outputs.cpu().detach().numpy()

    logits = embedding.dot(outputs)

    # 对logits进行排序，得到原始模型下的top-k个预测结果的索引（topk_raw）和对应的logits值（logits_raw）。
    old_rank = np.argsort(-logits).tolist()
    old_logits = np.sort(-logits).tolist()
    topk_raw = old_rank[:k]
    logits_raw = [-x for x in old_logits[:k]]

    # averaged hidden state debiased gpt2 model
    # 对于去偏版词嵌入debiased_embedding，将其与原始模型输出avg_outputs进行点积运算，得到新的logits值avg_logits
    avg_outputs = outputs
    avg_logits = debiased_embedding.dot(avg_outputs)

    # 对于top-k个预测结果，提取其对应的avg_logits值，并计算与原始模型下的logits值logits_raw之间的KL散度。
    # 得到KL散度值KL1和KL2。
    logits_new = []
    for i, token in enumerate(topk_raw):
        logits_new.append(avg_logits[token])
    logits_new = np.array(logits_new)

    KL1 = scipy.stats.entropy(logits_raw, logits_new)
    KL2 = scipy.stats.entropy(logits_new, logits_raw)

    return (KL1 + KL2) / 2

# 这段代码通过引入投影矩阵和向量归一化的方法，优化了去偏过程，使去偏后的结果更加合理和有效，从而更好地体现去偏效果。
def topk_kl_overlap_subspace1(prompt_text, k, tokenizer, model, embedding, mode, device):
    # 用于对向量进行归一化处理
    def normalize(x):
        norm = np.linalg.norm(x)
        return x if norm == 0 else x / norm

    # 通过投影矩阵将向量v投影到给定的子空间subspace上
    def project(v, subspace):
        subspace = normalize(subspace)
        if subspace.shape[1] != v.shape[0]:
            subspace = subspace.T
        projection_matrix = np.eye(v.shape[0]) - subspace @ subspace.T
        return projection_matrix @ v

    # 对整个嵌入矩阵进行去偏处理
    def debias_embeddings(embedding, subspace):
        debiased = []
        for vec in embedding:
            vec = normalize(vec)
            debiased_vec = project(vec, subspace)
            debiased.append(debiased_vec)
        return np.array(debiased)

    if mode[1] == "gender":
        if mode[0] == "direction":
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_direction.npy")
            gender_direction = gender_direction.reshape(-1, 1)  # 调整维度
            debiased_embedding = debias_embeddings(embedding, gender_direction)
        else:
            gender_subspace = np.load("../../data/bias_subspace/gpt2_gender_subspace.npy")
            debiased_embedding = debias_embeddings(embedding, gender_subspace)
    else:
        religion_dir1 = np.load("../../data/bias_subspace/religion_direction1.npy")
        religion_dir1 = religion_dir1.reshape(-1, 1)
        religion_dir2 = np.load("../../data/bias_subspace/religion_direction2.npy")
        religion_dir2 = religion_dir2.reshape(-1, 1)
        religion_dir3 = np.load("../../data/bias_subspace/religion_direction3.npy")
        religion_dir3 = religion_dir3.reshape(-1, 1)

        debiased_embedding = debias_embeddings(embedding, religion_dir1)
        debiased_embedding = debias_embeddings(debiased_embedding, religion_dir2)
        debiased_embedding = debias_embeddings(debiased_embedding, religion_dir3)

    # 使用tokenizer对提示文本进行编码
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
    # 使用模型的transformer部分对输入进行前向传播，得到最后一层隐藏状态的表示
    with torch.no_grad():
        outputs = model.transformer(input_ids=input_ids)[0][0][-1].cpu().numpy()

    # 得到对应于词汇表的logits。对logits进行排序，得到top-k预测结果的索引topk_raw和对应的logits值logits_raw。
    logits = embedding.dot(outputs)
    old_rank = np.argsort(-logits).tolist()
    old_logits = np.sort(-logits).tolist()
    topk_raw = old_rank[:k]
    logits_raw = [-x for x in old_logits[:k]]

    # 计算去偏后的logits
    avg_outputs = outputs
    avg_logits = debiased_embedding.dot(avg_outputs)

    # 对于top-k预测结果，提取其对应的新的logits值，并存储在logits_new中。
    logits_new = [avg_logits[token] for token in topk_raw]
    logits_new = np.array(logits_new)

    # 计算KL散度
    KL1 = scipy.stats.entropy(logits_raw, logits_new)
    KL2 = scipy.stats.entropy(logits_new, logits_raw)

    return (KL1 + KL2) / 2

"""
male_context 和 female_context 是两个包含上下文文本的列表或数组。这些上下文文本用于计算 KL 散度。
tokenizer 负责将文本转换为模型可以理解的输入格式，如将文本编码为 token IDs。
model 是一个已训练的模型对象，用于进行前向传播以生成输出。
embedding 是一个词嵌入矩阵，用于将模型的输出投影到 logits 空间，以便计算 KL 散度。
P 是一个投影矩阵，用于将模型的输出进行线性投影，以便计算 KL 散度。
A 是一个包含多个参数取值的列表或数组，用于进行线性插值。它控制了在计算 KL 散度时，原始模型输出和投影后的模型输出之间的权重。
device 用于指定计算设备，可以是 "cpu" 或 "cuda"。它决定了模型和数据在哪个设备上进行计算，以及是否使用 GPU 加速计算。
"""

def local_kl(male_context, female_context, tokenizer, model, embedding, P, A, device):
    kl1_avg = [0. for ii in range(len(A))]
    kl2_avg = [0. for ii in range(len(A))]
    for i in range(male_context.shape[0]):
        # 使用 tokenizer 对男性上下文进行编码，生成输入模型的 input_ids_m。
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_m = input_ids_m.to(device)

        # 将 outputs 移动到 CPU，并转换为 NumPy 数组。然后，通过将其与矩阵 P 进行点积运算，得到经过投影的输出 outputs_P。
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P = P.dot(outputs.T).T

        # 使用 tokenizer 对女性上下文进行编码，生成输入模型的 input_ids_f。
        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")

        # 使用 tokenizer 对女性上下文进行编码，生成输入模型的 input_ids_f。
        input_ids_f = input_ids_f.to(device)

        # 将 outputs_f 移动到 CPU，并转换为 NumPy 数组。
        # 然后，通过将其与矩阵 P 进行点积运算，得到经过投影的输出 outputs_P_f。
        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P_f = P.dot(outputs_f.T).T

        for a in range(len(A)):
            # 根据参数 A[a]，将 outputs_P 与 outputs 进行线性插值，得到新的 outputs_P。
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs

            # 使用词嵌入 embedding 与新的 outputs_P 进行点积运算，得到新的 logits new_logits。
            new_logits = embedding.dot(outputs_P)

            # 将 new_logits 转换为 PyTorch 的张量，并添加额外的维度。
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs_m = F.softmax(new_logits, dim=-1)

            outputs_P_f = (1 - A[a]) * outputs_P_f + A[a] * outputs_f
            new_logits_f = embedding.dot(outputs_P_f)
            new_logits_f = torch.from_numpy(new_logits_f).float()
            new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]

            # 将 new_logits 转换为 PyTorch 的张量，并添加额外的维度。
            probs_f = F.softmax(new_logits_f, dim=-1)

            # 计算 probs_m 和 probs_f 之间的 KL 散度 KL1，
            # 计算 probs_f 和 probs_m 之间的 KL 散度 KL2。
            KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
            KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

            kl1_avg[a] += KL1
            kl2_avg[a] += KL2

    return kl1_avg, kl2_avg


def local_Hellinger(male_context, female_context, tokenizer, model, embedding, P, A, device):
    # 这段代码创建了一个名为 kl1_avg 的列表，并初始化其所有元素为 0.0
    kl1_avg = [0. for ii in range(len(A))]
    kl2_avg = [0. for ii in range(len(A))]
    for i in range(male_context.shape[0]):
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_m = input_ids_m.to(device)
        outputs = model.transformer(input_ids=input_ids_m)[0][0][
            -1].cpu().detach().numpy()  # (2, batch, len, dim), embedding for male context

        # 通过将outputs与矩阵P进行点积运算，得到经过投影的输出outputs_P。
        # 通过点积运算，能够实现对于男性上下文的去偏
        # TODO
        outputs_P = P.dot(outputs.T).T  # debiased embedding for male context

        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_f = input_ids_f.to(device)
        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][
            -1].cpu().detach().numpy()  # (2, batch, len, dim), embedding for female context
        outputs_P_f = P.dot(outputs_f.T).T  # debiased embedding for female context

        for a in range(len(A)):
            # 将去偏表示(outputs_P)与原始表示(outputs)进行加权平均，使得在两者之间实现平滑过渡
            # TODO 调整列表A中的权重以实现对于去偏的优化
            ''' 
            通过调整列表A中的权重，可以探索不同的插值方案和权衡，以获得最佳的公平性去偏效果。
            这个线性插值操作的目的是在去偏表示和原始表示之间找到一个平衡点，以平滑且有效地解决公平性偏差问题。
            '''
            # 男性
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            '''
            倾向于去偏表示
            通过将较小的权重值（例如0.2）分配给alpha，可以使去偏表示(outputs_P)在加权平均中占据较小的比例，从而倾向于去偏。
            outputs_P = (1 - 0.2) * outputs_P + 0.2 * outputs
            '''
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]

            '''
            得到男性上下文的概率分布
            Softmax 是一种常用的激活函数，它可以将一个向量转换为概率分布。
            在这个特定的情况下，new_logits 是一个逻辑回归向量，其中的每个元素对应于男性上下文的概率得分。
            通过应用 softmax 函数，可以对 new_logits 进行归一化，将其转换为概率分布。softmax 函数的作用是对向量中的每个元素进行指数化，并将它们除以所有元素的总和，以确保得到的概率值在 0 到 1 之间，并且所有概率值的总和为 1。
            '''
            probs_m = F.softmax(new_logits, dim=-1)

            # 女性
            outputs_P_f = (1 - A[a]) * outputs_P_f + A[a] * outputs_f
            new_logits_f = embedding.dot(outputs_P_f)
            new_logits_f = torch.from_numpy(new_logits_f).float()
            new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
            probs_f = F.softmax(new_logits_f, dim=-1)

            '''
            Hellinger 距离是一种用于度量概率分布之间差异的统计指标。它的取值范围在 0 到 1 之间，其中 0 表示两个分布完全相同，1 表示两个分布完全不同。通过使用概率分布的平方根和乘法，上述代码计算了 Hellinger 距离的近似值，用于衡量男性上下文和女性上下文之间的差异程度
            计算了男女上下文的差异可以得到偏差程度
            并且反向计算KL2，
            '''
            hell1 = np.sqrt(1 - np.sum(np.sqrt(probs_m[0].detach().numpy() * probs_f[0].detach().numpy())))
            hell2 = np.sqrt(1 - np.sum(np.sqrt(probs_f[0].detach().numpy() * probs_m[0].detach().numpy())))
            # KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
            # KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

            kl1_avg[a] += hell1
            kl2_avg[a] += hell2

            # print(hell1)

    return kl1_avg, kl2_avg


def local_Hellinger1(male_context, female_context, tokenizer, model, embedding, P, A, device):
    kl1_avg = [0. for _ in range(len(A))]
    kl2_avg = [0. for _ in range(len(A))]

    for i in range(male_context.shape[0]):
        # 编码和获取嵌入
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt").to(device)
        outputs_m = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()

        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt").to(device)
        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()

        # 去偏
        outputs_P_m = P.dot(outputs_m.T).T
        outputs_P_f = P.dot(outputs_f.T).T

        # TODO 计算上下文相似度，用于调整权重
        sim_m = np.linalg.norm(outputs_m - outputs_P_m)
        sim_f = np.linalg.norm(outputs_f - outputs_P_f)
        # 动态调整权重
        adaptive_A = [min(1, sim_m / (sim_m + sim_f)) * a for a in A]

        for a in range(len(adaptive_A)):
            # 插值
            interpolated_m = (1 - A[a]) * outputs_P_m + A[a] * outputs_m
            interpolated_f = (1 - A[a]) * outputs_P_f + A[a] * outputs_f

            # 计算概率分布
            logits_m = embedding.dot(interpolated_m)
            logits_f = embedding.dot(interpolated_f)

            # 不一样
            probs_m = F.softmax(torch.from_numpy(logits_m).float(), dim=-1)
            probs_f = F.softmax(torch.from_numpy(logits_f).float(), dim=-1)

            # Hellinger 距离
            hell1 = np.sqrt(1 - np.sum(np.sqrt(probs_m[0].detach().numpy() * probs_f[0].detach().numpy())))
            hell2 = np.sqrt(1 - np.sum(np.sqrt(probs_f[0].detach().numpy() * probs_m[0].detach().numpy())))

            # TODO 其他度量标准，一起对于KL进行衡量
            js_divergence = 0.5 * (scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy()) +
                                   scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy()))

            kl1_avg[a] += hell1 + js_divergence
            kl2_avg[a] += hell2 + js_divergence

    kl1_avg = [val / male_context.shape[0] for val in kl1_avg]
    kl2_avg = [val / female_context.shape[0] for val in kl2_avg]

    return kl1_avg, kl2_avg

def local_Hellinger_sensitive(male_context, female_context, tokenizer, model, embedding, P, device):
    stop_word = np.loadtxt(open("../../data/stopword.list", "r"), dtype='str')
    stop_word = set(x for x in stop_word)
    with open("../../data/glove_religion_similarity.json") as ff:
        similarity = json.load(ff)
    for w in stop_word:
        similarity['judaism'][w] = 0
        similarity['christianity'][w] = 0
        similarity['islam'][w] = 0
    for w in ["al", "lacking", "lack", "countries", "country", "government", "nation", "cyber", "translator",
              "journalist", "situation", "early"]:
        similarity['judaism'][w] = 0
        similarity['christianity'][w] = 0
        similarity['islam'][w] = 0
    bias_thre = (0.16, 0.15, 0.17)

    kl1_avg = 0.
    kl2_avg = 0.
    for i in range(male_context.shape[0]):
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_m = input_ids_m.to(device)
        # sensitive ----
        model_inputs = model.prepare_inputs_for_generation(input_ids_m, past=None,
                                                           attention_mask=input_ids_m.new_ones(input_ids_m.shape),
                                                           use_cache=True)

        outputs = model(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]  # batch * vocab
        logits_filter = top_k_top_p_filtering(next_token_logits, top_p=0.7)  # batch * vocab
        top_p_mask = logits_filter.eq(-float("Inf"))

        top_k_tokens = []
        ii = 0
        tmp = (top_p_mask == False)[ii].nonzero().cpu().detach().numpy().tolist()  # batch tuple
        top_k_tokens.append([int(x) for x in torch.sort(-logits_filter[ii])[1][:len(tmp)]])
        probs_bias = F.softmax(logits_filter, dim=-1).cpu().detach().numpy()  # batch * vocab

        normalize = 0
        jew_token, christian_token, muslim_token = [], [], []
        jew_prob, christian_prob, muslim_prob = 0, 0, 0
        token_num = 0
        for t in top_k_tokens[ii]:
            glove_token = tokenizer.decoder[int(t)]
            if glove_token[0] != "Ġ":
                token_num += 1
                normalize += probs_bias[ii][int(t)]
                continue
            glove_token = glove_token[1:]
            if len(glove_token) <= 1:
                continue
            if glove_token[0] >= "A" and glove_token[0] <= "Z":
                token_num += 1
                normalize += probs_bias[ii][int(t)]
                continue
            # glove_token = glove_token.lower()
            if glove_token in similarity["judaism"].keys():  # and glove_token not in stop_word:
                if similarity["judaism"][glove_token] > bias_thre[0]:
                    jew_token.append(tokenizer.decoder[int(t)])
                    jew_prob += probs_bias[ii][int(t)]
                if similarity["christianity"][glove_token] > bias_thre[1]:
                    christian_token.append(tokenizer.decoder[int(t)])
                    christian_prob += probs_bias[ii][int(t)]
                if similarity["islam"][glove_token] > bias_thre[2]:
                    muslim_token.append(tokenizer.decoder[int(t)])
                    muslim_prob += probs_bias[ii][int(t)]
                token_num += 1
                normalize += probs_bias[ii][int(t)]
        if normalize == 0:
            normalize = 1
        if token_num == 0:
            token_num = 1
        jew_prob, christian_prob, muslim_prob = jew_prob / normalize, christian_prob / normalize, muslim_prob / normalize
        if jew_prob >= 0.1 or muslim_prob >= 0.1 or christian_prob >= 0.1 or len(
                jew_token) / token_num >= 0.3 or len(muslim_token) / token_num >= 0.3 or len(
            christian_token) / token_num >= 0.3:  # or tokenizer.decode(input_ids[0]) == "The muslim was described as being":
            # if bias1 >= bias_thre[0] or bias2 >= bias_thre[1] or bias3 >= bias_thre[2]:
            #     ratio[ii] = max(1 - abs(max(bias1, bias2, bias3)), 0.6)
            A_m = 0.7
        else:
            A_m = 1

        # ---------

        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_f = input_ids_f.to(device)

        # sensitive ----
        model_inputs = model.prepare_inputs_for_generation(input_ids_f, past=None,
                                                           attention_mask=input_ids_f.new_ones(input_ids_f.shape),
                                                           use_cache=True)

        outputs = model(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]  # batch * vocab
        logits_filter = top_k_top_p_filtering(next_token_logits, top_p=0.7)  # batch * vocab
        top_p_mask = logits_filter.eq(-float("Inf"))

        top_k_tokens = []
        ii = 0
        tmp = (top_p_mask == False)[ii].nonzero().cpu().detach().numpy().tolist()  # batch tuple
        top_k_tokens.append([int(x) for x in torch.sort(-logits_filter[ii])[1][:len(tmp)]])
        probs_bias = F.softmax(logits_filter, dim=-1).cpu().detach().numpy()  # batch * vocab

        normalize = 0
        jew_token, christian_token, muslim_token = [], [], []
        jew_prob, christian_prob, muslim_prob = 0, 0, 0
        token_num = 0
        for t in top_k_tokens[ii]:
            glove_token = tokenizer.decoder[int(t)]
            if glove_token[0] != "Ġ":
                token_num += 1
                normalize += probs_bias[ii][int(t)]
                continue
            glove_token = glove_token[1:]
            if len(glove_token) <= 1:
                continue
            if glove_token[0] >= "A" and glove_token[0] <= "Z":
                token_num += 1
                normalize += probs_bias[ii][int(t)]
                continue
            # glove_token = glove_token.lower()
            if glove_token in similarity["judaism"].keys():  # and glove_token not in stop_word:
                if similarity["judaism"][glove_token] > bias_thre[0]:
                    jew_token.append(tokenizer.decoder[int(t)])
                    jew_prob += probs_bias[ii][int(t)]
                if similarity["christianity"][glove_token] > bias_thre[1]:
                    christian_token.append(tokenizer.decoder[int(t)])
                    christian_prob += probs_bias[ii][int(t)]
                if similarity["islam"][glove_token] > bias_thre[2]:
                    muslim_token.append(tokenizer.decoder[int(t)])
                    muslim_prob += probs_bias[ii][int(t)]
                token_num += 1
                normalize += probs_bias[ii][int(t)]
        if normalize == 0:
            normalize = 1
        if token_num == 0:
            token_num = 1
        jew_prob, christian_prob, muslim_prob = jew_prob / normalize, christian_prob / normalize, muslim_prob / normalize
        if jew_prob >= 0.1 or muslim_prob >= 0.1 or christian_prob >= 0.1 or len(
                jew_token) / token_num >= 0.3 or len(muslim_token) / token_num >= 0.3 or len(
            christian_token) / token_num >= 0.3:  # or tokenizer.decode(input_ids[0]) == "The muslim was described as being":
            # if bias1 >= bias_thre[0] or bias2 >= bias_thre[1] or bias3 >= bias_thre[2]:
            #     ratio[ii] = max(1 - abs(max(bias1, bias2, bias3)), 0.6)
            A_f = 0.7
        else:
            A_f = 1
        print(A_f, A_m)

        # ---------

        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P = P.dot(outputs.T).T

        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P_f = P.dot(outputs_f.T).T

        outputs_P = (1 - A_m) * outputs_P + A_m * outputs
        new_logits = embedding.dot(outputs_P)
        new_logits = torch.from_numpy(new_logits).float()
        new_logits = new_logits.unsqueeze(0)  # [1, 50257]
        probs_m = F.softmax(new_logits, dim=-1)

        outputs_P_f = (1 - A_f) * outputs_P_f + A_f * outputs_f
        new_logits_f = embedding.dot(outputs_P_f)
        new_logits_f = torch.from_numpy(new_logits_f).float()
        new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
        probs_f = F.softmax(new_logits_f, dim=-1)

        hell1 = np.sqrt(1 - np.sum(np.sqrt(probs_m[0].detach().numpy() * probs_f[0].detach().numpy())))
        hell2 = np.sqrt(1 - np.sum(np.sqrt(probs_f[0].detach().numpy() * probs_m[0].detach().numpy())))
        # KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
        # KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

        kl1_avg += hell1
        kl2_avg += hell2

    return kl1_avg, kl2_avg


'''
male_context：男性上下文的输入
female_context：女性上下文的输入
tokenizer：用于对文本进行编码的分词器
model：语言模型
embedding：输入文本的嵌入表示
mode：模式参数，指定了公平性的类型和方法
device：运行代码的设备（如 CPU 或 GPU）
'''
def local_Hellinger_subspace(male_context, female_context, tokenizer, model, embedding, mode, device):
    # 根据 mode 参数的值，确定公平性的类型（性别或宗教）和方法（方向或子空间）
    # 对输入文本的嵌入表示进行去偏操作，得到去偏后的嵌入向量 debiased_embedding。
    if mode[1] == "gender":
        if mode[0] == "direction":
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_direction.npy")
            debiased_embedding = np.array([drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        else:
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_subspace.npy")
            debiased_embedding = np.array(
                [dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        # self.embedding.to(self.args.device)
    elif mode[1] == "religion":
        religion_dir1 = np.load("../../data/bias_subspace/religion_direction1.npy")
        religion_dir2 = np.load("../../data/bias_subspace/religion_direction2.npy")
        religion_dir3 = np.load("../../data/bias_subspace/religion_direction3.npy")
        debiased_embedding = np.array([drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

    kl1_avg = 0.
    kl2_avg = 0.
    for i in range(male_context.shape[0]):
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_m = input_ids_m.to(device)
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        # outputs_P = P.dot(outputs.T).T
        new_logits = debiased_embedding.dot(outputs)
        new_logits = torch.from_numpy(new_logits).float()
        new_logits = new_logits.unsqueeze(0)  # [1, 50257]
        probs_m = F.softmax(new_logits, dim=-1)

        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_f = input_ids_f.to(device)
        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        # outputs_P_f = P.dot(outputs_f.T).T
        new_logits_f = debiased_embedding.dot(outputs_f)
        new_logits_f = torch.from_numpy(new_logits_f).float()
        new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
        probs_f = F.softmax(new_logits_f, dim=-1)

        hell1 = np.sqrt(1 - np.sum(np.sqrt(probs_m[0].detach().numpy() * probs_f[0].detach().numpy())))
        hell2 = np.sqrt(1 - np.sum(np.sqrt(probs_f[0].detach().numpy() * probs_m[0].detach().numpy())))
        # KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
        # KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

        kl1_avg += hell1
        kl2_avg += hell2

    return kl1_avg, kl2_avg


def local_Hellinger_subspace1(male_context, female_context, tokenizer, model, embedding, mode, device):
    # 根据 mode 参数的值，确定公平性的类型（性别或宗教）和方法（方向或子空间）
    if mode[1] == "gender":
        if mode[0] == "direction":
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_direction.npy")
            debiased_embedding = np.array([drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        else:
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_subspace.npy")
            debiased_embedding = np.array(
                [dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
    elif mode[1] == "religion":
        religion_dir1 = np.load("../../data/bias_subspace/religion_direction1.npy")
        religion_dir2 = np.load("../../data/bias_subspace/religion_direction2.npy")
        religion_dir3 = np.load("../../data/bias_subspace/religion_direction3.npy")
        debiased_embedding = np.array([drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

    kl1_avg = 0.
    kl2_avg = 0.
    js_avg = 0.

    for i in range(male_context.shape[0]):
        # 编码和获取嵌入
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt").to(device)
        outputs_m = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()

        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt").to(device)
        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()

        # 计算概率分布
        new_logits_m = debiased_embedding.dot(outputs_m)
        new_logits_f = debiased_embedding.dot(outputs_f)

        probs_m = F.softmax(torch.from_numpy(new_logits_m).float().unsqueeze(0), dim=-1)
        probs_f = F.softmax(torch.from_numpy(new_logits_f).float().unsqueeze(0), dim=-1)

        # Hellinger 距离
        hell1 = np.sqrt(1 - np.sum(np.sqrt(probs_m[0].detach().numpy() * probs_f[0].detach().numpy())))
        hell2 = np.sqrt(1 - np.sum(np.sqrt(probs_f[0].detach().numpy() * probs_m[0].detach().numpy())))

        # JS 散度
        js_divergence = 0.5 * (scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy()) +
                               scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy()))

        kl1_avg += hell1 + js_divergence
        kl2_avg += hell2 + js_divergence
        js_avg += js_divergence

    kl1_avg /= male_context.shape[0]
    kl2_avg /= female_context.shape[0]
    js_avg /= male_context.shape[0]

    return kl1_avg, kl2_avg

'''
weat_dataset：WEAT数据集
weat_pos：WEAT数据集中的正例索引
model：语言模型
embedding：词嵌入表示
A：一组权重系数
P：一个偏置矩阵
p：用于 Top-k Top-p 过滤的参数
device：运行代码的设备（如 CPU 或 GPU）
topk：一个布尔值，指示是否使用 Top-k 过滤

用于计算 WEAT（Word Embedding Association Test）数据集的真实标签。
'''
def weat_true_label(weat_dataset, weat_pos, model, embedding, A, P, p, device, topk=False):
    if topk:
        weat_topk = 0.
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            logits = embedding.dot(outputs)
            logits_filter = torch.from_numpy(logits).float().clone()
            logits_filter = logits_filter.unsqueeze(0)
            logits_filter = top_k_top_p_filtering(logits_filter, top_p=p)
            top_p_mask = logits_filter.eq(-float("Inf"))

            outputs_P = P.dot(outputs.T).T
            new_logits = embedding.dot(outputs_P)
            logits_filter = torch.from_numpy(new_logits).float().clone()
            logits_filter = logits_filter.unsqueeze(0)
            logits_filter.masked_fill_(top_p_mask, -float("Inf"))
            probs_m = F.softmax(logits_filter, dim=-1).detach().numpy()

            weat_topk += probs_m[0][weat_pos[i]]
            count += 1
        return weat_topk / count

    weat_avg = [0. for ii in range(len(A))]
    count = 0
    for i in range(len(weat_dataset)):
        # 将输入文本转换为张量 input_ids_m，并将其发送到设备上。
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)

        # 使用语言模型进行前向传播，得到输出 outputs。
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        # 通过权重矩阵 P 对输出表示 outputs 进行加权平均，得到加权后的输出表示 outputs_P。
        outputs_P = P.dot(outputs.T).T
        for a in range(len(A)):
            # 根据当前权重系数 A[a] 对加权后的输出表示 outputs_P 进行更新，得到更新后的加权输出表示。
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs

            # 将 outputs_P 与词嵌入表示 embedding 进行点积运算，得到新的 logits。
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]

            # 将新的 logits 转换为概率分布，使用 softmax 函数得到概率分布 probs_m。
            probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

            # 遍历权重系数列表 A，对于每个权重系数 a，计算该权重下的平均概率值，并将结果累加到 weat_avg 列表的相应位置。
            weat_avg[a] += probs_m[0][weat_pos[i]]

        # 增加计数器 count 的值，表示已处理的输入文本数量。
        count += 1

    return [x / count for x in weat_avg]

def weat_true_label1(weat_dataset, weat_pos, model, embedding, A, P, p, device, topk=False):
    if topk:
        weat_topk = 0.
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = model.transformer(input_ids=input_ids_m).last_hidden_state[0][-1].cpu().detach().numpy()
            logits = embedding.dot(outputs)
            logits_filter = torch.from_numpy(logits).float().clone().unsqueeze(0)
            logits_filter = top_k_top_p_filtering(logits_filter, top_p=p)
            top_p_mask = logits_filter.eq(-float("Inf"))

            outputs_P = P.dot(outputs.T).T
            new_logits = embedding.dot(outputs_P)
            logits_filter = torch.from_numpy(new_logits).float().clone().unsqueeze(0)
            logits_filter.masked_fill_(top_p_mask, -float("Inf"))
            probs_m = F.softmax(logits_filter, dim=-1).detach().numpy()

            weat_topk += probs_m[0][weat_pos[i]]
            count += 1
        return weat_topk / count

    weat_avg = torch.zeros(len(A))
    count = 0
    similarity_threshold = 0.9  # 多样性约束的阈值
    diversity_penalty = 0.1  # 多样性约束的惩罚项

    for i in range(len(weat_dataset)):
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)

        outputs = model.transformer(input_ids=input_ids_m).last_hidden_state[0][-1].cpu().detach().numpy()
        outputs_P_initial = P.dot(outputs.T).T
        outputs = torch.from_numpy(outputs).float()
        outputs_P_initial = torch.from_numpy(outputs_P_initial).float()

        context_vector = outputs.mean(dim=0)  # 增加上下文向量

        prev_outputs_P = None


        for idx, a in enumerate(A):
            # TODO 使用矩阵运算进行加权平均，并引入上下文敏感性
            outputs_P = (1 - a) * outputs_P_initial + a * (outputs + context_vector)

            if prev_outputs_P is not None:
                # TODO 计算当前输出与前一个输出的余弦相似度
                cos_sim = cosine_similarity(outputs_P.numpy().reshape(1, -1), prev_outputs_P.numpy().reshape(1, -1))[0][0]
                if cos_sim > similarity_threshold:
                    # 引入多样性约束的惩罚项
                    outputs_P = outputs_P - diversity_penalty * cos_sim * (outputs_P - prev_outputs_P)

            prev_outputs_P = outputs_P.clone()

            new_logits = embedding.dot(outputs_P.numpy())
            new_logits = torch.from_numpy(new_logits).float().unsqueeze(0)
            new_logits = new_logits / new_logits.sum(dim=-1, keepdim=True)  # 归一化处理

            probs_m = F.softmax(new_logits, dim=-1).detach()

            weat_avg[idx] += probs_m[0, weat_pos[i]]

        count += 1

    return (weat_avg / count).tolist()


class AdversarialModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdversarialModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x.squeeze())


def weat_true_label_improved1(weat_dataset, weat_pos, model, embedding, A, P, p, device, topk=False):
    # 对抗性模型
    adversarial_model = AdversarialModel(input_dim=embedding.shape[1], hidden_dim=256,
                                         output_dim=embedding.shape[1]).to(device)

    if topk:
        weat_topk = 0.
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = model.transformer(input_ids=input_ids_m).last_hidden_state[0][-1].cpu().detach().numpy()
            logits = embedding.dot(outputs)
            logits_filter = torch.from_numpy(logits).float().clone().unsqueeze(0)
            logits_filter = top_k_top_p_filtering(logits_filter, top_p=p)
            top_p_mask = logits_filter.eq(-float("Inf"))

            outputs_P = P.dot(outputs.T).T
            new_logits = embedding.dot(outputs_P)
            logits_filter = torch.from_numpy(new_logits).float().clone().unsqueeze(0)
            logits_filter.masked_fill_(top_p_mask, -float("Inf"))
            probs_m = F.softmax(logits_filter, dim=-1).detach().numpy()

            weat_topk += probs_m[0][weat_pos[i]]
            count += 1
        return weat_topk / count

    weat_avg = torch.zeros(len(A))
    count = 0
    similarity_threshold = 0.9  # 多样性约束的阈值
    diversity_penalty = 0.1  # 多样性约束的惩罚项

    for i in range(len(weat_dataset)):
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)

        outputs = model.transformer(input_ids=input_ids_m).last_hidden_state[0][-1].cpu().detach().numpy()
        outputs_P_initial = P.dot(outputs.T).T
        outputs = torch.from_numpy(outputs).float().to(device)
        outputs_P_initial = torch.from_numpy(outputs_P_initial).float().to(device)

        context_vector = outputs.mean(dim=0)  # 增加上下文向量

        prev_outputs_P = None

        for idx, a in enumerate(A):
            outputs_P = (1 - a) * outputs_P_initial + a * (outputs + context_vector)

            if prev_outputs_P is not None:
                cos_sim = \
                cosine_similarity(outputs_P.cpu().numpy().reshape(1, -1), prev_outputs_P.cpu().numpy().reshape(1, -1))[
                    0][0]
                if cos_sim > similarity_threshold:
                    outputs_P = outputs_P - diversity_penalty * cos_sim * (outputs_P - prev_outputs_P)

            prev_outputs_P = outputs_P.clone()

            new_logits = embedding.dot(outputs_P.cpu().numpy())
            new_logits = torch.from_numpy(new_logits).float().unsqueeze(0).to(device)
            new_logits = new_logits / new_logits.sum(dim=-1, keepdim=True)

            probs_m = F.softmax(new_logits, dim=-1).detach()

            weat_avg[idx] += probs_m[0, weat_pos[i]]

        count += 1

    return (weat_avg / count).tolist()

def weat_true_label_improved(weat_dataset, weat_pos, model, embedding, A, P, p, device, topk=False):
    # 对抗性模型
    adversarial_model = AdversarialModel(input_dim=embedding.shape[1], hidden_dim=256,
                                         output_dim=embedding.shape[1]).to(device)
    adversarial_optimizer = torch.optim.Adam(adversarial_model.parameters(), lr=0.001)

    if topk:
        weat_topk = 0.
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = model.transformer(input_ids=input_ids_m).last_hidden_state[0][-1].cpu().detach().numpy()
            logits = embedding.dot(outputs)
            logits_filter = torch.from_numpy(logits).float().clone().unsqueeze(0)
            logits_filter = top_k_top_p_filtering(logits_filter, top_p=p)
            top_p_mask = logits_filter.eq(-float("Inf"))

            outputs_P = P.dot(outputs.T).T
            new_logits = embedding.dot(outputs_P)
            logits_filter = torch.from_numpy(new_logits).float().clone().unsqueeze(0)
            logits_filter.masked_fill_(top_p_mask, -float("Inf"))
            probs_m = F.softmax(logits_filter, dim=-1).detach().numpy()

            weat_topk += probs_m[0][weat_pos[i]]
            count += 1
        return weat_topk / count

    weat_avg = torch.zeros(len(A))
    count = 0
    similarity_threshold = 0.9  # 多样性约束的阈值
    diversity_penalty = 0.1  # 多样性约束的惩罚项

    for i in range(len(weat_dataset)):
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)

        outputs = model.transformer(input_ids=input_ids_m).last_hidden_state[0][-1].cpu().detach().numpy()
        outputs_P_initial = P.dot(outputs.T).T
        outputs = torch.from_numpy(outputs).float().to(device)
        outputs_P_initial = torch.from_numpy(outputs_P_initial).float().to(device)

        context_vector = outputs.mean(dim=0)  # 增加上下文向量

        prev_outputs_P = None

        for idx, a in enumerate(A):
            outputs_P = (1 - a) * outputs_P_initial + a * (outputs + context_vector)

            if prev_outputs_P is not None:
                cos_sim = \
                cosine_similarity(outputs_P.cpu().numpy().reshape(1, -1), prev_outputs_P.cpu().numpy().reshape(1, -1))[
                    0][0]
                if cos_sim > similarity_threshold:
                    outputs_P = outputs_P - diversity_penalty * cos_sim * (outputs_P - prev_outputs_P)

            prev_outputs_P = outputs_P.clone()

            # 对抗性训练部分
            adversarial_model.train()
            adversarial_optimizer.zero_grad()
            adv_outputs_P = adversarial_model(outputs_P)
            adv_loss = F.mse_loss(adv_outputs_P, outputs_P)
            adv_loss.backward()
            adversarial_optimizer.step()

            with torch.no_grad():
                adv_outputs_P = adversarial_model(outputs_P)

            new_logits = embedding.dot(adv_outputs_P.cpu().numpy())
            new_logits = torch.from_numpy(new_logits).float().unsqueeze(0).to(device)
            new_logits = new_logits / new_logits.sum(dim=-1, keepdim=True)

            probs_m = F.softmax(new_logits, dim=-1).detach()

            weat_avg[idx] += probs_m[0, weat_pos[i]]

        count += 1

    return (weat_avg / count).tolist()

def weat_true_label_sensitive(weat_dataset, weat_pos, model, embedding, mode, p, device, topk=False):
    if topk:
        weat_topk = 0.
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            logits = embedding.dot(outputs)
            logits_filter = torch.from_numpy(logits).float().clone()
            logits_filter = logits_filter.unsqueeze(0)
            logits_filter = top_k_top_p_filtering(logits_filter, top_p=p)
            top_p_mask = logits_filter.eq(-float("Inf"))

            outputs_P = P.dot(outputs.T).T
            new_logits = embedding.dot(outputs_P)
            logits_filter = torch.from_numpy(new_logits).float().clone()
            logits_filter = logits_filter.unsqueeze(0)
            logits_filter.masked_fill_(top_p_mask, -float("Inf"))
            probs_m = F.softmax(logits_filter, dim=-1).detach().numpy()

            weat_topk += probs_m[0][weat_pos[i]]
            count += 1
        return weat_topk / count

    weat_avg = [0. for ii in range(len(A))]
    count = 0
    for i in range(len(weat_dataset)):
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P = P.dot(outputs.T).T
        for a in range(len(A)):
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

            weat_avg[a] += probs_m[0][weat_pos[i]]
        count += 1
    return [x / count for x in weat_avg]

# TODO 引入多样性约束
def diversity_constraint(embedding, alpha=0.5):
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    normed_embedding = embedding / norm
    similarity_matrix = np.dot(normed_embedding, normed_embedding.T)
    diversity_penalty = np.sum(np.triu(similarity_matrix, 1))
    return embedding - alpha * diversity_penalty
def dynamic_drop(embedding, direction, context_weight):
    return embedding - context_weight * np.dot(embedding, direction) * direction
def weat_true_label_subspace(weat_dataset, weat_pos, model, embedding, mode, p, device, topk=False):
    def load_and_apply_subspace_with_weight(embedding, direction_path, weight):
        direction = np.load(direction_path)
        return np.array([dynamic_drop(e, direction, weight) for e in embedding])

    # 根据mode参数的设置，函数加载预定义的偏置子空间
    if mode[1] == "gender":
        if mode[0] == "direction":
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_direction.npy")
            debiased_embedding = np.array([drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        else:
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_subspace.npy")
            debiased_embedding = np.array(
                [dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        # self.embedding.to(self.args.device)
    elif mode[1] == "religion":
        religion_dir1 = np.load("../../data/bias_subspace/religion_direction1.npy")
        religion_dir2 = np.load("../../data/bias_subspace/religion_direction2.npy")
        religion_dir3 = np.load("../../data/bias_subspace/religion_direction3.npy")
        debiased_embedding = np.array([drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

    # TODO 引入多样性约束
    debiased_embedding = diversity_constraint(debiased_embedding)

    # 初始化weat_avg为0，用于计算WEAT指标的总和。count用于跟踪处理的文本样本数量。
    weat_avg = 0.
    count = 0
    for i in range(len(weat_dataset)):
        # 对于WEAT数据集中的每个样本，将输入文本转换为张量input_ids_m并将其发送到指定的设备。
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)

        # 通过模型对输入文本进行前向传播，得到输出张量outputs。这里的outputs是模型在输入文本上的最后一层隐藏状态。
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        # outputs_P = P.dot(outputs.T).T
        # for a in range(len(A)):
        #     outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
        new_logits = debiased_embedding.dot(outputs)
        new_logits = torch.from_numpy(new_logits).float()
        new_logits = new_logits.unsqueeze(0)  # [1, 50257]
        probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

        # weat_avg[a] += probs_m[0][weat_pos[i]]
        '''
        将目标单词的概率值（probs_m[0][weat_pos[i]]）累加到weat_avg中。
        probs_m[0]表示第一个样本的概率分布。
        weat_pos[i]表示在WEAT数据集中的目标单词的索引。
        '''
        weat_avg += probs_m[0][weat_pos[i]]
        count += 1
    return weat_avg / count


def weat_true_label_subspace1(weat_dataset, weat_pos, model, embedding, mode, p, device, topk=False):
    def load_and_apply_subspace_with_weight(embedding, direction_path, weight):
        direction = np.load(direction_path)
        return np.array([dynamic_drop(e, direction, weight) for e in embedding])

    # 根据mode参数的设置，函数加载预定义的偏置子空间
    if mode[1] == "gender":
        if mode[0] == "direction":
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_direction.npy")
            debiased_embedding = np.array([drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        else:
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_subspace.npy")
            debiased_embedding = np.array(
                [dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
    elif mode[1] == "religion":
        religion_dir1 = np.load("../../data/bias_subspace/religion_direction1.npy")
        religion_dir2 = np.load("../../data/bias_subspace/religion_direction2.npy")
        religion_dir3 = np.load("../../data/bias_subspace/religion_direction3.npy")
        debiased_embedding = np.array([drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

    # 初始化weat_avg为0，用于计算WEAT指标的总和。count用于跟踪处理的文本样本数量。
    weat_avg = 0.
    count = 0
    similarity_threshold = 0.9  # 多样性约束的阈值
    diversity_penalty = 0.1  # 多样性约束的惩罚项

    for i in range(len(weat_dataset)):
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)

        outputs = model.transformer(input_ids=input_ids_m).last_hidden_state[0][-1].cpu().detach().numpy()
        outputs = torch.from_numpy(outputs).float()

        context_vector = outputs.mean(dim=0)  # 增加上下文向量

        prev_outputs_P = None

        for a in [0.1, 0.3, 0.5, 0.7, 0.9]:  # 假设A是从0.1到0.9的一个集合
            # 使用矩阵运算进行加权平均，并引入上下文敏感性
            outputs_P = (1 - a) * outputs + a * (outputs + context_vector)

            if prev_outputs_P is not None:
                # 计算当前输出与前一个输出的余弦相似度
                cos_sim = cosine_similarity(outputs_P.numpy().reshape(1, -1), prev_outputs_P.numpy().reshape(1, -1))[0][
                    0]
                if cos_sim > similarity_threshold:
                    # 引入多样性约束的惩罚项
                    outputs_P = outputs_P - diversity_penalty * cos_sim * (outputs_P - prev_outputs_P)

            prev_outputs_P = outputs_P.clone()

            new_logits = debiased_embedding.dot(outputs_P.numpy())
            new_logits = torch.from_numpy(new_logits).float().unsqueeze(0)
            new_logits = new_logits / new_logits.sum(dim=-1, keepdim=True)  # 归一化处理

            probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

            weat_avg += probs_m[0][weat_pos[i]]
            count += 1

    return weat_avg / count

def local_kl_reverse(occ_context, tokenizer, model, embedding, pairs_id, A, P, device):
    kl = [0. for ii in range(len(A))]
    for i in range(occ_context.shape[0]):
        input_ids = tokenizer.encode(occ_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(device)
        outputs = model.transformer(input_ids=input_ids)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P = P.dot(outputs.T).T

        for a in range(len(A)):
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs = F.softmax(new_logits, dim=-1)
            probs = probs.cpu().detach().numpy()

            occ_prob1 = 0.
            occ_prob2 = 0.
            for p1, p2 in pairs_id:
                occ_prob1 += probs[0][p1]
                occ_prob2 += probs[0][p2]

            tmp_kl1 = 0.
            tmp_kl2 = 0.
            for p1, p2 in pairs_id:
                tmp_kl1 += probs[0][p1] / occ_prob1 * np.log(probs[0][p1] / occ_prob1 / probs[0][p2] * occ_prob2)
                tmp_kl2 += probs[0][p2] / occ_prob2 * np.log(probs[0][p2] / occ_prob2 / probs[0][p1] * occ_prob1)
            kl[a] += (tmp_kl1 + tmp_kl2) / 2

    return kl


def local_kl_reverse_geometry(occ_context, tokenizer, model, embedding, pairs_id, num_components=2, device="cpu"):
    def doPCA(pairs, num_components=10):
        matrix = []
        for a, b in pairs:
            center = (a + b) / 2
            norm_a = a - center
            norm_b = b - center
            norm_a, norm_b = norm_a.detach().numpy(), norm_b.detach().numpy()
            # norm_a, norm_b = norm_a/np.linalg.norm(norm_a), norm_b/np.linalg.norm(norm_b)
            matrix.append(norm_a)
            matrix.append(norm_b)
        matrix = np.array(matrix)
        pca = PCA(n_components=num_components, svd_solver="full")
        pca.fit(matrix)  # Produce different results each time...
        return pca

    def dropspace(u, V):
        # u, V = u.detach().numpy(), V.detach().numpy()
        norm_sqrd = np.sum(V * V, axis=-1)
        vecs = np.divide(V @ u, norm_sqrd)[:, None] * V
        subspace = np.sum(vecs, axis=0)
        return u - subspace

    pairs = []
    for female, male in pairs_id:
        female_feat, male_feat = embedding[female], embedding[male]
        female_feat, male_feat = female_feat / np.linalg.norm(female_feat), male_feat / np.linalg.norm(male_feat)
        if type(male_feat) is np.ndarray:
            female_feat, male_feat = torch.from_numpy(female_feat), torch.from_numpy(male_feat)
        pairs.append((female_feat, male_feat))
    pca_res = doPCA(pairs, num_components=num_components)
    print("pca_res.explained_variance_ratio_: ", pca_res.explained_variance_ratio_)
    print("pca shape", pca_res.components_.shape)
    gender_dir1 = torch.from_numpy(pca_res.components_[0])
    gender_dir2 = torch.from_numpy(pca_res.components_[1])
    # gender_dir = torch.from_numpy(pca_res.components_[:num_components])
    gender_dir = pca_res.components_[:num_components]

    # kl = [0. for ii in range(len(A))]
    kl = 0.
    for i in range(occ_context.shape[0]):
        input_ids = tokenizer.encode(occ_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(device)
        outputs = model.transformer(input_ids=input_ids)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        logits = embedding.dot(outputs)
        logits = torch.from_numpy(logits).float()
        logits = logits.unsqueeze(0)
        probs = F.softmax(logits, dim=-1)
        probs = probs.cpu().detach().numpy()

        occ_prob1 = 0.
        occ_prob2 = 0.
        for p1, p2 in pairs_id:
            occ_prob1 += probs[0][p1]
            occ_prob2 += probs[0][p2]

        tmp_kl1 = 0.
        tmp_kl2 = 0.
        for p1, p2 in pairs_id:
            tmp_kl1 += probs[0][p1] / occ_prob1 * np.log(probs[0][p1] / occ_prob1 / probs[0][p2] * occ_prob2)
            tmp_kl2 += probs[0][p2] / occ_prob2 * np.log(probs[0][p2] / occ_prob2 / probs[0][p1] * occ_prob1)
        kl += (tmp_kl1 + tmp_kl2) / 2

    tmp = model.lm_head.weight.data
    model.lm_head.weight.data = torch.from_numpy(
        np.array([dropspace(embedding[i], gender_dir) for i in range(embedding.shape[0])]))

    kl_debias = 0.
    for i in range(occ_context.shape[0]):
        input_ids = tokenizer.encode(occ_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(device)
        outputs = model.transformer(input_ids=input_ids)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        logits = embedding.dot(outputs)
        logits = torch.from_numpy(logits).float()
        logits = logits.unsqueeze(0)
        probs = F.softmax(logits, dim=-1)
        probs = probs.cpu().detach().numpy()

        occ_prob1 = 0.
        occ_prob2 = 0.
        for p1, p2 in pairs_id:
            occ_prob1 += probs[0][p1]
            occ_prob2 += probs[0][p2]

        tmp_kl1 = 0.
        tmp_kl2 = 0.
        for p1, p2 in pairs_id:
            tmp_kl1 += probs[0][p1] / occ_prob1 * np.log(probs[0][p1] / occ_prob1 / probs[0][p2] * occ_prob2)
            tmp_kl2 += probs[0][p2] / occ_prob2 * np.log(probs[0][p2] / occ_prob2 / probs[0][p1] * occ_prob1)
        kl_debias += (tmp_kl1 + tmp_kl2) / 2

        # outputs_P = P.dot(outputs.T).T

        # for a in range(len(A)):
        #     outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
        #     new_logits = embedding.dot(outputs_P)
        #     new_logits = torch.from_numpy(new_logits).float()
        #     new_logits = new_logits.unsqueeze(0)  # [1, 50257]
        #     probs = F.softmax(new_logits, dim=-1)
        #     probs = probs.cpu().detach().numpy()

    model.lm_head.weight.data = tmp

    return kl, kl_debias
