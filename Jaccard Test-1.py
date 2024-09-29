# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/9/14 17:01

"""
观察到 杰卡德相似度系数在随机旋转后的结果始终为0 这主要是由于杰卡德相似度不适合用于捕捉像旋转变换这样的几何变换
具体能够得到这种结论的理论分析如下：
1. 杰卡德相似度的定义与二值化处理的限制：
杰卡德相似度 是基于集合交集与并集的比例 它关注的是两个二值向量的重叠部分 即两个向量中相同位置的1（交集） 与总共存在1的位置（并集）的比率
当你对文本数据进行旋转操作时 文档中的特征会被 "移位" 导致两者的相同相同位置几乎不再存在
也就是说：旋转后的文档向量和原始文档向量几乎不可能有交集 特别是在二值化后（大部分位置要么是0，要么是1）
这种几何操作改变了数据的位置关系 但并没有在特征层面产生一致性 因此 杰卡德相似度对这种操作测量效果很差 几乎总是返回0
2. 旋转变换与文本向量化的关系：
文本向量化生成的是一个基于单词的高维向量 每个维度表示的是词汇（特征）的出现
旋转这种操作通常适用于图像等空间数据，对文本数据来说没有明确的物理含义
旋转操作会导致向量的每个元素“位移” 但这不是对文本特征的一种自然变换 反而更像是对向量的一种随机扰动
旋转后的向量不再与原始向量共享特征 这会导致杰卡德相似度接近 0
3. 杰卡德相似度的局限性：
杰卡德相似度 更适合衡量文本向量中的共同特征（即相同单词）的存在情况
对于像旋转这样改变整体结构的变换 杰卡德相似度不能很好地捕捉这种变化 因此输出为 0是预期的
"""

from sklearn.datasets import fetch_20newsgroups
from datasketch import MinHash
import numpy as np
import random
import string

# 加载20 Newsgroups数据集
newsgroups_train = fetch_20newsgroups(subset='train')

# 输出数据集中样本的数量
print(f"数据集中样本的数量: {len(newsgroups_train.data)}")


# 定义 n-gram 提取函数
def generate_ngrams(text, n=7):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]


# 定义 MinHash 函数（基于 n-gram）
def create_minhash(doc, num_perm=128, n=7):
    m = MinHash(num_perm=num_perm)
    ngrams = generate_ngrams(doc, n)
    for ngram in ngrams:
        m.update(ngram.encode('utf8'))
    return m


# 定义模拟旋转的词序变换函数（循环移位）
def rotate_words(doc):
    words = doc.split()
    if len(words) > 1:
        shift = np.random.randint(1, len(words) // 2)  # 随机决定移位步长
        words = words[shift:] + words[:shift]  # 循环移位
    return ' '.join(words)


# 定义噪声添加变换（插入随机字符作为噪声，包括字母、标点符号和数字）
def add_random_noise(doc, noise_level=0.02):
    noisy_doc = list(doc)
    num_noise_chars = int(len(noisy_doc) * noise_level)
    for _ in range(num_noise_chars):
        noisy_doc.insert(np.random.randint(0, len(noisy_doc)),
                         random.choice(string.ascii_letters + string.punctuation + string.digits))
    return ''.join(noisy_doc)


# 定义亮度调整变换（随机重复某些单词）
def adjust_brightness(doc, repeat_level=0.1):
    words = doc.split()
    num_repeats = int(len(words) * repeat_level)
    for _ in range(num_repeats):
        pos = np.random.randint(0, len(words))
        words.insert(pos, words[pos])  # 重复某些单词
    return ' '.join(words)


# 仿射变换（随机插入或删除单词）
def affine_transformation(doc, insert_prob=0.1, delete_prob=0.1):
    words = doc.split()
    new_words = []

    # 进行插入或删除操作
    for word in words:
        # 插入操作
        if random.random() < insert_prob:
            new_words.append(random.choice(words))  # 随机插入一个已有单词
        new_words.append(word)
        # 删除操作
        if random.random() < delete_prob:
            continue  # 随机跳过当前单词，相当于删除

    return ' '.join(new_words)


# 随机剪切（随机删除单词）
def random_shear(doc, delete_prob=0.2):
    words = doc.split()
    sheared_words = [word for word in words if random.random() > delete_prob]  # 随机删除单词
    return ' '.join(sheared_words)


# 随机选择一部分数据进行变换，带参数控制选择数量
def apply_transform_to_random_subset(documents, num_docs, transform_func):
    selected_indices = random.sample(range(len(documents)), num_docs)
    documents_transformed = documents.copy()

    # 对随机选择的部分文档进行指定变换
    for idx in selected_indices:
        documents_transformed[idx] = transform_func(documents_transformed[idx])

    return documents_transformed


# 定义 n 次相似度计算
def calculate_similarity(n_iterations, num_docs, n_gram_size, transform_func):
    similarities_across_iterations = []

    # 首先计算完整原始数据集的 MinHash 签名
    full_minhashes = [create_minhash(doc, n=n_gram_size) for doc in newsgroups_train.data]

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}:")

        # 选择部分数据集进行变换
        transformed_documents = apply_transform_to_random_subset(newsgroups_train.data, num_docs, transform_func)

        # 为变换后的数据集计算 MinHash 签名
        transformed_minhashes = [create_minhash(doc, n=n_gram_size) for doc in transformed_documents]

        # 计算整个变换后的数据集与原始数据集的杰卡德相似度
        jacquard_similarities = []
        for orig_mh, trans_mh in zip(full_minhashes, transformed_minhashes):
            similarity = orig_mh.jaccard(trans_mh)
            jacquard_similarities.append(similarity)

        # 计算本次循环的平均杰卡德相似度
        mean_jacquard_similarity = np.mean(jacquard_similarities)
        print(f"Mean Jacquard similarity for iteration {iteration + 1}: {mean_jacquard_similarity:.7f}")
        similarities_across_iterations.append(mean_jacquard_similarity)

    # 计算 n 次计算后的总体平均值
    overall_mean_similarity = np.mean(similarities_across_iterations)
    print(f"Overall mean Jacquard similarity after {n_iterations} iterations: {overall_mean_similarity:.7f}\n")
    return similarities_across_iterations, overall_mean_similarity


# 人为控制参数
n_iterations = 5  # 控制相似度计算的次数
num_docs = 1000  # 每次选择变换的文档数量
n_gram_size = 3  # n-gram 的大小

# 测试不同的变换
print("Transform 1: Rotation (word sequence shift)")
similarities_rotation, overall_mean_rotation = calculate_similarity(n_iterations, num_docs, n_gram_size, rotate_words)

print("\nTransform 2: Add Gaussian-like noise (random characters)")
similarities_noise, overall_mean_noise = calculate_similarity(n_iterations, num_docs, n_gram_size, add_random_noise)

print("\nTransform 3: Adjust Brightness (word repetition)")
similarities_brightness, overall_mean_brightness = calculate_similarity(n_iterations, num_docs, n_gram_size, adjust_brightness)

print("\nTransform 4: Affine Transformation (random insertion/deletion of words)")
similarities_affine, overall_mean_affine = calculate_similarity(n_iterations, num_docs, n_gram_size, affine_transformation)

print("\nTransform 5: Random Shear (random deletion of words)")
similarities_shear, overall_mean_shear = calculate_similarity(n_iterations, num_docs, n_gram_size, random_shear)











