# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/8/25 9:29

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard, cityblock

# 加载20newsgroups数据集
newsgroups = fetch_20newsgroups(subset='all')
texts = newsgroups.data

# 创建文本的词频矩阵
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 随机选择一个文档索引
def get_random_index(max_index):
    return np.random.randint(max_index)

# 随机选择n个文档索引
def get_random_indices(max_index, n):
    return np.random.choice(max_index, size=n, replace=False)

# 计算余弦相似度
def compute_cosine_similarity(X, index, random_indices):
    doc_vector = X[index]
    all_similarities = cosine_similarity(doc_vector, X[random_indices])
    return all_similarities.flatten()

# 计算杰卡德相似系数
def compute_jacquard_similarity(X, index, random_indices):
    doc_vector = X[index].toarray().flatten() > 0
    all_similarities = np.zeros(len(random_indices))
    for i, rand_index in enumerate(random_indices):
        other_vector = X[rand_index].toarray().flatten() > 0
        all_similarities[i] = 1 - jaccard(doc_vector, other_vector)
    return all_similarities

# 计算曼哈顿距离
def compute_manhattan_distance(X, index, random_indices):
    doc_vector = X[index].toarray().flatten()
    all_distances = np.zeros(len(random_indices))
    for i, rand_index in enumerate(random_indices):
        other_vector = X[rand_index].toarray().flatten()
        all_distances[i] = cityblock(doc_vector, other_vector)
    return all_distances

# 用户设置的参数n
n = 3  # 可以根据需要设置参数n的值

# 获取随机选择的文档索引
index = get_random_index(X.shape[0])
random_indices = get_random_indices(X.shape[0], n)

# 输出随机选择的文档索引
print(f"随机选择一个用来对比的文档 其索引为: {index}")
print(f"随机选择{n}个用来做对比的文档 它们的索引分别是: {random_indices}")
print(" ")

# 计算并输出相似度和距离
print(f"输出各自对比后计算的结果：")
cosine_sims = compute_cosine_similarity(X, index, random_indices)
print(f"Cosine Similarities with document {index}:")
print(cosine_sims)
print(" ")

jacquard_sims = compute_jacquard_similarity(X, index, random_indices)
print(f"Jacquard Similarities with document {index}:")
print(jacquard_sims)
print(" ")

manhattan_dists = compute_manhattan_distance(X, index, random_indices)
formatted_distances = ' '.join([str(int(d)) for d in manhattan_dists])
print(f"Manhattan Distances with document {index}:")
print(f"[{formatted_distances}]")


