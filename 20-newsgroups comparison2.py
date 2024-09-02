# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/8/25 17:51

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

# 随机选择两个不同的文档索引
def get_two_different_indices(max_index):
    indices = np.random.choice(max_index, 2, replace=False)
    return indices

# 计算余弦相似度
def compute_cosine_similarity(X, indices):
    return cosine_similarity(X[indices], X[indices])[0, 1]

# 计算杰卡德相似系数
def compute_jacquard_similarity(X, indices):
    x_binary = X[indices[0]].toarray().flatten()
    y_binary = X[indices[1]].toarray().flatten()
    return 1 - jaccard(x_binary > 0, y_binary > 0)

# 计算曼哈顿距离
def compute_manhattan_distance(X, indices):
    return cityblock(X[indices[0]].toarray().flatten(), X[indices[1]].toarray().flatten())

# 获取两个随机选择的文档索引
indices = get_two_different_indices(X.shape[0])

# 输出随机选择的文档索引
print(f"随机选择的两个文档的索引: {indices}")

# 计算并输出相似度和距离
cosine_sim = compute_cosine_similarity(X, indices)
print(f"两个文档之间的余弦相似度为: {cosine_sim:.7f}")

jacquard_sim = compute_jacquard_similarity(X, indices)
print(f"两个文档之间的杰卡德相似度系数为: {jacquard_sim:.7f}")

manhattan_dist = compute_manhattan_distance(X, indices)
print(f"两个文档之间的曼哈顿距离为: {manhattan_dist}")

