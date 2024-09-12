# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/7/20 22:33
"""
调研一里面 欧几里得距离不用实现 其他都可以实现
除了针对文档数据集的，其他都用MNIST数据集做实验（http://yann.lecun.com/exdb/mnist/）
对于文档数据集 做实验的时候可以在网上找找合适的 调研一的内容主要就是对比两个数据集之间的差异
以MNIST数据集实验为例可以针对MNIST数据集先进行平移、旋转、亮度变换等操作来生成一个新的数据集
然后再将这个新的数据集与原来的MNIST数据集进行相似度对比
平移、旋转、亮度变换等操作参见网上资料（例如https://blog.csdn.net/Keep_Trying_Go/article/details/121589832）
（1）几何变换包括：随机旋转图像或文本（旋转）、随机调整图像或文本的大小（缩放）
随机移动图像或文本的位置（平移）、水平或垂直翻转图像或文本（镜像翻转）
（2）色彩变幻包括：随机调整图像或文本的亮度（亮度调整）、随机调整图像或文本的对比度（对比度调整）
随机调整图像或文本的色调（色调变换）、随机调整图像或文本的饱和度（饱和度调整）
（3）变形包括：对图像或文本应用随机的弹性变形（弹性变形）、应用随机的仿射变换来改变图像或文本的形状（仿射变换）
（4）噪声添加：向图像或文本中添加随机高斯噪声（高斯噪声）、图像或文本中添加随机椒盐噪声（随机将像素值设为最大或最小值）
（5）随机剪切：对图像或文本应用随机的剪切变换
"""
# 两个集合A和B的交集元素在A和B的并集中所占的比例，称为两个集合的杰卡德相似系数（Jacquard Similarity Coefficient）
# 杰卡德相似系数是衡量两个集合的相似度一种指标。与杰卡德相似系数相反的概念是杰卡德距离（Jacquard Distance）
# 杰卡德距离用两个集合中不同元素占所有元素的比例来衡量两个集合的区分度

# from sklearn.metrics import jacquard_score
# 这里sklearn库中的杰卡德相关函数调用老是出现严重警告类型错误
# 所以多次调试无果后决定放弃使用库中函数 改用自定义的杰卡德相似度系数以及杰卡德距离函数

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.ndimage import rotate

# 加载20 Newsgroups数据集
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

# 只选择一部分数据进行演示
documents = newsgroups_train.data[:1000]

# 初始化文本向量化器
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents).toarray()

# 添加随机噪声来模拟数据变换
X_transformed = X + np.random.normal(0, 1.5, size=X.shape)  # 添加正态分布噪声


# 定义杰卡德相似度系数
def jacquard_similarity(vec1, vec2):
    intersection = np.logical_and(vec1, vec2)
    union = np.logical_or(vec1, vec2)
    return np.sum(intersection) / np.sum(union)

# 计算变换前后数据集的平均杰卡德相似系数和杰卡德距离
jacquard_similarities = []
jacquard_distances = []
for i in range(len(X)):
    similarity = jacquard_similarity(X[i], X_transformed[i])
    distance = 1-similarity
    # print(f"Distance {i + 1}: {distance}")
    jacquard_similarities.append(similarity)
    jacquard_distances.append(distance)

# 结算结果格式化
# formatted_jacquard_similarity = ["{:.7f}".format(similarity) for similarity in jacquard_similarities]
# formatted_jacquard_distance = ["{:.7f}".format(distance) for similarity in jacquard_distances]

# 计算结果求平均
mean_jacquard_similarity = np.mean(jacquard_similarities)
mean_jacquard_distance = np.mean(jacquard_distances)

# 计算结果输出
# print(f"经过添加随机噪声之后的杰卡德相似度系数: {formatted_jacquard_similarity}")
# print(f"经过添加随机噪声之后的杰卡德距离: {formatted_jacquard_distance}")
print(f"经过添加随机噪声之后的平均杰卡德相似系数: {mean_jacquard_similarity:.7f}")
print(f"经过添加随机噪声之后的平均杰卡德距离: {mean_jacquard_distance:.7f}")

# 定义随机旋转函数
def random_rotate(X, max_angle=25):
    angles = np.random.uniform(low=-max_angle, high=max_angle, size=len(X))
    X_rotated = np.zeros_like(X)
    for i in range(len(X)):
        X_rotated[i] = rotate(X[i].reshape(-1, 1), angles[i], reshape=False).flatten()
    return X_rotated

# 对数据集进行随机旋转变换
X_rotated = random_rotate(X)

# 计算变换后数据集的平均杰卡德相似度和距离
jacquard_similarities_rotated = []
jacquard_distances_rotated = []
for i in range(len(X)):
    similarity = jacquard_similarity(X[i], X_rotated[i])
    distance = 1 - similarity
    jacquard_similarities_rotated.append(similarity)
    jacquard_distances_rotated.append(distance)

# 格式化输出结果
# formatted_jacquard_similarity_rotated = ["{:.7f}".format(similarity) for similarity in jacquard_similarities_rotated]
# formatted_jacquard_distance_rotated = ["{:.7f}".format(distance) for distance in jacquard_distances_rotated]

# 计算平均值
mean_jacquard_similarity_rotated = np.mean(jacquard_similarities_rotated)
mean_jacquard_distance_rotated = np.mean(jacquard_distances_rotated)

# 输出结果
# print(f"经过随机旋转之后的杰卡德相似度系数: {formatted_jacquard_similarity_rotated}")
# print(f"经过随机旋转之后的杰卡德距离: {formatted_jacquard_distance_rotated}")
print(f"经过随机旋转之后的平均杰卡德相似系数: {mean_jacquard_similarity_rotated:.7f}")
print(f"经过随机旋转之后的平均杰卡德距离: {mean_jacquard_distance_rotated:.7f}")

# 定义随机亮度调整函数
def random_brightness(X, max_delta=0.5):
    delta = np.random.uniform(low=-max_delta, high=max_delta, size=X.shape)
    X_adjusted = X + delta
    return X_adjusted

# 对数据集进行随机亮度调整
X_adjusted = random_brightness(X)

# 计算调整前后每个文档的杰卡德相似度
jacquard_similarities = []
for i in range(len(X)):
    similarity = jacquard_similarity(X[i], X_adjusted[i])
    jacquard_similarities.append(similarity)

# 计算平均杰卡德相似度
mean_jacquard_similarity = np.mean(jacquard_similarities)

# 输出结果
print(f"经过随机亮度调整后的平均杰卡德相似系数: {mean_jacquard_similarity:.7f}")