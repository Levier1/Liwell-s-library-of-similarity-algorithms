# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/7/7 9:15

"""
调研一里面 欧几里得距离不用实现 其他都可以实现
除了针对文档数据集的，其他都用MNIST数据集做实验（http://yann.lecun.com/exdb/mnist/）
对于文档数据集 做实验的时候可以在网上找找合适的 调研一的内容主要就是对比两个数据集之间的差异
以MNIST数据集实验为例可以针对MNIST数据集先进行平移、旋转、亮度变换等操作来生成一个新的数据集
然后再将这个新的数据集与原来的MNIST数据集进行相似度对比
平移、旋转、亮度变换等操作参见网上资料（例如https://blog.csdn.net/Keep_Trying_Go/article/details/121589832）
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from skimage.transform import rotate
from scipy.spatial.distance import cosine
from sklearn.preprocessing import scale
from sklearn.preprocessing import FunctionTransformer

# 加载20 Newsgroups数据集
newsgroups_train = fetch_20newsgroups(subset='train')

# 输出数据集中样本的数量
print(f"数据集中样本的数量: {len(newsgroups_train.data)}")

# 定义余弦相似度函数
def cosine_similarity(vec1, vec2):
    # 计算两个向量的模长
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 检查模长是否为零
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # 如果其中一个向量为零向量，直接返回相似度为0

    # 否则计算余弦相似度
    return 1 - cosine(vec1, vec2)

# n 次执行的控制变量
n = 50  # 可以根据需要更改

# num_documents 控制每次执行时选取的数据量
num_documents = 100  # 可以根据需要更改

# 1. 随机噪声变换
all_mean_cosine_similarities_noise = []

for iteration in range(n):
    # 每次迭代重新选取不同数量的数据
    documents = newsgroups_train.data[:num_documents]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents).toarray()

    X_transformed = X + np.random.normal(0, 0.3, size=X.shape)
    cosine_similarities = [cosine_similarity(X[i], X_transformed[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_noise.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次计算的平均余弦相似度（随机噪声）: {mean_cosine_similarity:.7f}")

# 计算 n 次后的总平均值
overall_mean_cosine_similarity_noise = np.mean(all_mean_cosine_similarities_noise)
print(f"{n} 次计算后的平均余弦相似度（随机噪声）: {overall_mean_cosine_similarity_noise:.7f}")
print("\n")

# 2. 旋转变换
all_mean_cosine_similarities_rotation = []

for iteration in range(n):
    documents = newsgroups_train.data[:num_documents]
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_tfidf = tfidf_vectorizer.fit_transform(documents).toarray()
    image_data = X_tfidf[:, :400].reshape(-1, 20, 20)

    rotated_images = [rotate(image, np.random.uniform(-20, 20), mode='edge') for image in image_data]
    rotated_images = np.array(rotated_images).reshape(len(image_data), -1)
    cosine_similarities = [cosine_similarity(X_tfidf[i, :400], rotated_images[i, :400]) for i in range(len(X_tfidf))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_rotation.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次计算的平均余弦相似度（旋转）: {mean_cosine_similarity:.7f}")

# 计算 n 次后的总平均值
overall_mean_cosine_similarity_rotation = np.mean(all_mean_cosine_similarities_rotation)
print(f"{n} 次计算后的平均余弦相似度（旋转）: {overall_mean_cosine_similarity_rotation:.7f}")
print("\n")

# 3. 亮度调整（标准化）
all_mean_cosine_similarities_brightness = []

for iteration in range(n):
    documents = newsgroups_train.data[:num_documents]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents).toarray()

    X_noisy = X + np.random.normal(0, 0.01, size=X.shape)
    X_scaled = scale(X_noisy)
    cosine_similarities = [cosine_similarity(X[i], X_scaled[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_brightness.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次计算的平均余弦相似度（亮度调整）: {mean_cosine_similarity:.7f}")

# 计算 n 次后的总平均值
overall_mean_cosine_similarity_brightness = np.mean(all_mean_cosine_similarities_brightness)
print(f"{n} 次计算后的平均余弦相似度（亮度调整）: {overall_mean_cosine_similarity_brightness:.7f}")
print("\n")

# 4. 仿射变换
all_mean_cosine_similarities_affine = []

for iteration in range(n):
    documents = newsgroups_train.data[:num_documents]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents).toarray()

    affine_transformer = FunctionTransformer(lambda x: x + np.random.normal(0, 0.1, size=x.shape), validate=False)
    X_affine_transformed = affine_transformer.transform(X)
    cosine_similarities = [cosine_similarity(X[i], X_affine_transformed[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_affine.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次计算的平均余弦相似度（仿射变换）: {mean_cosine_similarity:.7f}")

# 计算 n 次后的总平均值
overall_mean_cosine_similarity_affine = np.mean(all_mean_cosine_similarities_affine)
print(f"{n} 次计算后的平均余弦相似度（仿射变换）: {overall_mean_cosine_similarity_affine:.7f}")
print("\n")

# 5. 随机剪切变换
all_mean_cosine_similarities_crop = []

def random_crop(x, crop_ratio=0.5):
    num_features = x.shape[1]
    crop_size = int(num_features * crop_ratio)
    crop_start = np.random.randint(0, num_features - crop_size + 10)
    x_cropped = x.copy()
    x_cropped[:, crop_start:crop_start + crop_size] = 0.5
    return x_cropped

for iteration in range(n):
    documents = newsgroups_train.data[:num_documents]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents).toarray()

    X_cropped = random_crop(X)
    cosine_similarities = [cosine_similarity(X[i], X_cropped[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_crop.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次计算的平均余弦相似度（随机剪切）: {mean_cosine_similarity:.7f}")

# 计算 n 次后的总平均值
overall_mean_cosine_similarity_crop = np.mean(all_mean_cosine_similarities_crop)
print(f"{n} 次计算后的平均余弦相似度（随机剪切）: {overall_mean_cosine_similarity_crop:.7f}")
print("\n")
