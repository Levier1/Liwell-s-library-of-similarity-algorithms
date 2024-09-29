# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/9/18 15:44

# 这是实验思路一增添了新的基线对比方式的实验代码设置
# 其代码设置的大部分都没有变化 只不过多了一组基线对比的代码设置

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
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return 1 - cosine(vec1, vec2)


# n 次执行的控制变量
n = 10

# num_documents 控制每次执行时选取的数据量 如果为None则使用所有文档。
num_documents = None
if num_documents is None:
    num_documents = len(newsgroups_train.data)  # 使用全部数据集样本

# 1. 随机噪声变换
all_mean_cosine_similarities_noise = []

for iteration in range(n):
    documents = newsgroups_train.data[:num_documents]
    vectorizer = CountVectorizer(max_features=10000)  # 限制特征数量
    X = vectorizer.fit_transform(documents).toarray()

    # 加噪声
    X_transformed = X + np.random.normal(0, 0.3, size=X.shape)

    # 计算噪声变换后的余弦相似度
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

    # 只选取前400个特征
    image_data = X_tfidf[:, :400].reshape(-1, 20, 20)
    rotated_images = [rotate(image, np.random.uniform(-20, 20), mode='edge') for image in image_data]
    rotated_images = np.array(rotated_images).reshape(len(image_data), -1)

    # 计算旋转变换后的余弦相似度
    cosine_similarities = [cosine_similarity(X_tfidf[i, :400], rotated_images[i, :400]) for i in range(len(X_tfidf))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_rotation.append(mean_cosine_similarity)

    print(f"第 {iteration + 1} 次计算的平均余弦相似度（旋转）: {mean_cosine_similarity:.7f}")

overall_mean_cosine_similarity_rotation = np.mean(all_mean_cosine_similarities_rotation)
print(f"{n} 次计算后的平均余弦相似度（旋转）: {overall_mean_cosine_similarity_rotation:.7f}")
print("\n")

# 3. 亮度调整（标准化）
all_mean_cosine_similarities_brightness = []

for iteration in range(n):
    documents = newsgroups_train.data[:num_documents]
    vectorizer = CountVectorizer(max_features=10000)
    X = vectorizer.fit_transform(documents).toarray()

    # 亮度调整（标准化）
    X_noisy = X + np.random.normal(0, 0.01, size=X.shape)
    X_scaled = scale(X_noisy)

    # 计算亮度调整后的余弦相似度
    cosine_similarities = [cosine_similarity(X[i], X_scaled[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_brightness.append(mean_cosine_similarity)

    print(f"第 {iteration + 1} 次计算的平均余弦相似度（亮度调整）: {mean_cosine_similarity:.7f}")

overall_mean_cosine_similarity_brightness = np.mean(all_mean_cosine_similarities_brightness)
print(f"{n} 次计算后的平均余弦相似度（亮度调整）: {overall_mean_cosine_similarity_brightness:.7f}")
print("\n")

# 4. 仿射变换
all_mean_cosine_similarities_affine = []

for iteration in range(n):
    documents = newsgroups_train.data[:num_documents]
    vectorizer = CountVectorizer(max_features=10000)
    X = vectorizer.fit_transform(documents).toarray()

    # 仿射变换
    affine_transformer = FunctionTransformer(lambda x: x + np.random.normal(0, 0.1, size=x.shape), validate=False)
    X_affine_transformed = affine_transformer.transform(X)

    # 计算仿射变换后的余弦相似度
    cosine_similarities = [cosine_similarity(X[i], X_affine_transformed[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_affine.append(mean_cosine_similarity)

    print(f"第 {iteration + 1} 次计算的平均余弦相似度（仿射变换）: {mean_cosine_similarity:.7f}")

overall_mean_cosine_similarity_affine = np.mean(all_mean_cosine_similarities_affine)
print(f"{n} 次计算后的平均余弦相似度（仿射变换）: {overall_mean_cosine_similarity_affine:.7f}")
print("\n")

# 5. 随机剪切变换
all_mean_cosine_similarities_crop = []


def random_crop(x, crop_ratio=0.5):
    num_features = x.shape[1]
    crop_size = int(num_features * crop_ratio)
    crop_start = np.random.randint(0, num_features - crop_size + 1)
    x_cropped = x.copy()
    x_cropped[:, crop_start:crop_start + crop_size] = 0.5
    return x_cropped


for iteration in range(n):
    documents = newsgroups_train.data[:num_documents]
    vectorizer = CountVectorizer(max_features=10000)
    X = vectorizer.fit_transform(documents).toarray()

    # 随机剪切变换
    X_cropped = random_crop(X)

    # 计算剪切变换后的余弦相似度
    cosine_similarities = [cosine_similarity(X[i], X_cropped[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_crop.append(mean_cosine_similarity)

    print(f"第 {iteration + 1} 次计算的平均余弦相似度（随机剪切）: {mean_cosine_similarity:.7f}")

overall_mean_cosine_similarity_crop = np.mean(all_mean_cosine_similarities_crop)
print(f"{n} 次计算后的平均余弦相似度（随机剪切）: {overall_mean_cosine_similarity_crop:.7f}")
print("\n")

# 单独的基线对比（随机选取两组不同文档，未做任何变换）
all_mean_cosine_similarities_baseline = []

for iteration in range(n):
    # 每次随机选取两组不同的文档
    random_indices1 = np.random.choice(len(newsgroups_train.data), num_documents, replace=False)
    random_indices2 = np.random.choice(len(newsgroups_train.data), num_documents, replace=False)
    documents1 = [newsgroups_train.data[i] for i in random_indices1]
    documents2 = [newsgroups_train.data[i] for i in random_indices2]

    vectorizer = CountVectorizer(max_features=10000)  # 限制特征数量

    # 在同一个 CountVectorizer 上调用 fit_transform，以确保两组数据的向量化维度相同
    documents_combined = documents1 + documents2
    X_combined = vectorizer.fit_transform(documents_combined).toarray()

    # 将向量化后的矩阵拆分为两组
    X1 = X_combined[:num_documents]
    X2 = X_combined[num_documents:num_documents * 2]

    # 基线：比较两个不同集合的相似性
    baseline_similarities = [cosine_similarity(X1[i], X2[i]) for i in range(len(X1))]
    mean_baseline_similarity = np.mean(baseline_similarities)
    all_mean_cosine_similarities_baseline.append(mean_baseline_similarity)

    print(f"第 {iteration + 1} 次计算的基线平均余弦相似度（随机选取两组文档）: {mean_baseline_similarity:.7f}")

# 计算 n 次后的总平均值
overall_mean_baseline_similarity = np.mean(all_mean_cosine_similarities_baseline)
print(f"{n} 次计算后的基线平均余弦相似度（随机选取两组文档）: {overall_mean_baseline_similarity:.7f}")




