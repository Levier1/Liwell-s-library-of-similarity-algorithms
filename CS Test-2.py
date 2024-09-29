# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/9/17 15:22

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from skimage.transform import rotate
from scipy.spatial.distance import cosine
from sklearn.preprocessing import scale
from sklearn.preprocessing import FunctionTransformer

# 加载20 Newsgroups数据集
# 查看所有可用类别
all_categories = fetch_20newsgroups(subset='train').target_names
print("所有可用的类别：", all_categories)

# 定义新的类别列表
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
    'rec.sport.hockey',  # 添加新类别
    'soc.religion.christian',  # 添加新类别
    'sci.med',  # 添加新类别
    'rec.motorcycles',   # 添加新类别
    'misc.forsale'   # 添加新类别
]

# 加载数据集
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

# 输出数据集的大小
print(f"数据集的总样本数: {len(newsgroups_train.data)}")

# 定义余弦相似度函数
def cosine_similarity(vec1, vec2):
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 检查模长是否为零
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # 如果其中一个向量为零向量，直接返回相似度为0

    # 否则计算余弦相似度
    return 1 - cosine(vec1, vec2)


# n 次执行的控制变量
n = 50  # 可以根据需要更改

# 固定的文档数量，可以手动修改
num_documents = 5000  # 可以根据需要手动更改

# 1. 随机噪声变换
all_mean_cosine_similarities_noise = []

for iteration in range(n):
    documents = newsgroups_train.data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents).toarray().astype(np.float64)  # 转换为 float64 类型

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X.shape[0], num_documents, replace=False)

    # 只对随机选取的文档进行变换操作
    X_transformed = X.copy()
    X_transformed[random_indices] += np.random.normal(0, 0.3, size=X[random_indices].shape)

    # 在整体数据集上计算相似度
    cosine_similarities = [cosine_similarity(X[i], X_transformed[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_noise.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次（随机噪声, {num_documents} 个文档）计算的平均余弦相似度: {mean_cosine_similarity:.7f}")

overall_mean_cosine_similarity_noise = np.mean(all_mean_cosine_similarities_noise)
print(f"{n} 次计算后的平均余弦相似度（随机噪声）: {overall_mean_cosine_similarity_noise:.7f}")
print("\n")

# 2. 旋转变换
all_mean_cosine_similarities_rotation = []

for iteration in range(n):
    documents = newsgroups_train.data
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_tfidf = tfidf_vectorizer.fit_transform(documents).toarray().astype(np.float64)  # 转换为 float64 类型

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X_tfidf.shape[0], num_documents, replace=False)

    # 只对随机选取的文档进行旋转变换
    image_data = X_tfidf[random_indices, :400].reshape(-1, 20, 20)
    rotated_images = [rotate(image, np.random.uniform(-20, 20), mode='edge') for image in image_data]
    rotated_images = np.array(rotated_images).reshape(len(image_data), -1)

    X_transformed = X_tfidf.copy()
    X_transformed[random_indices, :400] = rotated_images

    cosine_similarities = [cosine_similarity(X_tfidf[i, :400], X_transformed[i, :400]) for i in range(len(X_tfidf))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_rotation.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次（旋转, {num_documents} 个文档）计算的平均余弦相似度: {mean_cosine_similarity:.7f}")

overall_mean_cosine_similarity_rotation = np.mean(all_mean_cosine_similarities_rotation)
print(f"{n} 次计算后的平均余弦相似度（旋转）: {overall_mean_cosine_similarity_rotation:.7f}")
print("\n")

# 3. 亮度调整（标准化）
all_mean_cosine_similarities_brightness = []

for iteration in range(n):
    documents = newsgroups_train.data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents).toarray().astype(np.float64)  # 转换为 float64 类型

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X.shape[0], num_documents, replace=False)

    # 只对随机选取的文档进行标准化变换
    X_noisy = X[random_indices] + np.random.normal(0, 0.01, size=X[random_indices].shape)
    X_scaled = scale(X_noisy)

    X_transformed = X.copy()
    X_transformed[random_indices] = X_scaled

    cosine_similarities = [cosine_similarity(X[i], X_transformed[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_brightness.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次（亮度调整, {num_documents} 个文档）计算的平均余弦相似度: {mean_cosine_similarity:.7f}")

overall_mean_cosine_similarity_brightness = np.mean(all_mean_cosine_similarities_brightness)
print(f"{n} 次计算后的平均余弦相似度（亮度调整）: {overall_mean_cosine_similarity_brightness:.7f}")
print("\n")

# 4. 仿射变换
all_mean_cosine_similarities_affine = []

for iteration in range(n):
    documents = newsgroups_train.data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents).toarray().astype(np.float64)  # 转换为 float64 类型

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X.shape[0], num_documents, replace=False)

    # 只对随机选取的文档进行仿射变换
    affine_transformer = FunctionTransformer(lambda x: x + np.random.normal(0, 0.1, size=x.shape), validate=False)
    X_affine_transformed = affine_transformer.transform(X[random_indices])

    X_transformed = X.copy()
    X_transformed[random_indices] = X_affine_transformed

    cosine_similarities = [cosine_similarity(X[i], X_transformed[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_affine.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次（仿射, {num_documents} 个文档）计算的平均余弦相似度: {mean_cosine_similarity:.7f}")

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
    documents = newsgroups_train.data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents).toarray().astype(np.float64)  # 转换为 float64 类型

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X.shape[0], num_documents, replace=False)

    # 只对随机选取的文档进行剪切变换
    X_cropped = random_crop(X[random_indices])

    X_transformed = X.copy()
    X_transformed[random_indices] = X_cropped

    cosine_similarities = [cosine_similarity(X[i], X_transformed[i]) for i in range(len(X))]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_crop.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次（随机剪切, {num_documents} 个文档）计算的平均余弦相似度: {mean_cosine_similarity:.7f}")

overall_mean_cosine_similarity_crop = np.mean(all_mean_cosine_similarities_crop)
print(f"{n} 次计算后的平均余弦相似度（随机剪切）: {overall_mean_cosine_similarity_crop:.7f}")
print("\n")