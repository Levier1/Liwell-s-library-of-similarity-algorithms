# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/9/16 22:11

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from skimage.transform import rotate
from sklearn.preprocessing import scale
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import lil_matrix, csr_matrix

"""
待解决问题：数据集过大 计算开销上升导致计算速度过慢 
经过对数据的一些特殊处理之后 计算效率仍然没有什么显著提升
所以暂时先停留在这里 然后在 CS Test-2 二中将数据集减半先进行实验
"""
# 加载20 Newsgroups数据集
newsgroups_train = fetch_20newsgroups(subset='train')

# 输出数据集中样本的数量
print(f"数据集中样本的数量: {len(newsgroups_train.data)}")

# 定义余弦相似度函数，支持稀疏矩阵
def cosine_similarity_sparse(vec1, vec2):
    norm_vec1 = np.sqrt(vec1.multiply(vec1).sum())
    norm_vec2 = np.sqrt(vec2.multiply(vec2).sum())

    # 检查模长是否为零
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # 如果其中一个向量为零向量，直接返回相似度为0

    dot_product = vec1.multiply(vec2).sum()
    return dot_product / (norm_vec1 * norm_vec2)

# n 次执行的控制变量
n = 50  # 可以根据需要更改

# 固定的文档数量，可以手动修改
num_documents = 100  # 可以根据需要手动更改

# 1. 随机噪声变换
all_mean_cosine_similarities_noise = []

for iteration in range(n):
    documents = newsgroups_train.data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)  # 保持为稀疏矩阵

    # 将 csr_matrix 转换为 lil_matrix 以便进行修改
    X_transformed = X.tolil()

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X.shape[0], num_documents, replace=False)

    # 只对随机选取的文档进行变换操作
    X_transformed[random_indices] += csr_matrix(np.random.normal(0, 0.3, size=X[random_indices].shape))

    # 修改完成后再转换回 csr_matrix
    X_transformed = X_transformed.tocsr()

    # 在整体数据集上计算相似度
    cosine_similarities = [cosine_similarity_sparse(X[i], X_transformed[i]) for i in range(X.shape[0])]
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
    X_tfidf = tfidf_vectorizer.fit_transform(documents)  # 保持为稀疏矩阵

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X_tfidf.shape[0], num_documents, replace=False)

    # 只对随机选取的文档进行旋转变换
    image_data = X_tfidf[random_indices, :400].toarray().reshape(-1, 20, 20)
    rotated_images = [rotate(image, np.random.uniform(-20, 20), mode='edge') for image in image_data]
    rotated_images = np.array(rotated_images).reshape(len(image_data), -1)

    X_transformed = X_tfidf.copy().toarray()
    X_transformed[random_indices, :400] = rotated_images
    X_transformed = csr_matrix(X_transformed)

    cosine_similarities = [cosine_similarity_sparse(X_tfidf[i, :400], X_transformed[i, :400]) for i in range(X_tfidf.shape[0])]
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
    X = vectorizer.fit_transform(documents)  # 保持为稀疏矩阵

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X.shape[0], num_documents, replace=False)

    # 使用 lil_matrix 进行修改
    X_transformed = X.tolil()

    # 只对随机选取的文档进行标准化变换
    X_noisy = X[random_indices] + csr_matrix(np.random.normal(0, 0.01, size=X[random_indices].shape))
    X_scaled = scale(X_noisy.toarray())
    X_transformed[random_indices] = csr_matrix(X_scaled)

    # 修改完成后转换回 csr_matrix
    X_transformed = X_transformed.tocsr()

    cosine_similarities = [cosine_similarity_sparse(X[i], X_transformed[i]) for i in range(X.shape[0])]
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
    X = vectorizer.fit_transform(documents)  # 保持为稀疏矩阵

    # 使用 lil_matrix 以提高修改效率
    X_transformed = X.tolil()

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X.shape[0], num_documents, replace=False)

    # 只对随机选取的文档进行仿射变换
    affine_transformer = FunctionTransformer(lambda x: x + np.random.normal(0, 0.1, size=x.shape), validate=False)
    X_affine_transformed = csr_matrix(affine_transformer.transform(X[random_indices].toarray()))
    X_transformed[random_indices] = X_affine_transformed

    # 修改完成后转换回 csr_matrix
    X_transformed = X_transformed.tocsr()

    cosine_similarities = [cosine_similarity_sparse(X[i], X_transformed[i]) for i in range(X.shape[0])]
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
    X = vectorizer.fit_transform(documents)  # 保持为稀疏矩阵

    # 随机选取 num_documents 个文档的索引
    random_indices = np.random.choice(X.shape[0], num_documents, replace=False)

    # 使用 lil_matrix 进行修改
    X_transformed = X.tolil()

    # 只对随机选取的文档进行剪切变换
    X_cropped = csr_matrix(random_crop(X[random_indices].toarray()))
    X_transformed[random_indices] = X_cropped

    # 修改完成后转换回 csr_matrix
    X_transformed = X_transformed.tocsr()

    cosine_similarities = [cosine_similarity_sparse(X[i], X_transformed[i]) for i in range(X.shape[0])]
    mean_cosine_similarity = np.mean(cosine_similarities)
    all_mean_cosine_similarities_crop.append(mean_cosine_similarity)
    print(f"第 {iteration + 1} 次（随机剪切, {num_documents} 个文档）计算的平均余弦相似度: {mean_cosine_similarity:.7f}")

overall_mean_cosine_similarity_crop = np.mean(all_mean_cosine_similarities_crop)
print(f"{n} 次计算后的平均余弦相似度（随机剪切）: {overall_mean_cosine_similarity_crop:.7f}")
print("\n")





