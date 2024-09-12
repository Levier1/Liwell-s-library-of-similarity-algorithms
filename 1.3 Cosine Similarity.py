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
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

# 只选择一部分数据进行演示
documents = newsgroups_train.data[:100]

# 初始化文本向量化器
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents).toarray()
# 添加随机噪声来模拟数据变换
X_transformed = X + np.random.normal(0, 0.3, size=X.shape)  # 添加正态分布噪声


# 定义余弦相似度函数1（只适用比较两个向量之间的相似度）
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)


# 计算余弦相似度数值
cosine_similarities = []
for i in range(len(X)):
    similarity = cosine_similarity(X[i], X_transformed[i])
    cosine_similarities.append(similarity)

# 格式化余弦相似度为小数点后7位的字符串
formatted_cosine_similarities = ["{:.7f}".format(similarity) for similarity in cosine_similarities]

# 计算所有余弦相似度的总和
sum_cosine_similarities = sum(cosine_similarities)

# 计算平均余弦相似度
mean_cosine_similarity = sum_cosine_similarities / len(cosine_similarities)

# 输出计算结果
print(f"经过添加随机噪声之后的余弦相似度: {formatted_cosine_similarities}")
print(f"经过添加随机噪声之后的平均余弦相似度: {mean_cosine_similarity:.7f}")

# 使用TF-IDF向量化文本数据
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf_vectorizer.fit_transform(newsgroups_train.data).toarray()
# 将TF-IDF矩阵reshape成图像形状，这里简单地使用前400个特征作为图像的像素
image_data = X_tfidf[:, :400].reshape(-1, 20, 20)  # 假设图像大小为20x20
# 随机旋转图像数据
rotated_images = []
for image in image_data:
    angle = np.random.uniform(-20, 20)  # 随机生成旋转角度
    rotated = rotate(image, angle, mode='edge')  # 应用旋转变换，使用边缘填充模式
    rotated_images.append(rotated)
# 将旋转后的图像数据转换回原始形状
rotated_images = np.array(rotated_images).reshape(len(image_data), -1)

# 进行对比
# 对比原始数据和旋转后的数据
# 将图像数据展平为一维数组
original_data = X_tfidf[:, :400]
rotated_data = rotated_images[:, :400]

# 计算数据之间的相似度
similarity_scores1 = []
for i in range(len(original_data)):
    if np.std(original_data[i]) != 0 and np.std(rotated_data[i]) != 0:
        similarity = np.corrcoef(original_data[i], rotated_data[i])[0, 1]  # 使用皮尔逊相关系数进行去中心化
    else:
        similarity = 0  # 标准差为零时，相似度设为0或者其他处理方式
    similarity_scores1.append(similarity)

# 计算平均值
average_similarity1 = np.mean(similarity_scores1)

# 输出结果
print(f"经过随机旋转变换之后的平均皮尔逊相似度: {average_similarity1:.7f}")

# 添加随机噪声来模拟数据变换
X_noisy = X + np.random.normal(0, 0.01, size=X.shape)  # 添加正态分布噪声
# 新增亮度调整变换
X_scaled = scale(X_noisy)  # 使用scale函数进行标准化处理，模拟亮度调整

# 计算亮度调整变换前后数据集的平均余弦相似度
cosine_similarities1 = []
for i in range(len(X)):
    similarity = cosine_similarity(X[i], X_scaled[i])
    cosine_similarities1.append(similarity)

# 格式划计算结果
formatted_cosine_similarities1 = ["{:.7f}".format(similarity) for similarity in cosine_similarities1]
# 计算平均值
mean_cosine_similarity = np.mean(cosine_similarities1)

# 输出计算结果
print(f"经过随机亮度调整变换之后的余弦相似度: {formatted_cosine_similarities1}")
print(f"经过随机亮度调整变换之后的平均余弦相似度: {mean_cosine_similarity:.7f}")

# 添加随机噪声来模拟对数据进行修改
X_transformed = X + np.random.normal(0.5, 0.8, size=X.shape)  # 添加正态分布噪声
# 添加仿射变换过程
affine_transformer = FunctionTransformer(lambda x: x + np.random.normal(0, 0.1, size=x.shape), validate=False)
X_affine_transformed = affine_transformer.transform(X)

# 计算变换后数据集与原始数据集的平均余弦相似度
cosine_similarities_affine = []
for i in range(len(X)):
    similarity = cosine_similarity(X[i], X_affine_transformed[i])
    cosine_similarities_affine.append(similarity)

# 格式划计算结果
formatted_cosine_similarities_affine = ["{:.7f}".format(similarity) for similarity in cosine_similarities_affine]
# 计算平均值
mean_cosine_similarity_affine = np.mean(cosine_similarities_affine)

# 输出结果
print(f"经过随机仿射变换之后的余弦相似度: {formatted_cosine_similarities_affine}")
print(f"经过随机仿射变换之后的平均余弦相似度: {mean_cosine_similarity_affine:.7f}")


# 结论：经过反复调整参数 最终得出仿射变换的抗压能力最强
# 或者说是余弦相似度度量 在对经过仿射变换调整过的文本数据类型数据集进行相似性评估时 效果并不是很理想

# 定义随机剪切函数
def random_crop(x, crop_ratio=0.5):
    num_features = x.shape[1]
    crop_size = int(num_features * crop_ratio)
    crop_start = np.random.randint(0, num_features - crop_size + 10)
    x_cropped = x.copy()  # 复制原始向量，避免修改原始数据
    x_cropped[:, crop_start:crop_start + crop_size] = 0.5  # 将随机选择的部分置零
    return x_cropped


# 添加随机剪切操作
X_cropped = random_crop(X)

# 计算剪切后数据集与原始数据集的平均余弦相似度
cosine_similarities_cropped = []
for i in range(len(X)):
    similarity = cosine_similarity(X[i], X_cropped[i])
    cosine_similarities_cropped.append(similarity)

# 格式划计算结果
formatted_cosine_similarities_cropped = ["{:.7f}".format(similarity) for similarity in cosine_similarities_cropped]
# 计算平均值
mean_cosine_similarities_cropped = np.mean(cosine_similarities_cropped)

# 输出结果
print(f"经过随机剪切变换之后的余弦相似度: {formatted_cosine_similarities_cropped}")
print(f"经过随机剪切变换之后的平均余弦相似度: {mean_cosine_similarities_cropped:.7f}")
