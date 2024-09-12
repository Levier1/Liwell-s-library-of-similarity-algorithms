# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/7/21 20:19
import numpy as np
from sklearn.metrics import jaccard_score
from scipy.ndimage import affine_transform
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

"""
在仿射变换函数中，参数rotation_angle=0, translation=(0, 0)和是两个可以调整的函数
默认情况下：
1、rotation_angle（旋转角度）：
通常以度数为单位 范围可以从 -180 度到 180 度 这表示完整的反转和旋转范围
如果数据中不允许完全反转 可以限制范围到 -90 度到 90 度之间 
2、translation（平移量）：
平移量通常以像素为单位 可以是任何整数值或小数值 
对于一个典型的 20x20 的数据样本 平移量的范围可以在 -10 到 10 之间
或者更大一些 具体取决于数据的大小和变换的期望效果
但在这里我不知道为什么 每次调整其参数 就会产生报错：ValueError: Classification metrics can't handle a mix of multilabel-indicator and multiclass-multioutput targets
具体原因我还在调试 由于要写报告了 所以现在这里做好标记
"""

# 定义仿射变换函数
def apply_affine_transform(data, rotation_angle=0, translation=(0, 0)):
    # 创建一个仿射变换矩阵
    angle_rad = np.deg2rad(rotation_angle)
    transform_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), translation[0]],
                                 [np.sin(angle_rad), np.cos(angle_rad), translation[1]],
                                 [0, 0, 1]])

    # 应用仿射变换到每个样本的数据
    transformed_data = np.zeros_like(data)
    for i in range(len(data)):
        transformed_data[i] = affine_transform(data[i], transform_matrix)

    return transformed_data


# 定义随机裁剪函数
def apply_random_crop(data, crop_size=(10, 10)):
    cropped_data = np.zeros((data.shape[0], crop_size[0], crop_size[1]))
    for i in range(data.shape[0]):
        x_start = np.random.randint(0, data.shape[1] - crop_size[0] + 1)
        y_start = np.random.randint(0, data.shape[2] - crop_size[1] + 1)
        cropped_data[i] = data[i, x_start:x_start + crop_size[0], y_start:y_start + crop_size[1]]
    return cropped_data


# 确认随机种子
np.random.seed(0)

# 示例数据
data = np.random.randint(0, 2, size=(100, 10, 10))
data1 = np.random.randint(0, 2, size=(100, 10, 10))

# 应用仿射变换
transformed_data = apply_affine_transform(data)

# 应用随机裁剪
cropped_data = apply_random_crop(data1)

# 计算仿射变换后数据的杰卡德相似度
ground_truth = np.random.randint(0, 2, size=(100, 10, 10))
ground_truth1 = np.random.randint(0, 2, size=(100, 10, 10))

flat_ground_truth = ground_truth.reshape((ground_truth.shape[1], -1))
flat_ground_truth1 = ground_truth1.reshape((ground_truth1.shape[1], -1))
flat_transformed_data = transformed_data.reshape((transformed_data.shape[1], -1))
flat_cropped_data = cropped_data.reshape((cropped_data.shape[1], -1))

# 输出形状检查
print(f"Flat ground truth shape: {flat_ground_truth.shape}")
print(f"Flat transformed data shape: {flat_transformed_data.shape}")
print(f"Flat cropped data shape: {flat_cropped_data.shape}")

# 计算仿射变换之后的杰卡德相似度
jaccard_similarity_transformed = jaccard_score(flat_ground_truth.astype(int), flat_transformed_data.astype(int),
                                               average='micro')
print(f"经过仿射变换后的杰卡德相似度: {jaccard_similarity_transformed:7f}")

# 计算随机裁剪后数据的杰卡德相似度
jaccard_similarity_cropped = jaccard_score(flat_ground_truth1.astype(int), flat_cropped_data.astype(int),
                                           average='micro')
print(f"经过随机裁剪的杰卡德相似度: {jaccard_similarity_cropped:7f}")
