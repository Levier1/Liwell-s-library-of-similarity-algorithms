# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/7/28 20:42
# Don't get frustrated,keep going
# 开发者:郭汉卿

import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import affine_transform
from scipy.spatial.distance import cityblock
from torchvision import datasets, transforms


def load_mnist_data(sample_size=None):
    print("Loading MNIST data...")

    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 加载训练数据
    mnist_dataset = datasets.MNIST(root='D:/Levi的代码练习/相似度算法复现/data', train=True, download=True,
                                   transform=transform)

    # 提取数据和标签
    X = mnist_dataset.data.float().view(-1, 28 * 28).numpy()
    y = mnist_dataset.targets.numpy()

    if sample_size is not None:
        # 取样指定数量的数据
        X = X[:sample_size]
        y = y[:sample_size]

    print("MNIST data loaded")
    return X, y


# 添加随机噪声量
def add_random_noise(X, noise_level=0.1):
    noise = np.random.normal(scale=noise_level, size=X.shape)
    X_noisy = X + noise
    return X_noisy


# 随机旋转变换
def random_rotation(X, max_angle=25):
    angles = np.random.uniform(low=-max_angle, high=max_angle, size=len(X))
    X_rotated = np.zeros_like(X)
    for i in range(len(X)):
        X_rotated[i] = rotate(X[i].reshape(28, 28), angles[i], reshape=False).flatten()
    return X_rotated


# 亮度调整变换
def adjust_brightness(X, brightness_factor=0.5):
    X_adjusted = np.clip(X * brightness_factor, 0, 255).astype(np.uint8)
    return X_adjusted


# 仿射变换
def random_affine_transform(X, shear_range=0.2, scale_range=0.2, translation_range=0.2):
    X_transformed = np.zeros_like(X)
    for i in range(len(X)):
        shear = np.random.uniform(-shear_range, shear_range)
        scale = np.random.uniform(1 - scale_range, 1 + scale_range)
        translation = np.random.uniform(-translation_range, translation_range, 2)
        matrix = np.array([[scale, shear, translation[0]], [0, scale, translation[1]]])
        X_transformed[i] = affine_transform(X[i].reshape(28, 28), matrix, output_shape=(28, 28)).flatten()
    return X_transformed

# 随机剪切变换
def random_shear_transform(X, shear_range=0.2):
    X_transformed = np.zeros_like(X)
    for i in range(len(X)):
        shear = np.random.uniform(-shear_range, shear_range)
        matrix = np.array([[1, shear], [0, 1]])
        X_transformed[i] = affine_transform(X[i].reshape(28, 28), matrix, output_shape=(28, 28)).flatten()
    return X_transformed

# 计算变换前后的曼哈顿距离
def compute_distances(X1, X2):
    distances = []
    for i in range(len(X1)):
        distances.append(cityblock(X1[i], X2[i]))
    avg_distance = np.mean(distances)
    return avg_distance

# 计算部分样本与整个数据集的曼哈顿距离
def compute_sample_to_dataset_distances(X, sample_indices):
    distances = []
    for idx in sample_indices:
        sample = X[idx]
        for i in range(len(X)):
            distance = cityblock(sample, X[i])
            distances.append(distance)
    avg_distance = np.mean(distances)
    return avg_distance


# 设定用于计算的样本数目
sample_size = 1000  # 修改这里来设置使用的数据量，None表示使用全部数据
X, y = load_mnist_data(sample_size=sample_size)

# 添加随机噪声量
X_noisy = add_random_noise(X)

# 随机旋转变换
X_rotated = random_rotation(X)

# 亮度调整变换
X_brightness_adjusted = adjust_brightness(X)

# 仿射变换
X_affine_transformed = random_affine_transform(X)

# 随机剪切变换
X_shear_transformed = random_shear_transform(X)

# 随机选择样本索引
sample_indices = np.random.choice(len(X), size=10, replace=False)  # 从数据集中随机选择10个样本

# 计算各种变换后的曼哈顿距离
distance_noisy = compute_distances(X, X_noisy)
distance_rotated = compute_distances(X, X_rotated)
distance_brightness_adjusted = compute_distances(X, X_brightness_adjusted)
distance_affine_transformed = compute_distances(X, X_affine_transformed)
distance_shear_transformed = compute_distances(X, X_shear_transformed)

# 计算样本与数据集之间的曼哈顿距离
distance_sample_to_dataset = compute_sample_to_dataset_distances(X, sample_indices)

# 输出各个变换的计算结果
print(f"数据量为 {sample_size} 时，添加随机噪声后的曼哈顿距离: {distance_noisy:.3f}")
print(f"数据量为 {sample_size} 时，随机旋转变换后的曼哈顿距离: {distance_rotated:.3f}")
print(f"数据量为 {sample_size} 时，亮度调整变换后的曼哈顿距离: {distance_brightness_adjusted:.3f}")
print(f"数据量为 {sample_size} 时，仿射变换后的曼哈顿距离: {distance_affine_transformed:.3f}")
print(f"数据量为 {sample_size} 时，随机剪切变换后的曼哈顿距离: {distance_shear_transformed:.3f}")
print(f"数据量为 {sample_size} 时，部分样本与数据集之间的曼哈顿距离: {distance_sample_to_dataset:.3f}")
