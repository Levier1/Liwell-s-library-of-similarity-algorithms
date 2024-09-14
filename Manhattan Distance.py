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

# 定义加载MNIST数据集的函数
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

# 设定用于计算的样本数目
sample_size = None  # 可以通过修改这个值来调整数据量 None 默认为选取全部数据
n = 50  # n 表示要进行的计算次数

# 初始化列表用于保存每次计算的曼哈顿距离
all_distances_noisy = []
all_distances_rotated = []
all_distances_brightness_adjusted = []
all_distances_affine_transformed = []
all_distances_shear_transformed = []

for iteration in range(n):
    # 每次迭代重新加载数据
    X, y = load_mnist_data(sample_size=sample_size)

    # 添加随机噪声量
    X_noisy = add_random_noise(X)
    distance_noisy = compute_distances(X, X_noisy)
    all_distances_noisy.append(distance_noisy)

    # 随机旋转变换
    X_rotated = random_rotation(X)
    distance_rotated = compute_distances(X, X_rotated)
    all_distances_rotated.append(distance_rotated)

    # 亮度调整变换
    X_brightness_adjusted = adjust_brightness(X)
    distance_brightness_adjusted = compute_distances(X, X_brightness_adjusted)
    all_distances_brightness_adjusted.append(distance_brightness_adjusted)

    # 仿射变换
    X_affine_transformed = random_affine_transform(X)
    distance_affine_transformed = compute_distances(X, X_affine_transformed)
    all_distances_affine_transformed.append(distance_affine_transformed)

    # 随机剪切变换
    X_shear_transformed = random_shear_transform(X)
    distance_shear_transformed = compute_distances(X, X_shear_transformed)
    all_distances_shear_transformed.append(distance_shear_transformed)

    # 输出每次计算的曼哈顿距离
    print(f"第 {iteration + 1} 次计算: 经过随机噪声添加后的曼哈顿距离: {distance_noisy:.3f}")
    print(f"第 {iteration + 1} 次计算: 经过随机旋转操作后的曼哈顿距离: {distance_rotated:.3f}")
    print(f"第 {iteration + 1} 次计算: 经过亮度调整后的曼哈顿距离: {distance_brightness_adjusted:.3f}")
    print(f"第 {iteration + 1} 次计算: 经过仿射变换后的曼哈顿距离: {distance_affine_transformed:.3f}")
    print(f"第 {iteration + 1} 次计算: 经过随机剪切后的曼哈顿距离: {distance_shear_transformed:.3f}")
    print("\n")

# 计算 n 次计算后的平均曼哈顿距离
mean_distance_noisy = np.mean(all_distances_noisy)
mean_distance_rotated = np.mean(all_distances_rotated)
mean_distance_brightness_adjusted = np.mean(all_distances_brightness_adjusted)
mean_distance_affine_transformed = np.mean(all_distances_affine_transformed)
mean_distance_shear_transformed = np.mean(all_distances_shear_transformed)

# 输出各个变换类型 n 次计算的平均结果
print(f"（随机噪声） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_noisy:.3f}")
print(f"（随机旋转） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_rotated:.3f}")
print(f"（亮度调整） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_brightness_adjusted:.3f}")
print(f"（仿射变换） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_affine_transformed:.3f}")
print(f"（随机剪切） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_shear_transformed:.3f}")
