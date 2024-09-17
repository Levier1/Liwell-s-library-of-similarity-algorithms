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
def load_mnist_data():
    print("Loading MNIST data...")

    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 加载训练数据
    mnist_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)

    # 提取数据和标签
    X = mnist_dataset.data.float().view(-1, 28 * 28).numpy()  # MNIST数据
    y = mnist_dataset.targets.numpy()  # 标签

    print(f"数据集中共有 {X.shape[0]} 个样本")
    return X, y

# 添加随机噪声量
def add_random_noise(X, indices, noise_level=0.1):
    X_transformed = X.copy()
    noise = np.random.normal(scale=noise_level, size=X[indices].shape)
    X_transformed[indices] += noise
    return X_transformed

# 随机旋转变换
def random_rotation(X, indices, max_angle=25):
    X_transformed = X.copy()
    angles = np.random.uniform(low=-max_angle, high=max_angle, size=len(indices))  # 对选取的索引进行操作
    for i, idx in enumerate(indices):
        X_transformed[idx] = rotate(X[idx].reshape(28, 28), angles[i], reshape=False).flatten()  # 正确使用数组 reshape
    return X_transformed

# 亮度调整变换
def adjust_brightness(X, indices, brightness_factor=0.5):
    X_transformed = X.copy()
    X_transformed[indices] = np.clip(X[indices] * brightness_factor, 0, 255).astype(np.uint8)
    return X_transformed

# 仿射变换
def random_affine_transform(X, indices, shear_range=0.2, scale_range=0.2, translation_range=0.2):
    X_transformed = X.copy()
    for idx in indices:
        shear = np.random.uniform(-shear_range, shear_range)
        scale = np.random.uniform(1 - scale_range, 1 + scale_range)
        translation = np.random.uniform(-translation_range, translation_range, 2)
        matrix = np.array([[scale, shear, translation[0]], [0, scale, translation[1]]])
        X_transformed[idx] = affine_transform(X[idx].reshape(28, 28), matrix, output_shape=(28, 28)).flatten()
    return X_transformed

# 随机剪切变换
def random_shear_transform(X, indices, shear_range=0.2):
    X_transformed = X.copy()
    for idx in indices:
        shear = np.random.uniform(-shear_range, shear_range)
        matrix = np.array([[1, shear], [0, 1]])
        X_transformed[idx] = affine_transform(X[idx].reshape(28, 28), matrix, output_shape=(28, 28)).flatten()
    return X_transformed

# 计算变换前后的曼哈顿距离
def compute_distances(X1, X2):
    distances = []
    for i in range(len(X1)):
        distances.append(cityblock(X1[i].flatten(), X2[i].flatten()))  # 确保传入的是一维向量
    total_distance = np.mean(distances)  # 计算总的曼哈顿距离
    return total_distance

# 添加基线对比的函数，计算原始数据与自身的曼哈顿距离
def compute_baseline_distances(X):
    distances = []
    for i in range(len(X)):
        distances.append(cityblock(X[i].flatten(), X[i].flatten()))  # 自己与自己比较
    total_distance = np.sum(distances)
    return total_distance

# 设定用于计算的样本数目
sample_size = 40000  # 控制随机选取变换数据的数量 None 默认为使用全部数据集样本
n = 50  # n 表示要进行的计算次数

# 初始化列表用于保存每次计算的曼哈顿距离
all_distances_noisy = []
all_distances_rotated = []
all_distances_brightness_adjusted = []
all_distances_affine_transformed = []
all_distances_shear_transformed = []
all_distances_baseline = []  # 用于保存基线对比的曼哈顿距离

for iteration in range(n):
    # 每次迭代重新加载数据
    X, y = load_mnist_data()

    # 如果 sample_size 是 None，使用整个数据集
    if sample_size is None:
        sample_size = len(X)

    # 随机选取 sample_size 个数据的索引
    random_indices = np.random.choice(len(X), size=sample_size, replace=False)

    # 添加随机噪声量（只对 sample_size 个数据进行变换，但测量是对整个数据集）
    X_noisy = X.copy()  # 先复制整个数据集
    X_noisy[random_indices] = add_random_noise(X, random_indices)[random_indices]  # 部分数据被替换
    distance_noisy = compute_distances(X, X_noisy)
    all_distances_noisy.append(distance_noisy)

    # 随机旋转变换（同样的逻辑，部分数据变换）
    X_rotated = X.copy()
    X_rotated[random_indices] = random_rotation(X, random_indices)[random_indices]
    distance_rotated = compute_distances(X, X_rotated)
    all_distances_rotated.append(distance_rotated)

    # 亮度调整变换
    X_brightness_adjusted = X.copy()
    X_brightness_adjusted[random_indices] = adjust_brightness(X, random_indices)[random_indices]
    distance_brightness_adjusted = compute_distances(X, X_brightness_adjusted)
    all_distances_brightness_adjusted.append(distance_brightness_adjusted)

    # 仿射变换
    X_affine_transformed = X.copy()
    X_affine_transformed[random_indices] = random_affine_transform(X, random_indices)[random_indices]
    distance_affine_transformed = compute_distances(X, X_affine_transformed)
    all_distances_affine_transformed.append(distance_affine_transformed)

    # 随机剪切变换
    X_shear_transformed = X.copy()
    X_shear_transformed[random_indices] = random_shear_transform(X, random_indices)[random_indices]
    distance_shear_transformed = compute_distances(X, X_shear_transformed)
    all_distances_shear_transformed.append(distance_shear_transformed)

    # 计算基线对比：自己与自己
    distance_baseline = compute_baseline_distances(X)
    all_distances_baseline.append(distance_baseline)

    # 输出每次计算的曼哈顿距离
    print(f"第 {iteration + 1} 次计算: 经过随机噪声添加后的曼哈顿距离: {distance_noisy:.3f}")
    print(f"第 {iteration + 1} 次计算: 经过随机旋转操作后的曼哈顿距离: {distance_rotated:.3f}")
    print(f"第 {iteration + 1} 次计算: 经过亮度调整后的曼哈顿距离: {distance_brightness_adjusted:.3f}")
    print(f"第 {iteration + 1} 次计算: 经过仿射变换后的曼哈顿距离: {distance_affine_transformed:.3f}")
    print(f"第 {iteration + 1} 次计算: 经过随机剪切后的曼哈顿距离: {distance_shear_transformed:.3f}")
    print(f"基线对比曼哈顿距离: {distance_baseline:.3f}")
    print("\n")

# 计算 n 次计算后的平均曼哈顿距离
mean_distance_noisy = np.mean(all_distances_noisy)
mean_distance_rotated = np.mean(all_distances_rotated)
mean_distance_brightness_adjusted = np.mean(all_distances_brightness_adjusted)
mean_distance_affine_transformed = np.mean(all_distances_affine_transformed)
mean_distance_shear_transformed = np.mean(all_distances_shear_transformed)
mean_distance_baseline = np.mean(all_distances_baseline)

# 输出各个变换类型 n 次计算的平均结果
print(f"（随机噪声） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_noisy:.3f}")
print(f"（随机旋转） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_rotated:.3f}")
print(f"（亮度调整） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_brightness_adjusted:.3f}")
print(f"（仿射变换） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_affine_transformed:.3f}")
print(f"（随机剪切） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_shear_transformed:.3f}")
print(f"基线对比（自己与自己） {n} 次计算后的最终平均曼哈顿距离: {mean_distance_baseline:.3f}")
