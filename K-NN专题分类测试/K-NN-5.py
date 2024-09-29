# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/7/28 19:41
import numpy as np
import operator


def mahalanobis_distance(x, y, cov_inv):
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))


def classify(inX, dataSet, labels, k):
    num = len(dataSet)

    # 计算协方差矩阵和其逆
    cov_mat = np.cov(dataSet, rowvar=False)
    cov_inv = np.linalg.inv(cov_mat)

    # 计算马氏距离
    distances = [mahalanobis_distance(inX, dataSet[i], cov_inv) for i in range(num)]

    # 获得距离从小到大排序的索引
    sortedDistIndices = np.argsort(distances)

    classCount = {}
    # 选取距离最小的k个点
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 按票数从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


# 测试数据
dataSet = np.array([[250, 100], [270, 120], [111, 230], [130, 260], [200, 80], [70, 190], [90, 210], [140, 250]])
labels = ["理科生", "理科生", "文科生", "文科生", "理科生", "文科生", "文科生", "文科生"]
inX = np.array([240, 110])
print(classify(inX, dataSet, labels, 3))
