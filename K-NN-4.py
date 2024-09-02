# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/7/28 19:35
import numpy as np
import operator


def minkowski_distance(x, y, p):
    return np.power(np.sum(np.power(np.abs(x - y), p)), 1 / p)


def classify(inX, dataSet, labels, k, p=2):
    num = len(dataSet)
    # 计算闵可夫斯基距离
    distances = [minkowski_distance(inX, dataSet[i], p) for i in range(num)]
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

# 输出结果
print(classify(inX, dataSet, labels, 3, p=3))
