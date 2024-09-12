# Don't get frustrated,keep going
# 开发者:郭汉卿
# 开发时间:2024/7/15 17:32
"""
KNN算法流程如下:
　a.计算已知类别数据集中的点与当前点之间的距离；
　b.按照距离递增次序排序；
　c.选取与当前点距离最小的k个点；
　d.确定前k个点所在类别的出现频率；
　e.返回前k个点所出现频率最高的类别作为当前点的预测分类。
在kNN.py中，添加一个函数classify0作为 KNN 算法的核心函数，该函数的完整形式为：
def classify0(inX, dataSet, labels, k):其中各个参数的含义如下：
inX - 用于要进行分类判别的数据(来自测试集)
dataSet - 用于训练的数据(训练集)
labels - 分类标签
K - KNN算法参数,选择距离最小的k个点
在上述参数列表中，dataSet为所有训练数据的集合，也就是表示所有已知类别数据集中的所有点
dataSet为一个矩阵，其中每一行表示已知类别数据集中的一个点。inX为一个向量，表示当前要判别分类的点。
按照上述算法流程，我们首先应该计算inX这个要判别分类的点到dataSet中每个点之间的距离。
dataSet中每个点也是用一个向量表示的，点与点之间的距离怎么计算呢？
没错，就是求两向量之间的距离，这里有很多距离计算公式，包括但不限于：
欧氏距离(此处先选择欧氏距离实现）
曼哈顿距离
切比雪夫距离
闵可夫斯基距离
标准化欧氏距离
马氏距离
夹角余弦
汉明距离
杰卡德距离 & 杰卡德相似系数
信息熵
"""
import numpy as np
import operator
from os import listdir


def classify(inX, dataSet, labels, k):
    # Start Code Here
    # 返回dataSet的行数，即已知数据集中的所有点的数量
    num = len(dataSet)
    # 行向量方向上将inX复制m次，然后和dataSet矩阵做相减运算
    n = num - 1
    ans = np.asarray(inX)
    tmp = np.asarray(inX)
    while n != 0:
        ans = np.append(ans, tmp)
        n -= 1
    ans = np.reshape(ans, [num, 2])
    ans = ans - dataSet
    # 减完后，对每个数做平方
    ans = ans * ans
    # 平方后按行求和，axis=0表 示列相加，axis-1表示行相加
    dis = np.zeros([num, 1], dtype=int)
    # 开方计算出欧式距离
    for i in range(num):
        dis[i] = ans[i][0] + ans[i][1]
    ans = np.reshape(dis, [1, num])
    # 对距离从小到大排序，注意assort函数返回的是数组值从小到大的索引值2
    y = np.argsort(ans)
    # 用于类别/次数的字典，key为类别， value为次数
    y = list(y[0])
    # 取出第近的元素对应的类别
    dic = dict.fromkeys(list(set(labels)), 0)
    # 对类别次数进行累加
    for i in range(k):
        dic[labels[y[i]]] += 1
    # 根据字典的值从大到小排序
    dic = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return dic[0][0]
    # End Code Here


"""
实验数据带入尝试：
[]中的第一个数据为学生的理综成绩 第二个为学生的文综成绩
且设定上理科生理综成绩相对于文综成绩普遍较高 文科生文综成绩相对于理综成绩普遍较高
"""
dataSet = np.array([[250, 100], [270, 120], [111, 230], [130, 260], [200, 80], [70, 190], [90, 210], [140, 250]])
labels = ["理科生", "理科生", "文科生", "文科生", "理科生", "文科生", "文科生", "文科生"]
inX = [240, 110]
print(classify(inX, dataSet, labels, 3))
