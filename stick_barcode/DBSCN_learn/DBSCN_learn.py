# encoding:utf-8
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from sklearn import datasets

list_1 = []
list_2 = []


# 数据集一：随机生成散点图,参数为点的个数
# def scatter(num):
#     for i in range(num):
#         x = random.randint(0, 100)
#         list_1.append(x)
#         y = random.randint(0, 100)
#         list_2.append(y)
#     print(list_1)
#     print(list_2)
#     data = list(zip(list_1, list_2))
#     print(data)
#     #plt.scatter(list_1, list_2)
#     #plt.show()
#     return data
# scatter(50)

def loadDataSet(fileName, splitChar='\t'):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet


# 计算两个点之间的欧式距离，参数为两个元组
def dist(t1, t2):
    dis = math.sqrt((np.power((t1[0] - t2[0]), 2) + np.power((t1[1] - t2[1]), 2)))
    # print("两点之间的距离为："+str(dis))
    return dis


# dis = dist((1,1),(3,4))
# print(dis)


# DBSCAN算法，参数为数据集，Eps为指定半径参数，MinPts为制定邻域密度阈值
def dbscan(Data, Eps, MinPts):
    num = len(Data)  # 点的个数
    # print("点的个数："+str(num))
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    # print(unvisited)
    visited = []  # 已经访问的点的列表
    C = [-1 for i in range(num)]
    # C为输出结果，默认是一个长度为num的值全为-1的列表
    # 用k来标记不同的簇，k = -1表示噪声点
    k = -1
    # 如果还有没访问的点
    while len(unvisited) > 0:
        # 随机选择一个unvisited对象
        p = random.choice(unvisited)
        unvisited.remove(p)
        visited.append(p)
        # N为p的epsilon邻域中的对象的集合
        N = []
        for i in range(num):
            if (dist(Data[i], Data[p]) <= Eps):  # and (i!=p):
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts:
            k = k + 1
            # print(k)
            C[p] = k
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        if (dist(Data[j], Data[pi]) <= Eps):  # and (j!=pi):
                            M.append(j)
                    if len(M) >= MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1
    return C


# 数据集二：788个点
dataSet = loadDataSet('788points.txt', splitChar=',')
C = dbscan(dataSet, 2, 0)
print(C)
x = []
y = []
for data in dataSet:
    x.append(data[0])
    y.append(data[1])
plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(x, y, c=C, marker='o')
plt.show()