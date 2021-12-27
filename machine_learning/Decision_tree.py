"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2021/12/19 9:00 AM
"""
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

def decison_tree():
    # show data info
    data = load_iris()  # 加载 IRIS 数据集
    print('keys: \n', data.keys())  # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
    feature_names = data.get('feature_names')
    print('feature names: \n', data.get('feature_names'))  # 查看属性名称
    print('target names: \n', data.get('target_names'))  # 查看 label 名称
    x = data.get('data')  # 获取样本矩阵
    y = data.get('target')  # 获取与样本对应的 label 向量
    print(x.shape, y.shape)  # 查看样本数据
    print(data.get('DESCR'))

    # visualize the data
    f = []
    f.append(y == 0)  # 类别为第一类的样本的逻辑索引
    f.append(y == 1)  # 类别为第二类的样本的逻辑索引
    f.append(y == 2)  # 类别为第三类的样本的逻辑索引
    color = ['red', 'blue', 'green']
    fig, axes = plt.subplots(4, 4)  # 绘制四个属性之间的散点图
    for i, ax in enumerate(axes.flat):
        row = i // 4
        col = i % 4
        if row == col:
            ax.text(.1, .5, feature_names[row])
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        for k in range(3):
            ax.scatter(x[f[k], row], x[f[k], col], c=color[k], s=3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 设置间距
    plt.show()

    # 随机划分训练集和测试集
    num = x.shape[0]  # 样本总数
    ratio = 7 / 3  # 划分比例，训练集数目:测试集数目
    num_test = int(num / (1 + ratio))  # 测试集样本数目
    num_train = num - num_test  # 训练集样本数目
    index = np.arange(num)  # 产生样本标号
    np.random.shuffle(index)  # 洗牌
    x_test = x[index[:num_test], :]  # 取出洗牌后前 num_test 作为测试集
    y_test = y[index[:num_test]]
    x_train = x[index[num_test:], :]  # 剩余作为训练集
    y_train = y[index[num_test:]]

    # 构建决策树
    clf = tree.DecisionTreeClassifier()  # 建立决策树对象
    clf.fit(x_train, y_train)  # 决策树拟合

    # 预测
    y_test_pre = clf.predict(x_test)  # 利用拟合的决策树进行预测
    print('the predict values are', y_test_pre)  # 显示结果
    # 计算分类准确率
    acc = sum(y_test_pre == y_test) / num_test
    print('the accuracy is', acc)  # 显示预测准确率

if __name__ == '__main__':
    decison_tree()