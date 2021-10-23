
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
# from sklearn import datasets
from sklearn.datasets import load_iris


def get_normal_unnormal_data(normal_data_path,unnormal_path):
    """"get the the normal list and the unnormal list"""
    with open(normal_data_path,'r') as f:
        contents = f.readlines()
    normal_list = []
    for sample in contents:
        row = sample.strip().split()
        x1 = row[0]
        x2 = row[1]
        x3 = row[2]
        x4 = row[3]
        point_list = [x1,x2,x3,x4]
        # point_list = [x1, x2]
        normal_list.append(point_list)
    with open(unnormal_path,'r') as f:
        contents = f.readlines()
    unnormal_list = []
    for sample in contents:
        row = sample.strip().split()
        x1 = row[0]
        x2 = row[1]
        x3 = row[2]
        x4 = row[3]
        point_list = [x1,x2,x3,x4]
        # point_list = [x1, x2]
        unnormal_list.append(point_list)
    return normal_list,unnormal_list


def use_kmeans_to_success(normal_list, unnormal_list):

    X = np.array(normal_list)
    Y = np.array(unnormal_list)
    # 绘制数据分布图
    plt.legend(loc=2)
    # plt.show()

    estimator = KMeans(n_clusters=1)  # 构造聚类器
    estimator.fit(X)  # 聚类
    estimator.fit(Y)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    # 绘制k-means结果
    x0 = X[label_pred == 0]
    #x1 = X[label_pred == 1]
    y1 = Y[label_pred == 1]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='disturb')
    #plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='undisturb')

    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()



if __name__ == "__main__":
    normal_data_path = '/阴晓瑞/question4/normal.txt'
    unnormal_path = '/阴晓瑞/question4/unnormal.txt'
    normal_list,unnormal_list = get_normal_unnormal_data(normal_data_path, unnormal_path)
    use_kmeans_to_success(normal_list, unnormal_list)


