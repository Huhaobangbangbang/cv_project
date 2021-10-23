import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot

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
        # point_list = [x1,x2,x3,x4]
        point_list = [x1, x2]
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
        # point_list = [x1,x2,x3,x4]
        point_list = [x1, x2]
        unnormal_list.append(point_list)
    return normal_list,unnormal_list

def use_kmeans_model(normal_list,unnormal_list):
    """we use kmeans algorithm here"""
    X = np.array(normal_list+unnormal_list)
    # 把上面数据点分为两组（非监督学习）
    clf = KMeans(n_clusters=2)
    clf.fit(X)  # 分组

    centers = clf.cluster_centers_  # 两组数据点的中心点
    labels = clf.labels_  # 每个数据点所属分组
    print(labels)

    for i in range(len(labels)):
        pyplot.scatter(X[i][0], X[i][1], c=('r' if labels[i] == 0 else 'b'))
    pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', s=100)

    # 预测
    predict = [[2, 1], [6, 9]]
    label = clf.predict(predict)
    for i in range(len(label)):
        pyplot.scatter(predict[i][0], predict[i][1], c=('r' if label[i] == 0 else 'b'), marker='x')

    pyplot.show()


if __name__ == "__main__":
    normal_data_path = '/阴晓瑞/question4/normal.txt'
    unnormal_path = '/阴晓瑞/question4/unnormal.txt'
    normal_list,unnormal_list = get_normal_unnormal_data(normal_data_path, unnormal_path)
    use_kmeans_model(normal_list, unnormal_list)


 #
 #
 #
 #
 # # normal_list
 #    x = np.array(normal_list+unnormal_list)
 #    # 把上面数据点分为两组（非监督学习）
 #    clf = KMeans(n_clusters=2)
 #    clf.fit(x)  # 分组
 #    centers = clf.cluster_centers_  # 两组数据点的中心点
 #    labels = clf.labels_  # 每个数据点所属分组
 #
 #    for i in range(len(labels)):
 #        pyplot.scatter(x[i][0], x[i][1], c=('r' if labels[i] == 0 else 'b'))
 #    pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', s=100)
 #
 #    # 预测
 #    predict = [[2, 1], [6, 9]]
 #    label = clf.predict(predict)
 #    for i in range(len(label)):
 #        pyplot.scatter(predict[i][0], predict[i][1], c=('r' if label[i] == 0 else 'b'), marker='x')
 #
 #    pyplot.show()