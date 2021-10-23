import os

import numpy as np
import pandas as pd
import numpy
def get_data():
    new_dataset = []
    with open('./normal.txt','r') as f:
        coentents = f.readlines()
    for sample in coentents:
        sample_data = sample.strip().split()
        new_dataset.append("{} {} {} {} {}\n".format(sample_data[0],sample_data[1],sample_data[2],sample_data[3],'0'))
    with open('./normal.txt','r') as f:
        coentents = f.readlines()
    for sample in coentents:
        sample_data = sample.strip().split()
        new_dataset.append("{} {} {} {} {}\n".format(sample_data[0],sample_data[1],sample_data[2],sample_data[3],'1'))
    with open('end_data.txt', 'w') as f:
        f.writelines(new_dataset)



def get_csv():
    with open('./end_data.txt','r') as f:
        contents = f.readlines()
    new_data= []
    for sample in contents:
        sample_data = sample.strip().split()
        new_data.append(
            "{},{},{},{},{}\n".format(sample_data[0], sample_data[1], sample_data[2], sample_data[3], sample_data[4]))
    with open('end_data.csv','w') as f:
        f.writelines(new_data)
# get_csv()
def get_feature():
    with open('end_data.txt', 'r') as f:
        conetents =f.readlines()
    feature = []
    target = []
    for sample in conetents:
        sample_data = sample.strip().split()
        feature.append("{} {} {} {}".format(int(sample_data[0]),int(sample_data[1]),int(sample_data[2]),int(sample_data[3])))
        target.append(int(sample_data[4]))
    return feature,target
# feature ,target = get_feature()
def knn_model():
    # 1.  读取数据
    data = pd.read_csv('end_data.csv')
    print(data)
    # 获取特征
    feature = data[['feature1', 'feature2', 'feature3', 'feature4']]

    target = data['label']


    # 3. knn中特征数据是需要参与运算的，所以要保证特征数据必须为数值型的数据
    # 数据转换，将String类型数据转换为int
    #### map方法，进行数据转换

    dic = {}
    # unique()方法保证数据唯一
    # occ_arr = feature['occupation'].unique()
    # # 生成 字符对应数字的 关系表
    # for i in range(occ_arr.size):
    #     dic[occ_arr[i]] = i
    #
    #     # 数值替换字符串
    # feature['occupation'] = feature['occupation'].map(dic)

    # 4. 切片：训练数据和预测数据
    # 查看数据的形状 (训练的数据必须是二维数据)
    feature.shape

    # 训练数据
    x_train = feature[:549]
    y_train = target[:549]

    # 测试数据
    x_test = feature[549:]
    y_test = target[549:]

    # 5. 生成算法
    from sklearn.neighbors import KNeighborsClassifier
    # 实例化一个 knn对象,
    # 参数:n_neighbors可调,调到最终预测的是最好的结果.
    knn = KNeighborsClassifier(n_neighbors=10)
    # fit() 训练函数, (训练数据,训练数据的结果)
    knn.fit(x_train, y_train)

    # 对训练的模型进行评分 (测试数据,测试数据的结果)
    knn.score(x_test, y_test)

    # 6.预测数据
    print('真实的分类结果：', np.array(y_test))
    print('模型的分类结果：', knn.predict(x_test))
knn_model()