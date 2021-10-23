import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
def get_x_y_list():
    with open('./55z.txt','r') as f:
        contents = f.readlines()
    x_list = []
    y_list = []
    for sample in contents:
        sample_data = sample.strip().split()
        x = sample_data[2]
        y = sample_data[3]
        x_list.append(x)
        y_list.append(y)
        print(x,y)
    return x_list,y_list
x_list,y_list = get_x_y_list()


def get_track(x_axix,y_axix):


    # 这里导入你自己的数据

    # 开始画图
    # sub_axix = filter(lambda x: x % 200 == 0, x_axix)
    plt.title('track Analysis')
    # plt.plot(x_axix, y_axix, color='green', label='training accuracy')
    # for index in range(len(x_axix)):
    #     plt.scatter(x_axix[index], y_axix[index])
    plt.scatter(x_axix,y_axix,s=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # python 一个折线图绘制多个曲线
get_track(x_list,y_list)