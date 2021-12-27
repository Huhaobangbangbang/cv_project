"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2021/11/22 2:08 PM
"""

# 将这个正方形转换为灰度图
import cv2
import numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def get_gray_scale(img_path):
    # 将图像灰度化
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print('大小：{}'.format(img.shape))
    print("类型：%s" % type(img))
    print(img)
    return img.shape[0],img.shape[1],img
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

def scripy_orgin():
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    rng = np.random.default_rng()
    points = rng.random((1000, 2))
    values = func(points[:, 0], points[:, 1])

    grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
    plt.subplot(221)
    plt.imshow(func(grid_x, grid_y).T, extent=(0, 1, 0, 1), origin='lower')
    plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(grid_z0.T, extent=(0, 1, 0, 1), origin='lower')
    plt.title('Nearest')
    plt.subplot(223)
    plt.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower')
    plt.title('Linear')
    plt.subplot(224)
    plt.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower')
    plt.title('Cubic')
    plt.gcf().set_size_inches(6, 6)
    plt.show()

def scripy():
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    rng = np.random.default_rng()
    points = rng.random((1000, 2))
    values = func(points[:, 0], points[:, 1])
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    plt.subplot(221)
    plt.imshow(func(grid_x, grid_y).T, extent=(0, 1, 0, 1), origin='lower')
    plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(grid_z0.T, extent=(0, 1, 0, 1), origin='lower')
    plt.title('Nearest')
    plt.show()



def get_weight(x,y,img_length):
    img_x,img_y,img_matrix = get_gray_scale(img_path)
    x = int(img_length*x/img_x)
    y = int(img_length*y/img_y)
    weight = img_matrix[x,y]
    return weight

if __name__ == '__main__':
    img_path = './square.png'
    # get_gray_scale(img_path)
    # get_gray_scale(img_path)
    # scripy()
    get_weight()
