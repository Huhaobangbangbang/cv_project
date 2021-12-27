import cv2
import numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_gray_scale(img_path):
    # 将图像灰度化
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # print("类型：%s" % type(img))
    # 让书和背景分离，这里我们将图片二值化
    retVal, image = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # # 开始进行腐蚀操作
    # corrosion_img = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))  ##腐蚀预处理，确定处理核的大小,矩阵操作
    # img3 = cv2.erode(image, corrosion_img, iterations=10)  # 进行腐蚀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    expand_pic = cv2.dilate(img3, kernel)
    #
    pic_matrix = numpy.array(expand_pic)
    print(pic_matrix)

    cv2.imshow('grayimg', expand_pic)
    cv2.waitKey(0)


if __name__ == '__main__':
    # get the img_path
    sample_img_path = './sample1.jpg'
    get_gray_scale(sample_img_path)