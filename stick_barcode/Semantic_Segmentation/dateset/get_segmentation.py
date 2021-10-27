"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2021/10/25 8:16 PM
"""
# this script used by to get segmention picture
import os
import os.path as osp
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
def get_pic(pic_path):
    files = os.listdir(pic_path)
    img_list = []
    for img in files:
        if 'png' in img:
            img_list.append(img)
    return img_list


def get_label_information(pic_path, img_list):
    label_path = '/Users/huhao/Documents/cv_project/stick_barcode/Semantic_Segmentation/dateset/lib_dateset/ImageSets/label'
    segmentionclass_dir = '/Users/huhao/Documents/cv_project/stick_barcode/Semantic_Segmentation/dateset/lib_dateset/SegmentationClass'
    os.chdir(segmentionclass_dir)
    for img in tqdm(img_list):
        img_txt_path = osp.join(label_path, img[:-4]+'.txt')
        try:
            single_img_path = osp.join(pic_path, img)
            with open(img_txt_path, 'r') as fp:
                contents = fp.readlines()
            read_img = cv2.imread(single_img_path)
            # 读取图片的大小
            # cols指的是列，rows指的是行
            rows,cols,channels = read_img.shape
            # # 将图像转换为灰度图像
            # orig1 = read_img.copy()
            # orig = cv2.cvtColor(orig1,cv2.COLOR_BGR2GRAY)
            # 将图像转成指定的背景颜色 , 将图片转为双色图，背景颜色和object颜色
            orig = read_img.copy()

            for i in range(rows):
                for j in range(cols):
                    orig[i, j] = (128, 192, 0)  # 此处替换颜色，为BGR通道

            for sample in contents:
                sample_data = sample.strip().split()
                x = int(sample_data[0])
                y = int(sample_data[1])
                w = int(sample_data[2])
                h = int(sample_data[3])
                # 接下来我想执行利用给定x,y,w,h给图片画框
                cv2.rectangle(orig, (x, y), (w, h), (255, 185, 120), 2)
                # 接下来我想执行利用给定x,y,w,h给图片涂色,此处要防止数组越界
                for j in range(x, w):
                    for i in range(y, h):
                        try:
                            orig[i, j] = (255, 185, 120) # 此处替换颜色，为BGR通道
                        except IndexError:
                            pass
            # now I want to check the effect of the end pic
            # cv2.namedWindow("After process", 0)
            # cv2.resizeWindow("After process", 800, 640)
            # cv2.imshow("After process", orig)
            seg_img = osp.join(segmentionclass_dir, img)
            cv2.imwrite(seg_img, orig)

            # if cv2.waitKey(0) == 9:
            #     cv2.destroyAllWindows()

        except FileNotFoundError:
            print(img)


if __name__ == '__main__':
    pic_path = '/Users/huhao/Documents/cv_project/stick_barcode/Semantic_Segmentation/dateset/lib_dateset/JPEGImages'
    img_list = get_pic(pic_path)
    get_label_information(pic_path,img_list)