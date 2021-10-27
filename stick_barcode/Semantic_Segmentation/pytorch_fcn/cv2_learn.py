"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2021/10/26 11:13 AM
"""
import cv2
import  numpy as np

img=cv2.imread('/Users/huhao/Documents/cv_project/stick_barcode/Semantic_Segmentation/dateset/lib_dateset/JPEGImages/1.png')

rows,cols,channels = img.shape
print(rows,cols)
# img=cv2.resize(img,None,fx=0.5,fy=0.5)
# rows,cols,channels = img.shape
#cv2.imshow('img',img)
#
# #转换hsv
# hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# lower_blue=np.array([90,70,70])
# upper_blue=np.array([110,255,255])
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# cv2.imshow('Mask', mask)
#
# #腐蚀膨胀
# erode=cv2.erode(mask,None,iterations=1)
# cv2.imshow('erode',erode)
# dilate=cv2.dilate(erode,None,iterations=1)
# cv2.imshow('dilate',dilate)

#遍历替换
for i in range(rows):
    for j in range(cols):
        img[i,j]=(255, 185, 120)#此处替换颜色，为BGR通道
cv2.imshow('res',img)

cv2.waitKey(0)
cv2.destroyAllWindows()