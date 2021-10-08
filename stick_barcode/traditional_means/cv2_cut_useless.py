import cv2
import numpy as np


def match_loc(model_imgpath,test_imgpath): # 通过匹配获取目标图片相对位置
    model_img = cv2.imread(model_imgpath)
    test_img = cv2.imread(test_imgpath)

    b_model, g_model, r_model = cv2.split(model_img)
    b_test, g_test, r_test = cv2.split(test_img)
    w, h = g_model.shape[::-1]

    result = cv2.matchTemplate(g_test, g_model, cv2.TM_CCOEFF_NORMED)
    (min_val, score, min_loc, max_loc) = cv2.minMaxLoc(result)
    bottom_right = (max_loc[0] + w, max_loc[1] + h)
    print('max_loc',max_loc)
    kuang = cv2.rectangle(test_img, max_loc, bottom_right, 255, 2)
    cv2.imshow('test_img',kuang)
    cv2.waitKey(0)
    return test_img,max_loc, g_test


def Morph_exam(test_img,g_test):
    sobel = cv2.Sobel(g_test, cv2.CV_8U, 1, 0, ksize=3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow('binary',binary)
    cv2.waitKey(0)

    # 形态核：膨胀让轮廓突出--- 腐蚀去掉细节--再膨胀，让轮廓更明显
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    dilation = cv2.dilate(binary, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    dilation2 = cv2.dilate(erosion, element2, iterations=2)
    cv2.imshow('dilation2',dilation2)
    cv2.waitKey(0)

    # 查找轮廓和筛选文字区域
    region = []
    _,contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area < 800):
            continue
        rect = cv2.minAreaRect(cnt)
        print("rect is: ",rect)

        # 获取box四点坐标, 根据文字特征，筛选可能是文本区域的矩形。
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        if (height > width * 1.3):
            continue
        region.append(box)

    # 绘制轮廓
    for box in region:
        cv2.drawContours(test_img, [box], 0, (0, 255, 0), 2)
    cv2.imshow('img', test_img)
    cv2.waitKey(0)
    return region

def get_blank_area(img_path):
    # 将图像灰度化
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)



if __name__ == '__main__':
    model_imgpath = './sample2.jpg'
    test_imgpath = './sample2.jpg'
    test_img,max_loc, g_test = match_loc(model_imgpath, test_imgpath)
    Morph_exam(test_img,g_test)
    # get the img_path
    sample_img_path = 'sample2.jpg'
