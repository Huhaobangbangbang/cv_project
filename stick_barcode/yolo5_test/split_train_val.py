"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2021/12/23 6:11 PM
"""
import os
import random
from os.path import *


# --------------------------全局地址变量--------------------------------#
xml_path = '/yolo5_test/Annotations'


traintxt_path = os.path.join("/yolo5_test/ImagesSet", "train.txt")
valtxt_path = os.path.join("/yolo5_test/ImagesSet", "val.txt")

if os.path.exists(traintxt_path):
    os.remove(traintxt_path)
if os.path.exists(valtxt_path):
    os.remove(valtxt_path)
# --------------------------全局地址变量--------------------------------#



def create_imagesets(xml_full_path, traintxt_full_path, valtxt_full_path):
    train_percent = 0.8  # 需要改变比例就改这里
    val_percent = 0.2
    # test_percent = 0.1
    xml_path = xml_full_path
    total_xml = os.listdir(xml_path)

    num = len(total_xml)
    lists = list(range(num))

    num_train = int(num * train_percent)

    train_list = random.sample(lists, num_train)
    for i in train_list:
        lists.remove(i)
    val_list = lists

    ftrain = open(traintxt_full_path, 'w')
    fval = open(valtxt_full_path, 'w')

    for i in range(num):
        name = total_xml[i][:-4] + '\n'
        if i in train_list:
            ftrain.write(name)
        else:
            fval.write(name)

    ftrain.close()
    fval.close()



if __name__ == '__main__':
    create_imagesets(xml_path, traintxt_path, valtxt_path)