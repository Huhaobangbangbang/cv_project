"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 datetime： 2021/10/23 7:33 PM
"""
import os
import os.path as osp
# 这个代码用来处理voc数据集
import os
import shutil
import xml.dom.minidom
from xml.dom import minidom
import cv2 as cv
import xml.etree.ElementTree as ET
from tqdm import tqdm

def copy_xml_to_other_folder(pic_path,xml_path):
    """move xml file from pic_path to xml_path"""
    pic_xml_list = os.listdir(pic_path)
    for sample in tqdm(pic_xml_list):
        if 'xml' in sample:
            ori_xml_path = osp.join(pic_path,sample)
            end_xml_path = osp.join(xml_path,sample)
            shutil.copy(ori_xml_path,end_xml_path)

def xml_to_txt(xml_path, outdir):
    """读取xml文件，获得想要的内容存到txt文件中"""
    xml_list = os.listdir(xml_path)
    for xml_file in xml_list:
        xml_file_path = osp.join(xml_path,xml_file)
        if '.DS_Store' in xml_file_path:
            pass
        else:
            DOMTree = ET.parse(xml_file_path)
            root = DOMTree.getroot()
            for bndbox in root.iter('bndbox'):
                node = []
                for child in bndbox:
                    node.append(int(child.text))
                x1, y1 = node[0], node[1]
                x3, y3 = node[2], node[3]
                print(x1,y1,x3,y3)

    # for i, file in enumerate(xml_path):
    #     file_save = file.split('.')[0] + '.txt'
    #     file_txt = os.path.join(outdir, file_save)
    #     f_w = open(file_txt, 'w')
    #     # actual parsing
    #     DOMTree = xml.dom.minidom.parse(file)
    #     annotation = DOMTree.documentElement
    #     filename = annotation.getElementsByTagName("path")[0]
    #     imgname = filename.childNodes[0].data
    #     img_temp = imgname.split('\\')[-1]
    #     img_temp = os.path.join(image_dir, img_temp)
    #     image = cv.imread(imgname)
    #     cv.imwrite(img_temp, image)
    #     objects = annotation.getElementsByTagName("object")
    #     print(file)
    #     for object in objects:
    #         bbox = object.getElementsByTagName("robndbox")[0]
    #         cx = bbox.getElementsByTagName("cx")[0]
    #         x = float(cx.childNodes[0].data)
    #         print(x)
    #         cy = bbox.getElementsByTagName("cy")[0]
    #         y = float(cy.childNodes[0].data)
    #         print(y)
    #         cw = bbox.getElementsByTagName("w")[0]
    #         w = float(cw.childNodes[0].data)
    #         print(w)
    #         ch = bbox.getElementsByTagName("h")[0]
    #         h = float(ch.childNodes[0].data)
    #         print(h)
    #         cangel = bbox.getElementsByTagName("angle")[0]
    #         angle = float(cangel.childNodes[0].data)
    #         print(angle)
    #         cname = object.getElementsByTagName("name")[0]
    #         name = cname.childNodes[0].data
    #         print(name)
    #         x1 = x - w / 2.
    #         y1 = y - h / 2.
    #         x2 = x + w / 2.
    #         y2 = y - h / 2.
    #         x3 = x + w / 2.
    #         y3 = y + h / 2.
    #         x4 = x - w / 2.
    #         y4 = y + h / 2.
    #         temp = str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(x3) + ' ' + str(y3) + ' ' + str(
    #             x4) + ' ' + str(y4) + ' ' + name + '\n'
    #         f_w.write(temp)
    #     f_w.close()


def get_new_name(pic_path):
    """"change the name
        you can change the pic name what you want
    """
    pic_list = os.listdir(pic_path)
    index = 0
    for pic in pic_list:
        index = index + 1
        single_pic_path = osp.join(pic_path,pic)
        single_pic_path_end = osp.join(pic_path,str(index)+'.png')
        os.rename(single_pic_path,single_pic_path_end)




if __name__ == '__main__':
    pic_path = '/Users/huhao/Documents/cv_project/stick_barcode/Semantic_Segmentation/dateset/label_dateset/JPEGImages'
    # get_new_name(pic_path)
    # xml_path是存放xml文件的路径
    xml_path = '/Users/huhao/Documents/cv_project/stick_barcode/Semantic_Segmentation/dateset/label_dateset/Annotation'
    # outdir 为存放txt文件的路径
    outdir = '/Users/huhao/Documents/cv_project/stick_barcode/Semantic_Segmentation/dateset/label_dateset/ImageSets/Main'
    xml_to_txt(xml_path, outdir)
    # copy_xml_to_other_folder(pic_path, xml_path)