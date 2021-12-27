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
    """XML设计的核心是包含和传输数据"""
    xml_list = os.listdir(xml_path)
    for xml_file in tqdm(xml_list):
        xml_file_name = osp.basename(xml_file)[:-4]
        xml_file_path = osp.join(xml_path,xml_file)
        if '.DS_Store' in xml_file_path:
            pass
        else:
            DOMTree = ET.parse(xml_file_path)
            root = DOMTree.getroot()
            object_list = []
            for object in root.iter('object'):
                object_bndbox = object.iter('bndbox')
                object_name = object.iter('name')
                for bndbox in object_bndbox:
                    node = []
                    for child in bndbox:
                        node.append(int(child.text))
                    x, y = node[0], node[1]
                    w, h = node[2], node[3]
                for name in object_name:
                    name = name.text
                    object_list.append("{} {} {} {} {}\n".format(x, y, w, h,name))
        txt_file_path = osp.join(outdir,xml_file_name+'.txt')
        with open(txt_file_path,'w') as fp:
            fp.writelines(object_list)

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

def get_train_test():
    """get the test.txt and the train.txt"""
    xml_path = '/dateset/lib_dateset/Annotation'
    xml_list = os.listdir(xml_path)
    img_list = []
    for xml in xml_list:
        if '.DS' in xml:
            pass
        else:
            img_list.append(osp.basename(xml)[:-4])
    train_sum = 40
    train_list = []
    test_list = []
    for img in img_list:
        if len(train_list) <= train_sum:
            train_list.append("{}\n".format(img))
        else:
            test_list.append("{}\n".format(img))
    os.chdir('/dateset/lib_dateset/ImageSets/Main')
    with open('train.txt','w') as fp:
        fp.writelines(train_list)
    with open('test.txt','w') as fp:
        fp.writelines(test_list)


if __name__ == '__main__':
    pic_path = '/dateset/lib_dateset/JPEGImages'
    # get_new_name(pic_path)
    # xml_path是存放xml文件的路径
    xml_path = '/Users/huhao/Documents/cv_project/stick_barcode/yolo5_test/Annotations'
    # outdir 为存放txt文件的路径
    outdir = '/Users/huhao/Documents/cv_project/stick_barcode/yolo5_test/label'
    xml_to_txt(xml_path, outdir)
    # copy_xml_to_other_folder(pic_path, xml_path)
