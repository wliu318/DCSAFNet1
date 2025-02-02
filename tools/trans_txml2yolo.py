# -*- coding: utf-8 -*-
# change xml to txt
# ------------kevin---------------
import copy
from lxml.etree import Element, SubElement, tostring, ElementTree
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
from os import getcwd
import pickle
from os import listdir, getcwd
from os.path import join

sets = ['train', 'test']
classes = ["person", "people", "cyclist", "person?"]  # 改成自己的类别
# classes = ["0", "1", "2", "3"]  # 类别
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open('E:/deeplearning/dataset/LLVIP/labels_xml/%s.xml' % (image_id), encoding='UTF-8')
    # out_file_name = image_id[-26:-4]
    out_file = open('E:/deeplearning/dataset/LLVIP/labels_txt/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file.close()

if not os.path.exists('E:/deeplearning/dataset/LLVIP/labels_txt/'):
   os.makedirs('E:/deeplearning/dataset/LLVIP/labels_txt/')

xml_path = 'E:/deeplearning/dataset/LLVIP/labels_xml/'

# xml list
img_xmls = os.listdir(xml_path)
for img_xml in img_xmls:
    label_name = img_xml.split('.')[0]
    print(label_name)
    convert_annotation(label_name)
