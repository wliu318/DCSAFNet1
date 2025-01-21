# -*- coding:utf-8 -*-
# move all the xml to one folder
# ------------kevin---------------

import os
import shutil
from tqdm import tqdm




newfile = "E:\\deeplearning\\dataset\\\LLVIP\\\labels\\test\\gt_llvip_result.txt"
filepath = "E:\\deeplearning\\dataset\\\LLVIP\\\labels\\test"
files_1 = os.listdir(filepath)  # set
imageid = 1
newdata = []
width = 1280.0
height = 1024.0
with open(newfile, 'w', encoding='utf-8') as f_out:
    for filename_1 in tqdm(files_1):
        tmp_path_1 = os.path.join(filepath, filename_1)
        with open(tmp_path_1, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data = [float(ll) for ll in line.split(' ')]
                tmpdata = str(imageid) + "," + \
                          str((data[1] - data[3] / 2.0) * width) + "," + \
                          str((data[2] - data[4] / 2.0) * height) + "," + \
                          str(data[3] * width) + "," + \
                          str(data[4] * height) + ","  + \
                          str(1)
                f_out.write(tmpdata + "\n")

        imageid += 1  # label id+1

