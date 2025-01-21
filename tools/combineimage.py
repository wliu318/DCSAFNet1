# -*- coding:utf-8 -*-
# move all images to one folder
# ------------kevin---------------

import os
import shutil
from tqdm import tqdm

filepath = "/home/kevin/PycharmProjects/Detection/KAIST/images/test"
output_path = "/home/kevin/PycharmProjects/Detection/KAIST_processed/images/visible/val/"
files_1 = os.listdir(filepath)  # set

filetext = open("/home/kevin/PycharmProjects/Detection/KAIST_processed/images/visible/val.txt", "w")
for filename_1 in tqdm(files_1):
    tmp_path_1 = os.path.join(filepath, filename_1)
    if os.path.isdir(tmp_path_1):
        files_2 = os.listdir(tmp_path_1)  # Vxxx
        for filename_2 in tqdm(files_2):
            tmp_path_2 = os.path.join(tmp_path_1, filename_2)
            if os.path.isdir(tmp_path_2):  # lwir or visible
                files_3 = os.listdir(tmp_path_2)
                for filename_3 in files_3:
                    if filename_3 == "visible":  # choose lwir or visible
                        tmp_path_3 = os.path.join(tmp_path_2, filename_3)
                        files_4 = os.listdir(tmp_path_3)
                        for filename_4 in files_4:
                            tmp_path_4 = os.path.join(tmp_path_3, filename_4)
                            new_filename = filename_1 + filename_2 + filename_3 + filename_4
                            new_path = output_path + new_filename
                            shutil.copy(tmp_path_4, new_path)
                            content = new_path + '\n'
                            filetext.write(content)

filetext.close()