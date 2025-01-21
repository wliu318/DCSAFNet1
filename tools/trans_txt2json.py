import os
from pandas.io import json
from tqdm import tqdm


def txtToJson(path):
    image_id = 0  # 图片初始id
    annotation_id = 0  # 标注文件初始id

    coco_output = {
        "images": [],  # 存放所有图片信息
        "categories": [],  # 存放数据集类别信息
        "annotations": []  # 存放所有标注文件信息
    }

    categories = [
        {'id': 0, 'name': 'person'},
        {'id': 1, 'name': 'car'},
        {'id': 2, 'name': 'bicycle'},
        # {'id': 3, 'name': 'cyclist'},
        # {'id': 4, 'name': 'person?a'},
    ]

    coco_output['categories'] = categories
    height = 512
    width = 640

    filename = os.listdir(path)  # 获取path路径下的所有文件的名字(eg:123.txt)
    # filename=filename[:2]
    # print(len(filename),filename)
    filejson = dict()
    for fn in tqdm(filename):
        p = os.path.join(path, fn)
        image_name = fn.rstrip('.txt')

        image_dict = {
            "id": int(image_id),
            "im_name": f'{image_name}',
            "height": height,
            "width": width,
        }
        # 将当前图片信息加入到coco_output中
        coco_output['images'].append(image_dict)
        try:
            # 大多数文件都是utf-8格式的，少数文件是gbk格式，默认使用utf-8格式读取，为了防止gbk文件使程序中断,使用try catch处理特殊情况
            f = open(p, mode="r", encoding="utf-8")
            lines = f.readlines()
            for line in lines:
                data = [float(ll) for ll in line.split(' ')]
                category_id = data[0]  # image id
                bbox = [int((data[1] - data[3] / 2.0) * width), int((data[2] - data[4] / 2.0) * height), int(data[3] * width), int(data[4] * height)]  # bbox

                ann_dict = {
                    "id": annotation_id,
                    "image_id": int(image_id),
                    "category_id": 1,    #int(category_id),
                    "bbox": bbox,
                    "height": bbox[3],
                    "occlusion": 0,
                    "ignore": 0
                }
                coco_output["annotations"].append(ann_dict)  # 更新当前图片的object label信息
                annotation_id += 1  # label id+1

            image_id += 1  # 图片id+1

            filejson = coco_output
            f.close()
        except Exception:
            f = open(p, mode="r", encoding="gbk")
            data = f.read().replace(" ", "").replace("\n", "")
            filejson = data
            f.close()
    return filejson, len(filejson)


def saveInJsonFile(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(filejson, ensure_ascii=False))



# 要读取的文件夹路径
readpath = r"E:/deeplearning/dataset/FLIR-align-3class/labels/test/"
filejson, length = txtToJson(readpath)
print(filejson)

# 保存的文件路径 1.json可以更换成其他的名字
save_path = r"E:/deeplearning/dataset/FLIR-align-3class/labels/FLIR-align-3class_test.json"
saveInJsonFile(filejson, save_path)