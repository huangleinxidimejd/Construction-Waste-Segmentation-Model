import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

a_path = 'F:/HL/Construction waste/Building/data/pred/'
data_path = '延寿镇/19'
VOCdevkit_path = os.path.join(a_path, data_path)


if __name__ == "__main__":
    random.seed(0)

    temp_seg = os.listdir(VOCdevkit_path)
    total_seg = []
    for seg in temp_seg:
        total_seg.append(seg)

    num = len(total_seg)
    list = range(num)

    classes_nums = np.zeros([256], np.int)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(VOCdevkit_path, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。" % (png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。" % (name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。" % (name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    white_pixel = 0
    for i in range(1, 256):
        white_pixel = white_pixel + classes_nums[i]

    all_percent = white_pixel/classes_nums[0]
    print("{}年，全部 {} 张图片，白色像素点个数：{}, 黑色像素点个数：{}， 白色像素点占比：{}%".format(data_path, num, white_pixel, classes_nums[0],
                                                           all_percent*100))
