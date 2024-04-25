from PIL import Image
import os
from skimage import io
import numpy as np
import cv2
from mpmath import im

a_path = 'F:/HL/Construction waste/Building/data/pred/'
data_path = '东小口地区办事处/20'
path = os.path.join(a_path, data_path)

# 读入图像
data_dir = path
img_dir = os.path.join(data_dir)
data_list = os.listdir(img_dir)
data = []
count = 0
white_pixel_count = 0
black_pixel_count = 0
for it in data_list:
    it_name = it[:-4]
    it_ext = it[-4:]
    per_white_pixel_count = 0
    per_black_pixel_count = 0
    if (it_name[0]=='.'):
        continue
    if (it_ext == '.png'):
        img_path = os.path.join(img_dir, it)
        img = Image.open(img_path)
        # 获取图像数据
        gray_image = img.convert("L")
        width, height = gray_image.size
        for y in range(width):
            for x in range(height):
                r = gray_image.getpixel((x, y))
                if r < 50:
                    per_black_pixel_count += 1
                if 50 <= r <= 256:
                    per_white_pixel_count += 1
        percent = per_white_pixel_count/per_black_pixel_count
        print("{}，白色像素点个数：{}, 黑色像素点个数：{}， 白色像素点占比：{}%".format(it_name, per_white_pixel_count, per_black_pixel_count,
                                                               percent*100))

    white_pixel_count = white_pixel_count+ per_white_pixel_count
    black_pixel_count = black_pixel_count + per_black_pixel_count
    count += 1

all_percent = white_pixel_count/black_pixel_count
print("{}年，全部 {} 张图片，白色像素点个数：{}, 黑色像素点个数：{}， 白色像素点占比：{}%".format(data_path, count, white_pixel_count, black_pixel_count,
                                                           all_percent*100))
