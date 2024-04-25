import cv2 as cv
import os
from PIL import Image


dir = "E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/1/"
files = os.listdir(dir)
ResultPath = "E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/2"  # 定义转换格式后的保存路径
for file in files:  # 这里可以去掉for循环
    a, b = os.path.splitext(file)  # 拆分影像图的文件名称
    this_dir = os.path.join(dir + file)  # 构建保存 路径+文件名
    img = Image.open(this_dir, 'r')  # 读取tif影像
    img = img.convert("RGB")
    img.save(ResultPath + a, format="PNG")
