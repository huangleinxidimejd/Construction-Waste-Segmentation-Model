import cv2 as cv
import os
from PIL import Image


def Convert_To_Png_AndCut(dir):
    files = os.listdir(dir)
    ResultPath = "E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/1/"  # 定义转换格式后的保存路径
    # ResultPath2 = "./RS_Cut_Result/"  # 定义裁剪后的保存路径
    # ResultPath3 = "./RS_Cut_Result/"  # 定义裁剪后的保存路径
    for file in files:  # 这里可以去掉for循环
        a, b = os.path.splitext(file)  # 拆分影像图的文件名称
        this_dir = os.path.join(dir + file)  # 构建保存 路径+文件名
        img = Image.open(this_dir, "r")  # 读取tif影像
        # 第二个参数是通道数和位深的参数，
        # IMREAD_UNCHANGED = -1  # 不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
        # IMREAD_GRAYSCALE = 0  # 进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
        # IMREAD_COLOR = 1   # 进行转化为RGB三通道图像，图像深度转为8位
        # IMREAD_ANYDEPTH = 2  # 保持图像深度不变，进行转化为灰度图。
        # IMREAD_ANYCOLOR = 4  # 若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位

        img.save(ResultPath + a, format='PNG')  # 保存为png格式

        # # 下面开始裁剪-不需要裁剪tiff格式的可以直接注释掉
        # hight = img.shape[0]  # opencv写法，获取宽和高
        # width = img.shape[1]
        # # 定义裁剪尺寸
        # w = 480  # 宽度
        # h = 360  # 高度
        # _id = 1  # 裁剪结果保存文件名：0 - N 升序方式
        # i = 0
        # while (i + h <= hight):  # 控制高度,图像多余固定尺寸总和部分不要了
        #     j = 0
        #     while (j + w <= width):  # 控制宽度，图像多余固定尺寸总和部分不要了
        #         cropped = img[i:i + h, j:j + w]  # 裁剪坐标为[y0:y1, x0:x1]
        #         cv.imwrite(ResultPath2 + a + "_" + str(_id) + b, cropped)
        #         _id += 1
        #         j += w
        #     i = i + h


if __name__ == '__main__':
    path = r"E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/1/"  # 遥感tiff影像所在路径
    # 裁剪影像图
    Convert_To_Png_AndCut(path)
