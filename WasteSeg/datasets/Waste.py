import os
import cv2
import torch
import numpy as np
from skimage import io
from torch.utils import data
import tifffile as tiff
# from osgeo import gdal_array
import matplotlib.pyplot as plt
import WasteSeg.utils.transform as transform
from skimage.transform import rescale
from torchvision.transforms import functional as F

"""
    RS是Remote Sensing
    该代码实现了一个用于遥感图像分割的数据集类RS，其中包括了数据集读取、预处理和增强等功能。
    在该代码中，num_classes表示数据集中的类别数，COLORMAP是用于可视化标签的颜色映射，CLASSES表示数据集中各个类别的名称。
    MEAN和STD是用于归一化图像的均值和标准差。root表示数据集所在的路径。
    该代码中的函数包括：
    showIMG：用于展示图像。
    normalize_image和normalize_images：对单张和多张图像进行归一化。
    Colorls2Index、Color2Index和Index2Color：用于彩色标签和索引标签之间的相互转换。
    rescale_image和rescale_images：对单张和多张图像进行缩放。
    get_file_name：获取数据集中的文件名列表。
    read_RSimages：读取遥感图像和标签图像。
    RS类继承了torch.utils.data.Dataset类，其中__init__方法初始化数据集类，接收两个参数，mode表示数据集模式，random_flip表示是否进行随机翻转增强。
    该方法通过调用read_RSimages函数加载数据集。
    __getitem__方法定义了数据集的取数据方式，接收一个参数idx，表示数据集中的数据索引。
    该方法首先获取该索引对应的图像和标签数据，然后根据random_flip参数进行随机翻转增强，接着对图像数据进行归一化并转换为张量，最后返回数据和标签。
"""
num_classes = 4
COLORMAP = [[125, 125, 125], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]]
CLASSES = ['Invalid', 'Background', 'Empty', 'Waste', 'Facilities']
"""
MEAN：训练数据集的均值，是一个长度为3的NumPy数组，分别表示RGB三个通道的均值。
STD：训练数据集的标准差，也是一个长度为3的NumPy数组，分别表示RGB三个通道的标准差。
在进行数据预处理时，可以使用这些均值和标准差对图像进行归一化处理，
即将每个像素的RGB值减去均值，然后再除以标准差，以使得每个通道的像素值都集中在0附近，有利于模型的训练。
"""

# 中国公开数据集均值和标准差
# MEAN = np.array([94.87, 96.53, 98.60])
# STD = np.array([57.70, 52.40, 50.09])

# IEEE GRSS大赛数据集均值和标准差
# MEAN = np.array([81.36, 88.03, 72.17])
# STD = np.array([51.92, 49.32, 48.45])

# WasteSeg数据集均值和标准差
MEAN = np.array([123.38, 142.02, 131.88])
STD = np.array([68.33, 63.59, 66.03])

root = 'F:/HL/Construction waste/WasteSeg/data'


# 定义一个函数用于展示图像
def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


# 对单张图像进行归一化
def normalize_image(im):
    return (im - MEAN) / STD


# 对多张图像进行归一化
def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs


"""
索引标签图像（Index images）是一种常用的遥感图像分类标签形式，
其中每个像素的值表示该像素所属的类别，通常用整数编码表示。
例如，在遥感图像分类中，可以将陆地、水体、道路等不同类别分别编码为1、2、3等整数值，
并将这些值分别赋给遥感图像中相应像素的索引标签图像。
在训练遥感图像分类模型时，索引标签图像作为输入图像的标签用于计算损失函数和评估模型的准确率等指标。
"""

"""建立一个颜色索引查找表
0 [125, 125, 125]
1 [0, 0, 0]
2 [255, 255, 255]
3 [255, 0, 0]
4 [0, 0, 255]"""
colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


# 将索引标签图像转换为彩色标签图像
def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


# 将多张彩色标签图像转换为索引图像
def Colorls2Index(ColorLabels):
    for i, data in enumerate(ColorLabels):
        ColorLabels[i] = Color2Index(data)
    return ColorLabels


# 将单张彩色标签图像转换为索引图像(分类)
def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    # IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    # 对返回的索引映射设置阈值，以确保所有索引值都小于或等于最大类数。
    IndexMap = IndexMap * (IndexMap <= num_classes)
    # 将索引映射强制转换为数据类型。np.uint8
    return IndexMap.astype(np.uint8)


# 将多张图像按照指定比例和插值方法进行缩放
def rescale_images(imgs, scale, order):
    for i, im in enumerate(imgs):
        imgs[i] = rescale_image(im, scale, order)
    return imgs


# 将单张图像按照指定比例和插值方法进行缩放
def rescale_image(img, scale=1 / 8, order=0):
    flag = cv2.INTER_NEAREST
    if order == 1:
        flag = cv2.INTER_LINEAR
    elif order == 2:
        flag = cv2.INTER_AREA
    elif order > 2:
        flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)),
                             interpolation=flag)
    return im_rescaled


# 获取数据集中的文件名列表
def get_file_name(mode):
    data_dir = root
    # assert mode in ['train', 'test']
    mask_dir = os.path.join(data_dir, mode, 'images')

    data_list = os.listdir(mask_dir)
    for vi, it in enumerate(data_list):
        data_list[vi] = it[:-4]
    return data_list


# 读取遥感图像和标签图像
def read_RSimages(mode, rescale=False, rotate_aug=False):
    data_dir = root
    # assert mode in ['train', 'test']
    img_dir = os.path.join(data_dir, mode, 'images')
    mask_dir = os.path.join(data_dir, mode, 'label')

    data_list = os.listdir(img_dir)

    data, labels = [], []
    count = 0
    for it in data_list:
        it_name = it[:-4]
        it_ext = it[-4:]
        if (it_name[0]=='.'):
            continue
        if (it_ext == '.tif'):
            img_path = os.path.join(img_dir, it)

            # 根据标签图像格式选择对应函数
            mask_path = os.path.join(mask_dir, it_name + '.png')
            # mask_path = os.path.join(mask_dir, it_name + '.tif')

            img = io.imread(img_path)
            # label = gdal_array.LoadFile(mask_path)

            # 根据标签图像格式选择对应函数
            label = io.imread(mask_path)
            # label = tiff.imread(mask_path)

            data.append(img)
            labels.append(label)

            count += 1
            if not count % 500: print('%d/%d images loaded.' % (count, len(data_list)))
            # if count: break
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')

    return data, labels


# 读取遥感图像但是没有标签图像
def read_RSimages_nolabel(mode, rescale=False, rotate_aug=False):
    data_dir = root
    # assert mode in ['train', 'test']

    img_dir = os.path.join(data_dir, mode, 'images')

    data_list = os.listdir(img_dir)

    data = []
    count = 0
    for it in data_list:
        it_name = it[:-4]
        it_ext = it[-4:]
        if (it_name[0]=='.'):
            continue
        if (it_ext == '.tif'):
            img_path = os.path.join(img_dir, it)
            img = io.imread(img_path)
            data.append(img)
            count += 1
            if not count % 500: print('%d/%d images loaded.' % (count, len(data_list)))
            # if count: break
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
    return data


class RS(data.Dataset):
    """
    __init__方法：该方法初始化数据集类，接收两个参数，mode表示数据集模式，random_flip表示是否进行随机翻转增强。
    该方法通过调用read_RSimages函数加载数据集。
    """

    def __init__(self, mode, random_flip=False):
        self.mode = mode
        self.random_flip = random_flip
        data, labels = read_RSimages(mode, rescale=False)
        # data = read_RSimages_nolabel(mode, rescale=False)

        self.data = data
        self.labels = Colorls2Index(labels)

        self.len = len(self.data)

    """
    _getitem__方法：该方法定义了数据集的取数据方式，接收一个参数idx，表示数据集中的数据索引。
    该方法首先获取该索引对应的图像和标签数据，然后根据random_flip参数进行随机翻转增强，接着对图像数据进行归一化并转换为张量，
    最后返回数据和标签。
    """

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.random_flip:
            data, label = transform.rand_flip(data, label)
            # data = transform.rand_flip(data)
        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data, label
        # return data

    """
    __len__方法：该方法返回数据集的长度，即数据集中数据的个数。
    """
    def __len__(self):
        return self.len
