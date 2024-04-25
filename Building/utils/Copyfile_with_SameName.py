import os
import tifffile as tiff
from skimage import io

"""将文件夹中的同名文件复制到新文件夹中"""
# 定义源文件夹和三个子文件夹的路径
src_dir = 'F:/HL/大兴区建筑垃圾消纳场数据集/label/'

new_train_images = 'F:/HL/大兴区建筑垃圾消纳场数据集/train/'
# new_train_label = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/建筑垃圾消纳场/Construction waste/Building/data/train/label/'

new_val_images = 'F:/HL/大兴区建筑垃圾消纳场数据集/val/'
# new_val_label = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/建筑垃圾消纳场/Construction waste/Building/data/val/label/'

# new_test_images = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/IEEE_GRSS_Data/track2/new_test/images/'
# new_test_label = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/IEEE_GRSS_Data/track2/new_test/label/'
image_path = 'F:/HL/大兴区建筑垃圾消纳场数据集/image/'

for img_name in os.listdir(src_dir):
    # 标签图像地址
    name = img_name[0:-4]

    train_images = new_train_images + name + '.tif'
    label = io.imread(train_images)
    io.imsave(image_path + name + '.tif', label)

for img_name in os.listdir(src_dir):
    # 标签图像地址
    name = img_name[0:-4]

    val_images = new_val_images + name + '.tif'
    label = io.imread(val_images)
    io.imsave(image_path + name + '.tif', label)

# for img_name in os.listdir(new_val_images):
#     # 标签图像地址
#     name = img_name[0:-4]
#
#     label_path = src_dir + name + '.png'
#     label = io.imread(label_path)
#     io.imsave(new_val_label + name + '.png', label)

# for img_name in os.listdir(new_test_images):
#     # 标签图像地址
#     label_path = src_dir + img_name
#     label = tiff.imread(label_path)
#     tiff.imwrite(new_test_label + img_name[0:-4] + '.tif', label)
