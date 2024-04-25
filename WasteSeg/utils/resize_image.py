
import cv2
import os.path
import tifffile as tiff
from skimage import io
# 图片文件夹路径
images_dir = r'F:/HL/Construction waste/Building/data/val/images/'
label_dir = r'F:/HL/Construction waste/Building/data/val/label/'
new_images = r'F:/HL/Construction waste/Building/data/new_val/images/'
new_label = r'F:/HL/Construction waste/Building/data/new_val/label/'



for img_name in os.listdir(images_dir):
    name = img_name[0:-4]
    # 图像和标签地址
    img_path = images_dir + name + '.tif'
    label_path = label_dir + name + '.png'

    img = tiff.imread(img_path)
    # label = tiff.imread(label_path)
    label = io.imread(label_path)
    """裁剪一块区域，并且缩放"""

    # img.shape(512, 512, 3)
    width, height, channel = img.shape

    crop_width = 480
    crop_height = 480

    # 定义裁剪位置坐标
    left = int((width - crop_width)/2)
    top = int((height - crop_height)/2)
    right = left + crop_width
    bottom = top + crop_height

    cropped_image = img[top:bottom, left:right]
    cropped_label = label[top:bottom, left:right]


    # 保存缩放后的图像
    tiff.imwrite(new_images + img_name[0:-4]+'.tif', cropped_image)
    io.imsave(new_label + img_name[0:-4]+'.png', cropped_label)