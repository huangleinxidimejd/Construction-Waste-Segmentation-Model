import os
import math
import random
import numpy as np
from skimage import io
from osgeo import gdal_array

def create_crops(img, size):
    crop_imgs = []
    # crop_labels = []
    # label_dims = len(label.shape)
    
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    assert (h>=c_h and w>=c_w), "Cannot crop area."
    h_rate = h/c_h
    w_rate = w/c_w
    h_times = math.ceil(h_rate)
    w_times = math.ceil(w_rate)
    if h_times==1: stride_h=0
    else:
        stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
    if w_times==1: stride_w=0
    else:
        stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
    for j in range(h_times):
        for i in range(w_times):
            s_h = int(j*c_h - j*stride_h)
            if(j==(h_times-1)): s_h = h - c_h
            e_h = s_h + c_h
            s_w = int(i*c_w - i*stride_w)
            if(i==(w_times-1)): s_w = w - c_w
            e_w = s_w + c_w
            # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
            # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
            crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
            # if label_dims==2:
            #     crop_labels.append(label[s_h:e_h, s_w:e_w])
            # else:
            #     crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs
    
def save_crops(save_dir, dir_name, prefix, crop_imgs):
    data_len = len(crop_imgs)
    img_dir = os.path.join(save_dir, dir_name, 'images')
    # label_dir = os.path.join(save_dir, dir_name, 'label')
    if not os.path.exists(img_dir): os.makedirs(img_dir)
    # if not os.path.exists(label_dir): os.makedirs(label_dir)
    for i in range(data_len):
        crop_img = crop_imgs[i]
        # crop_label = crop_labels[i]
        img_path = os.path.join(img_dir, '%d.tif'%i)
        # label_path = os.path.join(label_dir, 'tif%d.png'%i)
        io.imsave(img_path, crop_img)
        # io.imsave(label_path, crop_label)
    print('%d '%data_len+' images saved.')

def main():
    
    src_path = r'F:/HL/Construction waste/Building/data/按行政区域切割/阳坊镇/20/1.tif'
    # label_path = r'D:\PIE_SDK\image\changping.tif'
    root = r'F:/HL/Construction waste/Building/data/按行政区域切割/阳坊镇/20/'
    crop_size = 500
    area_idx = '1'
    img = gdal_array.LoadFile(src_path)
    # label = gdal_array.LoadFile(label_path)
    img = np.transpose(img, (1, 2, 0))   
    # label = np.transpose(label, (1, 2, 0))
    print(img.shape)
    # print(label.shape)
    
    crop_imgs = create_crops(img, size=(crop_size, crop_size))
    save_crops(root, 'crops', area_idx, crop_imgs)

if __name__ == '__main__':
    main()