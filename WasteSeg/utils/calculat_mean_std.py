import os
import numpy as np
import tifffile as tiff

# 定义训练数据集的路径

# train_data_path = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/Building-max/Data/train/images/'
train_data_path = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/建筑垃圾消纳场/WasteSeg/data/train/images/'

files = os.listdir(train_data_path)

# 加载所有训练图像，并将它们存储在一个数组或列表中
train_data = []
for img_name in files:
    # print(img_name)
    # img = cv2.imread(os.path.join(train_data_path, img_name))
    img = tiff.imread(os.path.join(train_data_path, img_name))
    train_data.append(img)

print("%d张数据加载完成！" % len(files))

train_data = np.array(train_data)

# 计算每个通道的均值
mean_channel_1 = np.mean([img[:, :, 0] for img in train_data])
mean_channel_2 = np.mean([img[:, :, 1] for img in train_data])
mean_channel_3 = np.mean([img[:, :, 2] for img in train_data])
print("均值计算完成！")

# 计算每个通道的标准差
std_channel_1 = np.std([img[:, :, 0] for img in train_data])
std_channel_2 = np.std([img[:, :, 1] for img in train_data])
std_channel_3 = np.std([img[:, :, 2] for img in train_data])
print("标准差计算完成！")

# 将三个通道的均值组合成一个 RGB 均值向量
mean = [mean_channel_1, mean_channel_2, mean_channel_3]
std = [std_channel_1, std_channel_2, std_channel_3]

# 打印结果
print("RGB 均值：", mean)
print("RGB 标准差：", std)

with open('mean_std.txt', 'a') as file0:
    print('mean:%s' % mean, 'std:%s' % std, file=file0)
