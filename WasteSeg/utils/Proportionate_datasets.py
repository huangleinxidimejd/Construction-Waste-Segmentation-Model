import os
import shutil
import random

# 定义源文件夹和三个子文件夹的路径
src_dir = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/建筑垃圾消纳场/WasteSeg/data/all_data/images'

new_train = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/建筑垃圾消纳场/WasteSeg/data/train/images/'
new_val = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/建筑垃圾消纳场/WasteSeg/data/val/images/'
# new_test = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/IEEE_GRSS_Data/track2/new_test/images/'

# 创建三个子文件夹
os.makedirs(new_train, exist_ok=True)
os.makedirs(new_val, exist_ok=True)
# os.makedirs(new_test, exist_ok=True)

# 获取源文件夹中所有文件的路径
files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# 按照8:1:1的比例划分文件
num_files = len(files)

num_files1 = int(num_files * 0.8)
num_files2 = int(num_files * 0.2)
# num_files3 = num_files - num_files1 - num_files2

# 随机打乱文件列表
random.shuffle(files)

# 将文件拷贝到三个子文件夹中
for i in range(num_files1):
    shutil.copy(files[i], new_train)

for i in range(num_files1, num_files1 + num_files2):
    shutil.copy(files[i], new_val)

# for i in range(num_files1 + num_files2, num_files):
#     shutil.copy(files[i], new_test)
