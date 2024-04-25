import os, sys  # 导入模块


def add_prefix_subfolders():  # 定义函数名称
    mark = '20-'  # 准备添加的前缀内容
    old_names = os.listdir(path)  # 取路径下的文件名，生成列表
    old_names.sort(key=lambda x: int(x.split('.')[0]))
    i = 1
    for old_name in old_names:  # 遍历列表下的文件名
        if old_name != sys.argv[0]:  # 代码本身文件路径，防止脚本文件放在path路径下时，被一起重命名
            name = str(i)
            os.rename(os.path.join(path, old_name), os.path.join(path, mark + name + '.png'))  # 子文件夹重命名
            print(old_name, "has been renamed successfully! New name is: ", mark + name + '.png')
            i = i + 1


if __name__ == '__main__':
    path = r'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/建筑垃圾消纳场/WasteSeg/data/20/label/'
    # 运行程序前，记得修改主文件夹路径！
    add_prefix_subfolders()  # 调用定义的函数
