import numpy as np  # 用于进行科学计算和数值分析
import pydensecrf.densecrf as dcrf  # 导入pydensecrf库中的densecrf模块，用于图像分割和CRF后处理

"""定义一个函数，实现对图像进行密集条件随机场（dense CRF）的处理"""


def dense_crf(img, output_probs):
    h = output_probs.shape[0]  # 获取输出概率矩阵的高度
    w = output_probs.shape[1]  # 获取输出概率矩阵的宽度

    output_probs = np.expand_dims(output_probs, 0)  # 在概率矩阵的第0个维度上增加一个维度
    output_probs = np.append(1 - output_probs, output_probs, axis=0)  # 在概率矩阵的第0个维度上添加1-概率矩阵

    d = dcrf.DenseCRF2D(w, h, 2)  # 创建一个二维的dense CRF对象
    U = -np.log(output_probs)   # 对概率矩阵取对数的相反数作为一元势能
    U = U.reshape((2, -1))  # 将一元势能矩阵变形为二维的形式
    U = np.ascontiguousarray(U)  # 将一元势能矩阵转换为连续的内存布局，以便于在C/C++中处理
    img = np.ascontiguousarray(img)  # 将图像矩阵转换为连续的内存布局

    d.setUnaryEnergy(U)  # 设置一元势能

    d.addPairwiseGaussian(sxy=20, compat=3)  # 添加高斯核成对势能
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)  # 添加双边滤波核成对势能

    Q = d.inference(5)  # 对dense CRF进行5次迭代，得到最终的概率分割结果
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))  # 将分割结果矩阵转换为二维的形式

    return Q  # 返回概率分割结果矩阵
