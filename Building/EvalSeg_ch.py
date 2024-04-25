import math
import os
import time
import torch
import numpy as np
import torch.autograd
import torchvision
from matplotlib import pyplot as plt
from skimage import io
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
# from Building.modelss.DAMM_DeepLabv3_plus import DAMM_DeepLabv3_plus as Net
# from Building.modelss.DAMM_tandem_DeepLabv3_plus import DAMM_DeepLabv3_plus as Net
# from Building.modelss.DSAC_DeepLabv3_plus import DSAC_DeepLabv3_plus as Net
# from Building.modelss.Feature_fusion_DeeplabV3_plus import FCN_ASPP as Net
# from Building.modelss.traditional_DeepLabv3_plus import traditional_DeepLabv3_plus as Net
# from Building.modelss.RDSA_DeepLabv3_plus import RDSA_DeepLabv3_plus as Net
# from Building.modelss.Unet import UNet as Net
# from Building.modelss.SegNet import SegNet as Net
# from Building.modelss.PSPNet import Pspnet as Net
# from Building.modelss.deeplabv3_plus import DeepLab as Net
# from Building.modelss.DSATNet import Model as Net
# from Building.modelss.convlsrnet import Model as Net
from Building.modelss.SDSCUNet import create_shunted_unet_model as Net

from Building.datasets import Building_ch as RS
from utils.loss import CrossEntropyLoss2d
from utils.utils import binary_accuracy as accuracy
from utils.utils import compute_AP50 as AP50
from utils.utils import intersectionAndUnion, AverageMeter, CaclTP

#################################
DATA_NAME = 'Building'
#################################

working_path = os.path.dirname(os.path.abspath(__file__))  # E:\学习\研究生-北京工业大学信息学部软件学院\高分遥感图像识别\Building
# checkpoint_path = 'E:/学习/研究生-北京工业大学信息学部软件学院/高分遥感图像识别/Building-max/'
args = {
    'gpu': True,
    'batch_size': 1,
    'net_name': 'SDSCUNet',
    'load_path': os.path.join(working_path, 'checkpoints', 'Building', 'SDSC-UNet_169e_OA94.17_F75.44_IoU61.78.pth')
}


"""
这是一个Python函数，名为soft_argmax，它包含一个输入参数seg_map，这个参数应该是一个4维的张量（tensor）。
该函数的作用是对输入张量进行softmax操作，并输出softmax结果的4D张量。
具体而言，该函数首先检查输入张量的维度是否为4，然后定义一个参数alpha。
接着，该函数调用PyTorch的softmax函数F.softmax()对输入张量进行softmax操作，其中将输入张量乘以参数alpha。
最后，该函数返回softmax结果的4D张量。
在这个函数中，参数alpha是一个超参数，用于控制softmax操作中最大值的大小，从而使得softmax结果更加接近0或1。
通常，参数alpha的值越大，最大值的影响就越大，softmax结果越接近0或1。
"""


def soft_argmax(seg_map):
    assert seg_map.dim() == 4
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0 
    b,c,h,w, = seg_map.shape
    print(seg_map.shape)
    soft_max = F.softmax(seg_map*alpha, dim=1)
    return soft_max


"""
这是一个Python函数main()，其中定义了一个名为net的神经网络模型，加载了已经训练好的模型权重，并设置模型为评估模式。
该函数还定义了一个保存模型预测结果的路径pred_path，以及一个用于保存评估结果的文本文件info_txt_path。
然后，该函数使用函数RS.get_file_name()获取测试数据集中所有图像的名称列表，然后创建一个数据集对象test_set和数据加载器test_loader，
并将它们用于对神经网络模型进行评估。具体来说，函数predict()被调用，用于对数据加载器中的每个批次图像进行预测，并将预测结果保存到磁盘中。
需要注意的是，这段代码的args是一个字典类型的变量，它包含了函数的参数。
而Net是一个自定义的神经网络模型，它的输入通道数为3，输出类别数为1。
此外，在这段代码中，cuda()是一个PyTorch函数，它将模型的参数和缓存移动到GPU上进行计算。
eval()是一个PyTorch函数，它将模型的运行模式设置为评估模式。DataLoader是PyTorch提供的一个用于读取数据的工具类。
"""


def main():
    net = Net(pretrained=False, n_classes=1)
    # 它允许将保存的状态字典加载到模块中
    # 如果strict=True，则状态字典中的键必须与模块中的参数和缓冲区的键完全匹配。如果有任何键丢失或多余，将引发错误。
    # 如果strict=False，则该方法仅加载与之匹配的键，忽略任何多余的键或缺少的键。
    net.load_state_dict(torch.load(args['load_path']), strict=False)  # strict = False
    # net.upsample2 = Identity()
    net = net.cuda()
    net.eval()  # 它将模型的运行模式设置为评估模式
    print('Model loaded.')
    pred_path = os.path.join(RS.root, 'pred', args['net_name'])
    if not os.path.exists(pred_path): os.makedirs(pred_path)
    info_txt_path = os.path.join(pred_path, 'info.txt')
    f = open(info_txt_path, 'w+')

    # pred_name_list = RS.get_file_name('按行政区域切割/马池口地区办事处/20/crops')
    # test_set = RS.RS('按行政区域切割/马池口地区办事处/20/crops')
    pred_name_list = RS.get_file_name('512_val')
    test_set = RS.RS('512_val')
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], num_workers=4, shuffle=False)
    predict(net, test_loader, pred_path, pred_name_list, f)


"""
这个函数接受一个神经网络模型（net）、一个数据加载器（pred_loader）和其他输入参数。
它使用net模型在pred_loader中的数据上生成预测，评估预测的准确性，并将预测保存为图像。
该函数的输出是所有预测的平均F1分数。
下面是该函数更详细的分解：
output_info = f_out不为None：如果提供了f_out参数（即f_out不为None），
则output_info设置为True。否则，output_info设置为False。
"""


def predict(net, pred_loader, pred_path, pred_name_list, f_out=None):
    output_info = f_out is not None

    """acc_meter、precision_meter、recall_meter、F1_meter和IoU_meter都是自定义AverageMeter类的实例。
    这些对象用于跟踪每个预测的准确性、精确度、召回率、F1分数和IoU"""
    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    BER_meter = AverageMeter()

    # 记录precision、和recall的列表，为了计算AP50,其他参数是为了可视化图表
    acc_list = []
    precision_list = []
    recall_list = []
    F1_list = []
    IoU_list = []
    BER_list = []

    """total_iter是数据的总迭代次数，即数据加载器中批次的数量"""
    total_iter = len(pred_loader)
    """num_files是pred_name_list中文件的数量。"""
    num_files = len(pred_name_list)
    # crop_nums = int(total_iter/num_files)  原来的
    """
    crop_nums是每个图像的裁剪数量。
    它被计算为math.ceil(total_iter / num_files)，这确保了即使裁剪数量不能被文件数量整除，所有裁剪也会用于评估。
    """
    crop_nums = math.ceil(total_iter / num_files)

    """
    该函数迭代数据加载器（pred_loader），对于每一批图像：
    图像被发送到GPU，并通过神经网络（net）传递以生成预测（outputs）。
    然后将预测通过sigmoid函数传递，以将输出压缩到范围[0,1]。
    """

    for vi, data in enumerate(pred_loader):
        imgs, labels = data
        # imgs = data
        imgs = imgs.cuda().float()
        with torch.no_grad(): 
            # outputs, _ = net(imgs)
            outputs = net(imgs)
            # print("outputs",imgs.shape)
            outputs = F.sigmoid(outputs)
            print(outputs.shape)
        outputs = outputs.detach().cpu().numpy()

        """
        调用准确度函数计算每个预测的准确性、精确度、召回率、F1分数和IoU。
        这些值被添加到相应的AverageMeter对象中。
        """

        for i in range(args['batch_size']):
            idx = vi*args['batch_size'] + i
            file_idx = int(idx/crop_nums)
            crop_idx = idx % crop_nums
            if (idx>=total_iter): break
            pred = outputs[i]
            label = labels[i].detach().cpu().numpy()

            acc, precision, recall, F1, IoU, BER = accuracy(pred, label)

            acc_meter.update(acc)
            precision_meter.update(precision)
            recall_meter.update(recall)
            F1_meter.update(F1)
            IoU_meter.update(IoU)
            BER_meter.update(BER)

            # 将precision、recall记录成列表
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            F1_list.append(F1)
            IoU_list.append(IoU)
            BER_list.append(BER)

            """
            使用RS.Index2Color函数将预测的标签转换为彩色图像，
            并使用skimage.io模块中的imsave函数将其保存到磁盘上。
            保存的图像的文件名是使用pred_name_list和crop_idx构造的。
            """
            print(pred.shape)
            pred_color = RS.Index2Color(pred.squeeze())
            if crop_nums > 1: pred_name = os.path.join(pred_path, pred_name_list[file_idx]+'_%d.png'%crop_idx)
            else: pred_name = os.path.join(pred_path, pred_name_list[file_idx]+'.png')
            io.imsave(pred_name, pred_color)

            print('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f' % (idx, total_iter, acc*100, precision*100, recall*100, F1*100, IoU*100))
            if output_info:
                f_out.write('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f\n' % (idx, total_iter, acc*100, precision*100, recall*100, F1*100, IoU*100))

    """
    一旦所有预测都被评估，就计算平均准确度、精确度、召回率、F1分数和IoU, AP50，并将其打印到控制台上。
    如果output_info为True，则它们也被写入f_out文件中。
    """
    # 计算AP50
    ap50 = AP50(precision_list, recall_list, IoU_list)

    if output_info:
        f_out.write('precision %s\n' % precision_list)
        f_out.write('recall %s\n' % recall_list)
        f_out.write('acc %s\n' % acc_list)
        f_out.write('F1 %s\n' % F1_list)
        f_out.write('IoU %s\n' % IoU_list)
        f_out.write('BER %s\n' % BER_list)

    # 可视化
    # 绘制曲线
    x_1 = list(range(1, len(pred_loader) + 1))
    # plot the lines separately with their own settings and label
    plt.plot(x_1, precision_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("precision.png")
    plt.clf()

    x_2 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_2, recall_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("recall.png")
    plt.clf()

    x_3 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_3, acc_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("acc.png")
    plt.clf()

    x_4 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_4, F1_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("F1.png")
    plt.clf()

    x_5 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_5, IoU_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("IoU.png")
    plt.clf()

    x_6 = list(range(1, len(pred_loader) + 1))
    plt.plot(x_6, BER_list, color='green', marker='.', linestyle='solid', linewidth=1, markersize=2)
    plt.savefig("BER.png")
    plt.clf()

    print('avg Acc %.2f, Pre %.2f, AP50 %.2f, Recall %.2f, F1 %.2f, IOU %.2f, BER %.2f\n' % (acc_meter.avg*100, precision_meter.avg*100, ap50*100, recall_meter.avg*100, F1_meter.avg*100, IoU_meter.avg*100, BER_meter.avg*100))

    """
    如果output_info为True，则它们也被写入f_out文件中。
    ACC (Accuracy)：准确率，表示分类模型在所有样本中正确分类的比例。
    recall (召回率)：也称为查全率，表示模型正确识别为正样本的样本数量占所有正样本的数量的比例。
    precision (查准率)：表示模型正确识别为正样本的样本数量占所有识别为正样本的样本数量的比例。
    F1：F1得分，是precision和recall的调和平均值，用于衡量模型在查准率和查全率之间的平衡。
    IoU (Intersection over Union)：也称为Jaccard Index，是分割模型中常用的指标，用于衡量模型预测的区域和真实区域之间的重叠程度。其计算公式为IoU = (预测区域∩真实区域) / (预测区域∪真实区域)
    BER (Balanced Error Rate)：平衡错误率，是衡量二分类模型性能的指标，用于衡量正类和负类分类的平均错误率。
    """
    if output_info:
        f_out.write('Acc %.2f\n' % (acc_meter.avg*100))
        f_out.write('Avg Precision %.2f\n' % (precision_meter.avg*100))
        f_out.write('AP50 %.2f\n' % (ap50 * 100))
        f_out.write('Avg Recall %.2f\n' % (recall_meter.avg*100))
        f_out.write('Avg F1 %.2f\n' % (F1_meter.avg*100))
        f_out.write('mIoU %.2f\n' % (IoU_meter.avg*100))
        f_out.write('mBER %.2f\n' % (BER_meter.avg*100))
    return F1_meter.avg


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    main()
