import os
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch import optim
import torch.autograd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from skimage import io
from Building.datasets import Building_ch as RS
# from modelss.RDSA_DeepLabv3_plus import RDSA_DeepLabv3_plus as Net
# from modelss.Feature_fusion_DeeplabV3_plus import FCN_ASPP as Net
# from modelss.DSAC_DeepLabv3_plus import FCN_ASPP as Net
# from modelss.DAMM_DeepLabv3_plus_162 import DAMM_DeepLabv3_plus_162 as Net
# from modelss.DAMM_tandem_DeepLabv3_plus_171 import DAMM_DeepLabv3_plus_162 as Net
# from modelss.DSAC_DeepLabv3_plus import DSAC_DeepLabv3_plus as Net
# from modelss.traditional_DeepLabv3_plus import traditional_DeepLabv3_plus as Net
# from modelss.RDSA_DeepLabv3_plus import RDSA_DeepLabv3_plus as Net
# from modelss.Unet import UNet as Net
# from modelss.SegNet import SegNet as Net
# from modelss.PSPNet import Pspnet as Net
# from modelss.deeplabv3_plus import DeepLab as Net
# from modelss.DSATNet import Model as Net
# from modelss.convlsrnet import Model as Net
from modelss.SDSCUNet import create_shunted_unet_model as Net
# from torch.nn.modules.loss import CrossEntropyLoss as CEloss
from utils.utils import binary_accuracy as accuracy
from utils.utils import intersectionAndUnion, AverageMeter
# from modelss.DiResSeg import DiResSeg as Net

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

NET_NAME = 'SDSC-UNet'
DATA_NAME = 'Building'
working_path = os.path.abspath('.')

'''
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='FCN.layer0.0.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[wname] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='FCN.layer0.0.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
'''

args = {
    'train_batch_size': 4,
    'val_batch_size': 4,
    'train_crop_size': 512,
    'val_crop_size': 512,
    'lr': 0.001,  # 训练期间使用的学习率。它决定了模型从数据中学习的速度
    'epochs': 200,
    'gpu': True,
    'weight_decay': 5e-4,  # 一种正则化参数，有助于防止过拟合。它控制应用于模型权重的L2正则化的量
    'momentum': 0.9,  # 一种超参数，它通过将先前的权重更新的一部分添加到当前更新中，帮助模型更快地收敛
    'print_freq': 100,  # 训练期间更新打印的频率
    'predict_step': 5,  # 训练期间进行预测的频率。在这种情况下，每5个epochs进行一次预测。
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),  # 保存模型预测的目录。
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),  # 保存模型检查点的目录
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME),  # 保存模型日志的目录
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'xx.pth')  # 如适用，加载先前训练模型的路径。
}

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
writer = SummaryWriter(args['log_dir'])


def main():
    # 3个输入通道和1个输出类别（二分类）
    net = Net(pretrained=False, n_classes=1).cuda()
    '''
            net.load_state_dict(torch.load(args['load_path']), strict=False)
            for param in net.parameters():
                param.requires_grad = False'''

    train_set = RS.RS('512_train', random_flip=True)
    val_set = RS.RS('512_val')

    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    # weight = torch.tensor(1.0)
    # 定义损失函数为二分类交叉熵损失BCEWithLogitsLoss
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()

    # 定义优化器为 SGD，并设置学习率、权重衰减、动量等参数，同时使用 StepLR 学习率调度器对学习率进行调整。
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

    # optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=args['weight_decay'])

    train(train_loader, net, criterion, optimizer, scheduler, 0, args, val_loader)
    writer.close()
    print('Training finished.')
    # predict(net, args)


def train(train_loader, net, criterion, optimizer, scheduler, curr_epoch, train_args, val_loader):
    # 可视化展示
    Acc = []
    train_F1 = []
    val_F1 = []
    train_loss = []
    val_loss = []
    cu_ep = []

    bestaccT = 0
    bestF = 0
    bestIoU = 0
    bestloss = 1
    begin_time = time.time()
    # 迭代总数，用于调整学习率
    all_iters = float(len(train_loader) * args['epochs'])
    # fgm = FGM(net)
    # criterionVGG = VGGLoss()
    while True:
        # 清除GPU缓存
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        F1_meter = AverageMeter()
        train_main_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_learning_rate(optimizer, running_iter, all_iters, args)
            imgs, labels = data
            if args['gpu']:
                imgs = imgs.cuda().float()
                labels = labels.unsqueeze(1).cuda().float()
            # labels_s = F.interpolate(label, scale_factor=1/8, mode='area')

            """得到输出结果 outputs 和辅助输出结果 aux。
            这里的辅助输出是一种用于辅助训练的技术，通常用于提高网络的性能和鲁棒性。"""
            optimizer.zero_grad()
            # outputs, aux = net(imgs)
            outputs = net(imgs)

            # assert outputs.shape[1] == RS.num_classes + 1

            # loss_main = criterion(outputs, labels)
            # loss_aux = criterion(aux, labels)
            # loss = loss_main + loss_aux * 0.3

            loss = criterion(outputs, labels)

            out_sigmoid = F.sigmoid(outputs)

            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            preds = out_sigmoid.cpu().detach().numpy()
            # _, preds = torch.max(outputs, dim=1)
            # preds = out_sigmoid.cpu().detach().numpy()
            # batch_valid_sum = 0

            F1_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, precision, recall, F1, IoU, BER = accuracy(pred, label)
                if F1 > 0:
                    F1_curr_meter.update(F1)
            if F1_curr_meter.avg is not None:
                F1_meter.update(F1_curr_meter.avg)
            else:
                F1_meter.update(0)
            train_main_loss.update(loss.cpu().detach().numpy())

            # train_aux_loss.update(aux_loss, batch_pixel_sum)

            curr_time = time.time() - start

            if (i + 1) % train_args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f F1 %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_main_loss.val, F1_meter.avg * 100))
                writer.add_scalar('train loss', train_main_loss.val, running_iter)
                rex_loss = train_main_loss.val
                writer.add_scalar('train F1', F1_meter.avg, running_iter)
                # writer.add_scalar('train_aux_loss', train_aux_loss.avg, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        val_F, val_acc, val_IoU, loss_v = validate(val_loader, net, criterion)

        Acc.append(val_acc)
        val_F1.append(val_F)
        val_loss.append(loss_v)
        train_loss.append(train_main_loss.avg)
        train_F1.append(F1_meter.avg)
        cu_ep.append(curr_epoch)

        if val_F > bestF:
            bestF = val_F
            bestloss = loss_v
            bestIoU = val_IoU
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME + '_%de_OA%.2f_F%.2f_IoU%.2f.pth' % (
                curr_epoch, val_acc * 100, val_F * 100, val_IoU * 100)))
        print('Total time: %.1fs Best rec: Val %.2f, Val_loss %.4f BestIOU: %.2f' % (
            time.time() - begin_time, bestF * 100, bestloss, bestIoU * 100))
        curr_epoch += 1

        # scheduler.step()
        # 可视化展示ACC,F1,LOSS图像
        if curr_epoch >= train_args['epochs']:
            print('Acc', Acc)
            print('train_F1', train_F1)
            print('val_F1', val_F1)
            print('train_loss', train_loss)
            print('val_loss', val_loss)

            # with open('experimental_record/DSAC_DeepLabv3_plus/loss_main 0.7_0.3loss_aux.txt', 'a') as file0:
            with open('experimental_record/SDSC-UNet/train_all_eval.txt', 'a') as file0:
                print('Acc:%s\n' % Acc,
                      'train_F1:%s\n' % train_F1,
                      'val_F1:%s\n' % val_F1,
                      'train_loss:%s\n' % train_loss,
                      'val_loss:%s\n' % val_loss, file=file0)
            """
            这段代码使用Matplotlib库创建一个图形并将其保存为PNG文件。
            使用plt.plot()函数创建图形，参数如下：
            x1和y1：分别是要在x轴和y轴上绘制的数据点。
            color='green'：将绘制的线条颜色设置为绿色。
            marker='o'：将每个数据点的标记样式设置为圆圈（'o'）。
            linestyle='dashed'：将绘制的线条样式设置为虚线。
            linewidth=2：将绘制的线条宽度设置为2个点。
            markersize=12：将数据点标记的大小设置为12个点。
            创建图形后，使用plt.savefig()函数将图形保存为名为“acc.png”的PNG文件。
            请注意，在使用Matplotlib之前，必须先使用语句import matplotlib.pyplot as plt导入该库。
            """
            # 绘制5条曲线曲线
            x_all = cu_ep
            # create four lists of (x, y) pairs using zip()
            line1 = list(zip(x_all, val_loss))
            line2 = list(zip(x_all, train_loss))
            line3 = list(zip(x_all, Acc))
            line4 = list(zip(x_all, val_F1))
            line5 = list(zip(x_all, train_F1))
            # plot the lines separately with their own settings and label
            plt.plot([p[0] for p in line1], [p[1] for p in line1], color='green', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='val_loss')
            plt.plot([p[0] for p in line2], [p[1] for p in line2], color='red', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='train_loss')
            plt.plot([p[0] for p in line3], [p[1] for p in line3], color='blue', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='Acc')
            plt.plot([p[0] for p in line4], [p[1] for p in line4], color='black', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='val_F1')
            plt.plot([p[0] for p in line5], [p[1] for p in line5], color='yellow', marker='.', linestyle='solid',
                     linewidth=1, markersize=2, label='train_F1')
            # add a legend to the plot
            plt.legend()
            # plt.show()
            plt.savefig("train_all_eval.png")
            plt.clf()
            return


def validate(val_loader, model_seg, criterion, save_pred=True):
    # the following code is written assuming that batch size is 1
    model_seg.eval()
    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs, labels = data

        if args['gpu']:
            imgs = imgs.cuda().float()
            labels = labels.cuda().float().unsqueeze(1)

        with torch.no_grad():
            # out, aux = model_seg(imgs)
            out = model_seg(imgs)
            loss = criterion(out, labels)
            out_bn = F.sigmoid(out)  # soft_argmax(out)[:,1,:,:]
        val_loss.update(loss.cpu().detach().numpy())

        preds = out_bn.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        for (pred, label) in zip(preds, labels):
            acc, precision, recall, F1, IoU, BER = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)
        if save_pred and vi == 0:
            pred_color = RS.Index2Color(preds[0].squeeze())
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '.png'), pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f, F1: %.2f, Accuracy: %.2f' % (
        curr_time, val_loss.average(), F1_meter.avg * 100, Acc_meter.average() * 100))

    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg


def adjust_learning_rate(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 1.5)
    running_lr = args['lr'] * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    start = time.time()
    main()
