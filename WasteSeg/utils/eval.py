import torch
import torch.nn.functional as F
import torch.nn as nn
from dice_loss import dice_coeff
"""
eval_net(net, dataset, gpu=True)：用于评估网络在数据集上的性能。计算每个样本的交叉熵损失，最后返回平均损失。
其中，输入的dataset是一个迭代器，每次迭代返回一个形如(images, label)的元组，其中image是输入图像，label是对应的真实标签。
eval_net_BCE(net, dataset, gpu=True)：与eval_net类似，但使用二元交叉熵损失。
其中，输入的dataset同样是一个迭代器，每次迭代返回一个形如(images, label)的元组，其中image是输入图像，label是对应的真实标签。
这些函数接收一个net对象，即模型实例，可以在训练后用于评估模型在新数据上的性能。如果gpu=True，则使用GPU计算。"""


def eval_net(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    n=len(dataset)
    for i, b in enumerate(dataset):
        # img = b[0]
        # true_mask = b[1]
        # img = torch.from_numpy(img).unsqueeze(0)
        # true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        #
        # if gpu:
        #     img = img.cuda()
        #     true_mask = true_mask.cuda()

        img = torch.from_numpy(b[0]).unsqueeze(0).float()
        label = torch.from_numpy(b[1]).unsqueeze(0).long()
        if gpu:
            img = img.cuda()
            label = label.cuda()

        pred = net(img)
        loss = nn.CrossEntropyLoss()
        loss = loss(pred, label)

        tot += loss.item()
    return tot / n

def eval_net_BCE(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    n=len(dataset)
    for i, b in enumerate(dataset):
        # img = b[0]
        # true_mask = b[1]
        # img = torch.from_numpy(img).unsqueeze(0)
        # true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        #
        # if gpu:
        #     img = img.cuda()
        #     true_mask = true_mask.cuda()

        img = torch.from_numpy(b[0]).unsqueeze(0).float()
        label = torch.from_numpy(b[1]).unsqueeze(0).float()
        if gpu:
            img = img.cuda()
            label = label.cuda()

        pred = net(img)
        pred_flat = pred.view(-1)
        labels_flat = label.view(-1)
        loss = nn.BCEWithLogitsLoss()
        loss = loss(pred_flat, labels_flat)

        tot += loss.item()
    return tot / n
