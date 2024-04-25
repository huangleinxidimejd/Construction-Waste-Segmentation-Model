import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch
from modelss.ResNet import FCN_res101 as FCN

BN_MOMENTUM = 0.01

"""DAMM与ASPP串联"""


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PositionalAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionalAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Calculate queries, keys, and values
        """
        .view(batch_size, -1, height * width)：这将重塑上一步的输出张量。
        生成的张量具有(batch_size, channels, height * width)
        .permute(0, 2, 1)交换第二维和第三维，产生形状张量。(batch_size, height * width, channels)
        """
        queries = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        keys = self.key_conv(x).view(batch_size, -1, height * width)
        values = self.value_conv(x).view(batch_size, -1, height * width)

        # Calculate positional attention map
        attention_map = torch.bmm(queries, keys)
        attention_map = self.softmax(attention_map)

        # Apply attention map to values
        attended_values = torch.bmm(values, attention_map.permute(0, 2, 1))
        attended_values = attended_values.view(batch_size, -1, height, width)

        # result = self.alpha*attended_values + x
        result = attended_values + x
        return result


class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        # self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        feat_a = x.view(batch_size, -1, height * width)  # (b, c, N)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)  # (b, N, c)

        # attention 只是表示两个特征之间的相似度或关联性
        attention = torch.bmm(feat_a, feat_a_transpose)  # (b, c, c)

        # 这一步操作可能是为了在注意力矩阵中减小最大值的影响，以增加对其他相似度的关注。
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention

        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        # out = self.beta * feat_e + x
        out = feat_e + x

        return out


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(6, 12, 18)):
        super(ASPPModule, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))

        self.branch5_conv = nn.Conv2d(features, inner_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch5_bn = nn.BatchNorm2d(inner_features, momentum=0.95)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_features, momentum=0.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        """
        此行沿输入张量的空间维度（高度和宽度）执行全局平均池化。
        它通过对空间维度求平均值来计算每个通道的平均值，
        从而得到一个形状为.x(batch_size, num_channels, 1, 1)
        """
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)

        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)

        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([feat1, feat2, feat3, feat4, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DAMM_DeepLabv3_plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(DAMM_DeepLabv3_plus, self).__init__()

        self.FCN = FCN(in_channels, num_classes)

        self.pam = PositionalAttentionModule(2048)
        self.cam = ChannelAttentionModule()
        self.head = ASPPModule(2048, 64, 512)

        # self.head1 = BAM(512)
        self.low = nn.Sequential(conv1x1(256, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.low1 = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.fuse = nn.Sequential(conv3x3(64 + num_classes, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.fuse1 = nn.Sequential(conv3x3(128, 64), nn.BatchNorm2d(64), nn.ReLU())

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.classifier_aux = nn.Sequential(conv1x1(512, 64), nn.BatchNorm2d(64), nn.ReLU(), conv1x1(64, num_classes))

    def forward(self, x):
        x_size = x.size()

        # 改进后
        x0 = self.FCN.layer0(x)  # 1/2
        x = self.FCN.maxpool(x0)  # 1/4
        x1 = self.FCN.layer1(x)  # 1/4
        x = self.FCN.layer2(x1)  # 1/8
        x = self.FCN.layer3(x)  # 1/16
        x = self.FCN.layer4(x)  # 1/16

        # 进入DAMM
        x_pam = self.pam(x)
        x_cam = self.cam(x)
        x_DAMM = x_pam + x_cam

        # 进入ASPP
        x = self.head(x_DAMM)

        # Encoder层输出
        aux = self.classifier_aux(x)

        x1 = self.low(x1)
        x = torch.cat((F.interpolate(aux, x1.size()[2:], mode='bilinear'), x1), 1)
        fuse = self.fuse(x)
        # out = self.classifier(fuse)

        x0 = self.low1(x0)
        x = torch.cat((F.interpolate(fuse, x0.size()[2:], mode='bilinear'), x0), 1)
        fuse = self.fuse1(x)
        out = self.classifier(fuse)

        return F.interpolate(out, x_size[2:], mode='bilinear'), F.interpolate(aux, x_size[2:], mode='bilinear')
