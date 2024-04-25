import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch
from WasteSeg.modelss.ResNet import FCN_res101 as FCN


BN_MOMENTUM = 0.01


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def unconv(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=2, padding=0, output_padding=0, bias=True)


def unconv1(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=4, padding=0, output_padding=0, bias=True)


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


class FCN_ASPP(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_ASPP, self).__init__()
        self.FCN = FCN(in_channels, num_classes)        
        self.head = ASPPModule(2048, 64, 512)

        #self.head1 = BAM(512)

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
        x = self.FCN.layer4(x)

        # 进入ASPP
        x = self.head(x)

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

        # x1 = self.low(x1)
        # x = torch.cat((F.interpolate(x, x1.size()[2:], mode='bilinear'), x1), 1)
        # fuse = self.fuse(x)
        # out = self.classifier(fuse)
        return F.interpolate(out, x_size[2:], mode='bilinear'), F.interpolate(aux, x_size[2:], mode='bilinear')





        '''
        x0 = self.FCN.layer0(x) #1/2
        x = self.FCN.maxpool(x0) #1/4
        x1 = self.FCN.layer1(x) #1/4
        x = self.FCN.layer2(x1) #1/8
        x = self.FCN.layer3(x) #1/16
        x = self.FCN.layer4(x)
        x = self.head(x)
        aux = self.classifier_aux(x)

        x1 = self.low(x1)
        x = torch.cat((F.upsample(aux, x1.size()[2:], mode='bilinear'), x1), 1)
        fuse = self.fuse(x)
        out = self.classifier(fuse)
        return F.upsample(out, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:], mode='bilinear')
        

        

        


class ADSN_DeepLabv3_plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(ADSN_DeepLabv3_plus, self).__init__()
        self.FCN = FCN(in_channels, num_classes)
        self.head = DN_ASPPModule(2048, 64, 512)
        self.low = nn.Sequential(conv1x1(256, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.low1 = conv1x1(1024, 256)
        self.low2 = conv1x1(512, 256)
        self.con = unconv(256,256)
        self.con1 = unconv(1,256)
        self.con2 = unconv(512,64)
        self.fuse = nn.Sequential(conv3x3(128, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.classifier_aux = nn.Sequential(conv1x1(512, 64), nn.BatchNorm2d(64), nn.ReLU(), conv1x1(64, num_classes))

    def forward(self, x):
        x_size = x.size()

        x0 = self.FCN.layer0(x)  # 1/2
        x5 = self.FCN.maxpool(x0)  # 1/4
        x1 = self.FCN.layer1(x5)  # 1/4
        x2 = self.FCN.layer2(x1)  # 1/8
        x3 = self.FCN.layer3(x2)  # 1/16
        x4 = self.FCN.layer4(x3)  # 1/16
        x = self.head(x4)  #8,512,32,32
        #print('x',x.size())
        aux = self.classifier_aux(x) #8,1,32,32
       
        x3 = self.low1(x3)
        #print('x3_low',x3.size())
        x3 = self.con(x3)
       #x3_up = F.upsample(x3,scale_factor=2, mode='bilinear')
        #print('x3',x3.size())
        x2 = self.low2(x2)
        #print(x2.size())
        #x2_up = F.upsample(x2,scale_factor=2, mode='bilinear')
        xx = x2 + x3
        #print('xx',xx.size())    #8,256,63,63
        #print('aux',aux.size())  #8,1,32,32
        #print('con',self.con1(aux).size())
        #print((F.upsample(aux,scale_factor=2, mode='bilinear').size()))
        aux_up = torch.cat((self.con1(aux), xx), 1)
        x1 = self.low(x1)
        #print(x1.size())  #8,64,125,125
        print(self.con2(aux_up).size())#8，64，125，125
        #aux_up1 = F.upsample(aux_up,scale_factor=2, mode='bilinear')
        #print(aux_up1.size())
        x = torch.cat((self.con2(aux_up), x1), 1)
        fuse = self.fuse(x)
        out = self.classifier(fuse)
        '''
