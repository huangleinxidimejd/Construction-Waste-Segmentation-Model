import torch.nn as nn
from torchvision import models
import torch


# 打印resnet101网络结构
resnet_detail = models.resnet101()
print(resnet_detail.__dict__)


class FCN_res101(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(FCN_res101, self).__init__()
        # 载入预训练的ResNet-101网络
        resnet = models.resnet101(pretrained=False)
        # 新建卷积层，用于将输入图像通道数对齐到ResNet的预训练模型中, 前面改通道数，后面参数主要是使图像大小减半
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 定义输出特征图经过的卷积层
        self.head = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        # 定义分类器
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)


    def forward(self, x):
        # 前向传播过程
        x = self.layer0(x)  # size:1/2 strid=2  256
        x = self.maxpool(x)  # size:1/4 strid=2  128
        x = self.layer1(x)  # size:1/4 downSample中strid=1  128

        x = self.layer2(x)  # size:1/8 strid=2  64
        x = self.layer3(x)  # size:1/16 strid=2 32
        x = self.layer4(x)  # 1/32  16


        x = self.head(x)
        x = self.classifier(x)
        # 返回分类结果和上采样后的特征图
        return x


restnet = FCN_res101()
input = torch.ones(3, 3, 512, 512)
output = restnet(input)
print(output.shape)

print(restnet.__dict__)
