# from modelss.FCN_8s import FCN_res101 as FCN
import torch.nn as nn
import torch
from torch.nn import functional as F

BN_MOMENTUM = 0.01


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]  # 前两维全取，在h和w的维度上以stride的步长取值


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)  # 将tensor中的值都填充为0.5


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


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


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# 具体流程可以参考图1，通道注意力机制
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        return x


class Att(nn.Module):

    def __init__(self, channels, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)
        return x_out1


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(6, 12, 18), no_spatial=True):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   nn.BatchNorm2d(inner_features, momentum=0.95),
                                   nn.ReLU(inplace=True))
        self.no_spatial = no_spatial
        self.nam = Att(out_features)
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))
        self.point_conv = nn.Conv2d(
            in_channels=inner_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 4, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features, momentum=0.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        # feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        # feat2 = self.conv2(x)
        '''
        feat2 = self.conv2(x)
        feat2 = self.nam(feat2)
        feat3 = self.conv3(x)
        feat3 = self.point_conv(feat3)
        feat3 = self.nam(feat3)
        feat4 = self.conv4(x)
        feat4 = self.point_conv(feat4)
        feat4 = self.nam(feat4)
        feat5 = self.conv5(x)
        feat5 = self.point_conv(feat5)
        feat5 = self.nam(feat5)
        feat6 = self.conv1(x)
        out = torch.cat((feat2, feat3, feat4, feat5), 1)  # feat1,
        bottle = self.bottleneck(out)
        bottle = self.nam(bottle)
        '''
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat3 = self.point_conv(feat3)
        feat4 = self.conv4(x)
        feat4 = self.point_conv(feat4)
        feat5 = self.conv5(x)
        feat5 = self.point_conv(feat5)
        feat6 = self.conv1(x)
        out = torch.cat((feat2, feat3, feat4, feat5), 1)  # feat1,
        bottle = self.bottleneck(out)
        bottle = self.nam(bottle)
        return bottle


class FCN_ASPP(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_ASPP, self).__init__()
        self.FCN = FCN(in_channels, num_classes)
        self.head = ASPPModule(2048, 64, 64)
        # self.head1 = BAM(512)
        # self.l_ac = nn.Sequential(ACmix(256, 64), nn.BatchNorm2d(64), nn.ReLU())
        # self.l_ac1 = nn.Sequential(ACmix(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.low = nn.Sequential(conv1x1(256, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.low1 = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.fuse = nn.Sequential(conv3x3(65, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.fuse1 = nn.Sequential(conv3x3(128, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        # self.classifier1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.classifier_aux = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64), nn.ReLU(), conv1x1(64, num_classes))

    def forward(self, x):
        x_size = x.size()

        # 改进后
        x0 = self.FCN.layer0(x)  # 1/2
        x = self.FCN.maxpool(x0)  # 1/4
        x1 = self.FCN.layer1(x)  # 1/4
        x = self.FCN.layer2(x1)  # 1/8
        x = self.FCN.layer3(x)  # 1/16
        x = self.FCN.layer4(x)
        x = self.head(x)
        # x = self.head1(x)
        aux = self.classifier_aux(x)

        x1 = self.low(x1)
        x = torch.cat((F.upsample(aux, x1.size()[2:], mode='bilinear'), x1), 1)
        fuse = self.fuse(x)
        # out = self.classifier(fuse)

        x0 = self.low1(x0)
        x = torch.cat((F.upsample(fuse, x0.size()[2:], mode='bilinear'), x0), 1)
        fuse = self.fuse1(x)
        out = self.classifier(fuse)

        return F.upsample(out, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:], mode='bilinear')
