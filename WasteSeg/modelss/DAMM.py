"""
针对 DeepLabv3+网络在遥感影像上呈现出拟合速度慢，
边缘目标分割不精确，大尺度目标分割类内不一致、存在孔洞等缺陷，
提出在该网络中引入双注意力机制模块 （Dual Attention Mechanism Module,DAMM）

设计并实现了将DAMM结构与ASPP （Atous Spatial Pyramid Pooling）
层串联或并联的2种不同连接方式网络模型 ，串联连接方式中先将特征图送入 DAMM 后，再经过 ASPP结构；
并联连接方式中将双注意力机制层与 ASPP层并行连接，网络并行处理主干网提取特征图，再融合两层处理特征信息。
将改进的 2种方法通过 INRIA Aerial Image 高分辨率遥感影像数据集验证，
结果表明，串联或并联方式 2 种网络都能有效改善Deeplabv3+的不足，并联方式网络性能更好，
"""

import torch
import torch.nn as nn


class PositionalAttention(nn.Module):
    def __init__(self, in_channels):
        super(PositionalAttention, self).__init__()
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
        attended_values = attended_values.view(batch_size, channels, height, width)

        # result = self.alpha * attended_values + x
        result = attended_values + x
        return result

# Example usage
in_channels = 64
x = torch.randn(1, in_channels, 32, 32)
pos_att = PositionalAttention(in_channels)
print(x.shape)
print(pos_att)


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

# Example usage
in_channels = 64
x = torch.randn(1, in_channels, 32, 32)
ca = ChannelAttentionModule()
print(x.shape)
output = ca(x)
