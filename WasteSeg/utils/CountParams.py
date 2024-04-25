import time
import torch
import numpy as np
from thop import profile, clever_format

# 导入您的神经网络模型
# from WasteSeg.modelss.Unet import UNet as Net
# from WasteSeg.modelss.SegNet import SegNet as Net
# from WasteSeg.modelss.PSPNet import Pspnet as Net
# from WasteSeg.modelss.DAMM_DeepLabv3_plus import DAMM_DeepLabv3_plus as Net
# from WasteSeg.modelss.deeplabv3_plus import DeepLab as Net
# from WasteSeg.modelss.DSATNet import Model as Net
# from WasteSeg.modelss.convlsrnet import Model as Net
# from WasteSeg.modelss.convnext import ConvNeXt as Net
from WasteSeg.modelss.SDSCUNet import create_shunted_unet_model as Net
# from WasteSeg.modelss.MFNet.MFNet import MFNet as Net
# from WasteSeg.modelss.UANet.UANet import UANet_Res50 as Net
# 创建神经网络模型
model = Net(pretrained=False, n_classes=4)  # 请替换为您的模型的实例化方式
device = torch.device("cuda")
model.to(device)

# 定义输入数据（示例：1个通道，高度为256，宽度为256的图像）
batch_size = 1
channels = 3
height = 512
width = 512
input_data = torch.randn(batch_size, channels, height, width).to(device)
print(input_data)

# 生成一个随机分割标签图像，假设图像尺寸为256x256
num_classes = 4 # 假设有2个类别
random_segmentation = np.random.randint(0, num_classes, size=(batch_size, height, width))
random_segmentation = torch.tensor(random_segmentation).to(device)
random_segmentation = random_segmentation.long()
print(random_segmentation)


# 计算FLOPs和参数数量
macs, params = profile(model, inputs=(input_data,))
macs, params = clever_format([macs, params], "%.3f")

# # 训练时间
# def train(model, input_data):
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = torch.nn.CrossEntropyLoss()
#     num_epochs = 1
#
#     start_time = time.time()
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(input_data)
#         loss = criterion(outputs, random_segmentation)
#         loss.backward()
#         optimizer.step()
#
#     training_time = time.time() - start_time
#     return training_time

# # 预测时间
# def inference(model, input_data):
#     model.eval()
#     with torch.no_grad():
#         start_time = time.time()
#         _ = model(input_data)
#         inference_time = time.time() - start_time
#     return inference_time

# 输出结果
print(f"FLOPs: {macs}")
print(f"Parameters: {params}")
# print(f"Training Time (s): {train(model, input_data)}")
# print(f"Inference Time (s): {inference(model, input_data)}")
