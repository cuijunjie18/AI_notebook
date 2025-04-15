import cv2
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from my_frame import*
# from d2l import torch as d2l # 非必要不要

# 导入预训练好的卷积网络模型
pretrained_net = torchvision.models.shufflenet_v2_x1_0(pretrained = True)

dataset_dir = 'E:/My_note_book/torch_learning/semantic_segmentation/images'

# 生成训练样本
img = cv2.imread(os.path.join(dataset_dir,'4052_o.jpg'))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # 转RGB
feature_single = torch.tensor(img)
# train_data = train_data.permute(2,0,1).unsqueeze(0) # 3维转4维度(包含批尺度，此处批大小为1)，可输入网络的样本
feature_single = feature_single.permute(2,0,1) # 先不转4维
features = [feature_single] # 数据压缩到列表，符合后续数据集生成
print(feature_single.shape) # torch.Size([1, 3, 319, 500])

# 加载上面样本的标签
img = cv2.imread(os.path.join(dataset_dir,'4052_l.png'))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # 转RGB
label_single = torch.tensor(img)
label_single = label_single.permute(2,0,1)
labels = [label_single]
print(label_single.shape) # torch.Size([3, 319, 500])

# 要分割任务分割颜色
"""注意是RGB格式!!!!"""
color_map = [[0,0,0],[128,0,0]] # 简单测试，仅有两类
classes = ['background','plane']


# 检测飞机对应分割区域是否被成功打上对应的类别索引
# y = my_label_indices(labels[0],my_colormap2label(color_map))
# print(y[154:175,161:219])


# 生成数据集迭代器
crop_size = (288, 480)
train_data = MySegDataset(features, labels, crop_size, color_map) # 测试，预期结果：1个数据

batch_size = 1 # 因为数据集中只有一个数据，batch_size只能取1
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=0) # num_workers为读取数据集的进程数


# 网络连接
net = nn.Sequential(*list(pretrained_net.children())[:-1]) # 去掉shufflenet_v2_x1_0最后一层的全连接层
num_classes = 2 # 待分割的有两类，对应len(colormap)
net.add_module('final_conv',nn.Conv2d(1024,num_classes,kernel_size = 1)) # 1x1卷积另类全连接
net.add_module('transpose_conv',nn.ConvTranspose2d(num_classes,num_classes,
                                   kernel_size = 64,padding = 16,stride = 32)) # 转置卷积层恢复图像原大小

# 测试网络卷积部分的前向传播
# X = torch.rand(1,3,288,480)
# print(net(X).shape)

