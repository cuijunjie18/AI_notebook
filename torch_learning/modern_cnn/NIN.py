import torch
from torch import nn

def nin_block(in_channels,out_channels,kernel_size,strides,padding):
    """单个NIN块的生成"""
    blk = []
    blk.append(nn.Conv2d(in_channels,out_channels,kernel_size,strides,padding))
    blk.append(nn.ReLU())
    blk.append(nn.Conv2d(out_channels,out_channels,kernel_size = 1))
    blk.append(nn.ReLU())
    blk.append(nn.Conv2d(out_channels,out_channels,kernel_size = 1))
    blk.append(nn.ReLU())
    return nn.Sequential(*blk)

def nin(in_channels,class_nums):
    """NIN网络架构"""
    net = nn.Sequential(
    nin_block(in_channels, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, class_nums, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)), # 全局池化，将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())
    return net
