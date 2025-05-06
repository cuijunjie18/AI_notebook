import torch
from torch import nn

def vgg_block(num_convs,in_channels,out_channels) -> nn.Sequential:
    """单个vgg块的生成"""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,
                                kernel_size = 3,padding = 1)) # 保持宽高
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size = 2,stride = 2)) # 分辨率减半
    return nn.Sequential(*layers)

def vgg(conv_arch,in_H = 32,in_W = 32,class_nums = 10):
    """我的vgg网络，VGG-11加上一个softmax，处理分类问题"""
    conv_blks = []
    in_channels = 3     #初始为rgb三通道
    for num_convs,out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels = out_channels
        in_H = int(in_H / 2)
        in_W = int(in_W / 2)
    return nn.Sequential(
        *conv_blks,nn.Flatten(),
        nn.Linear(out_channels * in_H * in_W,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,class_nums)
                         ) # softmax可以不加