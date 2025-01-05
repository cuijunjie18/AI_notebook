import torch
import numpy as np
from torch import nn
# from d2l import torch as d2l
import time
import matplotlib.pyplot as plt
import torchvision
from torch.nn import functional as F
from anchor import*

# GPU使用
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# 通过卷积后的图像通道来预测，减少参数量
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)

# 预测边界框
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1) # 4为框的定位坐标(x1,y1,x2,y2)

# # 块的前向传播
# def forward(x, block):
#     return block(x)

# transpose并展开成(批大小，其他)
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1) # torch.permute类似numpy.transpose

# 将展开后的不同预测层连接
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# 高宽减半层，有块架构可知，感受野扩大6倍，1x1 -> 6x6
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

def base_net():
    blk = [] # 基本网络块列表(由多个减半块组成)
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

# 获取模型中的块
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X,blk,size,ratio,cls_predictor, bbox_predictor):
    """块的前向传播
    输入
    X : 图像
    blk : 当前尺度下的块
    size,ratio : 生成锚框要用的参数
    cls_predictor,bbox_predictor : 类别预测层、预测框预测层

    输出
    (CNN特征图Y,当前尺度即Y下的锚框,类别预测,预测框)
    """
    Y = blk(X)
    anchors = multibox_prior(Y,sizes = size,ratios = ratio) # 当前特征图下生成锚框
    cls_preds = cls_predictor(Y) # 获取输出的预测类别的层
    bbox_preds = bbox_predictor(Y) # 获取预测锚框(即定位框)的层
    return (Y,anchors,cls_preds,bbox_preds)

# 定义块的大小及比例数据
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# 目标检测模型
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1) # 将不同尺度上的锚框拼接
        cls_preds = concat_preds(cls_preds) # 将不同尺度上的预测类别拼接
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds) # 将不同尺度上的预测框进行拼接
        return anchors, cls_preds, bbox_preds
    

