import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from my_frame import*

# 搭建网络
pretrained_net = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
net = nn.Sequential(*list(pretrained_net.children())[:-1]) # 去掉最后一层全连接
num_classes = 21
net.add_module('final_conv', nn.Conv2d(1024, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# 加载数据集
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

train_features, train_labels = read_voc_images(voc_dir, True)
crop_size = (320, 480)
voc_train = MySegDataset(train_features,train_labels,crop_size,VOC_COLORMAP)
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=0) # num_workers为读取数据集的进程数

num_epochs, lr, wd, device = 5, 0.001, 1e-3, d2l.try_gpu()    
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

# # 训练
# net = net.to(device)
# use_times = 0
# plt_loss = []
# for epoch in range(num_epochs):
#     start = time.perf_counter() # 相比time.time()更加精确
#     net.train() # 切换回训练模式
#     loss_save = 0
#     for feature,target in train_iter:
#         # 清空梯度
#         trainer.zero_grad()

#         # 切换硬件
#         X = feature.to(device)
#         Y = target.to(device)

#         # 前向传播
#         y_pred = net(X)
#         _loss = loss(y_pred,Y)

#         # 反向传播及更新梯度
#         _loss.sum().backward()
#         trainer.step()

#         # 记录损失值
#         loss_save += _loss.sum().item()
#     loss_save = loss_save / len(train_iter)
#     plt_loss.append(loss_save)
#     end = time.perf_counter()
#     print(f"epoch{epoch} use time:{end-start}!")
#     use_times += end - start

# print(f"Model train use {use_times} seconds")
# plt.plot(np.arange(0,len(plt_loss),1),plt_loss)
# plt.show()

# torch.save(net.state_dict(),'data/shufflenet-v0.params')

img = train_features[0]
print(img.shape)
pred = predict(img,net,device)
print(pred.shape)
img_pred = idc2color(pred,VOC_COLORMAP)
print(img_pred.shape)
img_pred = img_pred.numpy()
img_pred = cv2.cvtColor(img_pred,cv2.COLOR_RGB2BGR)
cv2.imshow('Predict',img_pred)

img = train_features[0].permute(1,2,0).numpy()
label = train_labels[0].permute(1,2,0).numpy()
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
label = cv2.cvtColor(label,cv2.COLOR_RGB2BGR)
cv2.imshow('Origin',img)
cv2.imshow('Label',label)
cv2.waitKey(0)
cv2.destroyAllWindows()
