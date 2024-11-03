# 导入包
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

# 生成简单线性数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 生成参考值及简单数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 随机获取batch_size数据
def get_batch(batch_size,features,labels):
    num = len(features)
    index = torch.randperm(num)
    return features[index[:batch_size]],labels[index[:batch_size]]

# 定义前向传播
def forward(x):
    return x.mm(train_w) + train_b

# 定义损失函数
def Square_loss(x,labels):
    y = forward(x)
    loss = (y - labels).pow(2).sum()
    return loss

# 初始化权重参数
train_w = torch.rand((2,1),requires_grad = True)
train_b = torch.zeros(1,requires_grad = True)

# 设置超参数
train_times = 1000
batch_size = 10
learning_rate = 0.03

# 绘图
loss_save = []

# 模型训练
for i in range(train_times):
    features_batch,labels_batch = get_batch(batch_size,features,labels)
    loss = Square_loss(features_batch,labels_batch)
    loss_save.append(loss.detach().numpy())
    if (i % 100 == 0 ):
        print(f"loss = {loss} in epoch {i/100}\n")
    loss.backward()

    with torch.no_grad():
        train_w -= learning_rate * train_w.grad
        train_b -= learning_rate * train_b.grad
        train_w.grad.zero_()
        train_b.grad.zero_()

# 绘制训练图
plt.plot(range(train_times),loss_save)
plt.show()
