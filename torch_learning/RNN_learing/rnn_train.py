import torch
from torch import nn
from torch import optim
from net_frame import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import glob

# 计算损失
def evaluate(net,data_iter,loss_fn,device):
    net.eval() # 设置评估模式
    loss = 0
    state = None
    for x,y in data_iter:
        if state == None:
            state = net.begin_state(device,x.shape[0]) # 初始化隐状态
        x = x.to(device)
        y_pred,state = net(x,state)
        y = F.one_hot(y.T,y_pred.shape[-1]) 
        y = y.reshape(y_pred.shape).to(device) # 转化为与输出一致
        y = y.to(torch.float32) # 转化dtype
        loss += loss_fn(y_pred,y).item()
    return loss

# 训练
def train(net,data_iter,learning_rate = 1,weight_decay = 0,epochs = 1000,device = 'cpu'):
    optimizer = optim.SGD(net.parameters(),lr=learning_rate,weight_decay=weight_decay) # 设置优化器
    loss_fn = nn.CrossEntropyLoss() # 设置损失函数
    net.train() # 设置训练模式
    loss_plt = []
    # loop = tqdm(data_iter,desc = 'Train')
    for _ in range(epochs):
        print(f"{_}th epoch begin!")
        state = None # 隐藏状态，初始化为None
        for x,y in data_iter:
            # 初始化隐状态
            if state is None:
                state = net.begin_state(device,x.shape[0])
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # state对于nn.GRU是个张量
                    state.detach_()
                else:
                    # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                    for s in state:
                        s.detach_()

            # 输入数据移动
            # print(f"x shape : {x.shape}")
            x = x.to(device)

            # 前向传播
            y_pred,state = net(x,state)
            # print(f"y_pred shape : {y_pred.shape}")

            # labels数据移动与格式转化
            y = F.one_hot(y.T,y_pred.shape[-1]) 
            y = y.reshape(y_pred.shape).to(device) # 转化为与输出一致
            y = y.to(torch.float32) # 转化dtype
            # print(f"y shape : {y.shape}")

            # 反向传播
            loss = loss_fn(y_pred,y)
            optimizer.zero_grad() # 清空梯度
            loss.backward()

            # 梯度裁剪
            grad_clipping(net,1)

            # 更新权重
            optimizer.step()
        loss_plt.append(evaluate(net,data_iter,loss_fn,device))

    # print(len(loss_plt))
    # print(type(loss_plt[0]))
    # print(loss_plt[0])

    # 可视化loss
    plt.plot(np.arange(len(loss_plt)),loss_plt)
    plt.show()

# 获取数据迭代器
dir = 'poems_data/'
batch_size, num_steps = 2, 5
train_iter,vocab = load_data(batch_size,num_steps,dir)

# 验证每个样本的shape = (批大小，时间步)
# for x,y in train_iter:
#     print(x.shape,y.shape)

# 定义一个rnn-layer
num_hiddens = 20 # 隐藏层单元数量
rnn_layer = nn.RNN(len(vocab),num_hiddens)
state = torch.zeros((1,batch_size,num_hiddens)) # (隐藏层数量，批大小，隐藏单元数)

# 定义一整个RNN网络架构
device = try_gpu()
net = RNNModel(rnn_layer,vocab_size = len(vocab))
net = net.to(device)

# 训练(采用默认参数)
train(net,train_iter,device = device)

# 保存权重
models_save_path = "models/rnn_poems.pt"
torch.save(net,models_save_path)