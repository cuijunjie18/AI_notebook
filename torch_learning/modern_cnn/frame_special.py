import torch
import torchvision
from torch.utils import data
from torch.nn import functional
import pandas as pd
import os
import cv2
from tqdm import tqdm
from torch import nn

def try_gpu(i = 0) -> torch.device:
    """获取设备"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device('cpu')

def get_map() -> dict:
    object_map = {}
    with open("object_map.txt","r",encoding = 'utf-8') as file:
        for line in file:
            name,idx = line.split()
            object_map[name] = int(idx)
    return object_map

def load_data(data_path = 'train') -> tuple:
    """获取原始的img和对应的labels"""
    train_nums = 50000
    train_labels = pd.read_csv('trainLabels.csv')
    mode = torchvision.io.image.ImageReadMode.RGB # 设置模型读入为RGB!!!(使用torchvision读入的是通道优先)
    img_array = []
    labels_array = []

    object_map = get_map()
    loop = tqdm(range(train_nums),desc = 'Load train data')
    for i in loop:
        labels_array.append(object_map[train_labels.iloc[i,1]])
        img_array.append(torchvision.io.read_image(
            os.path.join(data_path,str(i + 1) + '.png'),mode
        ))
    return img_array,labels_array # 返回的是list格式

def make_data_iter(arrays,batch_size = 64) -> data.DataLoader:
    """获得数据迭代器"""
    data_arrays = data.TensorDataset(*arrays) # 构造Dataset
    return data.DataLoader(data_arrays,batch_size,shuffle = True) # 注意第一个参数不要给成了data_arrays

class TrainDataset(data.Dataset):
    """自定义的Dataset类，加载图片数据集"""
    def __init__(self,features,labels):
        self.transform = torchvision.transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        ) # RGB格式的通道归一化参数
        self.obj2idx = get_map()
        self.idx2obj = [key for key in self.obj2idx.keys()]
        self.obj_nums = len(self.idx2obj)
        self.features = [self.normalize_img(feature) 
                         for feature in tqdm(features,total = len(features),desc = 'Load-Data-iter')]
        self.labels = functional.one_hot(torch.tensor(labels),self.obj_nums).to(torch.float32)
        print(f"load {len(self.labels)} images!")

    def normalize_img(self,img):
        """图像归一化"""
        return self.transform(img.float() / 255)
    
    def __getitem__(self, index):
        """定义数据迭代，即下标访问类"""
        feature,label = self.features[index],self.labels[index]
        return feature,label
    
    def __len__(self):
        """定义类的长度"""
        return len(self.features)

def load_test(data_path = 'test'):
    mode = torchvision.io.image.ImageReadMode.RGB # 设置模型读入为RGB!!!
    test_arrays = []
    test_nums = 300000
    loop = tqdm(range(test_nums),desc = 'Load test data')
    for i in loop:
        test_arrays.append(torchvision.io.read_image(
            os.path.join(data_path,str(i + 1) + '.png'),mode
        ))
    return test_arrays

def normalize_img(img):
    transform = torchvision.transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        ) # RGB格式的通道归一化参数
    return transform(img.float() / 255)

class TestDataset(data.Dataset):
    """测试的自定义数据集，无labels"""
    def __init__(self,features):
        self.transform = torchvision.transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        ) # RGB格式的通道归一化参数
        self.features = [self.normalize_img(feature) 
                         for feature in tqdm(features,total = len(features),desc = 'Load-Data-iter')]
        print(f"load {len(self.features)} images!")

    def normalize_img(self,img):
        """图像归一化"""
        return self.transform(img.float() / 255)
    
    def __getitem__(self, index):
        """定义数据迭代，即下标访问类"""
        feature = self.features[index]
        return feature
    
    def __len__(self):
        """定义类的长度"""
        return len(self.features)
    
def train(net : nn.Sequential,train_iter,lr,num_epochs,loss_fn):
    """训练函数"""
    # devices_nums = torch.cuda.device_count()
    # devices = []
    # for i in range(devices_nums):
    #     devices.append(try_gpu(i = i))
    device = try_gpu(i = 0)
    net = net.to(device)
    net.train()
    optimzer = torch.optim.Adam(net.parameters(),lr = lr)
    loss_plt = []
    for epoch in range(num_epochs):
        loss_temp = 0
        total_nums = 0
        loop = tqdm(enumerate(train_iter),total = len(train_iter))
        for batch_idx,batch in loop:
            X,Y = batch
            X = X.to(device)
            Y = Y.to(device)
            total_nums += X.shape[0]

            optimzer.zero_grad()
            y_pred = net(X)
            loss = loss_fn(y_pred,Y)
            loss.sum().backward()
            optimzer.step()

            loss_temp += loss.sum().item()
            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix({"LOSS" : loss_temp / total_nums})
        loss_plt.append(loss_temp / total_nums)
    return loss_plt

def count_accurancy(net,train_iter):
    """训练集上准确率测试"""
    device = try_gpu(i = 0)
    net.eval()
    net.to(device)
    total_nums = 0
    ans = 0
    loop = tqdm(train_iter,total = len(train_iter),desc = "Eval")
    for X,Y in loop:
        total_nums += X.shape[0]
        X = X.to(device)
        Y = Y.to(device).argmax(dim = 1)
        y_pred = net(X).argmax(dim = 1)
        ans += (y_pred == Y).sum().item()
    return ans / total_nums
        
def predict(net,test_iter,idx2obj):
    """测试集上预测推理"""
    device = try_gpu(i = 0)
    net.eval()
    net.to(device)
    loop = tqdm(test_iter,desc = "Predict")
    idx = 0
    with open ("result.txt","w",encoding = 'utf-8') as file:
        for X in loop:
            X = X.to(device)
            pred = net(X).argmax(dim = 1)
            for k in pred:
                idx += 1
                file.write(str(idx) + "," + idx2obj[k.item()] + "\n")
        