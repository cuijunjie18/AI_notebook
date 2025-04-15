import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import os

voc_dir = 'E:/My_resources/d2l-zh/pytorch/data/VOCdevkit/VOC2012'

#@save
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels


#@save
def my_colormap2label(color_map):
    """构建RGB到分割任务的类别索引"""
    colormap2label = torch.zeros(256**3,dtype = torch.long)
    for i,color in enumerate(color_map):
        colormap2label[color[0]*256*256 + color[1]*256 + color[2]] = i
    return colormap2label

#@save
def my_label_indices(data,colormap2label):
    """将标签中的RGB值映射到对应的类别索引
    输入：
        data：输入图像数据，单张输入
    输出：
        colormap2label[idx]，返回对应位置每个像素的类别
    """
    # print(data.shape)
    data = data.permute(1,2,0).numpy().astype('int32') # 将输入数据转化图片的(H,W,C)格式，因为之前为了适应网络输入转换成其他了
    # print(data.shape)
    idx = data[:,:,0] * 256 * 256 + data[:,:,1] * 256 + data[:,:,2]
    return colormap2label[idx]

#@save
def rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像
    输入：
        feature：特征图像
        label：标签图像
        height、width：裁剪的大小
    """
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label    

#@save
def bilinear_kernel(in_channels, out_channels, kernel_size):
    """用于转置卷积层初始化的函数"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


#@save
class MySegDataset(torch.utils.data.Dataset):
    """一个用于加载"自己的数据集的":自定义数据集"""

    def __init__(self, features, labels,crop_size,color_map):
        """
        crop_size : 将数据集裁剪的大小
        """
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # RGB值标准化
        self.crop_size = crop_size
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = my_colormap2label(color_map)
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, my_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
    
# 定义网络的损失函数
def loss(inputs,targets):
    return F.cross_entropy(inputs,targets,reduction='none').mean(1).mean(1)

def predict(img,net,device):
    """预测函数
    输入：
        img : 待分割的图像
    """
    img = torch.tensor(img.clone().detach(),dtype=torch.float32)
    img = img.unsqueeze(0) # 转化为可输入网络格式
    pred = net(img.to(device)).argmax(dim = 1)
    pred = pred.squeeze(0)
    return pred

def idc2color(img,color_map):
    """类别标签图像反映射回RGB颜色
    输入：
        img : 待恢复颜色的图像(H,W)
    输出：
        img : (H,W,通道)
    """
    temp = img.clone().detach().numpy()
    map_tensor = torch.tensor(color_map,dtype = torch.float32)
    img = map_tensor[temp]
    return img