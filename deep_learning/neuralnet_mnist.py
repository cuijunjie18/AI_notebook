#导入包
import cv2
import numpy as np
import pickle

#导入数据集加载函数
from mnist import load_mnist


# 任务:权重参数评估,即对训练结果进行评估

#展示图像
def show(img,title = 'Test'):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#激活函数sigmoid
def sigmoid(a):
    x = 1/(1 + np.exp(-a))
    return x


#改进的归一化输出层处理
def imporve_softmax(a):
    "适用于对于多输入层需要保持对应输出层概率的绝对性的,即和为1,非相对性"
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a,axis = 1)
    for i in range(a.shape[0]):
        exp_a[i] = exp_a[i]/sum_exp_a[i]
    return exp_a

#归一化输出层处理
def softmax(a):
    "适用于一般的batch批处理,仅需相对关系"
    c = np.max(a)
    exp_a = np.exp(a-c) # 防止溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a

    return y

#加载数据集,获取测试数据(非训练数据)
def get_data():
    "加载数据集,获取测试数据(非训练数据)"
    (x_train,t_train),(x_test,t_test) = \
    load_mnist(normalize = True,flatten = True,one_hot_label = False)
    return x_test,t_test

#初始化神经网络(获取已经训练好的了)
def init_network():
    "初始化神经网络"
    #一种简洁的打开并自动关闭读取文件的语句
    # rb:读取二进制格式
    # pickle是存储数据的库,类似json,但是json更广泛,且人类可读
    with open('MNIST_data/sample_weight.pkl','rb') as f:
        network = pickle.load(f) # network像之前学习的一样,是字典格式的,包含权重及偏置矩阵
    return network

#神经网络前向传播,即计算
def forword(network,x):
    "神经网络前向传播,即计算"
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    # print(W1.shape,b1.shape)
    # print(W2.shape,b2.shape)
    # print(W3.shape,b3.shape)
    a1 = np.dot(x,W1) + b1
    Z1 = sigmoid(a1)
    a2 = np.dot(Z1,W2) + b2
    Z2 = sigmoid(a2)
    a3 = np.dot(Z2,W3) + b3

    y1 = softmax(a3)
    y2 = imporve_softmax(a3)
    return y1,y2