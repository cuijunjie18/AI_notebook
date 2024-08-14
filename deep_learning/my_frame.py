import numpy as np

# 激活函数集合
def skip(x:np.ndarray) -> np.ndarray:
    """阶跃函数

    input:
    x:输入矩阵

    return:
    y:输出矩阵
    """
    y = x > 0 # 生成bool数组
    return y.astype(np.int64)

def sigmoid(x:np.ndarray) -> np.ndarray:
    """sigmoid函数

    input:
    x:输入矩阵

    return:
    y:输出矩阵
    """
    return 1/(1 + np.exp(-x))

def relu(x:np.ndarray) -> np.ndarray:
    """relu函数

    input:
    x:输入矩阵

    return:
    y:输出矩阵
    """
    return np.maximum(0,x)

def softmax_remove(a:np.ndarray) -> np.ndarray:
    """
    已经被舍弃!!!
    输出层的归一函数--相对概率

    input:
    a:初始输出层

    return:
    y:激活后的输出层
    """
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a 

def softmax(a:np.ndarray) -> np.ndarray:
    """
    输出层的归一化函数--我的保持列矩阵的绝对概率
    (利用官方的进行修改的,循环->矩阵运算,提高速度)
    input:
    a:初始输出层

    return:
    y:激活后的输出层
    """
    if a.ndim == 1:
        a = a.reshape(1,-1)
    a = a.T
    a = a - np.max(a,axis = 0)
    y = np.exp(a)/np.sum(np.exp(a),axis = 0) # 注意要有多维才能axis,所有之前对ndim == 1的进行reshape
    return y.T # 最后转置回去



# 损失函数集合
def mean_squared_error(y,t):
    """
    均方误差计算

    input:
    y:输出层矩阵
    t:输入的正确标签矩阵

    return:
    loss:损失函数值
    """
    return 0/5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    """
    交叉熵误差计算--支持batch批处理

    input:
    y:输出层的矩阵(可以多维)
    t:输入的正确标签矩阵(支持one_hot与非one_hot格式)

    return:
    loss:损失函数的平均值
    """
    if y.ndim == 1:
        y = y.reshape(1,-1)
        t = t.reshape(1,-1)

    batch_size = y.shape[0]

    # 处理非one_hot格式的t_label
    if t.size != y.size: # 利用元素个数是否相等判断one_hot
        return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7))/batch_size
    
    # 处理one_hot格式的t_label
    else:
        return -np.sum(t*np.log(y + 1e-7))/batch_size

def function_four(x):
      return x[0]**3 + 2*x[1] + 3*x[2] + 4*x[3]
# 数值微分梯度下降法相关集合
def numerical_gradient_1d(f,x):
    """
    计算一行矩阵的梯度,即一维数组的梯度,数学上来讲为计算多元函数梯度

    input:
    f:给定函数
    x:输入矩阵

    return:
    grad:梯度矩阵
    """
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        tmp_val = x[i]

        # 计算f(x+h)
        x[i] = tmp_val + h
        fxh1 = f(x)

        # 计算f(x-h)
        x[i] = tmp_val - h
        fxh2 = f(x)

        # 还原值
        x[i] = tmp_val
        grad[i] = (fxh1 - fxh2)/(2*h)

    return grad

def numerical_gradient_2d(f,x):
    """
    计算行矩阵的梯度

    input:
    f:给定函数
    x:输入矩阵

    return:
    grad:梯度矩阵
    """
    if x.ndim == 1:
        return numerical_gradient_1d(f,x)
    else:
        grad = np.zeros_like(x)
        for i in range(x.shape[0]):
            grad[i] = numerical_gradient_1d(f,x[i])
        
        return grad

def numerical_gradient(f, x):
    """
    照搬作者的数值微分梯度计算

    input:
    f:输入函数
    x:输入矩阵
    
    return:
    grad:梯度
    """
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad



# 误差反向传播相关节点层

class MulLayer:
    """简单乘法层"""
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y

        return out

    def backward(self,dout):
        dx = self.y*dout # 翻转信号
        dy = self.x*dout

        return dx,dy
    
class AddLayer:
    """加法层的实现"""
    def __init__(self):
        pass

    def forward(self,x,y):
        out = x + y
        return out
    
    def backward(self,dout):
        dx = dout*1
        dy = dout*1

        return dx,dy
    
class Relu:
    """Relu函数的计算层"""

    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0 # 这个用法要学会；将True位置数置零

        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

class Sigmoid:
    """sigmoid计算节点"""
    def __init__(self):
        self.out = None
    
    def forward(self,x):
        out = 1/(1 + np.exp(-x))
        self.out = out

        return out
    
    def backward(self,dout):
        dx = dout*self.out*(1.0 - self.out)

        return dx

# 兼容批处理的Affine层的实现
class Affine:
    """Affine仿射层"""

    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b

        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.W.T) # 注意这里的矩阵乘法，W进行了转置
        self.dW = np.dot(self.x.T,dout) # 同样要转置，对应矩阵求导公式
        self.db = np.sum(dout,axis = 0)

        return dx

# 构建Softmax-with-Loss层
class SoftmaxWithLoss:
    """Softmax正规化及交叉熵损失层"""

    def __init__(self):
        self.loss = None # 损失
        self.y = None # softmax的输出
        self.t = None # 监督数据(要求one-hot格式的)

    def forward(self,x,t):
        """x为未激活的输出层,t为监督数据"""
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)

        return self.loss
    
    def backward(self,dout = 1): # 由于是最后一层了，无上游导数，默认参数为1
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size # 计算单个输入数据的导数，应该叫做误差

        return dx

# 定义卷积层的类
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
    
    # 原理基本与Affine层一致，但是其中的矩阵转化有点复杂
    def backward(self, dout):
        """卷积层的反向传播"""
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
# 定义池化层的类
class Pooling:
    """池化层的实现"""
    def __init__(self,pool_h,pool_w,stride = 1,pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        # 传播所需
        self.x = None
        self.arg_max = None

    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1 + (H - self.pool_h)/self.stride)
        out_w = int(1 + (W - self.pool_w)/self.stride)

        # 展开
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)

        # 进行取最大值操作
        arg_max = np.argmax(col, axis=1)
        out = np.max(col,axis = 1)

        # 转换
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        # 保存用于backward
        self.x = x
        self.arg_max = arg_max

        return out
    
    # 反向传播基本原理与relu相似，但是代码难理解
    def backward(self,dout): 
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx



# 优化器合集
class SGD:
    """随机梯度下降算法"""
    def __init__(self,learning_rate = 0.01):
        self.lr = learning_rate

    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]

class Momentum:
    """Momentum算法"""
    def __init__(self,learning_rate = 0.01,momentum = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class AdaGrad:
    """学习率衰减算法"""
    def __init__(self,learning_rate = 0.01):
        self.lr = learning_rate
        self.h = None
    
    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key]*grads[key]
            params[key] -= self.lr*grads[key]/np.sqrt(self.h[key] + 1e-7) # 加上微小值防止除0错误



# 卷积im2col相关
# coding: utf-8
def smooth_curve(x):
    """用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """打乱数据集

    Parameters
    ----------
    x : 训练数据
    t : 监督数据

    Returns
    -------
    x, t : 打乱的训练数据和监督数据
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]