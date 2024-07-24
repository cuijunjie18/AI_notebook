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