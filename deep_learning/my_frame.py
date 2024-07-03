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
# 梯度下降法相关集合
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








"""
skip()
sigmoid()
relu()
softmax()
improve_softmax()
mean_squared_error()
cross_entropy_error()
"""
