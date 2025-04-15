import torch
from net_frame import *
import glob

# 获取设备
device = try_gpu()

# 导入模型
models_save_path = "models/rnn_simple3.pt"
model = torch.load(models_save_path,weights_only = False)

# dir = 'poems_data/'
dir = 'data/'

# 获取词表
my_vocab = my_load_corpus(max_tokens = 10000,dir = dir)[1]

# 构造输入
ask = "Do you love me?"
reply = predict_ch8(ask,33,model,my_vocab,device)
# index = reply.find('eeee')

print("==================================================================")
print(f"AI : {reply[len(ask):]}")