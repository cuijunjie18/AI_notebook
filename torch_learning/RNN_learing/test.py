from net_frame import *

# 获取数据迭代器
dir = 'poems_data/'
# dir = 'data/'
batch_size, num_steps = 2, 5
train_iter,vocab = load_data(batch_size,num_steps,dir)
corpus = train_iter.corpus
str = ''
for idx in corpus:
    str += vocab.idx_to_token[idx]
print(str)
