import torch

def sequence_mask(X, valid_len, value=0): #@save
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # print(mask)
    X[~mask] = value
    return X

batch_size,num_steps,vocab_size = 10,10,30
X = torch.rand(batch_size,num_steps,vocab_size)
valid_len = torch.randint(1,10,(batch_size,))
sequence_mask(X,valid_len)