from my_frame import*
import my_frame
from anchor import*
from d2l import torch as d2l

# 读取香蕉数据集
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

# 网络初始化
device, net = d2l.try_gpu(), TinySSD(num_classes = 1)
trainer = torch.optim.SGD(net.parameters(), lr = 0.2, weight_decay = 5e-4)

# 导入之前训练的模型参数
net.load_state_dict(torch.load("data/SSD.params"))

# 定义损失函数
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

nums_epoch = 1
net = net.to(device)
use_times = 0
plt_loss = []
for epoch in range(nums_epoch):
    start = time.perf_counter() # 相比time.time()更加精确
    net.train() # 切换回训练模式
    loss_save = 0
    for features,target in train_iter:
        # 清空梯度
        trainer.zero_grad()

        # 切换硬件
        X = features.to(device)
        Y = target.to(device) # Target为实际目标的位置

        # 网络mini-batch输入的前向传播
        anchors,cls_preds,bbox_preds = net(X)

        # 为生成的锚框标注类别与偏移量
        bbox_labels,bbox_masks,cls_labels = multibox_target(anchors,Y)

        # 根据类别、偏移量的预测和标注值计算损失函数
        loss = calc_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks)

        # 反向传播及更新权重参数
        loss.mean().backward()
        loss_save += loss.mean().item()
        trainer.step()
    plt_loss.append(loss_save / len(train_iter))
    end = time.perf_counter()
    print(f"epoch{epoch} use:{end - start} seconds")
    use_times += end - start

print(f"Model train use {use_times} seconds")
plt.plot(np.arange(0,len(plt_loss),1),plt_loss)
plt.show()
torch.save(net.state_dict(),"data/SSD.params")