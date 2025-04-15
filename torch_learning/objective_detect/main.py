import my_frame as mf
# from d2l import torch as d2l
import torch
import torchvision
from torch.nn import functional as F
from anchor import*
import cv2

# 预测函数
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

# # 展示函数
# def display(img, output, threshold):
#     d2l.set_figsize((5, 5))
#     fig = d2l.plt.imshow(img)
#     for row in output:
#         score = float(row[1])
#         if score < threshold:
#             continue
#         h, w = img.shape[0:2]
#         bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
#         d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

# 我的opencv展示
def my_show(path,objective_position,thresh = 0.9):
    img = cv2.imread(path)
    h,w = img.shape[:2]
    for i in range(len(objective_position)):
        score = objective_position[i][1]
        if score < thresh:
            continue
        x1,y1,x2,y2 = objective_position[i][2:]
        x1 = int (w * x1)
        x2 = int (w * x2)
        y1 = int (h * y1)
        y2 = int (h * y2)
        cv2.rectangle(img,(x1,y1),(x2,y2),[0,0,255],3)
    cv2.imshow("Result",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

device = mf.try_gpu()
net = mf.TinySSD(num_classes = 1)
net.load_state_dict(torch.load("data/SSD.params"))

# 读入待检测图片并化为可输入到网络的格式
img_path = "E:/My_note_book/torch_learning/objective_detect/images/banana.jpg"
X = torchvision.io.read_image('E:/My_note_book/torch_learning/objective_detect/images/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
output = predict(X)
nums_objective = torch.sum(output[:,1] >= 0.9).item()
print(f"This img have {nums_objective} obejcets!")
my_show(img_path,output)
# display(img, output.cpu(), threshold=0.9)

