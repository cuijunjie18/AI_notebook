import torchvision
import os
from torchinfo import summary
save_path = 'E:/My_note_book/torch_learning/semantic_segmentation'
pretrained_net = torchvision.models.shufflenet_v2_x1_0()
summary(pretrained_net,(1, 3, 1080, 720))
# with open(os.path.join(save_path,'net.txt'),'w') as f: 
#     for x in list(pretrained_net.children()):
#         f.write(str(x) + '\n\n')