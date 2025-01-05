import cv2
import numpy as np
import os

def show(img,title = 'Test'):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = 'E:/My_note_book/torch_learning/semantic_segmentation/images/4052_l.png'
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
save = cv2.threshold(img,10,255,cv2.THRESH_BINARY)[1]
cv2.imwrite('E:/My_note_book/torch_learning/semantic_segmentation/images/4052_l.jpg',save)