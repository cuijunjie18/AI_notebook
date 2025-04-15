import cv2
import numpy as np

def response(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f"({x},{y}) : {img[y,x]}")

path = 'E:/My_note_book/torch_learning/semantic_segmentation/images/4052_l.png'
img = cv2.imread(path)
cv2.namedWindow("Click")
cv2.setMouseCallback('Click',response)

while(True):
    cv2.imshow('Click',img)
    if cv2.waitKey(1) == 27:
        break