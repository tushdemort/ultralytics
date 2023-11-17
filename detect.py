from ultralytics import YOLO
import numpy as np
import random
import cv2
import os
import time 



labels=open('./coco.txt','r')

labels=labels.read().split('\n')

detect_color=[]
for i in range(len(labels)):
    r=random.randint(0,255)
    g=random.randint(0,255)
    b=random.randint(0,255)
    detect_color.append((b,g,r))

model = YOLO("yolov8n.pt",'v8') 

width=640
height=480

capture=cv2.VideoCapture(0)
while (capture.isOpened()):
    path='./runs/detect/predict'
    files=os.listdir(path)
    for file in files:
        os.remove(os.path.join(path, file))
    os.rmdir(path)
    ret, frame= capture.read()
    if frame.any():
        cv2.imwrite('./frames/frame.jpg',frame)

        pred=model.predict(source='./frames/frame.jpg',conf=0.45,save=True)
        #pred=pred[0].numpy()
        #print(pred[0].numpy())
        #if len(pred)!=0:
         #   for params in pred:
        '''cv2.rectangle(frame,(int(params[0]),int(params[1])),(int(params[2]),int(params[3])),detect_color[int(params[5])],3)
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,labels[int(params[5])]+' '+str(round(params[4],3))+"%",(int(params[0]),int(params[1])-10),font,1,(255,255,255),2)
                cv2.imshow('detected',frame)'''
        detected=cv2.imread('./runs/detect/predict/frame.jpg')
        cv2.imshow('detected',detected)
        cv2.waitKey(100)



