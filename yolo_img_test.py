import cv2
import numpy as np
#YOLO_network
net = cv2.dnn.readNetFromDarknet("F:\\program1\\yolov4\\darknet-master\\build\\darknet\\x64\\cap\\yolov4-tiny.cfg","F:\\program1\\yolov4\\darknet-master\\build\\darknet\\x64\\cap\\yolov4-tiny_last_610.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open("F:\\program1\\yolov4\\darknet-master\\build\\darknet\\x64\\cap\\cap.names")]
colors = [(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0)]
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

def yolo_detect(frame):
    # forward propogation
    img = frame
    #print(type(img))
    #img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    #print(type(img))<class 'numpy.ndarray'>
    height, width, channels = img.shape 
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # get detection boxes
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            tx, ty, tw, th, confidence = detection[0:5]
            scores = detection[5:]
            class_id = np.argmax(scores)  
            if confidence > 0.6:   
                center_x = int(tx * width)
                center_y = int(ty * height)
                w = int( tw * width)#tw *
                h = int( th * height)#th *

                # 取得箱子方框座標
                x = int(center_x - w/2 )
                y = int(center_y - h/2 )
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # draw boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 1) #劃出框框
            #print(x,y,(x+w),(y+h))
            image = img[y - 10: y + h + 10, x - 10: x + w + 10]
            
            cv2.putText(img, label, (x, y -5), font, 1, color, 1)
    return img

import cv2
import imutils
import time

img=cv2.imread("F:\\Dataset\\cap\\k_cimg\\c20.jpg")

    
img = yolo_detect(img)
cv2.imshow("Frame", imutils.resize(img, width=1200,height=960))#imutils.resize(img), width=, width=860
    
cv2.waitKey(0)
cv2.destroyAllWindows()
