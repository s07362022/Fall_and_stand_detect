import cv2
import numpy as np
#YOLO_network
net = cv2.dnn.readNetFromDarknet("F:\\program1\\yolo\\darknet-master\\build\\darknet\\x64\\cfg\\yolov3.cfg","F:\\program1\\yolo\\darknet-master\\build\\darknet\\x64\\yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open("F:\\program1\\yolo\\darknet-master\\build\\darknet\\x64\\data\\coco.names")]
colors = [(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0),(0,0,255),(255,0,0),(0,255,0)]
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
##################
img_path= "F:\\pic"
count=1
##################
def yolo_detect(frame,path_name,count):
    # forward propogation
    img = frame
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
            if count%10==0:
                img_name = '%s//%d.jpg'%(path_name, count)
                print("儲存成功")
            cv2.imwrite(img_name, image)
            cv2.putText(img, label, (x, y -5), font, 1, color, 1)
            '''
            print("偵測到")
            if label=="stand":
                print("站著")
            elif label=="fall":
                print("跌倒")
            '''    
    return img

import cv2
import imutils
import time

VIDEO_IN = cv2.VideoCapture(1)

while True:
    hasFrame, frame = VIDEO_IN.read()
    
    img = yolo_detect(frame,path_name=img_path,count=count)
    cv2.imshow("Frame", imutils.resize(img, width=1200,height=960))#imutils.resize(img), width=, width=860
    count=count+1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
VIDEO_IN.release()
cv2.destroyAllWindows()
