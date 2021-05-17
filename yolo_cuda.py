import cv2
import numpy as np
from influxdb import InfluxDBClient
import time 
import datetime
client = InfluxDBClient('localhost', 8086, '', '', 'stand_fall')  #SQL
all_value = [0] #全部的class狀況
stand_value= [0]
fall_value= [0]
count = 1

# GPU內存#############################
from GPU_memory import *
#solve_cudnn_error()
########################################



#CNN######################################
import imutils
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten ,BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import keras
IMAGE_SIZE =640
train_list = []
train_ary= np.zeros(shape=(100,640,640,3))
model = load_model('C:\\Users\\User.DESKTOP-IIINHE5\\Desktop\\stand_fall3.h5') #model_load
##############################################
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    h, w, _ = image.shape
    
    longest_edge = max(h, w)    
    
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    BLACK = [0, 0, 0]
    
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    return cv2.resize(constant, (height, width))


#Gmail#################################
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
import smtplib
def smtp(img_name,label):
    content = MIMEMultipart()  #建立MIMEMultipart物件
    content["subject"] = "detector to  Stand or fall"  #郵件標題
    content["from"] = "gish1040403@gmail.com"  #寄件者
    content["to"] = "qaz3661537@gmail.com" #收件者
    timenow = datetime.datetime.now()
    timenow = str(timenow)
    content.attach(MIMEText(timenow + " " +label))  #郵件內容
    phto = Path(img_name).read_bytes()
    content.attach(MIMEImage(phto))
    #文字
    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:  # 設定SMTP伺服器
        try:
            smtp.ehlo()  # 驗證SMTP伺服器
            smtp.starttls()  # 建立加密傳輸
            smtp.login("gish1040403@gmail.com", "cffakphyrorydcti")  # 登入寄件者gmail
            smtp.send_message(content)  # 寄送郵件
            print("Complete!")
        except Exception as e:
            print("Error message: ", e)
#######################################


#YOLO################################
net = cv2.dnn.readNetFromDarknet("F:\\program1\\yolo\\darknet-master\\build\\darknet\\x64\\yolov3.cfg","F:\\program1\\yolo\\darknet-master\\build\\darknet\\x64\\yolov3_last.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open("F:\\program1\\yolo\\darknet-master\\build\\darknet\\x64\\obj.names")]
colors = [(0,0,255),(255,0,0),(0,255,0)]
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
###########################################



#DB########################################
def sql(all_value,stand_value,count):
    data = [
        {
            "measurement": "stand_fall",
            "tags": {
                "topic": "Sensor/yolo"
            },
            "fields": {
                "value": all_value[-1]
            }
        }
    ]
    client.write_points(data)
    #if count ==0:
        #client.write_points(data)
        #print(all_value[-1])
    #elif (count %100) == 0:
        #client.write_points(data)
        #print(all_value[-1])
    data2 = [
        {
            "measurement": "stand_fall",
            "tags": {
                "topic": "Sensor/stand"
            },
            "fields": {
                "value": stand_value[-1]
            }
        }
    ]
    #client.write_points(data2)


def sql_fall(fall_value):
    data3 = [
        {
            "measurement": "stand_fall",
            "tags": {
                "topic": "Sensor/fall"
            },
            "fields": {
                "value": fall_value[-1]
            }
        }
    ]
    client.write_points(data3)
    #time.sleep()
###########################################

path_name = "F:\\program1\\influxdb\\img"
def yolo_detect(frame,count,path_name):
    # forward propogation
    img = cv2.resize(frame, None, fx=0.5, fy=0.5)
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
            cv2.putText(img, label, (x, y -5), font, 1, color, 1)
            #print("偵測到")
            if label=="stand":
                all_value.append("stand")
                stand_value.append("stand")
                sql(all_value,stand_value,count)#存DB
                print("站著",count)
                #下載照片並傳送                
                if count %15 == 0:
                    image = img[y +10: y + h -10, x+10 : x + w -10]
                    img_name = '%s//%d.jpg'%(path_name, count)
                    cv2.imwrite(img_name, image)
                    #CNN第二層判斷##################
                    train = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                    train_list.append(train)
                    global train_ary
                    train_ary = np.array(train_list, dtype=np.float32)
                    ID = model.predict_classes(train_ary)#_classes
                    #preds = model.predict(train_ary)
                    #ID = np.argmax(preds[0])
                    print(ID[-1])
                    #print(type(ID[-1]))
                    if ID[-1] == 0:
                        smtp(img_name,label)
                        time.sleep(5)

            elif label=="fall":
                all_value.append("fall")
                fall_value.append("fall")
                sql(all_value,stand_value,count)#存DB
                sql_fall(fall_value)#存DB         
                print("跌倒",count)
                #下載照片並傳送
                if count %15 == 0:
                    image = img[y +10: y + h -10, x+10 : x + w -10]
                    img_name = '%s//%d.jpg'%(path_name, count)
                    cv2.imwrite(img_name, image)
                    #CNN############
                    train = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                    train_list.append(train)
                    #global train_ary
                    train_ary = np.array(train_list, dtype=np.float32)
                    #preds = model.predict(train_ary)
                    #ID = np.argmax(preds[0])
                    ID = model.predict_classes(train_ary)#_classes
                    print(ID)
                    #print(type(ID[-1]))
                    if (ID[-1]==1) | (ID[-1] ==2):# | (ID[-1] ==2):
                        smtp(img_name,label)
                        time.sleep(5)
                    
            elif label=="fall1":
                all_value.append("fall")
                fall_value.append("fall")
                sql(all_value,stand_value,count)#存DB
                sql_fall(fall_value)#存DB
                print("跌倒",count)
                #下載照片並傳送
                if count %15 == 0:
                    image = img[y +10: y + h -10, x+10 : x + w -10]
                    img_name = '%s//%d.jpg'%(path_name, count)
                    cv2.imwrite(img_name, image)
                    #CNN###################
                    train = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                    train_list.append(train)
                    #global train_ary
                    train_ary = np.array(train_list, dtype=np.float32)
                    #preds = model.predict(train_ary)
                    #ID = np.argmax(preds[0])
                    ID = model.predict_classes(train_ary)#_classes
                    print(ID)
                    #print(type(ID[-1]))
                    if (ID[-1]==1) | (ID[-1] ==2):
                        smtp(img_name,label)
                        time.sleep(5)    
                
    return img

import cv2
import imutils
import time

VIDEO_IN = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    hasFrame, frame = VIDEO_IN.read()
    
    img = yolo_detect(frame,count,path_name)
    
    cv2.imshow("Frame", imutils.resize(img, width=1200,height=960))#imutils.resize(img), width=, width=860
    count+=1
    #GPU
    #G()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
VIDEO_IN.release()
cv2.destroyAllWindows()
