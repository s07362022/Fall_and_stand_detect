import cv2
import sys
from PIL import Image
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten ,BatchNormalization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from face_train import Model
import keras
#from keras.models import load_model
from tensorflow.keras.models import load_model

IMAGE_SIZE =48 #the model img size
train_list = []
train_ary= np.zeros(shape=(100,48,48,3)) 

#載入模型
#model = Sequential()
model = load_model('C:\\Harden_project\\project5_node.js\\CNN_fall.h5') #keras.models. CNN_fall.h5

def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #獲取影象尺寸
    h, w, _ = image.shape
    
    #對於長寬不相等的圖片，找到最長的一邊
    longest_edge = max(h, w)    
    
    #計算短邊需要增加多上畫素寬度使其與長邊等長
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
    
    #RGB顏色
    BLACK = [0, 0, 0]
    
    #給影象增加邊界，是圖片長、寬等長，cv2.BORDER_CONSTANT指定邊界顏色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    #調整影象大小並返回
    return cv2.resize(constant, (height, width))

def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)
    
    #視訊來源，可以來自一段已存好的視訊，也可以直接來自USB攝像頭
    cap = cv2.VideoCapture(camera_idx)                
    
    #告訴OpenCV使用人臉識別分類器
    #classfier = cv2.CascadeClassifier("D:\\Python\\mushrooms\\XLM\\6_21\\cascade.xml")
    
    #識別出人臉後要畫的邊框的顏色，RGB格式
    color = (0, 255, 0)

    #video_images = []
    #time_a = 10
    c = 0
    a = 0
    while cap.isOpened():
        ok, frame = cap.read() #讀取一幀資料
        if not ok:            
            break  

        #將當前幀轉換成灰度影象
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 
        
        #檢測，1.2和2分別為圖片縮放比例和需要檢測的有效點數
        
        #image = frame[y - 10: y + h + 10, x - 10: x + w + 10]

        #cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

        train = resize_image(frame, IMAGE_SIZE, IMAGE_SIZE)
                #print(train.shape)
        train_list.append(train)
        global train_ary 
        train_ary = np.array(train_list, dtype=np.float32)
        print(train_ary.shape)
        
        #train_ary = train_ary.reshape(train_ary.shape[0],-1)
        fallID = model.predict_classes(train_ary)
        print(faceID)
        print(type(faceID))
                #轉list
        fall = fallID.tolist()
                #試試看 數值變數  
        if fall[a] == 1:                                                        
            #cv2.rectangle(frame)
            #文字提示是誰
            cv2.putText(frame,'FALL' ,(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)                                     
        else:
            pass
              
        a +=1                    
        #顯示影象
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(100)
        if c & 0xFF == ord('q'):
            break
    #x = np.array(video_images, dtype=np.float32)
    #把list轉arry 
    #global train_ary       
    #train_ary= np.array(train_list, dtype=np.float32)
    #print(train_ary.shape)
    #預測
    #faceID = model.predict_classes(train_ary)
    #釋放攝像頭並銷燬所有視窗
    cap.release()
    cv2.destroyAllWindows()


'''
#把list轉arry        
train_ary= np.array(train_list, dtype=np.float32)
print(train_ary.shape)
#預測
faceID = model.predict_classes(train_ary)
'''

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("video", 0)
#print(train_ary.shape)
#預測
fallID = model.predict_classes(train_ary)
#print(fallID) 
