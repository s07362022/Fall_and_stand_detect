import cv2
import sys
from PIL import Image
import numpy as np
import imutils
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten ,BatchNormalization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from face_train import Model
import keras
#from keras.models import load_model
from tensorflow.keras.models import load_model

IMAGE_SIZE =48
train_list = []
train_ary= np.zeros(shape=(100,48,48,3))

#model = Sequential()
model = load_model('C:\\Users\\User.DESKTOP-IIINHE5\\Desktop\\stand_fall1.h5') #keras.models.

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

def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)
    
    cap = cv2.VideoCapture(camera_idx)                
    
    color = (0, 255, 0)

    c = 0
    a = 0
    while cap.isOpened():
        ok, frame = cap.read() 
        if not ok:            
            break  

        train = resize_image(frame, IMAGE_SIZE, IMAGE_SIZE)
        train_list.append(train)
        global train_ary 
        train_ary = np.array(train_list, dtype=np.float32)
        
        faceID = model.predict_classes(train_ary)
        
        face = faceID.tolist()
                  
        if face[a] == 1:                                                        
            cv2.putText(frame,'fall' ,(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        elif face[a] == 2:                                                        
            cv2.putText(frame,'fall1' ,(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        elif face[a] == 0:                                                        
            cv2.putText(frame,'stand' ,(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)                                     
        else:
            pass
              
        a +=1  

        cv2.imshow(window_name, imutils.resize(frame, width=1200,height=960))        
        c = cv2.waitKey(100)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("video", 0)
