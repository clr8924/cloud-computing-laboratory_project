import cv2
import numpy as np
import sys,os,dlib,glob,numpy
from skimage import io
import numpy as np
import sys,os,dlib,glob,numpy
from skimage import io
from keras.models import Model
import keras

#載入模型
model = keras.models.load_model('C:/Users/user/Cloud-Computing-Laboratory/gesture/modelv2.h5')

cap = cv2.VideoCapture(0)

CLIP_X1,CLIP_Y1,CLIP_X2,CLIP_Y2 = 160,140,400,360 # ROI's size
image_q = cv2.THRESH_BINARY 

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('C:/Users/user/Cloud-Computing-Laboratory/gesture/gesture.mp4',fourcc,10,(640,480))

while True:
    
    _, FrameImage = cap.read() 
    FrameImage = cv2.flip(FrameImage, 1) 
    
    cv2.rectangle(FrameImage, (CLIP_X1, CLIP_Y1), (CLIP_X2, CLIP_Y2), (0,255,0) ,1) # ROI位置
    ROI = FrameImage[CLIP_Y1:CLIP_Y2, CLIP_X1:CLIP_X2] # ROI大小
    ROI = cv2.resize(ROI, (32, 32))  # ROI RESIZE 
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY) # ROI 轉灰階
    SHOWROI = cv2.resize(ROI, (256,256)) # ROI resize 
    
    _, output = cv2.threshold(ROI,90, 255, image_q) # Black Background     
    _, output2 = cv2.threshold(SHOWROI,90, 255, image_q) # 顯示
    
    img = output2
    imgnew = cv2.resize(img,(32,32))
    data = (np.asarray(imgnew, dtype="uint8").astype('float32'))/255
    data = data.reshape((1, 32, 32, 1)) 
    gestureID = model.predict(data)
    print(gestureID)
    
    
    
    if gestureID[0][0] > 0.85 :
        
        cv2.putText(FrameImage,'stone', (100,120), cv2.FONT_HERSHEY_TRIPLEX, 4, (143,69,134), 2)
                    
    elif gestureID[0][1] > 0.85 :  
                
        cv2.putText(FrameImage,'one', (100, 120), cv2.FONT_HERSHEY_TRIPLEX, 4, (143,69,134), 2)
        
    elif gestureID[0][2] > 0.85 :  
                
        cv2.putText(FrameImage,'ya', (100, 120), cv2.FONT_HERSHEY_TRIPLEX, 4, (143,69,134), 2)
        
    elif gestureID[0][3] > 0.85 :  
                
        cv2.putText(FrameImage,'hi', (100, 120), cv2.FONT_HERSHEY_TRIPLEX, 4, (143,69,134), 2)
        
    else:

         cv2.putText(FrameImage,'where is your finger', (30, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (143,69,134), 2)


    
    out.write(output2)
    out.write(FrameImage)
    cv2.imshow("ROI", output2)
    cv2.imshow("Webcam", FrameImage) 
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('q'):
        break
     
    
cap.release()
out.release()
cv2.destroyAllWindows()