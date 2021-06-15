#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import glob
import PIL
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt


# In[ ]:


#二值化
myfiles = glob.glob('C:/Users/yutingweng/Desktop/gesture2/0_r/' + '*.jpg')
i = 0
for i,f in enumerate(myfiles):
    img = cv2.imread(f)
    (h,w,d) = img.shape
    center = (w//2,h//2)
    img_rotate = cv2.rotate(img,cv2.cv2.ROTATE_90_CLOCKWISE)
    img_new = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2GRAY)
    ret,th2 = cv2.threshold(img_new, 80, 255, cv2.THRESH_BINARY)    
    plt.imshow(th2, 'gray')
    plt.axis('off')
    plt.savefig('C:/Users/yutingweng/Desktop/gesture2/new0_r' + '/' +'0_' + str('{0:2d}').format(i+100) + '.jpg', bbox_inches='tight',pad_inches = 0)

    
#resize
myfiles = glob.glob('C:/Users/yutingweng/Desktop/0' + '/*.jpg')
i = 0
for i,f in enumerate(myfiles):
    img = Image.open(f)
    imgnew = img.resize((32,32),PIL.Image.ANTIALIAS)
    imgnew.save('C:/Users/yutingweng/Desktop/daya' + '/' +'L' + str('{0:2d}').format(i) + '.bmp')

