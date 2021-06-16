import os
import cv2
import numpy as np
import pandas as pd
import itertools
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

import warnings
warnings.filterwarnings("ignore")

# sklearn
from sklearn.model_selection import train_test_split #分割資料
from sklearn.metrics import confusion_matrix

# keras 
import keras
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

#影像的輸入大小
im_W = 32
im_H = 32

#影像的輸入維度
datachannel = 1

#影像的path
class_path = r"C:/Users/user/Cloud-Computing-Laboratory/gesture/dataset/gestureresize/"

#影像該位置的所有檔案名稱
class_filenames = os.listdir(class_path)

#建立空陣列並定義大小，將給予輸入存取的空間：根據影像數量與輸入大小及維度
data = np.empty((len(class_filenames), im_W, im_H, datachannel), dtype="uint8")

#建立皆為0且檔案數量的label
label = [0] * len(class_filenames)

#塞制空陣列的輸入與label
for i in range(len(class_filenames)):
    
    #label根據檔案名字前一個字，取代之前建利皆為0的label
    if(str(class_filenames[i][0])=='0'):
        label[i] = 0
    if(str(class_filenames[i][0])=='1'):
        label[i] = 1
    if(str(class_filenames[i][0])=='2'):
        label[i] = 2
    if(str(class_filenames[i][0])=='5'):
        label[i] = 3
    if(str(class_filenames[i][0])=='8'):
        label[i] = 4
        
    #每一影像讀取並轉呈灰階
    img = Image.open(class_path + "/" + class_filenames[i]).convert("L")
    
    #將該影像轉成陣列
    npimg = np.asarray(img, dtype="uint8")
    
    #持續將影像陣列蓋掉之前空陣列
    data[i,:,:,0] = npimg

#打印出影像的維度與標籤的維度
print("影像的維度：", data.shape)
print("標籤的維度：", len(label))

#影像分割訓練集與測試集
x_train, x_test, y_train_noonehot, y_test_noonehot = train_test_split(data, label, test_size = 0.3, random_state = 0) 

#打印出訓練集與測試集的每個類別數量
print("訓練集標籤數量：%s" %(len(y_train_noonehot)))
print("訓練集標籤的類別數量：",Counter(y_train_noonehot))
print("訓練集標籤數量：%s" %(len(y_test_noonehot)))
print("測試集標籤的類別數量：",Counter(y_test_noonehot))

#將Label轉為One-hot-encode
y_train = pd.get_dummies(pd.Series(y_train_noonehot)).values.astype('float32')
y_test = pd.get_dummies(pd.Series(y_test_noonehot)).values.astype('float32')

#影像正規化
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255

#宣告模型
model = Sequential()

#輸入層與第一個隱藏層
model.add(Conv2D(filters=8,kernel_size=(5, 5), input_shape=(im_W, im_H, datachannel),activation='relu'))
model.add(Dropout(0.3))

#第二個隱藏層
model.add(Conv2D(filters=16,kernel_size=(5, 5),activation='relu'))
model.add(Dropout(0.3))

#第三個隱藏層
model.add(Conv2D(filters=16,kernel_size=(5, 5),activation='relu'))
model.add(Dropout(0.3))

#第四個隱藏層
model.add(Conv2D(filters=32,kernel_size=(5, 5),activation='relu'))
model.add(Dropout(0.3))

#全連接層
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

#輸出層(共九個類別)
model.add(Dense(5 , activation='softmax'))

#定義使用的損失函數 與 優化器 以及 要顯示的指標
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

#打印出目前已宣告好的模型
print(model.summary())

print('x_train.shape:{}'.format(x_train.shape))
print('x_test.shape:{}'.format(x_test.shape))
print('y_train.shape:{}'.format(y_train.shape))
print('y_test.shape:{}'.format(y_test.shape))

#開始訓練模型
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=30, verbose=2)

# 繪出學習曲線
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.title("Training & Validation Acc")
plt.plot(np.arange(len(acc)), acc,color='b', label="Training set")
plt.plot(np.arange(len(val_acc)), val_acc,color='r', label="Validation set")
plt.legend(loc='lower right')
plt.show()

plt.title("Training & Validation Loss")
plt.plot(np.arange(len(loss)), loss,color='b', label="Training set")
plt.plot(np.arange(len(val_loss)), val_loss,color='r', label="Validation set")
plt.legend(loc='lower right')
plt.show()

#模型儲存
model.save(r"C:/Users/user/Cloud-Computing-Laboratory/gesture/modelv2.h5")

#訓練集
probabilities = model.predict(x_train)
predict = model.predict_classes(x_train)

cnf_matrix = confusion_matrix(y_train_noonehot, predict)
target_names = np.arange(0,5,1).astype(str)

plt.figure(figsize=(6,6))
plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix - Training set', cmap=plt.cm.Blues)

#測試集

probabilities = model.predict(x_test)
predict = model.predict_classes(x_test)

cnf_matrix = confusion_matrix(y_test_noonehot, predict)
target_names = np.arange(0,5,1).astype(str)

plt.figure(figsize=(6,6))
plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix - Test set', cmap=plt.cm.Greens)

