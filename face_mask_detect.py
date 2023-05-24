import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint




DIRECTORY = r"F:\mask_no mask dataset\test"
A = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
                                   
X = A.flow_from_directory( DIRECTORY,
                                                    batch_size=10, 
                                                    target_size=(150, 150))
model = Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit_generator(X, epochs=5)                               
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
rect_size = 4

ans={0:' mask',1:'without mask'}

color_dict={0:(0,255,0),1:(255,0,0)}

while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1,1) 
    
    rerect_size = cv2.resize(frame, (frame.shape[1] // rect_size, frame.shape[0] // rect_size))
    
    
    faces = face_cascade.detectMultiScale(rerect_size)
    for f in faces:
      (x, y, w, h) = [v * rect_size for v in f] 
      face_img = frame[y:y+h, x:x+w]
      rerect_sized=cv2.resize(face_img,(150,150)) 
      normal=rerect_sized/255.0
      reshaped=np.reshape(normal,(1,150,150,3))
      reshaped = np.vstack([reshaped])
      ans=model.predict(reshaped)


      label=np.argmax(ans,axis=1)[0]
      
      if label == 0:
        res = "   MASK"
        color = (0,255,0)
      else:
        res  = "  NO  MASK" 
        color = (255,0,0) 
      cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],5)
      cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],5)
      cv2.putText(frame,res,(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
      cv2.imshow('frame', frame)

      if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()