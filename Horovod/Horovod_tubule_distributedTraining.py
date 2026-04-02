#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import horovod.keras as hvd


# In[2]:


hvd.init()
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()],'GPU')


# In[3]:


import os, cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'
from glob import glob
import numpy as np
from utils import *
import matplotlib.pyplot as plt

T95_path = "../../../KMIT_Datasets/dataset_T95"
T95_images = sorted(glob(T95_path + "/images/*.png"))
T95_masks = sorted(glob(T95_path + "/masks/*.png"))

T94_path = "../../../KMIT_Datasets/dataset_T94"
T94_images = sorted(glob(T94_path + "/images/*.png"))
T94_masks = sorted(glob(T94_path + "/masks/*.png"))

T93_path = "../../../KMIT_Datasets/dataset_T93"
T93_images = sorted(glob(T93_path + "/images/*.png"))
T93_masks = sorted(glob(T93_path + "/masks/*.png"))

new_images=sorted(glob("../../../KMIT_Datasets/new_data/*/images/*.jpg"))
new_masks=sorted(glob("../../../KMIT_Datasets/new_data/*/masks/*.jpg"))
plot = True


# In[3]:


print(len(new_images))


# # T95 ImageViz

# In[4]:




x = []; y = []
i=0
for image, mask in zip(T95_images, T95_masks):
  img = cv2.imread(image)
  x.append(img)
  x.append(cv2.flip(img, 0))
  x.append(cv2.flip(img, -1))
  x.append(cv2.flip(img, 1))
  msk = (cv2.threshold(cv2.imread(mask, 0),127,255,cv2.THRESH_BINARY)[1]//255.0).reshape(256,256,1)
  y.append(msk)
  y.append(cv2.flip(msk, 0).reshape(256,256,1))
  y.append(cv2.flip(msk, -1).reshape(256,256,1))
  y.append(cv2.flip(msk, 1).reshape(256,256,1))
  i+=1
for image, mask in zip(T94_images, T94_masks):
  img = cv2.imread(image)
  x.append(img)
  x.append(cv2.flip(img, 0))
  x.append(cv2.flip(img, -1))
  x.append(cv2.flip(img, 1))
  msk = (cv2.threshold(cv2.imread(mask, 0),127,255,cv2.THRESH_BINARY)[1]//255.0).reshape(256,256,1)
  y.append(msk)
  y.append(cv2.flip(msk, 0).reshape(256,256,1))
  y.append(cv2.flip(msk, -1).reshape(256,256,1))
  y.append(cv2.flip(msk, 1).reshape(256,256,1))
  i+=1  
for image, mask in zip(T93_images, T93_masks):
  img = cv2.imread(image)
  x.append(img)
  x.append(cv2.flip(img, 0))
  x.append(cv2.flip(img, -1))
  x.append(cv2.flip(img, 1))
  msk = (cv2.threshold(cv2.imread(mask, 0),127,255,cv2.THRESH_BINARY)[1]//255.0).reshape(256,256,1)
  y.append(msk)
  y.append(cv2.flip(msk, 0).reshape(256,256,1))
  y.append(cv2.flip(msk, -1).reshape(256,256,1))
  y.append(cv2.flip(msk, 1).reshape(256,256,1))
  i+=1
print(i)
x = np.asarray(x); y = np.asarray(y)


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, shuffle = True)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle = True,) 
print('Data Split!')
print(len(x_train),len(x_test), len(x_valid))
print(x_train[0].shape, y_train[0].shape)
del x; del y


# In[12]:


# In[15]:
callbacks=[hvd.callbacks.BroadcastGlobalVariablesCallback(0),
           hvd.callbacks.MetricAverageCallback(),
           ]

if(hvd.local_rank()==0):
    callbacks.append(tf.keras.callbacks.CSVLogger("hvd2_history.csv",separator=",", append=False))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('hvd_model2.h5', save_best_only=True, monitor='val_mean_io_u', mode='max'))


import time
t1=time.time()
opt=tf.keras.optimizers.Adam(learning_rate=2e-4)
model=r2_unet(256, 256, 1)
opt=hvd.DistributedOptimizer(opt)
model.compile(optimizer=opt, loss="binary_crossentropy",experimental_run_tf_function=False, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
results = model.fit(x_train,y_train,verbose=1 if hvd.local_rank()==0 else 0,validation_data=(x_valid,y_valid),batch_size=16, epochs=250, callbacks=callbacks
   
)
#print(time.time()-t1)


# In[4]:

model.evaluate(x_test,y_test)

