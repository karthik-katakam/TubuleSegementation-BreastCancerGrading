import os
import cv2
import sys
import random
from skimage.io import imread,imshow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Concatenate

from keras.layers import BatchNormalization, Reshape, Permute, Input,add, multiply
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Concatenate, Activation
from tensorflow.keras.models import Model

from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
# from keras.layers import concatenate, core, Dropout
from tensorflow.python.keras.layers  import concatenate, core, Dropout
from tensorflow.python.keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from tensorflow.python.keras.layers.core import Lambda
import tensorflow.keras.backend as K

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

train_ratio = 0.70
validation_ratio = 0.10
test_ratio = 0.20

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def contour_img(image,mask):
    mask = mask.astype(np.uint8)
    edge = cv2.Canny(mask,100,200)
    contours, hierarchy = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(image, contours, -1,(0,255,0), 2)
    return image
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def mean_iou(mask, pred):
    intersection = np.logical_and(mask,pred)
    union = np.logical_or(mask, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    if(np.sum(union) == 0): return 100
    return iou_score*100

from tensorflow.keras.layers import Concatenate

def up_and_concate(down_layer, layer):
    in_channel = down_layer.shape[-1]
    up = UpSampling2D(size=(2, 2))(down_layer)
    concate = Concatenate(axis=-1)([up, layer])
    return concate


  
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1], padding='same'):
    input_shape = input_layer.shape
    input_n_filters = input_shape[3]  # Accessing the number of filters directly from the shape tuple

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding)(input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding)(layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding)(add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer

def r2_unet(img_w, img_h, n_label):
    inputs = Input((img_w, img_h, 3))
    x = inputs
    depth = 4
    features = 16
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        features = features * 2

    x = rec_res_block(x, features)
    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i])
        x = rec_res_block(x, features)

    conv6 = Conv2D(n_label, (1, 1), padding='same')(x)
    conv7 = Activation('sigmoid')(conv6)  # Make sure this uses TensorFlow's Activation
    model = Model(inputs=inputs, outputs=conv7)
    return model

def Hsequence(Sequence):
    seed=1
    def __init__(self,x_set,y_set,batch_size):
        #self.x,self.y=shuffle(np.array(self.x),np.array(self.y),random_state=seed)
        self.x=self.x[hvd.local_rank()*len(self.x)/hvd.size():(hvd.local_rank()+1)*len(self.x)/hvd.size()]
        self.y=self.y[hvd.local_rank()*len(self.y)/hvd.size():(hvd.local_rank()+1)*len(self.y)/hvd.size()]
        self.batch_size=batch_size
    def __len__(self):
        return math.ceil(len(self.x)/self.batch_size)
    def __getitem__(self,idx):
        
        batch_x=self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y=self.y[idx*self.batch_size:(idx+1)*self.batch_size]
        return np.array(batch_x),np.array(batch_y)
    def on_epoch_end(self):
        self.x,self.y=shuffle(np.array(self.x),np.array(self.y),random_state=seed)
        self.x=self.x[hvd.local_rank()*len(self.x)/hvd.size():(hvd.local_rank()+1)*len(self.x)/hvd.size()]
        self.y=self.y[hvd.local_rank()*len(self.y)/hvd.size():(hvd.local_rank()+1)*len(self.y)/hvd.size()]
        
        
