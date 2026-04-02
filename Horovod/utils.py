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

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input,add, multiply
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
from tensorflow.python.keras.layers import concatenate, core, Dropout
from tensorflow.keras.models import Model

from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
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

def mean_iou(mask, pred):
    intersection = np.logical_and(mask,pred)
    union = np.logical_or(mask, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    if(np.sum(union) == 0): return 100
    return iou_score*100


def up_and_concate(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[3]
    up = UpSampling2D(size=(2, 2))(down_layer)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])
    return concate
  
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],padding='same'):
  #print(input_layer.get_shape())
  input_n_filters = input_layer.get_shape().as_list()[3]

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
    inputs = Input((img_w, img_h,3))
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
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    return model