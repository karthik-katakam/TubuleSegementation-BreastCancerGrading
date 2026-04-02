import os, cv2
from glob import glob
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

images=sorted(glob("Data/*/images/*"))
masks=sorted(glob("Data/*/masks/*"))

print(len(images))
print(len(masks))

newsize = (256, 256)
x = []
y = []

# Process each image and its corresponding mask
for image, mask in zip(images, masks):
    img = Image.open(image).resize(newsize)
    # print(img)
    pil_image = img.copy().convert('RGB')
    open_cv_image = np.array(pil_image)[:, :, ::-1]  # Convert to OpenCV format (BGR)
    
    x.append(open_cv_image)
    x.append(cv2.flip(open_cv_image, 0))
    x.append(cv2.flip(open_cv_image, -1))
    x.append(cv2.flip(open_cv_image, 1))

    msk = Image.open(mask).resize(newsize).convert('RGB')
    open_cv_mask = np.array(msk)[:, :, ::-1]
    msk = cv2.cvtColor(open_cv_mask, cv2.COLOR_BGR2GRAY)
    msk = (cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)[1] // 255.0).reshape(256, 256, 1)

    y.append(msk)
    y.append(cv2.flip(msk, 0).reshape(256, 256, 1))
    y.append(cv2.flip(msk, -1).reshape(256, 256, 1))
    y.append(cv2.flip(msk, 1).reshape(256, 256, 1))

x = np.asarray(x)
y = np.asarray(y)


x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=1 - train_ratio, shuffle=True, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=test_ratio / (test_ratio + validation_ratio), shuffle=True, random_state=42)


print('Data Split!')
print(f'Train: {len(x_train)}, Test: {len(x_test)}, Validation: {len(x_valid)}')
print(f'x_train shape: {x_train[0].shape}, y_train shape: {y_train[0].shape}')

opt = Adam(learning_rate=2e-4)


model = r2_unet(256, 256, 1)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

initial_epochs = 10  # Reduced from 500 for initial testing
small_batch_size = 16  # Reduced from 32 to lessen memory load

results = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                    batch_size=small_batch_size, epochs=initial_epochs, verbose=2,
                    callbacks=[ModelCheckpoint('partial_model_070622.keras', save_best_only=True, monitor='val_mean_io_u_2', mode='max'),
                               CSVLogger("partial_history_070622.csv", separator=",", append=False)])

opt=tf.keras.optimizers.Adam(learning_rate=2e-4)
model1=r2_unet(256, 256, 1)
model1.compile(optimizer=opt, loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2),f1_metric])

model1.load_weights('./complete_model_6_11.h5')