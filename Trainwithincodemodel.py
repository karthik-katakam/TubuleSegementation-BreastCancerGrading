import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from multiprocessing import Pool

if tf.test.is_gpu_available():
    print("GPU is available and will be used for training.")
else:
    print("GPU not detected. Training will use CPU.")

def preprocess_image(image_path, mask_path, new_size=(256, 256), counter=None):
    img = Image.open(image_path).resize(new_size).convert('RGB')
    img_array = np.array(img)
    msk = Image.open(mask_path).resize(new_size).convert('L')  # Grayscale for masks
    mask_array = np.array(msk)

    if counter is not None:
        counter[0] += 1
        print(f"Processed {counter[0]} images", end='\r')

    return img_array, mask_array

def load_data(image_path_pattern, mask_path_pattern, new_size=(256, 256)):
    images = sorted(glob(image_path_pattern))
    masks = sorted(glob(mask_path_pattern))

    if len(images) != len(masks):
        raise ValueError("Number of images and masks must match.")

    counter = [0]  

    with Pool() as pool:
        results = pool.starmap(preprocess_image, zip(images, masks, [new_size] * len(images), [counter] * len(images)))

    x, y = zip(*results)

    x = np.array(x)
    y = np.array(y)
    x_aug = np.concatenate([x, np.flip(x, axis=1), np.flip(x, axis=2)], axis=0)
    y_aug = np.concatenate([y, np.flip(y, axis=1), np.flip(y, axis=2)], axis=0)

    print(f"\nAll images processed.")
    return x_aug, y_aug

def build_r2unet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    def recurrent_block(x, filters):
        for _ in range(2):
            x1 = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
            x = tf.keras.layers.add([x, x1])
        return x

    def residual_recurrent_block(x, filters):
        x1 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", activation=None)(x)
        x = recurrent_block(x, filters)
        x = tf.keras.layers.add([x, x1])
        return x

    c1 = residual_recurrent_block(inputs, 64)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = residual_recurrent_block(p1, 128)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = residual_recurrent_block(p2, 256)

    u1 = tf.keras.layers.UpSampling2D((2, 2))(c3)
    u1 = tf.keras.layers.concatenate([u1, c2])
    c4 = residual_recurrent_block(u1, 128)

    u2 = tf.keras.layers.UpSampling2D((2, 2))(c4)
    u2 = tf.keras.layers.concatenate([u2, c1])
    c5 = residual_recurrent_block(u2, 64)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = tf.keras.models.Model(inputs, outputs)
    return model

def main():
    image_path = "Data/*/images/*"
    mask_path = "Data/*/masks/*"

    x, y = load_data(image_path, mask_path)
    print(f"Loaded {x.shape[0]} images and {y.shape[0]} masks.")

    x = x / 255.0  # Normalize images
    y = y / 255.0  # Normalize masks
    
    from sklearn.model_selection import train_test_split
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=42)

    model = build_r2unet(input_shape=(256, 256, 3))
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy()])

    callbacks = [
        ModelCheckpoint("model.h5", save_best_only=True, verbose=1),
        ReduceLROnPlateau(patience=5, factor=0.1, verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
    ]

    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=50,
        batch_size=16,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()
