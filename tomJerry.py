import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from enum import Enum
import PIL
import PIL.Image
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping 

# preparing data

directories = ["archive/tom_and_jerry/tom_and_jerry/jerry", 
               "archive/tom_and_jerry/tom_and_jerry/tom", 
               "archive/tom_and_jerry/tom_and_jerry/tom_jerry_0",
               "archive/tom_and_jerry/tom_and_jerry/tom_jerry_1"]
# print(os.listdir("archive/tom_and_jerry/tom_and_jerry/"))
# directories = sorted(os.listdir("archive/tom_and_jerry/tom_and_jerry/"))
# print(sorted(os.listdir("archive/tom_and_jerry/tom_and_jerry/")))

classToIndex = {"jerry_only" : 0, "tom_only" : 1, "neither" : 2, "tom_and_jerry" : 3}

train_ds = tf.keras.utils.image_dataset_from_directory(
  "archive/tom_and_jerry/tom_and_jerry/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(200, 400),
  batch_size=64)
print("train,", train_ds)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "archive/tom_and_jerry/tom_and_jerry/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(200, 400),
  batch_size=64)


# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(train_ds.class_names[labels[i]])
#     plt.axis("off")
# plt.show()


# print(train_ds.class_names)
# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break 

# for image_batch, labels_batch in val_ds:
#   print("val:", image_batch.shape)
#   print("val:", labels_batch.shape)
#   break 

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

tomJerryNetwork = Sequential([
    tf.keras.layers.Rescaling(1./255),
    Conv2D(filters=32, kernel_size=(4,4), strides=(2, 2), input_shape=[200,400,3], padding='valid', activation="relu"),
    MaxPooling2D(pool_size=(4,4)), 
    Conv2D(filters=64, kernel_size=(2,2), strides=(2, 2), padding='valid', activation="relu"),
    MaxPooling2D(pool_size=(4,4)), 
    # Conv2D(filters=64, kernel_size=(3,3), strides=(2, 2), padding='valid', activation="relu"),
    # MaxPooling2D(pool_size=(4,4)), 
    Flatten(), 
    Dropout(0.5), 
    Dense(1000, activation="relu"), 
    Dropout(0.5), 
    Dense(1000, activation="relu"), 
    Dropout(0.5), 
    Dense(4)
])
# tomJerryNetwork = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(4)
# ])

tomJerryNetwork.compile(optimizer='adam',
            #   loss=tf.keras.losses.binary_crossentropy,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])
# print("HERE!!!!:", tomJerryNetwork.summary())
callback = EarlyStopping(monitor="val_loss", patience=2)
history = tomJerryNetwork.fit(train_ds, 
                              epochs=60, 
                              batch_size=64, 
                              validation_data=val_ds, 
                              callbacks = callback)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = "val_loss")
plt.xlabel('loss')
plt.ylabel('val_loss')
plt.legend()
plt.show()
print("should be plotted")

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

tomJerryNetwork.save("tom_jerry_classifier_compressed_6.keras")