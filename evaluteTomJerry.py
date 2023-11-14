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
import random

model = keras.models.load_model('tom_jerry_classifier_compressed_5.keras')

val_ds = tf.keras.utils.image_dataset_from_directory(
  "archive/tom_and_jerry/tom_and_jerry/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(200, 400),
  batch_size=32)

loss, accuracy = model.evaluate(val_ds) 

# Make predictions on the validation dataset
allImages = []
predicted_labels = []
true_labels = []

for images, labels in val_ds:
    allImages.extend(images)
    true_labels.extend(labels.numpy())
    predictions = model.predict(images)
    predicted_labels.extend(np.argmax(predictions, axis=1))

# allImages = np.array(allImages)

# Find mislabelled images
mislabelled_indices = [i for i in range(len(true_labels)) if true_labels[i] != predicted_labels[i]]
random.shuffle(mislabelled_indices)
# Plot mislabelled images with their incorrect labels
plt.figure(figsize=(12, 12))
for i, idx in enumerate(mislabelled_indices[:12]):  # Change the range to display more mislabelled images
    ax = plt.subplot(3, 4, i + 1)
    print("idx:", idx, "len(images):", len(images))
    plt.imshow(allImages[idx].numpy().astype("uint8"))
    plt.title(f"True: {val_ds.class_names[true_labels[idx]]}\nPredicted: {val_ds.class_names[predicted_labels[idx]]}")
    plt.axis("off")
plt.show()