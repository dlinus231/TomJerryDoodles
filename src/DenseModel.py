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
from PIL import Image
import random


# returns a list of tuples where the elements are the edge-filtered image's path
# paired with the original image's path
def getFilePaths(): 
    dataset = []
    for inputDir, outputDir in zip(inputDirectories, outputDirectories): 
        fileNames = os.listdir(inputDir)
        for fileName in fileNames: 
            inputImagePath = os.path.join(inputDir, fileName)
            outputImagePath = os.path.join(outputDir, fileName)
            dataset.append((inputImagePath, outputImagePath))
    return dataset

# takes in two paths of paired images, 
# optionally, one can specify the desired resolution of the image
# whether to display the retrieved images, and whether to convert the images to RGB
# will return the two images as numpy arrays, normalized by dividing by 255
def loadImage(input_file_path, output_file_path, 
              image_shape=(400, 200), showImages=False, isRGB = True): 

    input_image = Image.open(input_file_path).resize(image_shape)
    output_image = Image.open(output_file_path).resize(image_shape)

    # convert to greyscale if desired
    if not isRGB: 
        input_image = input_image.convert("L")
        output_image = output_image.convert("L")

    # if desired, display the retrieved images
    if showImages: 
        # Display images for testing
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(input_image))
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(output_image))
        plt.title("Ground Truth Image")
        plt.axis('off')

        plt.show()
    return np.array(input_image)/255.0, np.array(output_image)/255.0

print("Working Directory:", os.getcwd())

# directories containing desired output, i.e. our y-labels, the ground truth 
outputDirectories = ["archive/tom_and_jerry/tom_and_jerry/jerry", 
                     "archive/tom_and_jerry/tom_and_jerry/tom", 
                     "archive/tom_and_jerry/tom_and_jerry/tom_jerry_0",
                     "archive/tom_and_jerry/tom_and_jerry/tom_jerry_1"]

# directories containing edge-filtered images
inputDirectories = ["archive/tom_and_jerry/tom_and_jerry_edge_detected/jerry_edge_detected", 
                    "archive/tom_and_jerry/tom_and_jerry_edge_detected/tom_edge_detected", 
                    "archive/tom_and_jerry/tom_and_jerry_edge_detected/tom_jerry_0_edge_detected",
                    "archive/tom_and_jerry/tom_and_jerry_edge_detected/tom_jerry_1_edge_detected"]


# Creation of training and testing datasets 
paths = getFilePaths()
result = loadImage(paths[0][0], paths[0][1], image_shape=(100,100), showImages=True, isRGB=False)
print(np.min(result[0]), np.max(result[0]))
print(np.min(result[1]), np.max(result[1]))


# randomize paths so images are randomly allocated into training/testing datasets
random.shuffle(paths)
validation_split = 0.2

x_train, y_train = [], []
x_test, y_test = [], []

imagesProcessed = 0
totalImagePairs = len(paths)
for inputImagePath, outputImagePath in paths: 
    inputImage, imageLabel = loadImage(inputImagePath, outputImagePath)

    # if validation_split = 0.2, put 80% of images into the training dataset
    if imagesProcessed / totalImagePairs > 1.0 - validation_split: 
        x_train.append(inputImage)
        y_train.append(imageLabel)
    # else, put the images into the testing dataset
    else: 
        x_test.append(inputImage)
        y_train.append(imageLabel)
        
    imagesProcessed += 1

# following guidance from the below link
# https://www.tensorflow.org/tutorials/load_data/numpy
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

    

# print(np.min(result[0]), np.max(result[0]))
# print(np.min(result[1]), np.max(result[1]))

# print(np.min(result[0]/255.0), np.max(result[0]/255.0))
# print(np.min(result[1]/255.0), np.max(result[1]/255.0))

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   "archive/tom_and_jerry/tom_and_jerry/",
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(200, 400),
#   batch_size=64)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#   "archive/tom_and_jerry/tom_and_jerry/",
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(200, 400),
#   batch_size=64)
# print("train,", train_ds)