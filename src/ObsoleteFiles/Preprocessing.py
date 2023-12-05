import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2 as cv
import matplotlib.pyplot as plt
import random
import tqdm

# print("HELLO:", os.getcwd())

base_path = "archive/tom_and_jerry/tom_and_jerry/"
imageFolders = os.listdir(base_path)

edge_string = "_edge_detected"
for imageFolder in imageFolders: 
    # name for destination folder
    dstFolderName = imageFolder + edge_string
    dstFolderPath = f"archive/tom_and_jerry/tom_and_jerry_edge_detected/{dstFolderName}"

    # make destination folder
    os.mkdir(dstFolderPath)
        
    # get all images in the current folder
    images = os.listdir(base_path + imageFolder)

    # iterate over all images, and put their edge filters in the destination folder
    for imageName in images: 
        print(base_path + imageFolder + "/" + imageName)
        image = cv.imread(base_path + imageFolder + "/" + imageName)
        
        # consider cropping, but images all need different croppings so it's inconsistent
        # only crop if we will brutally cut down images such that no images have borders
        # image = image[96:660, :]

        # Create the edge-detected image
        edges = cv.Canny(image, 100, 250)
        
        # Write it to the proper folder
        cv.imwrite(dstFolderPath + "/" + imageName, edges)
print("Folders successfully created and filled")
