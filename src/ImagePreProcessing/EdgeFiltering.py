"""
    The purpose of this file is to turn images into edge_detected versions using OpenCV
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2 as cv
import matplotlib.pyplot as plt
import random
import tqdm

# print("HELLO:", os.getcwd())

base_path = "ProgressPhotos/"
imageFolders = os.listdir(base_path)

edge_string = "_edge_detected"
for imageFolder in imageFolders: 
    # name for destination folder
    dstFolderName = imageFolder + edge_string
    dstFolderPath = f"{base_path}{dstFolderName}"

    # make destination folder
    os.mkdir(dstFolderPath)
        
    # get all images in the current folder
    images = os.listdir(base_path + imageFolder)

    # iterate over all images, and put their edge filters in the destination folder
    for imageName in images: 
        print(base_path + imageFolder + "/" + imageName)
        image = cv.imread(base_path + imageFolder + "/" + imageName)
        
        # Create the edge-detected image
        edges = cv.Canny(image, 100, 200)

        # could've explore thresholding the edge-filtered images, to remove noise within images/edges

        # edges = cv.resize(edges, (256, 256))
        # _, edges_thresholded = cv.threshold(edges, 5, 255, cv.THRESH_BINARY)
        
        
        # Write it to the proper folder
        cv.imwrite(dstFolderPath + "/" + imageName, edges)
print("Folders successfully created and filled")
