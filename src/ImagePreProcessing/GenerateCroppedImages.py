""" 
    The purpose of this file is to take the original Tom and Jerry Dataset and create cropped, square images based on
    the rectangles drawn by the Hugging Face classifier
    
"""
from transformers import pipeline
import os
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow import keras
import os
from enum import Enum
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import random
from tqdm import tqdm
from multiprocessing import Process, Manager
import time

print("cwd", os.getcwd())

# Global Vars
classifier = pipeline(model="facebook/detr-resnet-50")
base_path = "archive/tom_and_jerry/tom_and_jerry"
folders = ["jerry", "tom", "tom_jerry_1"]

def process_images(image_list, result_list, folder): 
    # go over each image
    # for i in tqdm(range(5)): 
    for i in tqdm(range(len(image_list))): 
        print("starting image")
        currFrame = image_list[i]
        path_to_image = f"{base_path}/{folder}/{currFrame}"

        im = Image.open(path_to_image)
        res = classifier(im)

        # if boxes were predicted, created cropped images from each box
        if res: 
            boxes = []
            for j in range(len(res)): 
                boxCoords = res[j]["box"]["xmin"], res[j]["box"]["ymin"], res[j]["box"]["xmax"], res[j]["box"]["ymax"]
                boxes.append(boxCoords)
            # draw the box
            for j in range(len(boxes)): 
                xmin, ymin, xmax, ymax = boxes[j]
                width, height = boxCoords[2]-boxCoords[0], boxCoords[3]-boxCoords[1]
                croppedImage = im.crop((xmin, ymin, xmax, ymax))
                result_list.append((folder, croppedImage))
        # print("finished with image")
        # print("result from process:", result_list)

# Create a square image with padding (so as to not augment the image)
# This function is originally from: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def saveImages(croppedImages): 
    # print("croppedImages:", croppedImages)
    squarePaddedImgs = []
    for j in range(len(croppedImages)): 
        folder, image = croppedImages[j]
        resizedImg = expand2square(image, (0, 0, 0))
        squarePaddedImgs.append((folder, resizedImg))
        

    # TODO: Retrieve original frame's name, and append an underscore, e.g. "frame1035_1.jpg"
    for j, folderImage in enumerate(squarePaddedImgs): 
        folder, image = folderImage
        image = image.resize((256, 256))
        image.save(f"ObjectDetectedDataset/{folder}{j}.jpg")

if __name__ == "__main__": 
    # it turned out parallelizing this was slower because the YOLO classifier
    # is already in parallel :/ 
    num_threads = 1
    with Manager() as manager:
        croppedImages = manager.list()  
        processes = []

        for folder in folders:
            currentFolder = base_path + "/" + folder
            pictures = os.listdir(currentFolder)

            # Split the work as before and create processes
            split_images = np.array_split(pictures, num_threads)
            for i in range(num_threads):
                p = Process(target=process_images, args=(split_images[i], croppedImages, folder))
                processes.append(p)
                p.start()

            # Wait for all processes to complete
            for p in processes:
                p.join()

        # Now croppedImages will have the results
        saveImages(list(croppedImages))