import os
import random
import shutil

print(os.getcwd())

print(os.listdir())

base_path = "tom_jerry_object_detected/"
allFileNames = []
for folder in os.listdir(base_path): 
    allFileNames.extend(os.listdir(base_path + folder))

# print(allFileNames[:10])
random.shuffle(allFileNames)
# print(allFileNames[:10])

splitIndex = int(len(allFileNames) * 0.8)

trainingImageNames = allFileNames[:splitIndex] 
testingImageNames = allFileNames[splitIndex:]

print(len(trainingImageNames))
print(len(testingImageNames))

imagesMade = 0
for trainImage in trainingImageNames: 
    # get src and destination for both edge_detected and original
    edge_src_path =     "tom_jerry_object_detected/ObjectDetectedDataset_edge_detected/" + trainImage
    original_src_path = "tom_jerry_object_detected/ObjectDetectedDataset/" + trainImage

    edge_dst_path =     "ObjectDetectedDataset/train/edge_detected/" + trainImage
    original_dst_path = "ObjectDetectedDataset/train/tom_jerry_images/" + trainImage

    
    shutil.copyfile(edge_src_path, edge_dst_path)
    # print("Edge detected copied", trainImage)

    shutil.copyfile(original_src_path, original_dst_path)
    # print("Original copied", trainImage)
    imagesMade += 1

for testImage in testingImageNames: 
    # get src and destination for both edge_detected and original
    edge_src_path =     "tom_jerry_object_detected/ObjectDetectedDataset_edge_detected/" + testImage
    original_src_path = "tom_jerry_object_detected/ObjectDetectedDataset/" + testImage

    edge_dst_path =     "ObjectDetectedDataset/test/edge_detected/" + testImage
    original_dst_path = "ObjectDetectedDataset/test/tom_jerry_images/" + testImage
    
    shutil.copyfile(edge_src_path, edge_dst_path)
    # print("Edge detected copied", testImage)

    shutil.copyfile(original_src_path, original_dst_path)
    # print("Original copied", testImage)
    imagesMade += 1
    
print(imagesMade, "images made")