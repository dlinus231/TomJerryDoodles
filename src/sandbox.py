import os

print(os.getcwd())
directories = os.listdir("archive/tom_and_jerry/tom_and_jerry/")
directories.sort()

# verifying that all images were edge-filtered
for folderName in directories: 
    print(folderName, len(os.listdir("archive/tom_and_jerry/tom_and_jerry/" + folderName)))