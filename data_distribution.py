# SEP 740 Final Project - Aerial Perspective Object Detection

# Authors: 
#   Jukai Hu (400485702)
#   Ray Albert Pangilinan (400065058)
#   Luke Vanden Broek (400486889)

# Data Distribution

import os.path
cwd = os.getcwd()

import matplotlib. pyplot as plt
import matplotlib.image as image
import numpy as np
import pandas as pd

labels_dir = cwd + "/../dataset/class_dict_seg.csv"
images_dir = cwd + "/../dataset/dataset/semantic_drone_dataset/original_images/"
masks_dir = cwd + "/../dataset/RGB_color_image_masks/RGB_color_image_masks/"

labels = pd.read_csv(labels_dir)
labels.insert(4, "count", 0)

print(labels)

images_list = os.listdir(images_dir)
masks_list = os.listdir(masks_dir)

images_list = list(map(lambda img : images_dir + img, images_list))
masks_list = list(map(lambda img : masks_dir + img, masks_list))

# img = image.imread(masks_list[0])
# print(np.array(img) * 255)
# plt.imshow(img)
# plt.show()
