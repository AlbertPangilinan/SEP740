# SEP 740 Final Project - Aerial Perspective Object Detection

# Authors: 
#   Jukai Hu (400485702)
#   Ray Albert Pangilinan (400065058)
#   Luke Vanden Broek (400486889)

# Data Distribution (113.png)

import os.path
cwd = os.getcwd()

import matplotlib. pyplot as plt
import matplotlib.image as image
import numpy as np
import pandas as pd

labels_dir = cwd + "/../dataset/class_dict_seg.csv"
masks_dir = cwd + "/../dataset/RGB_color_image_masks/RGB_color_image_masks/"

distribution = pd.read_csv(labels_dir)
rgb_arr = distribution.to_numpy()[:, 1:]
distribution.insert(4, "count", 0)

masks_list = os.listdir(masks_dir)
mask_0_filename = masks_list[0]

masks_list = list(map(lambda mask : masks_dir + mask, masks_list))
mask_0_rgb = np.array(image.imread(masks_list[0])) * 255

for row in mask_0_rgb:
  for pixel in row[:1]:
    i, = np.where(np.prod(rgb_arr == pixel, axis=-1))
    distribution.at[i[0], "count"] += 1

print(distribution)
labels = distribution.to_numpy()[:, :1].flatten()
rgb_tuples = list(map(lambda col : tuple(col / 255), rgb_arr))

plt.bar(range(len(labels)), distribution.loc[:, "count"], color=rgb_tuples)
plt.title("Pixel Class Distribution for " + mask_0_filename)
plt.xticks(range(len(labels)), labels)
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
