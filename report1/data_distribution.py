# SEP 740 Final Project - Aerial Perspective Object Detection

# Authors: 
#   Jukai Hu (400485702)
#   Ray Albert Pangilinan (400065058)
#   Luke Vanden Broek (400486889)

# Data Distribution (000.png)

# Imports

import os.path
cwd = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

# Reading color image mask files from disk

labels_dir = cwd + "/../dataset/class_dict_seg.csv"
masks_dir = cwd + "/../dataset/RGB_color_image_masks/RGB_color_image_masks/"

distribution = pd.read_csv(labels_dir)
rgb_arr = distribution.to_numpy()[:, 1:]
distribution.insert(4, "count", 0)

masks_paths = os.listdir(masks_dir)
masks_paths.sort()
mask_0_filename = masks_paths[0]

masks_paths = list(map(lambda mask : masks_dir + mask, masks_paths))
mask_0_rgb = np.array(Image.open(mask_0_filename)) * 255

# Plotting colour legend

labels = distribution.to_numpy()[:, :1].flatten()
rgb_tuples = list(map(lambda col : tuple(col / 255), rgb_arr))

fig, axs = plt.subplots(6, 4)

for i, ax in enumerate(fig.axes):
  ax.imshow([[rgb_tuples[i]]])
  ax.set_title(labels[i])
  ax.axis("off")

fig.suptitle("Colour Legend")
plt.show()

# Counting pixel distribution

for row in mask_0_rgb:
  for pixel in row:
    i, = np.where(np.prod(rgb_arr == pixel, axis=-1))
    distribution.at[i[0], "count"] += 1

print(distribution)

# Plotting pixel distribution

plt.bar(range(len(labels)), distribution.loc[:, "count"], color=rgb_tuples, width=1)
plt.title("Pixel Class Distribution for " + mask_0_filename)
plt.xticks(range(len(labels)), labels, rotation=90)
plt.xlabel("Class")
plt.ylabel("Count", rotation=90)
plt.show()
