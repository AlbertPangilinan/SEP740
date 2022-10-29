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

# Reading class reference table and labelled image files from disk

classes_dir = cwd + "/../../dataset/class_dict_seg.csv"
labels_dir = cwd + "/../../dataset/dataset/semantic_drone_dataset/label_images_semantic/"

distribution = pd.read_csv(classes_dir)
rgb_arr = distribution.to_numpy()[:, 1:]
distribution.insert(4, "count", 0)

labels_paths = os.listdir(labels_dir)
labels_paths.sort()
label_0_filename = labels_paths[0]

labels_paths = list(map(lambda label : labels_dir + label, labels_paths))
label_0 = np.array(Image.open(labels_dir + label_0_filename))


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


# Plotting pixel distribution

for pixel in label_0.flatten():
  distribution.at[pixel, "count"] += 1

plt.bar(range(len(labels)), distribution.loc[:, "count"], color=rgb_tuples, width=1)
plt.title("Pixel Class Distribution for " + label_0_filename)
plt.xticks(range(len(labels)), labels, rotation=90)
plt.xlabel("Class")
plt.ylabel("Count", rotation=90)
plt.show()
