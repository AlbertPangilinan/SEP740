# SEP 740 Final Project - Aerial Perspective Object Detection

# Authors: 
#   Jukai Hu (400485702)
#   Ray Albert Pangilinan (400065058)
#   Luke Vanden Broek (400486889)

# Data Distribution (runs out of memory when loading >275 images, using 16GB RAM)

# Imports

import os.path
cwd = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd


# Reading class reference table from disk

classes_dir = cwd + "/../../dataset/class_dict_seg.csv"
classes_pd = pd.read_csv(classes_dir)
classes = classes_pd.to_numpy()[:, :1].flatten()
rgb_arr = classes_pd.to_numpy()[:, 1:]
rgb_tuples = list(map(lambda col : tuple(col / 255), rgb_arr))


# Plotting colour legend

fig, axs = plt.subplots(6, 4)

for i, ax in enumerate(fig.axes):
  ax.imshow([[rgb_tuples[i]]])
  ax.set_title(classes[i])
  ax.axis("off")

fig.suptitle("Colour Legend")
plt.show()


# Reading labelled image files from disk

labels_dir = cwd + "/../../dataset/dataset/semantic_drone_dataset/label_images_semantic/"
labels_paths = os.listdir(labels_dir)
labels_paths.sort()
labels_paths = np.array_split(labels_paths, 4)


# Calculating data distribution

print("Calculating distribution for images 1-100...")
labels = [np.array(Image.open(labels_dir + label)) for label in labels_paths[0]]
histogram, bin_edges = np.histogram(labels, bins=24, range=(0, 24))

for i in range(1, 4):
  print("Calculating distribution for images " + str(i * 100 + 1) + "-" + str((i + 1) * 100) + "...")
  labels = [np.array(Image.open(labels_dir + label)) for label in labels_paths[i]]
  histogram += np.histogram(labels, bins=24, range=(0, 24))[0]



# Plotting pixel distribution


plt.bar(bin_edges[0:-1], histogram, color=rgb_tuples, width=1)
plt.title("Pixel Class Distribution")
plt.xticks(range(len(classes)), classes, rotation=90)
plt.xlabel("Class")
plt.ylabel("Count", rotation=90)
plt.show()
