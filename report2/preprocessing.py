# SEP 740 Final Project - Aerial Perspective Object Detection

# Authors: 
#   Jukai Hu (400485702)
#   Ray Albert Pangilinan (400065058)
#   Luke Vanden Broek (400486889)

# Data Distribution

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
rgb_tuples = [tuple(col / 255) for col in rgb_arr]


# Plotting colour legend

# fig, axs = plt.subplots(6, 4)

# for i, ax in enumerate(fig.axes):
#   ax.imshow([[rgb_tuples[i]]])
#   ax.set_title(classes[i])
#   ax.axis("off")

# fig.suptitle("Colour Legend")
# plt.show()


# Reading image files from disk

images_dir = cwd + "/../../dataset/dataset/semantic_drone_dataset/original_images/"
images_paths = os.listdir(images_dir)
images_paths.sort()
images_paths = np.array_split(images_paths, 4)

labels_dir = cwd + "/../../dataset/dataset/semantic_drone_dataset/label_images_semantic/"
labels_paths = os.listdir(labels_dir)
labels_paths.sort()
labels_paths = np.array_split(labels_paths, 4)



# Dividing original images into smaller images of equal size 

def divide_image(image, tile_height, tile_width):
    if (len(image.shape) == 3):
      image_height, image_width, channels = image.shape

      tiles_arr = image.reshape(image_height // tile_height,
                                  tile_height,
                                  image_width // tile_width,
                                  tile_width,
                                  channels)

      tiles_arr = tiles_arr.swapaxes(1, 2)
      num_tiles = (image_height // tile_height) * (image_width // tile_width)
      tiles_arr = tiles_arr.reshape(num_tiles, tile_height, tile_width, channels)

    elif (len(image.shape) == 2):
      image_height, image_width = image.shape

      tiles_arr = image.reshape(image_height // tile_height,
                                  tile_height,
                                  image_width // tile_width,
                                  tile_width)

      tiles_arr = tiles_arr.swapaxes(1, 2)
      num_tiles = (image_height // tile_height) * (image_width // tile_width)
      tiles_arr = tiles_arr.reshape(num_tiles, tile_height, tile_width)

    return tiles_arr

image = np.array(Image.open(images_dir + images_paths[0][0]))
image_tiles_4 = divide_image(image, 2000, 3000)
image_tiles_16 = divide_image(image, 1000, 1500)

plt.imshow(image)
plt.axis("off")
plt.title("Original Image")

fig, ax = plt.subplots(2, 2)

for i, ax in enumerate(fig.axes):
  ax.imshow(image_tiles_4[i])
  ax.axis("off")

fig.suptitle("Image / 4")

fig, ax = plt.subplots(4, 4)

for i, ax in enumerate(fig.axes):
  ax.imshow(image_tiles_16[i])
  ax.axis("off")

fig.suptitle("Image / 16")

plt.show()


# Dividing original label images into smaller images of equal size (16 tiles seems to cause errors in labels)

label = np.array(Image.open(labels_dir + labels_paths[0][0]))
label_tiles_4 = divide_image(label, 2000, 3000)
label_tiles_16 = divide_image(label, 1000, 1500)

plt.imshow(label)
plt.axis("off")
plt.title("Original Label")

fig, ax = plt.subplots(2, 2)

for i, ax in enumerate(fig.axes):
  ax.imshow(label_tiles_4[i])
  ax.axis("off")

fig.suptitle("Label / 4")

fig, ax = plt.subplots(4, 4)

for i, ax in enumerate(fig.axes):
  ax.imshow(label_tiles_16[i])
  ax.axis("off")

fig.suptitle("Label / 16")

plt.show()


# Calculating data distribution

# histogram, bin_edges = [], []

# for i in range(len(images_paths)):
#   print("Calculating distribution for images " + str(i * 100 + 1) + "-" + str((i + 1) * 100) + "...")
#   images = [np.array(Image.open(images_dir + image)) for image in images_paths[i]]
#   if (i == 0):
#     histogram, bin_edges = np.histogram(images, bins=24, range=(0, 24))
#   else:
#     histogram += np.histogram(images, bins=24, range=(0, 24))[0]


# Plotting pixel distribution

# plt.bar(bin_edges[0:-1], histogram, color=rgb_tuples, width=1)
# plt.title("Pixel Class Distribution")
# plt.xticks(range(len(classes)), classes, rotation=90)
# plt.xlabel("Class")
# plt.ylabel("Count", rotation=90)
# plt.show()
