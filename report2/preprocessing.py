# SEP 740 Final Project - Aerial Perspective Object Detection

# Authors: 
#   Jukai Hu (400485702)
#   Ray Albert Pangilinan (400065058)
#   Luke Vanden Broek (400486889)

# Data Preprocessing - 000.png (normalization and image resizing)

# Imports

import os.path
cwd = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd


# Reading image and label files from disk (4 batches of 100 images each)

images_dir = cwd + "/../../dataset/dataset/semantic_drone_dataset/original_images/"
images_paths = os.listdir(images_dir)
images_paths.sort()
images_paths = np.array_split(images_paths, 4)

labels_dir = cwd + "/../../dataset/dataset/semantic_drone_dataset/label_images_semantic/"
labels_paths = os.listdir(labels_dir)
labels_paths.sort()
labels_paths = np.array_split(labels_paths, 4)


# divide_image() helper function

def divide_image(image, tile_height, tile_width):
  # RGB images
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

  # Single channel images (labels)
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


# Normalizing image pixels and dividing original images into 4 images of equal size 

image_name = images_paths[0][0]
image = np.array(Image.open(images_dir + image_name)) / 255
image_tiles = divide_image(image, 2000, 3000)


# Plotting original image and quartered image for comparison

plt.imshow(image)
plt.axis("off")
plt.title(image_name + " - Original")

fig, ax = plt.subplots(2, 2)

for i, ax in enumerate(fig.axes):
  ax.imshow(image_tiles[i])
  ax.axis("off")

fig.suptitle(image_name + " - 4 tiles")

plt.show()


# Dividing original label images into 4 images of equal size

label_name = labels_paths[0][0]
label = np.array(Image.open(labels_dir + label_name))
label_tiles = divide_image(label, 2000, 3000)


# Plotting original label image and quartered label image for comparison

plt.imshow(label)
plt.axis("off")
plt.title(label_name + " - Original")

fig, ax = plt.subplots(2, 2)

for i, ax in enumerate(fig.axes):
  ax.imshow(label_tiles[i])
  ax.axis("off")

fig.suptitle(label_name + " - 4 tiles")

plt.show()
