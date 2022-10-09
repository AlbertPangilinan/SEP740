# SEP 740 Final Project - Aerial Perspective Object Detection

# Authors: 
#   Jukai Hu (400485702)
#   Ray Albert Pangilinan (400065058)
#   Luke Vanden Broek (400486889)

# Random Forest Classifier

# Imports

import os.path
cwd = os.getcwd()

import matplotlib. pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# Reading color image mask files from disk

images_dir = cwd + "/../dataset/dataset/semantic_drone_dataset/original_images/"
labels_dir = cwd + "/../dataset/dataset/semantic_drone_dataset/label_images_semantic/"
masks_dir = cwd + "/../dataset/RGB_color_image_masks/RGB_color_image_masks/"

images_paths = os.listdir(images_dir)
images_paths.sort()
image_0_filename = images_paths[0]

labels_paths = os.listdir(labels_dir)
labels_paths.sort()
label_0_filename = labels_paths[0]

masks_paths = os.listdir(masks_dir)
masks_paths.sort()
mask_0_filename = masks_paths[0]

images_paths = list(map(lambda img : images_dir + img, images_paths))
labels_paths = list(map(lambda img : labels_dir + img, labels_paths))
masks_paths = list(map(lambda mask : masks_dir + mask, masks_paths))

images_paths.sort()
labels_paths.sort()
masks_paths.sort()

images = list(map(lambda img : np.array(Image.open(img)), images_paths[:5]))
labels = list(map(lambda img : np.array(Image.open(img)), labels_paths[:5]))
masks = list(map(lambda mask : np.array(Image.open(mask)), masks_paths[:5]))

image_0 = images[0]
label_0 = labels[0]
mask_0 = masks[0]

image_0_reshape = image_0.reshape(-1, image_0.shape[-1])
label_0_flatten = label_0.flatten()

print(np.amin(labels[1]))
print(np.amax(labels[1]))

fig, ax = plt.subplots(1, 3)

ax[0].imshow(image_0)
ax[0].set_title("Image")

ax[1].imshow(label_0)
ax[1].set_title("Labels")

ax[2].imshow(masks[0])
ax[2].set_title("Mask")

plt.show()

rf = RandomForestClassifier(max_depth=2)
rf.fit(image_0_reshape, label_0_flatten)

print("model fitted")

image_1 = images[1]
label_1 = labels[1]

image_1_reshape = image_1.reshape(-1, image_1.shape[-1])
label_1_flatten = label_1.flatten()

label_1_predicted = rf.predict(image_1_reshape)
label_1_predicted_reshape = np.reshape(label_1_predicted, (4000, -1))

print("image 1 predicted")

fig, ax = plt.subplots(1, 3)

ax[0].imshow(image_1)
ax[0].set_title("Image")

ax[1].imshow(label_1)
ax[1].set_title("Labels")

ax[2].imshow(label_1_predicted_reshape)
ax[2].set_title("Predicted")

plt.show()
