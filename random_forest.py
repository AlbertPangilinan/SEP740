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
# masks_dir = cwd + "/../dataset/RGB_color_image_masks/RGB_color_image_masks/"

images_paths = os.listdir(images_dir)
images_paths.sort()
image_0_filename = images_paths[0]

labels_paths = os.listdir(labels_dir)
labels_paths.sort()
label_0_filename = labels_paths[0]

# masks_paths = os.listdir(masks_dir)
# masks_paths.sort()
# mask_0_filename = masks_paths[0]

images_paths = list(map(lambda img : images_dir + img, images_paths))
labels_paths = list(map(lambda img : labels_dir + img, labels_paths))
# masks_paths = list(map(lambda mask : masks_dir + mask, masks_paths))

images_paths.sort()
labels_paths.sort()
# masks_paths.sort()

images = list(map(lambda img : np.array(Image.open(img)), images_paths[:10]))
labels = list(map(lambda img : np.array(Image.open(img)), labels_paths[:10]))
# masks = list(map(lambda mask : np.array(Image.open(mask)), masks_paths[:10]))

rf = RandomForestClassifier(max_depth=2)

for i in range(len(images) - 1):

  image = images[i]
  label = labels[i]
  # mask = masks[0]

  image_reshape = image.reshape(-1, image.shape[-1])
  label_flatten = label.flatten()

  rf.fit(image_reshape, label_flatten)

  print("image " + str(i) + " fitted to model")

print("all images fitted")

image_predict = images[-1]
label_predict = labels[-1]

image_predict_reshape = image_predict.reshape(-1, image_predict.shape[-1])
label_predict_flatten = label_predict.flatten()

label_predict_result = rf.predict(image_predict_reshape)
label_predict_result_reshape = np.reshape(label_predict_result, (4000, -1))

print("image 1 predicted")

fig, ax = plt.subplots(1, 3)

ax[0].imshow(image_predict)
ax[0].set_title("Image")

ax[1].imshow(label_predict)
ax[1].set_title("Labels")

ax[2].imshow(label_predict_result_reshape)
ax[2].set_title("Predicted")

plt.show()
