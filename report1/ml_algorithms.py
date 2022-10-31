# SEP 740 Final Project - Aerial Perspective Object Detection

# Authors: 
#   Jukai Hu (400485702)
#   Ray Albert Pangilinan (400065058)
#   Luke Vanden Broek (400486889)

# ML Algorithms (Random Forest, K-nearest Neighbours, Naive Bayes)

# Imports

import os.path
cwd = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score


# Reading raw image and class label files from disk

images_dir = cwd + "/../../dataset/dataset/semantic_drone_dataset/original_images/"
labels_dir = cwd + "/../../dataset/dataset/semantic_drone_dataset/label_images_semantic/"

images_paths = os.listdir(images_dir)
images_paths.sort()

labels_paths = os.listdir(labels_dir)
labels_paths.sort()

images = [np.array(Image.open(images_dir + img)) for img in images_paths[:10]]
labels = [np.array(Image.open(labels_dir + img)) for img in labels_paths[:10]]


# Defining ML algorithms

rf = RandomForestClassifier(max_depth=2)
knn = KNeighborsClassifier(n_neighbors = 3)
nb = GaussianNB()


# Fitting each image to each model

for i in range(len(images) - 1):
  print("Fitting image " + str(i) + "...")
  image = images[i]
  label = labels[i]

  image_reshape = image.reshape(-1, image.shape[-1])
  label_flatten = label.flatten()

  rf.fit(image_reshape, label_flatten)
  knn.fit(image_reshape, label_flatten)
  nb.fit(image_reshape, label_flatten)

  print("Image " + str(i) + " fitted to all models.")


# Predicting labels for one image using trained models

print("Predicting labels for image " + str(len(images) - 1) + "...")

image_predict = images[-1]
label_predict = labels[-1]

image_predict_reshape = image_predict.reshape(-1, image_predict.shape[-1])
label_predict_flatten = label_predict.flatten()

label_predict_rf = rf.predict(image_predict_reshape)
label_predict_knn = knn.predict(image_predict_reshape)
label_predict_nb = nb.predict(image_predict_reshape)

label_predict_rf_reshape = np.reshape(label_predict_rf, (4000, -1))
label_predict_knn_reshape = np.reshape(label_predict_knn, (4000, -1))
label_predict_nb_reshape = np.reshape(label_predict_nb, (4000, -1))

rf_accuracy = round(accuracy_score(label_predict_flatten, label_predict_rf) * 100, 2)
knn_accuracy = round(accuracy_score(label_predict_flatten, label_predict_knn) * 100, 2)
nb_accuracy = round(accuracy_score(label_predict_flatten, label_predict_nb) * 100, 2)

rf_f1 = round(f1_score(label_predict_flatten, label_predict_rf, average="weighted") * 100, 2)
knn_f1 = round(f1_score(label_predict_flatten, label_predict_knn, average="weighted") * 100, 2)
nb_f1 = round(f1_score(label_predict_flatten, label_predict_nb, average="weighted") * 100, 2)

print("Labels for image " + str(len(images) - 1) + " predicted.")


# Plotting prediction results

fig, ax = plt.subplots(2, 3)

ax[0, 0].imshow(image_predict)
ax[0, 0].set_title("Image")
ax[0, 0].axis("off")

ax[0, 1].imshow(label_predict)
ax[0, 1].set_title("Labels")
ax[0, 1].axis("off")

ax[0, 2].axis("off")

ax[1, 0].imshow(label_predict_rf_reshape)
ax[1, 0].set_title("Random Forest\nAccuracy: " + str(rf_accuracy) + "%\nF1: "+ str(rf_f1) + "%")
ax[1, 0].axis("off")

ax[1, 1].imshow(label_predict_knn_reshape)
ax[1, 1].set_title("K-nearest Neighbours\nAccuracy: " + str(knn_accuracy) + "%\nF1: "+ str(knn_f1) + "%")
ax[1, 1].axis("off")

ax[1, 2].imshow(label_predict_nb_reshape)
ax[1, 2].set_title("Naive Bayes\nAccuracy: " + str(nb_accuracy) + "%\nF1: "+ str(nb_f1) + "%")
ax[1, 2].axis("off")

plt.show()
