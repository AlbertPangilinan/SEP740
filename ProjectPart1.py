#!/usr/bin/env python
# coding: utf-8

# In[5]:


#importing modules and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import time

import cv2
import os
from os import listdir
import PIL
import PIL.Image
from PIL import Image
import glob

import tensorflow as tf
print(tf.__version__)
print(cv2.__version__)

#import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC  
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#set up path for loading file
import os.path
cwd = os.getcwd()
print(cwd)


# In[7]:


#testing on indiviudal image loading via cv2
path=cwd + "\dataset\semantic_drone_dataset\original_images\\031.jpg"
print(path)
img = cv2.imread(path, cv2.IMREAD_COLOR)

plt.imshow(img)
img.shape




# In[8]:


RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(RGB_img)


# In[ ]:





# In[9]:


#loading all images with glob
path1 = cwd + "\dataset\semantic_drone_dataset\original_images\*.jpg"

image_list = []
for filename in glob.glob(path1):
    im=Image.open(filename)
    image_list.append(im)


# In[13]:


plt.imshow(image_list[0])


# In[17]:


#csv histogram
rgb_data=cwd+"\class_dict_seg.csv"
print(rgb_data)
rgb_data=pd.read_csv(rgb_data)
print("Data in Table form")
print(rgb_data)


# In[71]:


plt.figure(figsize=(12,12))
rgb_val=rgb_data.values
plt.subplot(3, 1, 1)
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 90
plt.ylabel("Red")
plt.bar(rgb_val[:,0],rgb_val[:,1], color='r')

plt.subplot(3, 1, 2)
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 90
plt.ylabel("Green")
plt.bar(rgb_val[:,0],rgb_val[:,2], color='g')

plt.subplot(3, 1, 3)
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 90
plt.ylabel("Blue")
plt.bar(rgb_val[:,0],rgb_val[:,3], color='b')

plt.tight_layout(h_pad=5)

plt.suptitle("Visualization of RGB Values for Each Class",y=1.05)

