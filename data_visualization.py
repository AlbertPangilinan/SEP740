#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing modules and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

import cv2
import os
from os import listdir
import PIL
import PIL.Image
from PIL import Image
import glob

#set up path for loading file
import os.path
cwd = os.getcwd()
print(cwd)


# In[2]:


#loading all images with glob
original_path = cwd + "\dataset\semantic_drone_dataset\original_images\*.jpg"
semantic_path = cwd + "\dataset\semantic_drone_dataset\label_images_semantic\*.png"
colormap_path = cwd + "\RGB_color_image_masks\RGB_color_image_masks\*.png"

image_list_o = []
for filename in glob.glob(original_path):
    im=Image.open(filename)
    image_list_o.append(im)


image_list_s = []
for filename in glob.glob(semantic_path):
    im=Image.open(filename)
    image_list_s.append(im)
    
image_list_c = []
for filename in glob.glob(colormap_path):
    im=Image.open(filename)
    image_list_c.append(im)


# In[12]:


plt.figure(figsize=(12,12))

#choose which image to show
image_number=0

#plot original image
plt.subplot(3,1,1)
plt.title("Original Image")
plt.imshow(image_list_o[image_number])

#plot semantic image
plt.subplot(3,1,2)
plt.title("Semantic Image")
plt.imshow(image_list_s[image_number])

#plot colormap image
plt.subplot(3,1,3)
plt.title("Colormap Image")
plt.imshow(image_list_c[image_number])


# In[4]:


#csv histogram of RGB values for each class
rgb_data=cwd+"\class_dict_seg.csv"
print(rgb_data)

#print the data in a table format
rgb_data=pd.read_csv(rgb_data)
print("Data in Table form")
print(rgb_data)

#convert to array
rgb_val=rgb_data.values


# In[5]:


#plot histogram
plt.figure(figsize=(12,12))

#r values
plt.subplot(3, 1, 1)
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 90
plt.ylabel("Red")
plt.bar(rgb_val[:,0],rgb_val[:,1], color='r')

#g values
plt.subplot(3, 1, 2)
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 90
plt.ylabel("Green")
plt.bar(rgb_val[:,0],rgb_val[:,2], color='g')

#b values
plt.subplot(3, 1, 3)
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 90
plt.ylabel("Blue")
plt.bar(rgb_val[:,0],rgb_val[:,3], color='b')

plt.tight_layout(h_pad=5)
plt.suptitle("Visualization of RGB Values for Each Class",y=1.05)

