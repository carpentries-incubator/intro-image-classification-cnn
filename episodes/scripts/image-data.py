# -*- coding: utf-8 -*-
"""
Episode 02 Introduction to Image Data

"""

from tensorflow import keras
from keras.utils import img_to_array
from keras.utils import load_img
import os
import numpy as np

#### Pre-existing image data

# load the CIFAR-10 dataset included with the keras packages
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()

# normalize the RGB values to be between 0 and 1
train_images = train_images / 255.0
val_images = val_images / 255.0

#### Custom image data

# specify the image path
new_img_path = "../data/Jabiru_TGS.JPG" # path to image

# read in the image with default arguments
new_img_pil = load_img(new_img_path)

# confirm the data class and size
print('The new image is of type :', new_img_pil.__class__, 'and has the size', new_img_pil.size)

### Image Dimensions - Resizing

# read in the new image and specify the target size to be the same as our training images
new_img_pil_small = load_img(new_img_path, target_size=(32,32))

# confirm the data class and shape
print('The new image is still of type:', new_img_pil_small.__class__, 'but now has the same size', new_img_pil_small.size, 'as our training data.')# convert the Image into an array for processing
new_img_arr = img_to_array(new_img_pil)

### Normalization

# convert the Image into an array for normalization
new_img_arr = img_to_array(new_img_pil_small)

# confirm the data class and shape
print('The new image is now of type :', new_img_arr.__class__, 'and has the shape', new_img_arr.shape)

# extract the min, max, and mean pixel values BEFORE
print('The min, max, and mean pixel values are', new_img_arr.min(), ',', new_img_arr.max(), ', and', new_img_arr.mean().round(), 'respectively.')

# normalize the RGB values to be between 0 and 1
new_img_arr_norm = new_img_arr / 255.0

# extract the min, max, and mean pixel values AFTER
print('After normalization, the min, max, and mean pixel values are', new_img_arr_norm.min(), ',', new_img_arr_norm.max(), ', and', new_img_arr_norm.mean().round(), 'respectively.')

#### CINIC-10 Test Dataset Preparation

# Load multiple images into single object to be able to process multiple images at the same time

# main_directory/
# ...class_a/
# ......image_1.jpg
# ......image_2.jpg
# ...class_b/
# ......image_1.jpg
# ......image_2.jpg

# set the mian directory  
test_image_dir = 'D:/20230724_CINIC10/test_images'

# make two lists of the subfolders (ie class or label) and filenames
test_filenames = []
test_labels = []

for dn in os.listdir(test_image_dir):
    
    for fn in os.listdir(os.path.join(test_image_dir, dn)):
        
        test_filenames.append(fn)
        test_labels.append(dn)

# prepare the images
# create an empty numpy array to hold the processed images
test_images = np.empty((len(test_filenames), 32, 32, 3), dtype=np.float32)

# use the dirnames and filenanes to process each 
for i in range(len(test_filenames)):
    
    # set the path to the image
    img_path = os.path.join(test_image_dir, test_labels[i], test_filenames[i])
    
    # load the image and resize at the same time
    img = load_img(img_path, target_size=(32,32))
    
    # convert to an array
    img_arr = img_to_array(img)
    
    # normalize
    test_images[i] = img_arr/255.0

print(test_images.shape)
print(test_images.__class__)
  
########################################################
# Challenge TRAINING AND TEST SETS

# Q1
print('The training set is of type', train_images.__class__)
print('The training set has', train_images.shape[0], 'samples.\n')

print('The number of labels in our training set and the number images in each class are:\n')
print(np.unique(train_labels, return_counts=True))

# Q2
print('The test set is of type', test_images.__class__)
print('The test set has', test_images.shape[0], 'samples.\n')

print('The number of labels in our test set and the number images in each class are:\n')
print(np.unique(test_labels, return_counts=True))
########################################################

########################################################
# Challenge Data Splitting Example

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, target, test_size=0.2, random_state=42, shuffle=True, stratify=target)

########################################################

