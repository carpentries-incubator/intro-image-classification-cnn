# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:59:45 2023

@author: bellf
"""

#### Pre-existing image data

# load the cifar dataset included with the keras packages
from tensorflow import keras
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()

# EG Still trying to decide which image dataset to use

#### Custom image data

from keras.utils import img_to_array
from keras.utils import load_img

# specify the image path
new_img_path = "../data/Jabiru_TGS.JPG" # path to image

# read in the image with default arguments
new_img_pil = load_img(new_img_path)

# confirm the data class and size
print('The new image is of type :', new_img.__class__, 'and has the size', new_img.size)

# convert the Image into an array for processing
new_img_arr = img_to_array(new_img_pil)

# confirm the data class and shape
print('The new image is of type :', new_img_arr.__class__, 'and has the shape', new_img_arr.shape)

# read in the new image and specify the size to be the same as our training images
new_img_pil_small = load_img(path_to_img, target_size=(32,32))

# confirm the data class and shape
print('The new image is still of type:', new_img_pil_small.__class__, 'and has the reduced shape', new_img_pil_small.shape)


# TODO show how to load multiple images