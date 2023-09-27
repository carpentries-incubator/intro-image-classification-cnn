# -*- coding: utf-8 -*-
"""
Episode 02 Image Data

"""

#### Pre-existing image data

# load the cifar dataset included with the keras packages
from tensorflow import keras

(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()

#### Custom image data

# load the libraries required
from keras.utils import img_to_array
from keras.utils import load_img

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

#### Load multiple images at the same time

from keras.utils import image_dataset_from_directory 
test_image_dir = 'D:/20230724_CINIC10/test_images'
test_images = image_dataset_from_directory(test_image_dir, labels='inferred', batch_size=None, image_size=(32,32), shuffle=False)

from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(test_images, test_size=1.0)

