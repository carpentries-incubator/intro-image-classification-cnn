# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 02 Introduction to Image Data

"""

#%%

# load the required packages

from tensorflow import keras # data and neural network
from sklearn.model_selection import train_test_split # data splitting
from keras.utils import img_to_array # image processing
from keras.utils import load_img # image processing

#%%

#### Custom image data

### Working with Pixels

# specify the image path
new_img_path = "../data/Jabiru_TGS.JPG"

# read in the image with default arguments
new_img_pil = load_img(new_img_path)

# check the image class and size
print('Image class :', new_img_pil.__class__)
print('Image size', new_img_pil.size)

#%%

### Image Dimensions - Resizing

# read in the image and specify the target size
new_img_pil_small = load_img(new_img_path, target_size=(32,32))

# confirm the image class and size
print('Resized image class :', new_img_pil_small.__class__)
print('Resized image size', new_img_pil_small.size) 

#%%

### Normalisation

# first convert the image into an array for normalisation
new_img_arr = img_to_array(new_img_pil_small)

# confirm the image class and size
print('Converted image class  :', new_img_arr.__class__)
print('Converted image shape', new_img_arr.shape)

#%%

# inspect pixel values before and after normalisation

# extract the min, max, and mean pixel values BEFORE
print('BEFORE normalization')
print('Min pixel value ', new_img_arr.min()) 
print('Max pixel value ', new_img_arr.max())
print('Mean pixel value ', round(new_img_arr.mean(), 2))

# normalize the RGB values to be between 0 and 1
new_img_arr_norm = new_img_arr / 255.0

# extract the min, max, and mean pixel values AFTER
print('AFTER normalization') 
print('Min pixel value ', new_img_arr_norm.min()) 
print('Max pixel value ', new_img_arr_norm.max())
print('Mean pixel value ', round(new_img_arr_norm.mean(), 2))


#%%

#### Pre-existing image data

# load the data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# create a list of classnames associated with each CIFAR-10 label
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#%%

## CHALLENGE Create a function to prepare the dataset

# def prepare_dataset(_____, _____):
    
#     # normalize the RGB values to be between 0 and 1
#     _____
    
#     # one hot encode the training labels
#     _____
    
#     # split the training data into training and validation set
#     _____

#     return _____

#%%

## SOLUTION

# function to prepare the dataset
def prepare_dataset(train_images, train_labels):
    
    # normalize the RGB values to be between 0 and 1
    train_images = train_images / 255.0
    
    # one hot encode the training labels
    train_labels = keras.utils.to_categorical(train_labels, len(class_names))
    
    # split the training data into training and validation set
    train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size = 0.2, random_state=42)

    return train_images, val_images, train_labels, val_labels

#%%

# Investigate labels BEFORE one-hot encoding

print()
print('train_labels BEFORE one hot encoding')
print(train_labels)

#%%

# prepare the dataset for training
train_images, val_images, train_labels, val_labels = prepare_dataset(train_images, train_labels)

#%%

# Investigate labels AFTER one-hot encoding

print()
print('train_labels AFTER one hot encoding')
print(train_labels)


#%%

# CHALLENGE TRAINING AND VALIDATION

print()
print('Number of training set images:', train_images.shape[0])
print('Number of images in each class:\n', train_labels.sum(axis=0))

print()
print('Number of validation set images:', val_images.shape[0])
print('Nmber of images in each class:\n', val_labels.sum(axis=0))














