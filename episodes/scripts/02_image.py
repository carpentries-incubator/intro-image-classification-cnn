# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 02 Introduction to Image Data

"""

#%%

# load the required packages
import tensorflow as tf # neural network

#%%
# create training dataset from folder of cifar images
train_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/cifar10_images_small/train",
    image_size = (32, 32),
    batch_size = 32,
    shuffle = True,
    seed = 32
)

print("Train_ds:", train_ds)

#%%
# extract the list of class names
class_names = train_ds.class_names
print(class_names)

   
#%%
# inspect one batch of images and labels
for images, labels in train_ds.take(1):
    print("Train images batch shape:", images.shape)
    print("Train labels batch shape:", labels.shape)
    
 #%%
 # inspect image data types and pixel values
for images, labels in train_ds.take(1):
    print("Data type: ", images.dtype)
    print("Pixel value range:", images[0].numpy().min(), images[0].numpy().max())


#%%
 # CHALLENGE Inspect the `test` dataset
 
# create test dataeset from folder of cifar images
test_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/cifar10_images_small/test",
    image_size = (32, 32),
    batch_size = 32,
    shuffle = False,
)

# class names
print("Test class names: ", test_ds.class_names)

# dimensions of the images and labels
for images, labels in test_ds.take(1):
    print("Test images batch shape:", images.shape)
    print("Test labels batch shape:", labels.shape)

# total number of images
# 250 as noted previously

# images in each class
# ????
