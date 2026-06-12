# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:03:43 2026

@author: emg

"""

import tensorflow as tf # neural network

# create a function to prepare the training dataset
def prepare_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "../data/cifar10_images_small/train",
        image_size = (32, 32),
        batch_size = 32,
        shuffle = True,
        seed = 32
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "../data/cifar10_images_small/val",
        image_size = (32, 32),
        batch_size = 32,
        shuffle = True,
        seed = 32
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "../data/cifar10_images_small/test",
        image_size = (32, 32),
        batch_size = 32,
        shuffle = True,
        seed = 32
    )

    return train_ds, val_ds, test_ds

# function that defines an introductory convolutional neural network
def create_model_intro(input_shape=(32,32,3), num_classes=5):
    
    # CNN Part 1
    # Input layer of 32x32 images with three channels (RGB)
    inputs_intro = tf.keras.Input(shape=input_shape)
    
    # CNN Part 2
    x_intro = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs_intro)
    x_intro = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x_intro)
    x_intro = tf.keras.layers.Flatten()(x_intro)
    
    # CNN Part 3
    # Output layer with one unit for each class and softmax activation
    outputs_intro = tf.keras.layers.Dense(units = num_classes, activation='softmax')(x_intro)
    
    # create the model
    model_intro = tf.keras.Model(inputs = inputs_intro,
                                 outputs = outputs_intro, 
                                 name = "cifar_model_intro")
    
    return model_intro