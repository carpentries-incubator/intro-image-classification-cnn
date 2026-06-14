# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 03 Build a Convolutional Neural Network

"""

#%%

# load the required packages
import tensorflow as tf # neural network
import matplotlib.pyplot as plt # plotting
import icwcnn_functions as icfn # pre-defined helpers

#%%

### Step 4. Build a new architecture

#### CNN Part 1. Input Layer

inputs_intro = tf.keras.Input(shape=(32,32,3))

#### CNN Part 2. Hidden Layer
x_intro = tf.keras.layers.Conv2D(filters=16, 
                                 kernel_size=(3,3), 
                                 activation='relu')(inputs_intro)                              
x_intro = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x_intro)
x_intro = tf.keras.layers.Flatten()(x_intro)

#### CNN Part 3. Output Layer
outputs_intro = tf.keras.layers.Dense(units = 5,
                                      activation='softmax')(x_intro)

model_intro = tf.keras.Model(inputs = inputs_intro,
                             outputs = outputs_intro, 
                             name = "cifar_model_intro")
                                   
#%%                                   
                                   
## Putting it all together

## CHALLENGE Turn your neural network into a function

## SOLUTION

# function that defines an introductory convolutional neural network
def create_model_intro(input_shape=(32,32,3), num_classes=5):
    
    # CNN Part 1
    # Input layer of 32x32 images with three channels (RGB)
    inputs_intro = tf.keras.Input(shape=input_shape)
    
    # CNN Part 2
    x_intro = tf.keras.layers.Conv2D(filters=16, 
                                     kernel_size=(3,3), 
                                     activation='relu')(inputs_intro)
    x_intro = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x_intro)
    x_intro = tf.keras.layers.Flatten()(x_intro)
    
    # CNN Part 3
    # Output layer with one unit for each class and softmax activation
    outputs_intro = tf.keras.layers.Dense(units=num_classes, 
                                          activation='softmax')(x_intro)
    
    # create the model
    model_intro = tf.keras.Model(inputs = inputs_intro,
                                 outputs = outputs_intro, 
                                 name = "cifar_model_intro")
    
    return model_intro

#%%

# create the introduction model
model_intro = create_model_intro()

# view model summary
model_intro.summary()
