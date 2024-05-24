# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 03 Build a Convolutional Neural Network

"""

#%%

# load the required packages

from tensorflow import keras # data and neural network
from sklearn.model_selection import train_test_split # data splitting
import matplotlib.pyplot as plt # plotting
import seaborn as sns # specialised plotting
import pandas as pd # handles dataframes

#%%

# create a function to prepare the dataset

def prepare_dataset(train_images, train_labels):
    
    # normalize the RGB values to be between 0 and 1
    train_images = train_images / 255
    
    # one hot encode the training labels
    train_labels = keras.utils.to_categorical(train_labels, len(class_names))
    
    # split the training data into training and validation set
    train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size = 0.2, random_state=42)

    return train_images, val_images, train_labels, val_labels

#%%

# load the data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# create a list of classnames
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# prepare the dataset for training
train_images, val_images, train_labels, val_labels = prepare_dataset(train_images, train_labels)

#%%

### Step 4. Build a new architecture from scratch

#### CNN Part 1. Input Layer

# recall the shape of the images in our dataset
print(train_images.shape)

#%%

# ## CHALLENGE Create the input layer

# # CNN Part 1
# # Input layer of 32x32 images with three channels (RGB)
# inputs_intro = keras.Input(_____)

#%%

# #### CNN Part 2. Hidden Layers

# ##### **Convolutional Layers**

# ## CHALLENGE Create a 2D convolutional layer for our network

# # CNN Part 2
# # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
# x_intro = keras.layers.Conv2D(filters=_____, kernel_size=_____, activation=_____)(_____)
                              
#%%

# ##### **Pooling Layers**

# ## CHALLENGE Create a Pooling layer for our network

# # Pooling layer with input window sized 2,2
# x_intro = keras.layers.MaxPooling2D(pool_size=_____)(_____)
                                    
#%%

# ##### **Dense layers**

# ## CHALLENGE Create a Dense layer for our network

# # Dense layer with 64 neurons and ReLU activation
# x_intro = keras.layers.Dense(units=_____, activation=_____)(_____)

#%%

# ##### **Reshaping Layers: Flatten**        

# ## CHALLENGE Create a Flatten layer for our network

# # Flatten layer to convert 2D feature maps into a 1D vector
# x_intro = keras.layers.Flatten()(_____)


#%%

## CHALLENGE Using the four layer types above, create a hidden layer architecture

# TODO decide what to put here

#%%

# #### CNN Part 3. Output Layer

# ## CHALLENGE Create an Output layer for our network

# # CNN Part 3
# # Output layer with 10 units (one for each class) and softmax activation
# outputs_intro = keras.layers.Dense(units=_____, activation=_____)(_____)
                                   
#%%                                   
                                   
## Putting it all together

## CHALLENGE Create a function that defines a CNN using the input, hidden, and output layers in previous challenges.

## SOLUTION

#### Define the Model

def create_model_intro():
    
    # CNN Part 1
    # Input layer of 32x32 images with three channels (RGB)
    inputs_intro = keras.Input(shape=train_images.shape[1:])
    
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs_intro)
    # Pooling layer with input window sized 2x2
    x_intro = keras.layers.MaxPooling2D(pool_size=(2,2))(x_intro)
    # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x_intro)
    # Second Pooling layer with input window sized 2x2
    x_intro = keras.layers.MaxPooling2D(pool_size=(2,2))(x_intro)
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_intro = keras.layers.Flatten()(x_intro)
    # Dense layer with 64 neurons and ReLU activation
    x_intro = keras.layers.Dense(units=64, activation='relu')(x_intro)
    
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_intro = keras.layers.Dense(units=10, activation='softmax')(x_intro)
    
    # create the model
    model_intro = keras.Model(inputs = inputs_intro, 
                              outputs = outputs_intro, 
                              name = "cifar_model_intro")
    
    return model_intro

#%%

# create the introduction model
model_intro = create_model_intro()

# view model summary
model_intro.summary()
