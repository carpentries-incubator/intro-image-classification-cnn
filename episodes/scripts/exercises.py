# -*- coding: utf-8 -*-
"""

The Image Classification with Convolutional Neural Networks workshop is designed to provide users with a useful program at the end of it, a separate exercises python script file will be used.

This exercises template file is designed to provide the required initialisation code to complete the exercises within the workshop.

@author: Jason Bell – September 2023

"""

# This section of the code helps to set things up ready for the various exercices

# load the cifar dataset included with the keras packages
from tensorflow import keras
from icwithcnn_functions import prepare_image_icwithcnn

(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()

print('Train: Images=%s, Labels=%s' % (train_images.shape, train_labels.shape))
print('Validate: Images=%s, Labels=%s' % (val_images.shape, val_labels.shape))

# CINAC-10 uses the same class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


"""

Introduction to Deep Learning Exercies

"""

### Challenge Load the CIFAR-10 dataset ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/01-introduction.html#challenge-load-the-cifar-10-dataset

# Uncomment the 2 lines of code below
#print('Train: Images=%s, Labels=%s' % (train_images.shape, train_labels.shape))
#print('Validate: Images=%s, Labels=%s' % (val_images.shape, val_labels.shape))


"""

Introduction to Image Data

"""

### TRAINING AND TEST SETS ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/02-image-data.html#challenge1

# Still TODO


"""

Build a Convolutional Neural Network

"""

# Load the saved the model from the intro
# The line of code below can be run and uncommented after the intro model
# has been saved from the "Introduction to Deep Learning" episode.

model_intro = keras.models.load_model('fit_outputs/01_intro_model.h5')

### Number of parameters ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/03-build-cnn.html#number-of-parameters

# Suppose we create a single Dense (fully connected) layer with 100 hidden 
# units that connect to the input pixels, how many parameters does this 
# layer have?

# Uncomment the 6 lines of code below and add your solution
#width, height = 
#n_hidden_neurons = 
#n_bias = 
#n_input_items = 
#n_parameters = 
#print(n_parameters)

# We can also check this by building the layer in Keras:

# Uncomment the 4 lines of code below and add your solution
#inputs_ex = 
#outputs_ex = 
#model_ex = 
#model_ex.summary()   

### Convolutional Neural Network ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/03-build-cnn.html#challenge-network-depth


### Challenge Network depth ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/04-fit-cnn.html#challenge-the-training-curve

inputs_cnd = keras.Input(shape=train_images.shape[1:])
x_cnd = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_cnd)
x_cnd = keras.layers.MaxPooling2D((2, 2))(x_cnd)
x_cnd = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_cnd)
x_cnd = keras.layers.MaxPooling2D((2, 2))(x_cnd)
# Add your extra layer here
x_cnd = keras.layers.Flatten()(x_cnd)
x_cnd = keras.layers.Dense(50, activation='relu')(x_cnd)
outputs_cnd = keras.layers.Dense(10)(x_cnd)


"""
Compile and Train a Convolutional Neural Network

"""

### Challenge The Training Curve ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/04-fit-cnn.html#challenge-the-training-curve



### Vary dropout rate ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/04-fit-cnn.html#vary-dropout-rate


"""
Compile and Train a Convolutional Neural Network

"""

### Challenge 1 ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/05-evaluate-predict-cnn.html#challenge1


### Challenge 1 ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/05-evaluate-predict-cnn.html#challenge2


### Challenge Confusion Matrix ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/05-evaluate-predict-cnn.html#challenge-confusion-matrix


### Challenge ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/05-evaluate-predict-cnn.html#challenge4

# specify a new image and prepare it to match CIFAR-10 dataset


new_img_path = "../data/Jabiru_TGS.JPG" # path to image
new_img_prepped = prepare_image_icwithcnn(new_img_path)

# predict the classname (Section can be uncomment once has been model_intro in previous exercises)
#result_intro = model_intro.predict(new_img_prepped) # make prediction
#print(result_intro) # probability for each class
#print(class_names[result_intro.argmax()]) # class with highest probability


### Exercise: plot the training progress ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/05-evaluate-predict-cnn.html#exercise-plot-the-training-progress



### Open question: What could be next steps to further improve the model? ###
# https://carpentries-incubator.github.io/intro-image-classification-cnn/05-evaluate-predict-cnn.html#open-question-what-could-be-next-steps-to-further-improve-the-model


