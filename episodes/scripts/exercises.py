# -*- coding: utf-8 -*-
"""

The Image Classification with Convolutional Neural Networks workshop is designed to provide users with a useful program at the end of it, a separate exercises python script file will be used.

This exercises template file is designed to provide the required initialisation code to complete the exercises within the workshop.

@author: Jason Bell – September 2023

"""

# load the cifar dataset included with the keras packages
from tensorflow import keras
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()

print('Train: Images=%s, Labels=%s' % (train_images.shape, train_labels.shape))
print('Validate: Images=%s, Labels=%s' % (val_images.shape, val_labels.shape))

"""

Introduction to Deep Learning Exercies

"""

# Challenge Load the CIFAR-10 dataset - https://carpentries-incubator.github.io/intro-image-classification-cnn/01-introduction.html#challenge-load-the-cifar-10-dataset

# Uncomment the 2 lines of code below
# print('Train: Images=%s, Labels=%s' % (train_images.shape, train_labels.shape))
# print('Validate: Images=%s, Labels=%s' % (val_images.shape, val_labels.shape))

