# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:59:45 2023

@author: bellf
"""

# load the cifar dataset included with the keras packages
from tensorflow import keras
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()




#paste("This", "new", "lesson", "looks", "good")

print("This", "new", "lesson", "looks", "good")

result = " ".join(["This", "new", "lesson", "looks", "good"])
print(result)

