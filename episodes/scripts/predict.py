# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:27:34 2023

@author: jc140298
"""

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

# load a new image and prepare it to match cifar10 dataset
new_img_pil = load_img("01_Jabiru_TGS.JPG", target_size=(32,32)) # Image format
new_img_arr = img_to_array(new_img_pil) # convert to array for analysis
new_img_reshape = new_img_arr.reshape(1, 32, 32, 3) # reshape into single sample
new_img_float =  new_img_reshape.astype('float64') / 255.0 # normalize

# predict the classname
result = model.predict(new_img_float) # make prediction
print(result) # probability for each class
print(class_names[result.argmax()]) # class with highest probability


# test set
y_pred = model.predict(test_images)
