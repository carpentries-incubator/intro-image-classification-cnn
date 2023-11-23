# -*- coding: utf-8 -*-
"""
Define a function to prepare image for icwithcnn prediction

"""
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img

def prepare_image_icwithcnn(path_to_img):
    
    # TODO add code to check argument

    new_img_pil = load_img(path_to_img, target_size=(32,32)) # Image format
    new_img_arr = img_to_array(new_img_pil) # convert to array for analysis
    new_img_reshape = new_img_arr.reshape(1, 32, 32, 3) # reshape into single sample
    new_img_float = new_img_reshape.astype('float64') / 255.0 # normalize
    
    return new_img_float