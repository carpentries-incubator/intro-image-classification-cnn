# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:00:43 2023

@author: eg

Define function to prepare image for icwithcnn prediction

"""
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
import os
import numpy as np

def prepare_image_icwithcnn(path_to_img):
    
    # TODO add code to check argument

    new_img_pil = load_img(path_to_img, target_size=(32,32)) # Image format
    new_img_arr = img_to_array(new_img_pil) # convert to array for analysis
    new_img_reshape = new_img_arr.reshape(1, 32, 32, 3) # reshape into single sample
    new_img_float = new_img_reshape.astype('float64') / 255.0 # normalize
    
    return new_img_float

def prepare_testdataset_icwithcnn(absolute_path_to_img_dir):

    # make two lists of the subfolders (ie class or label) and filenames
    test_filenames = []
    test_class = []
    
    for dn in os.listdir(absolute_path_to_img_dir):
        
        for fn in os.listdir(os.path.join(absolute_path_to_img_dir, dn)):
            
            test_filenames.append(fn)
            test_class.append(dn)
    
    # prepare the images
    # create an empty numpy array to hold the processed images
    test_images = np.empty((len(test_filenames), 32, 32, 3), dtype=np.float32)
    
    # use the dirnames and filenanes to process each 
    for i in range(len(test_filenames)):
        
        # set the path to the image
        img_path = os.path.join(absolute_path_to_img_dir, test_class[i], test_filenames[i])
        
        # load the image and resize at the same time
        img = load_img(img_path, target_size=(32,32))
        
        # convert to an array
        img_arr = img_to_array(img)
        
        # normalize
        test_images[i] = img_arr/255.0
    
    return test_images, test_class