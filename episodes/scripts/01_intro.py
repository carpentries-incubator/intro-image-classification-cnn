# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 01 Introduction to Deep Learning

"""

#%%

# load the required packages
import tensorflow as tf # neural network
import matplotlib.pyplot as plt # plotting
import icwcnn_functions as icfn # pre-defined helpers


#%%

### Step 3. Prepare data

# create a list of class names associated with each CIFAR-10 label
class_names = ['airplane', 'bird', 'cat', 'dog', 'truck']

# load the data
train_ds, val_ds, test_ds = icfn.prepare_datasets()

#%%

#### Visualise a subset of the training dataset

# set up plot region, including width, height in inches
plt.figure(figsize=(5, 5))

# loop through images and plot first the first nine
for images, labels in train_ds.take(1):
    for i in range(9):
        
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# view plot
plt.show()

#%%

### Step 4. Build the model architecture

# create the introduction model
model_intro = icfn.create_model_intro()


#%%

### Step 5. Choose a loss function and optimizer and compile model

# compile model
model_intro.compile(optimizer = "adam",
                    loss = "sparse_categorical_crossentropy",
                    metrics =["accuracy"])

#%%

### Step 6. Train the model

# fit model
history_intro = model_intro.fit(x = train_ds)

#%%

### Step 7. Perform a Prediction/Classification
  
# extract image and label for first image
for images, labels in test_ds.take(1):
    first_image = images[0]
    first_label = labels[0]
    
# use the model to predict class
prediction = model_intro.predict(tf.expand_dims(first_image, axis=0))
print("Predict:", prediction)

# extract class name with highest probability
predicted_label = tf.argmax(prediction[0])

print("Predicted:", class_names[predicted_label])
print("True:", class_names[first_label])


#%%

# plot the first test image with its predicted label
plt.imshow(first_image.numpy().astype("uint8"))
plt.title('Predicted:' + class_names[predicted_label])
plt.axis("off")
plt.show() 

#%%

### Step 10. Share Model

# save model

import os
os.makedirs("../models", exist_ok=True)

model_intro.save('../models/cifar_model_intro.keras')




