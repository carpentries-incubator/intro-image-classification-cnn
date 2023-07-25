# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 09:37:03 2023

@author: bellf
"""

# load the cifar dataset included with the keras packages
from tensorflow import keras
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()


print('Train: Images=%s, Labels=%s' % (train_images.shape, train_labels.shape))
print('Validate: Images=%s, Labels=%s' % (val_images.shape, val_labels.shape))


# normalize the RGB values to be between 0 and 1
train_images = train_images / 255.0
val_images = val_images / 255.0

# create a list of classnames
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plot a subset of the images
import matplotlib.pyplot as plt

# create a figure object and specify width, height in inches
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(class_names[train_labels[i,0]])
plt.show()


# define the inputs, layers, and outputs of a cnn model
inputs_intro = keras.Input(shape=train_images.shape[1:])
x_intro = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_intro)
x_intro = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_intro)
x_intro = keras.layers.Flatten()(x_intro)
outputs_intro = keras.layers.Dense(10)(x_intro)

model_intro = keras.Model(inputs=inputs_intro, outputs=outputs_intro, name="cifar_model_intro")

# compile the model
model_intro.compile(optimizer = 'adam', 
                    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                    metrics = ['accuracy'])

# fit the model
history_intro = model_intro.fit(train_images, train_labels, 
                                epochs = 10, 
                                validation_data = (val_images, val_labels))

# save the model
model_intro.save('fit_outputs/01_intro_model.h5')

# specify a new image and prepare it to match cifar10 dataset
from icwithcnn_functions import prepare_image_icwithcnn

new_img_path = "../data/Jabiru_TGS.JPG" # path to image
new_img_prepped = prepare_image_icwithcnn(new_img_path)

# predict the classname
result_intro = model_intro.predict(new_img_prepped) # make prediction
print(result_intro) # probability for each class
print(class_names[result_intro.argmax()]) # class with highest probability

