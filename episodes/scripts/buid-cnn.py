# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:41:57 2023

@author: bellf
"""

from tensorflow import keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()


# define the inputs, layers, and outputs of a cnn model
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

# create the model
model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")


# recall the shape of the images in our dataset
print(train_images.shape)

# define the input
inputs = keras.Input(shape=train_images.shape[1:])

dim = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]
print(dim)

width, height = (32, 32)
n_hidden_neurons = 100
n_bias = 100
n_input_items = width * height * 3
n_parameters = (n_input_items * n_hidden_neurons) + n_bias
print(n_parameters)

inputs = keras.Input(shape=dim)
outputs = keras.layers.Dense(100)(inputs)
model_ex = keras.models.Model(inputs=inputs, outputs=outputs)
model_ex.summary()

outputs = keras.layers.Dense(10)(x)



inputs_sm = keras.Input(shape=train_images.shape[1:])
x_sm = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_sm)
x_sm = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_sm)
x_sm = keras.layers.Flatten()(x_sm)
outputs_sm = keras.layers.Dense(10)(x_sm)

model_sm = keras.Model(inputs=inputs_sm, outputs=outputs_sm, name="cifar_model_small")

model_sm.summary()




# x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)

# second layer
# x = keras.layers.MaxPooling2D((2, 2))(x)

#  $$output_shape = math.floor\frac{(input_shape - pool_size)}{strides} + 1$$ & when input_shape >= pool_size


# add layers
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)

# define outputs
outputs = keras.layers.Dense(10)(x)

# create the model
model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")

model.summary()



inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
# Add your extra layer here
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)



inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")
model.summary()