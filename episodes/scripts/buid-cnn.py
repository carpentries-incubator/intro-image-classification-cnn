# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:41:57 2023

@author: bellf
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()

# define the inputs, layers, and outputs of a convolutional neural network

#inputs_intro = keras.Input(shape=train_images.shape[1:])

#x_intro = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_intro)
#x_intro = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_intro)
#x_intro = keras.layers.Flatten()(x_intro)

#outputs_intro = keras.layers.Dense(10)(x_intro)

# create the model
model_intro = keras.Model(inputs=inputs_intro, outputs=outputs_intro, name="cifar_model_intro")

# recall the shape of the images in our dataset
print(train_images.shape)

# define the input
# Input layer of 32x32 images with three channels (RGB)
#inputs_intro = keras.Input(shape=train_images.shape[1:])

dim = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]
print(dim)

width, height = (32, 32)
n_hidden_neurons = 100
n_bias = 100
n_input_items = width * height * 3
n_parameters = (n_input_items * n_hidden_neurons) + n_bias
print(n_parameters)

inputs_ex = keras.Input(shape=dim)
outputs_ex = keras.layers.Dense(100)(inputs_ex)
model_ex = keras.models.Model(inputs=inputs_ex, outputs=outputs_ex)
model_ex.summary()

# Output layer with 10 units (one for each class)
#outputs = keras.layers.Dense(10)(x)

# Input layer of 32x32 images with three channels (RGB)
inputs_intro = keras.Input(shape=train_images.shape[1:])
# Convolutional layer with 50 filters, 3x3 kernel size, and ReLU activation
x_intro = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_intro)
# Second Convolutional layer
x_intro = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_intro)
# Flatten layer to convert 2D feature maps into a 1D vector
x_intro = keras.layers.Flatten()(x_intro)
# Output layer with 10 units (one for each class)
outputs_intro = keras.layers.Dense(10)(x_intro)

# create the model
model_intro = keras.Model(inputs=inputs_intro, outputs=outputs_intro, name="cifar_model_intro")

#x_intro = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_intro)
#x_intro = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_intro)

model_intro.summary()

model_intro.compile(optimizer = 'adam', loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics = ['accuracy'])
history_intro = model_intro.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# convert the history to a dataframe for plotting 
history_intro_df = pd.DataFrame.from_dict(history_intro.history)

# plot the loss and accuracy from the training process
fig, axes = plt.subplots(1, 2)
fig.suptitle('cifar_model_intro')
sns.lineplot(ax=axes[0], data=history_intro_df[['loss', 'val_loss']])
sns.lineplot(ax=axes[1], data=history_intro_df[['accuracy', 'val_accuracy']])

# pooling layer
#x_pool = keras.layers.MaxPooling2D((2, 2))(x_pool)

#  $$output_shape = math.floor\frac{(input_shape - pool_size)}{strides} + 1$$ & when input_shape >= pool_size

# dense layer
#x_pool = keras.layers.Dense(50, activation='relu')(x_pool)

# define the inputs, layers, and outputs of a cnn model with pooling
inputs_pool = keras.Input(shape=train_images.shape[1:])
x_pool = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_pool)
x_pool = keras.layers.MaxPooling2D((2, 2))(x_pool)
x_pool = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_pool)
x_pool = keras.layers.MaxPooling2D((2, 2))(x_pool)
x_pool = keras.layers.Flatten()(x_pool)
x_pool = keras.layers.Dense(50, activation='relu')(x_pool)
outputs_pool = keras.layers.Dense(10)(x_pool)

# create the pooling model
model_pool = keras.Model(inputs=inputs_pool, outputs=outputs_pool, name="cifar_model_pool")

model_pool.compile(optimizer = 'adam', loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics = ['accuracy'])

history_pool = model_pool.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

model_pool.summary()

# save pool model
model_pool.save('fit_outputs/model_pool.h5')

# TODO left off until next episode but might do now?
# fig, axes = plt.subplots(1, 2)
# fig.suptitle('cifar_model_pool')
# sns.lineplot(ax=axes[0], data=history_pool_df[['loss', 'val_loss']])
# sns.lineplot(ax=axes[1], data=history_pool_df[['accuracy', 'val_accuracy']])

inputs_cnd = keras.Input(shape=train_images.shape[1:])
x_cnd = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_cnd)
x_cnd = keras.layers.MaxPooling2D((2, 2))(x_cnd)
x_cnd = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_cnd)
x_cnd = keras.layers.MaxPooling2D((2, 2))(x_cnd)
# Add your extra layer here
x_cnd = keras.layers.Flatten()(x_cnd)
x_cnd = keras.layers.Dense(50, activation='relu')(x_cnd)
outputs_cnd = keras.layers.Dense(10)(x_cnd)

model_cnd = keras.Model(inputs=inputs_cnd, outputs=outputs_cnd, name="cifar_model_Challenge_network_depth")

model.summary()