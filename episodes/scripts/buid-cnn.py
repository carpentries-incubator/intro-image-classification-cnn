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

inputs_pool = keras.Input(shape=train_images.shape[1:])
x_pool = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_pool)
x_pool = keras.layers.MaxPooling2D((2, 2))(x_pool)
x_pool = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_pool)
x_pool = keras.layers.MaxPooling2D((2, 2))(x_pool)
x_pool = keras.layers.Flatten()(x_pool)
x_pool = keras.layers.Dense(50, activation='relu')(x_pool)
outputs_pool = keras.layers.Dense(10)(x_pool)

model_pool = keras.Model(inputs=inputs_pool, outputs=outputs_pool, name="cifar_model_pool")

model_pool.summary()

model_pool.compile(optimizer = 'adam', loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics = ['accuracy'])

history_pool = model_pool.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# convert the history to a dataframe for plotting 
history_pool_df = pd.DataFrame.from_dict(history_pool.history)

fig, axes = plt.subplots(1, 2)
fig.suptitle('cifar_model_pool')
sns.lineplot(ax=axes[0], data=history_pool_df[['loss', 'val_loss']])
sns.lineplot(ax=axes[1], data=history_pool_df[['accuracy', 'val_accuracy']])

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