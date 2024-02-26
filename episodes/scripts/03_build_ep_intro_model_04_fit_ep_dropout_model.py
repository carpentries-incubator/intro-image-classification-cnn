# -*- coding: utf-8 -*-
"""
Episode 03 Build a Convolutional Neural Network

"""

# load the required packages
from tensorflow import keras
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time

start = time.time()

# load the cifar dataset included with the keras packages
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# normalize the RGB values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# create a list of classnames
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# one-hot encode training labels
train_labels = keras.utils.to_categorical(train_labels, len(class_names))

# split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# recall the parts of the model in the introduction

# # CNN Part 1
# # Input layer of 32x32 images with three channels (RGB)
# inputs_intro = keras.Input(shape=train_images.shape[1:])

# # CNN Part 2
# # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
# x_intro = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs_intro)
# # Pooling layer with input window sized 2,2
# x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
# # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
# x_intro = keras.layers.Conv2D(32, (3, 3), activation='relu')(x_intro)
# # Second Pooling layer with input window sized 2,2
# x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
# # Flatten layer to convert 2D feature maps into a 1D vector
# x_intro = keras.layers.Flatten()(x_intro)
# # Dense layer with 64 neurons and ReLU activation
# x_intro = keras.layers.Dense(64, activation='relu')(x_intro)

# # CNN Part 3
# # Output layer with 10 units (one for each class) and softmax activation
# outputs_intro = keras.layers.Dense(10, activation='softmax')(x_intro)

# recall the shape of the images in our dataset
print(train_images.shape)

########################################################
# Challenge Number of parameters

width, height = (32, 32)
n_hidden_neurons = 100
n_bias = 100
n_input_items = width * height * 3
n_parameters = (n_input_items * n_hidden_neurons) + n_bias
print(n_parameters)

inputs = keras.Input(shape=n_input_items)
outputs = keras.layers.Dense(100)(inputs)
model = keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()

########################################################

## Putting it all together

#### Define the Model

# CNN Part 1
# Input layer of 32x32 images with three channels (RGB)
inputs_intro = keras.Input(shape=train_images.shape[1:])

# CNN Part 2
# Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
x_intro = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs_intro)
# Pooling layer with input window sized 2,2
x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
# Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
x_intro = keras.layers.Conv2D(32, (3, 3), activation='relu')(x_intro)
# Second Pooling layer with input window sized 2,2
x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
# Flatten layer to convert 2D feature maps into a 1D vector
x_intro = keras.layers.Flatten()(x_intro)
# Dense layer with 64 neurons and ReLU activation
x_intro = keras.layers.Dense(64, activation='relu')(x_intro)

# CNN Part 3
# Output layer with 10 units (one for each class) and softmax activation
outputs_intro = keras.layers.Dense(10, activation='softmax')(x_intro)

# create the model
model_intro = keras.Model(inputs=inputs_intro, outputs=outputs_intro, name="cifar_model_intro")

# view the model summary
model_intro.summary()


"""
Episode 04 Compile and Train (Fit) a Convolutional Neural Network

"""

# compile the model
model_intro.compile(optimizer = 'adam', 
                    loss = keras.losses.CategoricalCrossentropy(), 
                    metrics = ['accuracy'])

# fit the model
history_intro = model_intro.fit(train_images, train_labels, 
                                epochs = 10, 
                                validation_data = (val_images, val_labels),
                                batch_size = 32)

# save the model
model_intro.save('fit_outputs/model_intro.keras')


# convert the history to a dataframe for plotting 
history_intro_df = pd.DataFrame.from_dict(history_intro.history)

# plot the loss and accuracy from the training process
fig, axes = plt.subplots(1, 2)
fig.suptitle('cifar_model_intro')
sns.lineplot(ax=axes[0], data=history_intro_df[['loss', 'val_loss']])
sns.lineplot(ax=axes[1], data=history_intro_df[['accuracy', 'val_accuracy']])

### Improve Model Generalization (avoid Overfitting)

## Dropout

# Input layer of 32x32 images with three channels (RGB)
inputs_dropout = keras.Input(shape=train_images.shape[1:])

# CNN Part 2
# Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
x_dropout = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs_dropout)
# Pooling layer with input window sized 2,2
x_dropout = keras.layers.MaxPooling2D((2, 2))(x_dropout)
# Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
x_dropout = keras.layers.Conv2D(32, (3, 3), activation='relu')(x_dropout)
# Second Pooling layer with input window sized 2,2
x_dropout = keras.layers.MaxPooling2D((2, 2))(x_dropout)
# Second Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
x_dropout = keras.layers.Conv2D(64, (3, 3), activation='relu')(x_dropout)
# Dropout layer randomly drops 60% of the input units
x_dropout = keras.layers.Dropout(0.6)(x_dropout) # This is new!
# Flatten layer to convert 2D feature maps into a 1D vector
x_dropout = keras.layers.Flatten()(x_dropout)
# Dense layer with 128 neurons and ReLU activation
x_dropout = keras.layers.Dense(128, activation='relu')(x_dropout)

# CNN Part 3
# Output layer with 10 units (one for each class) and softmax activation
outputs_dropout = keras.layers.Dense(10, activation='softmax')(x_dropout)

# create the dropout model
model_dropout = keras.Model(inputs=inputs_dropout, outputs=outputs_dropout, name="cifar_model_dropout")

model_dropout.summary()

########################################################
# Challenge Compile, Fit, and Evaulate Dropout Model

# compile the dropout model
model_dropout.compile(optimizer = 'adam',
              loss = keras.losses.CategoricalCrossentropy(),
              metrics = ['accuracy'])

# fit the dropout model
history_dropout = model_dropout.fit(train_images, train_labels, 
                                    epochs = 10,
                                    validation_data = (val_images, val_labels),
                                    batch_size = 32)

# save dropout model
model_dropout.save('fit_outputs/model_dropout.keras')

# inspect the training results

# convert the history to a dataframe for plotting 
history_dropout_df = pd.DataFrame.from_dict(history_dropout.history)

# plot the loss and accuracy from the training process
fig, axes = plt.subplots(1, 2)
fig.suptitle('cifar_model_dropout')
sns.lineplot(ax=axes[0], data=history_dropout_df[['loss', 'val_loss']])
sns.lineplot(ax=axes[1], data=history_dropout_df[['accuracy', 'val_accuracy']])

########################################################

end = time.time()

print()
print()
print("Time taken to run program was:", end - start, "seconds")