# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 03 Build a Convolutional Neural Network

"""

#%%

# load the required packages

from tensorflow import keras # data and neural network
from sklearn.model_selection import train_test_split # data splitting
import matplotlib.pyplot as plt # plotting
import seaborn as sns # specialised plotting
import pandas as pd # handles dataframes
import time # track run time

#%%

# start timer
start = time.time()

#%%

# create a function to prepare the dataset

def prepare_dataset(train_images, train_labels):
    
    # normalize the RGB values to be between 0 and 1
    train_images = train_images / 255
    test_images = train_labels / 255
    
    # one hot encode the training labels
    train_labels = keras.utils.to_categorical(train_labels, len(class_names))
    
    # split the training data into training and validation set
    train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size = 0.2, random_state=42)

    return train_images, val_images, train_labels, val_labels

#%%

# load the data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# create a list of classnames
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# prepare the dataset for training
train_images, val_images, train_labels, val_labels = prepare_dataset(train_images, train_labels)

#%%

### Step 4. Build a new architecture from scratch

#### CNN Part 1. Input Layer

# recall the shape of the images in our dataset
print(train_images.shape)

#%%

## CHALLENGE Create the input layer

# Input layer of 32x32 images with three channels (RGB)
inputs_intro = keras.Input(shape=train_images.shape[1:])

#%%

#### CNN Part 2. Hidden Layers

##### **Convolutional Layers**



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
