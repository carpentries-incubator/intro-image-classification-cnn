# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 05 Evaluate a Convolutional Neural Network and Make Predictions (Classifications)

# Step 9. Tune hyperparameters

## CHALLENGE Tune Activation Function using For Loop

"""
#%%

# load the required packages

from tensorflow import keras # data and neural network
from sklearn.model_selection import train_test_split # data splitting
import matplotlib.pyplot as plt # plotting
import time # track run time

#%%

# start timer
start = time.time()

#%%

# function to prepare the training dataset

def prepare_dataset(train_images, train_labels):
    
    # normalize the RGB values to be between 0 and 1
    train_images = train_images / 255.0
    
    # one hot encode the training labels
    train_labels = keras.utils.to_categorical(train_labels, len(class_names))
    
    # split the training data into training and validation set
    train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size = 0.2, random_state=42)

    return train_images, val_images, train_labels, val_labels

#%%

# load the data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# create a list of class names associated with each CIFAR-10 label
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# prepare the dataset for training
train_images, val_images, train_labels, val_labels = prepare_dataset(train_images, train_labels)

#%%

### Step 9. Tune hyperparameters

## CHALLENGE Tune Activation Function using For Loop

# modify the intro model to sample activation functions
def create_model_act(activation_function):

    # CNN Part 1
    # Input layer of 32x32 images with three channels (RGB)
    inputs_act = keras.Input(shape=train_images.shape[1:])
    
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_act = keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation=activation_function)(inputs_act)
    # Pooling layer with input window sized 2x2
    x_act = keras.layers.MaxPooling2D((2, 2))(x_act)
    # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    x_act = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=activation_function)(x_act)
    # Second Pooling layer with input window sized 2x2
    x_act = keras.layers.MaxPooling2D(pool_size=(2,2))(x_act)
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_act = keras.layers.Flatten()(x_act)
    # Dense layer with 64 neurons and ReLU activation
    x_act = keras.layers.Dense(units=64, activation=activation_function)(x_act)
    
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_act = keras.layers.Dense(units=10, activation='softmax')(x_act)
    
    # create the model
    model_act = keras.Model(inputs = inputs_act, 
                              outputs = outputs_act, 
                              name="cifar_model_activation")
    
    # compile the model
    model_act.compile(optimizer = keras.optimizers.Adam(),
                      loss = keras.losses.CategoricalCrossentropy(),
                      metrics = keras.metrics.CategoricalAccuracy())

    return model_act

#%%

# create a ist of activation functions to try
activations = ['relu', 'sigmoid', 'tanh', 'selu', keras.layers.LeakyReLU()]

# create a dictionary object to store the training history
history_data = {} # dictionary

# train the model with each activation function and store the history
for activation in activations:
    
    # create the model
    model = create_model_act(activation)
    
    # fit the model
    history = model.fit(x = train_images, y = train_labels,
                        batch_size = 32,
                        epochs = 10, 
                        validation_data = (val_images, val_labels))
    
    # add training history to dictionary
    history_data[str(activation)] = history

# plot the validation accuracy for each activation function
plt.figure(figsize=(12, 6))

for activation, history in history_data.items():
    plt.plot(history.history['val_categorical_accuracy'], label=activation)

plt.title('Validation accuracy for different activation functions')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()

#%%

end = time.time()

print()
print()
print("Time taken to run program was:", end - start, "seconds")