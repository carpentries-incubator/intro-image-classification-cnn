# -*- coding: utf-8 -*-
"""
Evaluate a Convolutional Neural Network and Make Predictions (Classifications)

"""
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# load weights for your best model if not still in memory
model_dropout = keras.models.load_model('C:/Users/jc140298/Documents/Software_Carpentry/20230316_ML_AI/scripts/outputs_cinic10/model_dropout.h5')

# recreate test_images from 'image-data.py' if not still in memory

# check correct model is loaded
print('We are using ', model_dropout.name)

# check test dataset is loaded - images and labels
print('The number and shape of images in our test dataset is: ', test_images.shape)
print('The number of labels in our test dataset is: ', len(test_labels))

# use our current best model to predict probability of each class on new test set
predicted_prob = model_dropout.predict(test_images)

# convert probability predictions to table using class names for column names
prediction_df = pd.DataFrame(predicted_prob, columns=class_names)

# inspect 
print(prediction_df.head())

# now find the maximum probability for each image
predicted_labels = predicted_prob.argmax(axis=1)

### Step 8. Measuring Performance

# plot the predicted versus the true class

# training labels are numeric; want test labels to the same for plotting
# need the list of classnames to convert test_labels to test_values
# recall train_values were numeric, not strings
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# use element position in class_names to generate values
test_values = [] 
for i in range(len(test_labels)):
    test_values.append(class_names.index(test_labels[i]))
    
# make the plot
plt.scatter(test_labels_values, predicted_labels)
plt.xlabel('Test Class')
plt.ylabel('Predicted Class')
plt.xlim(0, 9)
plt.ylim(0, 9)
#plt.axline(xy1=(0,0), xy2=(9,9), linestyle='--') # expected
plt.show()

# confusion matrix

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(test_labels_values, predicted_labels)
print(conf_matrix)

# Convert to a pandas dataframe
confusion_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Set the names of the x and y axis, this helps with the readability of the heatmap.
confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'

# heatmap visualization of the confusion matrix
import seaborn as sns

sns.heatmap(confusion_df, annot=True, fmt='3g')

# Gridsearch

# Load data
from tensorflow import keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

def create_model():
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
    model = keras.Model(inputs=inputs_intro, outputs=outputs_intro, name="cifar_model_intro")
    model.compile(optimizer = 'adam', loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

#Wrap the model
model = KerasClassifier(build_fn=create_model, epochs=2, batch_size=32, verbose=0)  # epochs, batch_size, verbose can be adjusted as required. Using low epochs to save computation time and demonstration purposes only

# Define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adam']
param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(train_images, train_labels)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#### Assessing activiation function performance

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define a function to create a model with a given activation function
def create_model(activation_function):
    model = Sequential([
        Conv2D(32, (3, 3), activation=activation_function, input_shape=(32, 32, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation=activation_function),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# List of activation functions to try
activations = ['relu', 'sigmoid', 'tanh', 'selu', tf.keras.layers.LeakyReLU()]

history_data = {}

# Train a model with each activation function and store the history
for activation in activations:
    model = create_model(activation)
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    history_data[str(activation)] = history

# Plot the validation accuracy for each activation function
plt.figure(figsize=(12, 6))

for activation, history in history_data.items():
    plt.plot(history.history['val_accuracy'], label=activation)

plt.title('Validation accuracy for different activation functions')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()


# Step 10 Share the model
# TODO Here or separate script?

from icwithcnn_functions import prepare_image_icwithcnn

# load a saved model
pretrained_model = keras.models.load_model('C:/Users/jc140298/ext.h5')

new_img_path = "../data/Jabiru_TGS.JPG" # path to image
new_img_prepped = prepare_image_icwithcnn(new_img_path)

# predict the class name
y_pretrained_pred = pretrained_model.predict(new_img_prepped)
pretrained_predicted_class = class_names[y_pretrained_pred.argmax()]
print(pretrained_predicted_class)
