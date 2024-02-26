# -*- coding: utf-8 -*-
"""
Episode 05 Evaluate a Convolutional Neural Network and Make Predictions (Classifications)

"""
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Step 7. Perform a Prediction/Classification

# load your best model
model_best = keras.models.load_model('fit_outputs/model_dropout.keras')
print('We are using', model_best.name)

# load the CIFAR-10 dataset included with the keras library
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# normalize the RGB values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# create a list of classnames 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# one-hot encode training labels
train_labels = keras.utils.to_categorical(train_labels, len(class_names))

# split the training data into training and validation sets
# NOTE the function is train_test split but we're using it to split train into train and validation
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                      test_size = 0.2, 
                                                                      random_state = 42)

# check test image dataset is loaded - images and labels
print('The number and shape of images in our test dataset is:', test_images.shape)
print('The number of labels in our test dataset is:', len(test_labels))

# use our current best model to predict probability of each class on new test set
predictions = model_best.predict(test_images)

# convert probability predictions to table using class names for column names
prediction_df = pd.DataFrame(predictions, columns=class_names)

# inspect 
print(prediction_df.head())

# convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)

# Step 8. Measuring Performance

# evaluate the model on the test data set
test_acc = accuracy_score(test_labels, predicted_labels)
print('Accuracy:', round(test_acc,2))

# create a confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)
print(conf_matrix)

# Convert to a pandas dataframe
confusion_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Set the names of the x and y axis, this helps with the readability of the heatmap.
confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'

# heatmap visualization of the confusion matrix
sns.heatmap(confusion_df, annot=True, fmt='3g')

# Step 9. Tune hyperparameters

########################################################
## CHALLENGE Tune Dropout Rate (Model Build) using a For Loop

# specify range of dropout rates
dropout_rates = [0.15, 0.3, 0.45, 0.6, 0.75]

# create empty list to hold losses
val_losses_vary = [] 

for dropout_rate in dropout_rates:
    
    # Input layer of 32x32 images with three channels (RGB)
    inputs_vary = keras.Input(shape=train_images.shape[1:])
    
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_vary = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs_vary)
    # Pooling layer with input window sized 2,2
    x_vary = keras.layers.MaxPooling2D((2, 2))(x_vary)
    # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    x_vary = keras.layers.Conv2D(32, (3, 3), activation='relu')(x_vary)
    # Second Pooling layer with input window sized 2,2
    x_vary = keras.layers.MaxPooling2D((2, 2))(x_vary)
    # Second Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    x_vary = keras.layers.Conv2D(64, (3, 3), activation='relu')(x_vary)
    # Dropout layer randomly drops x% of the input units
    x_vary = keras.layers.Dropout(dropout_rate)(x_vary) # This is new!
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_vary = keras.layers.Flatten()(x_vary)
    # Dense layer with 128 neurons and ReLU activation
    x_vary = keras.layers.Dense(128, activation='relu')(x_vary)
    
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_vary = keras.layers.Dense(10, activation='softmax')(x_vary)

    model_vary = keras.Model(inputs = inputs_vary, outputs = outputs_vary, 
                             name ="cifar_model_vary_dropout")

    model_vary.compile(optimizer = 'adam',
                       loss = keras.losses.CategoricalCrossentropy(),
                       metrics = ['accuracy'])

    model_vary.fit(train_images, train_labels, 
                   epochs = 20,
                   validation_data = (val_images, val_labels),
                   batch_size = 32)

    val_loss_vary, val_acc_vary = model_vary.evaluate(val_images, val_labels)
    val_losses_vary.append(val_loss_vary)

loss_df = pd.DataFrame({'dropout_rate': dropout_rates, 'val_loss_vary': val_losses_vary})

sns.lineplot(data=loss_df, x='dropout_rate', y='val_loss_vary')

########################################################

########################################################
# Challenge Tune Optimizer using Grid Search

# use the intro model for gridsearch
def create_model():

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
    model_intro = keras.Model(inputs = inputs_intro, 
                              outputs = outputs_intro, 
                              name="cifar_model_intro")
    
    # compile the model
    model_intro.compile(optimizer = 'adam', 
                        loss = keras.losses.CategoricalCrossentropy(), 
                        metrics = ['accuracy'])

    return model_intro

# Wrap the model
model = KerasClassifier(model=create_model, epochs=2, batch_size=32, verbose=0)  # epochs, batch_size, verbose can be adjusted as required. Using low epochs to save computation time and demonstration purposes only

# Define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adam']
param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(train_images, train_labels)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

########################################################

########################################################
## CHALLENGE Tune Activation Function using For Loop

# use the intro model for activation function
def create_model(activation_function):

    # CNN Part 1
    # Input layer of 32x32 images with three channels (RGB)
    inputs_intro = keras.Input(shape=train_images.shape[1:])
    
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(16, (3, 3), activation=activation_function)(inputs_intro)
    # Pooling layer with input window sized 2,2
    x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
    # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    x_intro = keras.layers.Conv2D(32, (3, 3), activation=activation_function)(x_intro)
    # Second Pooling layer with input window sized 2,2
    x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_intro = keras.layers.Flatten()(x_intro)
    # Dense layer with 64 neurons and ReLU activation
    x_intro = keras.layers.Dense(64, activation=activation_function)(x_intro)
    
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_intro = keras.layers.Dense(10, activation='softmax')(x_intro)
    
    # create the model
    model_intro = keras.Model(inputs = inputs_intro, 
                              outputs = outputs_intro, 
                              name="cifar_model_intro")
    
    # compile the model
    model_intro.compile(optimizer = 'adam', 
                        loss = keras.losses.CategoricalCrossentropy(), 
                        metrics = ['accuracy'])

    return model_intro

# List of activation functions to try
activations = ['relu', 'sigmoid', 'tanh', 'selu', keras.layers.LeakyReLU()]

history_data = {} # dictionary

# Train a model with each activation function and store the history
for activation in activations:
    
    model = create_model(activation)
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
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

########################################################