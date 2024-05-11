# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 05 Evaluate a Convolutional Neural Network and Make Predictions (Classifications)

# Step 9. Tune hyperparameters

## CHALLENGE Tune Dropout Rate (Model Build) using a For Loop

"""
#%%

# load the required packages

from tensorflow import keras # data and neural network
from sklearn.model_selection import train_test_split # data splitting
import seaborn as sns # specialised plotting
import pandas as pd # handles dataframes
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

# create a list of classnames
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# prepare the dataset for training
train_images, val_images, train_labels, val_labels = prepare_dataset(train_images, train_labels)

# prepare test dataset
# normalize the RGB values to be between 0 and 1
test_images = test_images / 255.0

#%%

### Step 9. Tune hyperparameters

## CHALLENGE Tune Dropout Rate (Model Build) using a For Loop

#%%

# define new dropout function that accepts a dropout rate

def create_model_dropout_vary(dropout_rate):
    
    # Input layer of 32x32 images with three channels (RGB)
    inputs_vary = keras.Input(shape=train_images.shape[1:])
    
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_vary = keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs_vary)
    # Pooling layer with input window sized 2x2
    x_vary = keras.layers.MaxPooling2D(pool_size=(2,2))(x_vary)
    # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    x_vary = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x_vary)
    # Second Pooling layer with input window sized 2x2
    x_vary = keras.layers.MaxPooling2D(pool_size=(2,2))(x_vary)
    # Second Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    x_vary = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x_vary)
    # Dropout layer randomly drops x% of the input units
    x_vary = keras.layers.Dropout(rate=dropout_rate)(x_vary) # This is new!
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_vary = keras.layers.Flatten()(x_vary)
    
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_vary = keras.layers.Dense(units=10, activation='softmax')(x_vary)

    model_vary = keras.Model(inputs = inputs_vary, 
                             outputs = outputs_vary, 
                             name ="cifar_model_dropout_vary")

    return model_vary

#%%

# specify range of dropout rates
dropout_rates = [0.15, 0.3, 0.45, 0.6, 0.75]

# create empty list to hold losses
val_losses_vary = [] 

# use for loop to explore varying the dropout rate
for dropout_rate in dropout_rates:
    
    # create the model
    model_vary = create_model_dropout_vary(dropout_rate)
    
    # compile the model
    model_vary.compile(optimizer = keras.optimizers.Adam(),
                      loss = keras.losses.CategoricalCrossentropy(),
                      metrics = keras.metrics.CategoricalAccuracy())

    # fit the model
    model_vary.fit(x = train_images, y = train_labels,
                   batch_size = 32,
                   epochs = 10,
                   validation_data = (val_images, val_labels))

    # evaluate the model on the test data set
    val_loss_vary, val_acc_vary = model_vary.evaluate(val_images, val_labels)
    
    # save the evaulation metrics
    val_losses_vary.append(val_loss_vary)

# convert rates and metrics to dataframe for plotting
loss_df = pd.DataFrame({'dropout_rate': dropout_rates, 'val_loss_vary': val_losses_vary})

# plot the loss and accuracy from the training process
sns.lineplot(data=loss_df, x='dropout_rate', y='val_loss_vary')

#%%

end = time.time()

print()
print()
print("Time taken to run program was:", end - start, "seconds")
