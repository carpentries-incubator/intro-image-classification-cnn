# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 04 Compile and Train (Fit) a Convolutional Neural Network

"""

#%%

# load the required packages
import tensorflow as tf # neural network
import matplotlib.pyplot as plt # plotting
import icwcnn_functions as icfn # pre-defined helpers
import seaborn as sns # specialised plotting
import pandas as pd # handles dataframes


#%%

### Step 3. Prepare data

# create a list of class names associated with each CIFAR-10 label
class_names = ['airplane', 'bird', 'cat', 'dog', 'truck']

# load the data
train_ds, val_ds, test_ds = icfn.prepare_datasets()

#%%

### Step 4. Build the model architecture

# create the introduction model
model_intro = icfn.create_model_intro()
              
#%%

### Step 5. Choose a loss function and optimizer and compile model

## CHALLENGE Compile the model

## SOLUTION

# compile the model
model_intro.compile(optimizer = 'adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

#%%

### Step 6. Train (Fit) model
                                    
history_intro = model_intro.fit(x = train_ds,
                                epochs = 10,
                                validation_data = val_ds
)

#%%

# Monitor Training Progress (aka Model Evaluation during Training)

# convert the model history to a dataframe for plotting 
history_intro_df = pd.DataFrame.from_dict(history_intro.history)

# plot the loss and accuracy from the training process
fig, axes = plt.subplots(1, 2)
fig.suptitle('cifar_model_intro')

sns.lineplot(ax=axes[0], data=history_intro_df[['loss', 'val_loss']])
sns.lineplot(ax=axes[1], data=history_intro_df[['accuracy', 'val_accuracy']])

