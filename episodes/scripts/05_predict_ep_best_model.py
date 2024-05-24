# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 05 Evaluate a Convolutional Neural Network and Make Predictions (Classifications)

"""
#%%

# load the required packages

from tensorflow import keras # data and neural network
import seaborn as sns # specialised plotting
import pandas as pd # handles dataframes
import numpy as np # for argmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#%%

### Step 7. Perform a Prediction/Classification

#### Prepare test dataset

# load the CIFAR-10 dataset included with the keras library
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# create a list of classnames 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# normalize the RGB values to be between 0 and 1
test_images = test_images / 255.0

# check test image dataset is loaded - images and labels
print('Test: Images=%s, Labels=%s' % (test_images.shape, test_labels.shape))

#%%

#### Step 7. Perform a Prediction/Classification

# ## CHALLENGE Write the code to make class predictions on test data

# # load preferred model
# _____ = keras.models.load_model(_____)

# # use preferred model to predict
# _____ = _____.predict(x=_____)

#%%

## SOLUTION

# load preferred model
model_best = keras.models.load_model('fit_outputs/model_dropout.keras')
print('We are using', model_best.name)

# use preferred model to predict probability of each class on new test set
predictions = model_best.predict(x=test_images)

print(predictions)

#%%

# convert probability predictions to table using class names for column names
prediction_df = pd.DataFrame(data=predictions, columns=class_names)

# inspect 
print(prediction_df.head())

# convert predictions to class labels
predicted_labels = np.argmax(a=predictions, axis=1)
print(predicted_labels)

#%%

### Step 8. Measuring performance

# evaluate the model on the test data set
test_acc = accuracy_score(y_true=test_labels, y_pred=predicted_labels)
print('Accuracy:', round(test_acc,2))

#%%

#### Confusion matrix

# create a confusion matrix
conf_matrix = confusion_matrix(y_true=test_labels, y_pred=predicted_labels)
print(conf_matrix)

#%%

# Convert confustion matrix to a pandas dataframe
confusion_df = pd.DataFrame(data=conf_matrix, index=class_names, columns=class_names)

# Set the names of the x and y axis, this helps with the readability of the heatmap
confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'

# heatmap visualization of the confusion matrix
sns.heatmap(data=confusion_df, annot=True, fmt='3g')