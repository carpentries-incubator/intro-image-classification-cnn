# -*- coding: utf-8 -*-
"""
Image Classification with Convolutional Neural Networks

Episode 05 Evaluate a Convolutional Neural Network and Make Predictions (Classifications)

"""
#%%

# load the required packages
import tensorflow as tf # neural network
import matplotlib.pyplot as plt # plotting
import icwcnn_functions as icfn # pre-defined helpers
import numpy as np # for argmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns # specialised plotting

#%%

### Step 3. Prepare data

# create a list of class names associated with each CIFAR-10 label
class_names = ['airplane', 'bird', 'cat', 'dog', 'truck']

# load the data
train_ds, val_ds, test_ds = icfn.prepare_datasets()

#%%

# load pre-trained model
model_intro = tf.keras.models.load_model('../models/cifar_model_intro.keras')


#%%

#### Step 7. Perform a Prediction/Classification

# make predictions
predictions = model_intro.predict(x = test_ds)

# convert predictions to class labels
predicted_labels = tf.argmax(predictions, axis=1)

#%%

### Step 8. Measuring performance

test_labels = np.concatenate([y for x, y in test_ds], axis=0)
test_acc = accuracy_score(y_true=test_labels, y_pred=predicted_labels)
print('Accuracy:', round(test_acc,2))

#%%

#### Confusion matrix

# create a confusion matrix
conf_matrix = confusion_matrix(y_true=test_labels, y_pred=predicted_labels)
print(conf_matrix)

#%%

# heatmap visualization of the confusion matrix
sns.heatmap(data=conf_matrix, annot=True, fmt='d', 
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')

