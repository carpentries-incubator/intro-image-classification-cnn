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
print('We are using ', model_dropout.name, '\n')

# check test image dataset is loaded
print('The test image dataset has the shape: ', test_images.shape)

# use our current best model to predict probability of each class on new test set
predicted_prob = model_dropout.predict(test_images)

# create a dictionary to convert from string labels to numeric labels
#label_str_int_map = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
#TODO 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# convert probability predictions to table using class names for column names
prediction_df = pd.DataFrame(predicted_prob, columns=class_names)

# inspect 
print(prediction_df.head())

# now find the maximum probability for each image
predicted_labels = predicted_prob.argmax(axis=1)

### Step 8. Measuring Performance

# plot the predicted versus the true class
plt.plot(label_str_int_map[test_labels], predicted_labels)
plt.xlabel('Test Class')
plt.ylabel('Predicted Class')
plt.xlim(0, 9)
plt.ylim(0, 9)
#plt.axline(xy1=(0,0), xy2=(9,9), linestyle='--')
plt.show()