# -*- coding: utf-8 -*-
"""
Episode 01 Introduction to Deep Learning

"""

# load the required packages
from tensorflow import keras # for neural networks
from sklearn.model_selection import train_test_split # for splitting data into sets
import matplotlib.pyplot as plt # for plotting
import time

start = time.time()

# load the CIFAR-10 dataset included with keras
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
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

########################################################
# Challenge: Examine the CIFAR-10 dataset

print('Train: Images=%s, Labels=%s' % (train_images.shape, train_labels.shape))
print('Validate: Images=%s, Labels=%s' % (val_images.shape, val_labels.shape))
print('Test: Images=%s, Labels=%s' % (test_images.shape, test_labels.shape))

########################################################

#### Visualize a subset of the CIFAR-10 dataset

# create a figure object and specify width, height in inches
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i])
    plt.axis('off')
    plt.title(class_names[train_labels[i,].argmax()])
plt.show()


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
model_intro = keras.Model(inputs = inputs_intro, 
                          outputs = outputs_intro, 
                          name="cifar_model_intro")

# compile the model
model_intro.compile(optimizer = 'adam', 
                    loss = keras.losses.CategoricalCrossentropy(), 
                    metrics = ['accuracy'])

# fit the model
history_intro = model_intro.fit(train_images, train_labels, epochs = 10, 
                                validation_data = (val_images, val_labels),
                                batch_size = 32)


#### Perform a Prediction/Classification

# predict the classname of the first test image
result_intro = model_intro.predict(test_images[0].reshape(1,32,32,3))

print('The predicted probability of each class is: ', result_intro.round(4))
print('The class with the highest predicted probability is: ', class_names[result_intro.argmax()])

# plot the image with its true label
plt.imshow(test_images[0])
plt.title('True class:' + class_names[test_labels.item(0)])
plt.show()

# save the model
model_intro.save('fit_outputs/model_intro.keras')

end = time.time()

print()
print()
print("Time taken to run program was:", end - start, "seconds")


