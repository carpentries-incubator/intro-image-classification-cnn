---
title: "Introduction to Deep Learning"
teaching: 10
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions

- What is machine learning and what is it used for?
- What is deep learning?
- How do I use a neural network for image classification?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain the difference between artificial intelligence, machine learning and deep learning
- Explain how machine learning is used for regression and classification tasks
- Understand what algorithms are used for image classification
- Know difference between training, testing, and validation datasets
- Perform an image classification using a convolutional neural network (CNN)

::::::::::::::::::::::::::::::::::::::::::::::::

## What is machine learning?
Machine learning is a set of tools and techniques which let us find patterns in data. This lesson will introduce you to only one of these techniques, **Deep Learning** with **convolutional neural network**, abbreviated as **CNN**, but there are many more.

The techniques breakdown into two broad categories, predictors and classifiers. Predictors are used to predict a value (or set of values) given a set of inputs, for example trying to predict the cost of something given the economic conditions and the cost of raw materials or predicting a country’s GDP given its life expectancy. Classifiers try to classify data into different categories, or assign a label; for example, deciding what characters are visible in a picture of some writing or if a message is spam or not.

## Training Data

Many (but not all) machine learning systems “learn” by taking a series of input data and output data and using it to form a model. The maths behind the machine learning doesn’t care what the data is as long as it can represented numerically or categorised. Some examples might include:

- predicting a person’s weight based on their height
- predicting house prices given stock market prices
- classifying if an email is spam or not
- classifying an image as eg, person, place, or particular object

Typically we will need to train our models with hundreds, thousands or even millions of examples before they work well enough to do any useful predictions or classifications with them.


## Deep Learning, Machine Learning and Artificial Intelligence

Deep Learning (DL) is just one of many machine learning techniques and people often talk about machine learning being a form of artificial intelligence (AI). Definitions of artificial intelligence vary, but usually involve having computers mimic the behaviour of intelligent biological systems. Since the 1950s many works of science fiction have dealt with the idea of an artificial intelligence which matches (or exceeds) human intelligence in all areas. Although there have been great advances in AI and ML research recently we can only come close to human like intelligence in a few specialist areas and are still a long way from a general purpose AI. The image below shows some differences between artificial intelligence, Machine Learning and Deep Learning.

![The image above is by Tukijaaliwa, CC BY-SA 4.0, via Wikimedia Commons, [original source]](fig/01_AI_ML_DL_differences.png){alt='Three nested circles describing AI as the largest circle in dark blue; enclosing machine learning in medium blue; enclosing deep learning in even lighter blue'}

::::::::::::::::::::::::::::::::::::::::: callout
Concept: Differentiation between classical ML models and Deep Learning models

Traditional ML algorithms can only use one (possibly two layers) of data transformation to calculate an output (shallow models). With high dimensional data and growing feature space (possible set of values for any given feature), shallow models quickly run out of layers to calculate outputs. 

Deep neural networks (constructed with multiple layers of neurons) are the extension of shallow models with three layers: input, hidden, and outputs layers. The hidden layer is where learning takes place. As a result, deep learning is best applied to large datasets for training and prediction. As observations and feature inputs decrease, shallow ML approaches begin to perform noticeably better. 
:::::::::::::::::::::::::::::::::::::::::::::::::

## What is image classification?
![](fig/01_Fei-Fei_Li_Justin_Johnson_Serena_Young__CS231N_2017.png){alt='Four types of image classification tasks include semantic segmentation where every pixel is labelled; classification and localization that detects a single object like a cat; object detection that detects multiple objects like cats and dogs; and instance segmentation that detects each pixel of multiple objects'}

## Deep Learning Workflow
To apply Deep Learning to a problem there are several steps we need to go through:

### 1. Formulate/ Outline the problem
Firstly we must decide what it is we want our Deep Learning system to do. This lesson is all about image classification so our aim is to put an image into one of a few categories. Specifically in our case, we will be looking at 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck'

### 2. Identify inputs and outputs
Next we need to identify what the inputs and outputs of the neural network will be. In our case, the data is images and the inputs could be the individual pixels of the images. We are performing a classification problem and we will have one output for each potential class.

### 3. Prepare data
Many datasets are not ready for immediate use in a neural network and will require some preparation. Neural networks can only really deal with numerical data, so any non-numerical data (eg images) will have to be somehow converted to numerical data.

Next we will need to divide the data into multiple sets. One of these will be used by the training process and we will call it the **training set**. Another set, called the **validation set**, will be used during the training process to tune hyperparameters. A third **test set** is used to assess the final performance of the trained model.

For this lesson, we will be using an existing image dataset known as CIFAR-10 that we will discuss in more depth in the next episode.

### Load data

```python
# load the cifar dataset included with the keras packages
from tensorflow import keras

(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()
```

::::::::::::::::::::::::::::::::::::: challenge 

## Challenge Load the Cifar10 dataset

Explain the output of these commands?

```python
print('Train: Images=%s, Labels=%s' % (train_images.shape, train_labels.shape))
print('Validate: Images=%s, Labels=%s' % (val_images.shape, val_labels.shape))
```

:::::::::::::::::::::::: solution 

## Output
 
```output
Train: Images=(50000, 32, 32, 3), Labels=(50000, 1)
Validate: Images=(10000, 32, 32, 3), Labels=(10000, 1)
```
The training set consists of 50000 images of 32x32 pixels and 3 channels (RGB values) and labels.
The validation set consists of 10000 images of 32x32 pixels and 3 channels (RGB values) and labels.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

Image RGB values are between 0 and 255. For input of neural networks, it is better to have small input values. So we normalize our data between 0 and 1.

::::::::::::::::::::::::::::::::::::::::: callout
ChatGPT
Normalizing the RGB values to be between 0 and 1 is a common pre-processing step in machine learning tasks, especially when dealing with image data. This normalization has several benefits:

1. **Numerical Stability**: By scaling the RGB values to a range between 0 and 1, you avoid potential numerical instability issues that can arise when working with large values. Neural networks and many other machine learning algorithms are sensitive to the scale of input features, and normalizing helps to keep the values within a manageable range.

2. **Faster Convergence**: Normalizing the RGB values often helps in faster convergence during the training process. Neural networks and other optimization algorithms rely on gradient descent techniques, and having inputs in a consistent range aids in smoother and faster convergence.

3. **Equal Weightage for All Channels**: In RGB images, each channel (Red, Green, Blue) represents different color intensities. By normalizing to the range [0, 1], you ensure that each channel is treated with equal weightage during training. This is important because some machine learning algorithms could assign more importance to larger values.

4. **Generalization**: Normalization helps the model to generalize better to unseen data. When the input features are in the same range, the learned weights and biases can be more effectively applied to new examples, making the model more robust.

4. **Compatibility**: Many image-related libraries, algorithms, and models expect pixel values to be in the range of [0, 1]. By normalizing the RGB values, you ensure compatibility and seamless integration with these tools.

The normalization process is typically done by dividing each RGB value (ranging from 0 to 255) by 255, which scales the values to the range [0, 1].

For example, if you have an RGB image with pixel values (100, 150, 200), after normalization, the pixel values would become (100/255, 150/255, 200/255) ≈ (0.39, 0.59, 0.78).

Remember that normalization is not always mandatory, and there could be cases where other scaling techniques might be more suitable based on the specific problem and data distribution. However, for most image-related tasks in machine learning, normalizing RGB values to [0, 1] is a good starting point.
:::::::::::::::::::::::::::::::::::::::::::::::::::


```python
# normalize the RGB values to be between 0 and 1
train_images = train_images / 255.0
val_images = val_images / 255.0
```
The labels are a set of single numbers denoting the class and we map the class numbers back to the class names, taken from the documentation:

```python
# create a list of classnames
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

### Visualize a subset of the Cifar10 dataset

```python

import matplotlib.pyplot as plt

# create a figure object and specify width, height in inches
plt.figure(figsize=(10,10))

# plot a subset of the images 
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(class_names[train_labels[i,0]])
plt.show()
```

![](fig/01_cifar10.png){alt='Subset of 25 Cifar10 images displayed in five rows and five columns '}

### 4. Choose a pre-trained model or build a new architecture from scratch

Often we can use an existing neural network instead of designing one from scratch. Training a network can take a lot of time and computational resources. There are a number of well publicised networks which have been shown to perform well at certain tasks, if you know of one which already does a similar task well then it makes sense to use one of these.

If instead we decide we do want to design our own network then we need to think about how many input neurons it will have, how many hidden layers and how many outputs, what types of layers we use (we will explore the different types later on). This will probably need some experimentation and we might have to try tweaking the network design a few times before we see acceptable results.

TODO decide to simplify this model or present as is

#### Define the Model

```python
# define the inputs, layers, and outputs of a cnn model
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

# create the model
model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")
```

### 5. Choose a loss function and optimizer

The loss function tells the training algorithm how far away the predicted value was from the true value. We will look at choosing a loss function in more detail later on.

The optimizer is responsible for taking the output of the loss function and then applying some changes to the weights within the network. It is through this process that the “learning” (adjustment of the weights) is achieved.

```python
# compile the model
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
```

### 6. Train the model

We can now go ahead and start training our neural network. We will probably keep doing this for a given number of iterations through our training dataset (referred to as epochs) or until the loss function gives a value under a certain threshold.

```python
# fit the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# save the model
model.save('01_intro_model.h5')
``` 

### 7. Perform a Prediction/Classification

After training the network we can use it to perform predictions. This is the mode you would use the network in after you have fully trained it to a satisfactory performance. Doing predictions on a special hold-out set is used in the next step to measure the performance of the network.

```python
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

# load a new image and prepare it to match cifar10 dataset
new_img_pil = load_img("01_Jabiru_TGS.JPG", target_size=(32,32)) # Image format
new_img_arr = img_to_array(new_img_pil) # convert to array for analysis
new_img_reshape = new_img_arr.reshape(1, 32, 32, 3) # reshape into single sample
new_img_float =  new_img_reshape.astype('float64') / 255.0 # normalize

# predict the classname
result = model.predict(new_img_float) # make prediction
print(result) # probability for each class
print(class_names[result.argmax()]) # class with highest probability
```

```output
Result: [[-2.0185328   9.337507   -2.4551604  -0.4688053  -4.599108   -3.5822825
   6.427376   -0.09437321  0.82065487  1.2978227 ]]
Class name: automobile
```

Congratulations, you just created your first image classification model and used it to classify an image! 

Unfortunately the classification was incorrect. Why might that be?

::::::::::::::::::::::::::::::::::::: challenge 

Why did our model classify a bird as a plane?

:::::::::::::::::::::::: solution 

After resizing the original image to be the size expected by our model it looks like this:

![](fig/01_Jabiru_w_ReducedPixels.png){alt='Subset of 25 Cifar10 images displayed in five rows and five columns '}

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

What can we do about? 

There are many ways we can try to improve the accuracy of our model, such as adding or removing layers to the model definition and fine-tuning the hyperparameters, which takes us to the next steps in our workflow.

### 8. Measure Performance

Once we trained the network we want to measure its performance. To do this we use some additional data that was **not** part of the training; this is known as a test set. There are many different methods available for measuring performance and which one is best depends on the type of task we are attempting. These metrics are often published as an indication of how well our network performs.

### 9. Tune Hyperparameters

Hyperparameters are all the parameters set by the person configuring the machine learning instead of those learned by the algorithm itself. The hyperparameters include the number of epochs or the parameters for the optimizer. It might be necessary to adjust these and re-run the training many times before we are happy with the result.

### 10. Share Model
Now that we have a trained network that performs at a level we are happy with we can go and use it on real data to perform a prediction. At this point we might want to consider publishing a file with both the architecture of our network and the weights which it has learned (assuming we did not use a pre-trained network). This will allow others to use it as as pre-trained network for their own purposes and for them to (mostly) reproduce our result.

We will return to these workflow steps throughout this lesson and discuss each component in more detail.


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: instructor

Inline instructor notes can help inform instructors of timing challenges
associated with the lessons. They appear in the "Instructor View"

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Machine learning is the process where computers learn to recognise patterns of data
- Machine learning is used for regression and classification tasks
- Deep learning is a subset of machine learning, which is a subset of artificial intelligence
- Convolutional neural networks are well suited for image classification
- To use Deep Learning effectively we need to go through a workflow of: defining the problem, identifying inputs and outputs, preparing data, choosing the type of network, choosing a loss function, training the model, tuning Hyperparameters, measuring performance before we can classify data.
::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->
[original source]: https://en.wikipedia.org/wiki/File:AI-ML-DL.svg
