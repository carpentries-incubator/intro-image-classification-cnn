---
title: "Build a Convolutional Neural Network"
teaching: 45
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions

- What is a neural network?
- How is a convolutional neural network (CNN) different from an ANN?
- What are the types of layers used to build a CNN?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand how a convolutional neural network (CNN) differs from an artificial neural network (ANN).
- Explain the terms: kernel, filter.
- Know the different layers: convolutional, pooling, flatten, dense.

::::::::::::::::::::::::::::::::::::::::::::::::

In Episode 1, we used a pre-defined model to train our classifier.

In this episode, we’ll build that model ourselves, one step at a time.

We don’t need to understand all the maths behind neural networks — instead, we’ll focus on how to put the pieces together.


## Neural Networks

A **neural network** is a series of layers that transform input data step by step. 
A **convolutional neural network (CNN)** is a type of neural network designed specifically for images.

It learns by passing image data through a series of layers, each one transforming the data slightly, until it can make a final prediction.


### Step 4. Build an architecture

We build a CNN by stacking layers together.

Each layer takes some input, transforms it, and passes it to the next layer, i.e., the output from each layer becomes the input to the next layer.

There are three main components of a neural network:

- CNN Part 1. Input Layer
- CNN Part 2. Hidden Layers
- CNN Part 3. Output Layer

#### CNN Part 1. Input Layer

We start by telling Keras the shape of our images. Recall the size of our images are 32x32 pixels with 3 colour channels (RGB).

```python
# input layer 
inputs_intro = tf.keras.Input(shape=(32,32,3))
```

#### CNN Part 2. Hidden Layers

The next component consists of the so-called hidden layers of the network.

In a neural network, the input layer receives the raw data, and the output layer produces the final predictions or classifications. These layers' contents are directly observable because you can see the input data and the network's output predictions.

However, the hidden layers, which lie between the input and output layers, contain intermediate representations of the input data.

In a CNN, the hidden layers typically consist of convolutional, pooling, reshaping (e.g., Flatten), and dense layers.


#### **Convolutional Layers**

Convolutional layers look for simple patterns in images, such as edges or textures.

To create a convolutional layer, we need to specify:

- the number of features to learn, `filters`
- the size of the search window, `kernel_size`
	- smaller kernels are used to capture fine-grained features
	- odd-sized windows are common because they have a centre pixel
- the activation function to use, `activation`

When building a model, each layer takes the output from the previous layer as its input.

```python
# hidden layers
x_intro = tf.keras.layers.Conv2D(filters=16, 
                                 kernel_size=(3,3),
                                 activation='relu')(inputs_intro)
```

::::::::::::::::::::::::::::::::::::: challenge 
## Create a convolutional layer

Create a Conv2D layer with:
- 16 filters
- a 3x3 kernel size
- 'relu' activation function.

Use `inputs_intro` as the input.
```python
x_intro = tf.keras.layers.Conv2D(filters=_____, 
                                 kernel_size=_____, 
                                 activation=_____)(inputs_intro)
```

:::::::::::::::::::::::: solution 
```python
x_intro = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu')(inputs_intro)
```
:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::                              

#### **Pooling Layers**

Pooling layers reduce the size of the image, helping the model focus on the most important features.

To create a pooling layer, we need to specify how much to reduce the image by using `pool_size`. A pool size of (2, 2) reduces the width and height of the image by half.

```python
x_intro = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x_intro)
```

::::::::::::::::::::::::::::::::::::::::: callout
## Deep Learning

We often repeat this two-layer pattern to learn more complex features by, e.g. increasing the number of filters:

```python
x_intro = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(inputs_intro)
x_intro = tf.keras.layers.MaxPooling2D((2,2))(x_intro)

x_intro = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x_intro)
x_intro = tf.keras.layers.MaxPooling2D((2, 2))(x_intro)
```
:::::::::::::::::::::::::::::::::::::::::

Up to this point, our model has been working with 2D image data.

Next, we convert it into a format suitable for classification.

#### Reshaping Layers: **Flatten**

Flatten turns our 2D feature maps into a single list of numbers.

```python
x_intro = tf.keras.layers.Flatten()(x_intro)
```

#### CNN Part 3. Output Layer

#### **Dense layers**

To make a predcition, we use a dense (fully connected) layer after reshaping.  This is because `Dense` layers expect 1D input.

To create a dense layer, we need to specify:

- the number of outputs to produce, `units`
- the activation function, `activation`

For classification problems:
- `units` is the number of classes  
- `softmax` converts the outputs into probabilities that add up to 1  


```python
outputs_intro = tf.keras.layers.Dense(units=5, activation='softmax')(x_intro)
```

## Putting it all together

Once you decide on the initial architecture of your CNN, the last step to create the model is to bring all of the parts together:

```python
model_intro = tf.keras.Model(inputs = inputs_intro,
                             outputs = outputs_intro, 
                             name = "cifar_model_intro")
```

We now have a simple convolutional neural network! 

We can put this code inside a function definition like we used in Episode 1.

:::::::::::::::::::::::::::::::::::: challenge
## Turn your neural network into a function

Use the architecture from this episode to define a function called `create_model_intro`.

- Hint 1 Your function should take two arguments, `input_shape` and `num_classes`
- Hint 2 Your function should return a model object

:::::::::::::::::::::::: solution
```python
# function that defines an introductory convolutional neural network
def create_model_intro(input_shape=(32,32,3), num_classes=5):
    
    # CNN Part 1
    # Input layer of 32x32 images with three channels (RGB)
    inputs_intro = tf.keras.Input(shape=input_shape)
    
    # CNN Part 2
    x_intro = tf.keras.layers.Conv2D(filters=16, 
                                     kernel_size=(3,3), 
                                     activation='relu')(inputs_intro)
    x_intro = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x_intro)
    x_intro = tf.keras.layers.Flatten()(x_intro)
    
    # CNN Part 3
    # Output layer with one unit for each class and softmax activation
    outputs_intro = tf.keras.layers.Dense(units=num_classes,
                                          activation='softmax')(x_intro)
    
    # create the model
    model_intro = tf.keras.Model(inputs = inputs_intro,
                                 outputs = outputs_intro, 
                                 name = "cifar_model_intro")
    
    return model_intro
```
::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

### Viewing the model

Once a model is created, we can look at a summary of its structure.

```python
# create the introduction model
model_intro = create_model_intro()

# view model summary
model_intro.summary()
```
```output
Model: "cifar_model_intro"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 32, 32, 3)]       0         
                                                                 
 conv2d (Conv2D)             (None, 30, 30, 16)        448       
                                                                 
 max_pooling2d (MaxPooling2  (None, 15, 15, 16)        0         
 D)                                                              
                                                                 
 flatten (Flatten)           (None, 3600)              0         
                                                                 
 dense (Dense)               (None, 5)                 18005     
                                                                 
=================================================================
Total params: 18453 (72.08 KB)
Trainable params: 18453 (72.08 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

In the model summary you’ll see:

- the layers in order
- how the data shape changes
- how many parameters the model learns

Don’t worry about all the details — this is just a useful way to inspect your model.

:::::::::::::::::::::::::::::::::::::: callout
## How do we choose this architecture?

You might be wondering how we decided on the number of layers and their settings.

In practice, there’s no single “correct” answer — building models often involves some trial and error.

A common approach is to:

- start with a simple model  
- train it and see how it performs  
- gradually add layers and adjust settings to improve it  

We’ll explore this idea later in the lesson.

::::::::::::::::::::::::::::::::::::::::::::::

## We have a model — now what?

We now have a simple CNN that can take an image and produce a prediction.

At the moment, the model hasn’t learned anything yet — it still needs to be trained on our data.

In the next step, we’ll:

- tell the model how to learn (choose a loss function and optimizer)  
- train it on our training data  

This is where the model starts improving and learning patterns from the images.


::::::::::::::::::::::::::::::::::::: keypoints 
- Convolutional neural networks (CNNs) are designed for working with image data.
- CNNs are built by stacking layers, where each layer transforms the data and passes it to the next layer.
- Convolutional layers look for simple patterns in images (e.g. edges and textures).
- Pooling layers reduce the size of the data, helping the model focus on important features.
- The Flatten layer converts image data into a format suitable for classification.
- Dense layers are used to produce the final prediction.
::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->
[CC BY-SA 3.0]: https://creativecommons.org/licenses/by-sa/3.0
[original source]: https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg
[Layers API]: https://keras.io/api/layers/
[Image kernels explained]: https://setosa.io/ev/image-kernels/
[convolutional neural network cheat sheet]: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
[Keras Models API]: https://keras.io/api/models/

