---
title: "Build a Convolutional Neural Network"
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions

- What is a (artificial) neural network (ANN)?
- How is a convolutional neural network (CNN) different from an ANN?
- What are the types of layers used to build a CNN?
- How do you monitor the training process?
- What is underfitting?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand how a convolutional neural network (CNN) differs from an artificial neural network (ANN)
- Explain the terms: kernel, filter
- Know the different layers: convolutional, pooling, flatten, dense

::::::::::::::::::::::::::::::::::::::::::::::::

## Neural Networks

A neural network is an artificial intelligence technique loosely based on the way neurons in the brain work. A neural network consists of connected computational units called neurons. Each neuron ...

- has one or more inputs, e.g. input data expressed as floating point numbers
- most of the time, each neuron conducts 3 main operations:
    - take the weighted sum of the inputs
    - add an extra constant weight (i.e. a bias term) to this weighted sum
    - apply a non-linear function to the output so far (using a predefined activation function)
- return one output value, again a floating point number

![](fig/03_neuron.png){alt=''}

Multiple neurons can be joined together by connecting the output of one to the input of another. These connections are associated with weights that determine the 'strength' of the connection, the weights are adjusted during training. In this way, the combination of neurons and connections describe a computational graph, an example can be seen in the image below. In most neural networks neurons are aggregated into layers. Signals travel from the input layer to the output layer, possibly through one or more intermediate layers called hidden layers. The image below shows an example of a neural network with three layers, each circle is a neuron, each line is an edge and the arrows indicate the direction data moves in.

![The image above is by Glosser.ca, [CC BY-SA 3.0], via Wikimedia Commons, [original source]](fig/03_neural_net.png){alt=''}

Neural networks aren't a new technique, they have been around since the late 1940s. But until around 2010 neural networks tended to be quite small, consisting of only 10s or perhaps 100s of neurons. This limited them to only solving quite basic problems. Around 2010 improvements in computing power and the algorithms for training the networks made much larger and more powerful networks practical. These are known as deep neural networks or Deep Learning

## Convolutional Neural Networks

A convolutional neural network (CNN) is a type of artificial neural network (ANN) that is most commonly applied to analyze visual imagery. They are designed to recognize the spatial structure of images when extracting features.

### Step 4. Build an architecture from scratch or choose a pretrained model

Let us look at how to build a neural network from scratch. Although this sounds like a daunting task, with Keras it is surprisingly straightforward. With Keras you compose a neural network by creating layers and linking them together.

Let's look at our network from the introduction:

```
# CNN Part 1
# # CNN Part 1
# # Input layer of 32x32 images with three channels (RGB)
# inputs_intro = keras.Input(shape=train_images.shape[1:])

# # CNN Part 2
# # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
# x_intro = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs_intro)
# # Pooling layer with input window sized 2,2
# x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
# # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
# x_intro = keras.layers.Conv2D(32, (3, 3), activation='relu')(x_intro)
# # Second Pooling layer with input window sized 2,2
# x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
# # Flatten layer to convert 2D feature maps into a 1D vector
# x_intro = keras.layers.Flatten()(x_intro)
# # Dense layer with 64 neurons and ReLU activation
# x_intro = keras.layers.Dense(64, activation='relu')(x_intro)

# # CNN Part 3
# # Output layer with 10 units (one for each class) and softmax activation
# outputs_intro = keras.layers.Dense(10, activation='softmax')(x_intro)
```

### Parts of a neural network

Here we can see there are three main components of a neural network:  

- CNN Part 1. Input Layer
- CNN Part 2. Hidden Layers
- CNN Part 3. Output Layer

The output from each layer becomes the input to the next layer.

#### CNN Part 1. Input Layer

The Input in Keras gets special treatment when images are used. Keras automatically calculates the number of inputs and outputs a specific layer needs and therefore how many edges need to be created. This means we need to let Keras know how big our input is going to be. We do this by instantiating a `keras.Input` class and pass it a tuple that indicates the dimensionality of the input data.

In our case, the shape of an image is defined by its pixel dimensions and number of channels:

```python
# recall the shape of the images in our dataset
print(train_images.shape)
```
```output
(40000, 32, 32, 3) # number of images, image width in pixels, image height in pixels, number of channels (RGB)
```

The input layer is created with the `tf.keras.Input` function and its first parameter is the expected shape of the input. 

Because the shape of our input dataset includes the total number of images, we want to take a slice of the shape related to an individual image, hence:

```
# Input layer of 32x32 images with three channels (RGB)
#inputs_intro = keras.Input(shape=train_images.shape[1:])
```


#### CNN Part 2. Hidden Layers

The next component consists of the so-called hidden layers of the network. The reason they are referred to as hidden is because the true values of their nodes are unknown.

In a CNN, the hidden layers typically consist of convolutional, pooling, reshaping (e.g., Flatten), and dense layers. 

Check out the [Layers API] section of the Keras documentation for each layer type and its parameters.


##### **Convolutional Layers**

A **convolutional** layer is a fundamental building block in a CNN designed for processing structured grid data, such as images. It applies convolution operations to input data using learnable filters or kernels, extracting local patterns and features (e.g. edges, corners). These filters enable the network to capture hierarchical representations of visual information, allowing for effective feature learning.

To find the particular features of an image, CNN's make use of a concept from image processing that precedes Deep Learning.

A **convolution matrix**, or **kernel**, is a matrix transformation that we 'slide' over the image to calculate features at each position of the image. For each pixel, we calculate the matrix product between the kernel and the pixel with its surroundings. A kernel is typically small, between 3x3 and 7x7 pixels. We can for example think of the 3x3 kernel:

```
[[-1, -1, -1],
 [0,   0,  0]
 [1,   1,  1]]
```
This kernel will give a high value to a pixel if it is on a horizontal border between dark and light areas. Note that for RGB images, the kernel should also have a depth of 3, one for each colour channel.

In the following image, we see the effect of such a kernel on the values of a single-channel image. The red cell in the output matrix is the result of multiplying and summing the values of the red square in the input, and the kernel. Applying this kernel to a real image shows that it indeed detects horizontal edges.

![](fig/03_conv_matrix.png){alt=''}

![](fig/03_conv_image.png){alt=''}

Within our convolutional layer, the hidden units comprise multiple convolutional matrices, also known as kernels. The matrix values, serving as weights, are learned during the training process. The convolutional layer produces an 'image' for each kernel, representing the output derived by applying the kernel to each pixel.

There are several types of convolutional layers available in Keras depending on your application. We use the two-dimensional layer typically used for images, `tf.keras.layers.Conv2D`.

We define arguments for the number of filters, the kernel size, and the activation function.

```
# # Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
# x_intro = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs_intro)
```

The instantiation here has three parameters and a seemingly strange combination of parentheses, so let us take a closer look.

- The first parameter is the number of filters we want in this layer and this is one of the hyperparameters of our system and needs to be chosen carefully. 

The term *filter* in the context of CNN's is often used synonymously with kernel. However, a filter refers to the learned parameters (weights) that are applied during the convolution operation. For example, in a convolutional layer, you might have multiple filters (or kernels), each responsible for detecting different features in the input data. The parameter here specifies the number of output filters in the convolution.

It's good practice to start with a relatively small number of filters in the first layer to prevent overfitting and choosing a number of filters as a power of 2 (e.g., 32, 64, 128) is common.

- The second parameter is the kernel size which we already discussed. Smaller kernels are often used to capture fine-grained features and odd-sized filters are preferred because they have a centre pixel which helps maintain spatial symmetry during covolutions.

- The third parameter is the activation function to use; here we choose **relu** which is 0 for inputs that are 0 and below and the identity function (returning the same value) for inputs above 0. This is a commonly used activation function in deep neural networks that is proven to work well. We will discuss activation functions later in **Step 9. Tune hyperparameters** but to satisfy your curiosity, `ReLU` stands for Rectified Linear Unit (ReLU).

- Next we see an extra set of parenthenses with inputs in them, this means that after creating an instance of the Conv2D layer we call it as if it was a function. This tells the Conv2D layer to connect the layer passed as a parameter, in this case the inputs.

- Finally, we store a reference so we can pass it to the next layer.


:::::::::::::::::::::::::::::::::::::: callout

## Playing with convolutions

Convolutions applied to images can be hard to grasp at first. Fortunately, there are resources out there that enable users to interactively play around with images and convolutions:

- [Image kernels explained] shows how different convolutions can achieve certain effects on an image, like sharpening and blurring.

- The [convolutional neural network cheat sheet] shows animated examples of the different components of convolutional neural nets.
:::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

Border pixels

What do you think happens to the border pixels when applying a convolution?

:::::::::::::::::::::::: solution

There are different ways of dealing with border pixels. You can ignore them, which means that your output image is slightly smaller then your input. It is also possible to 'pad' the borders, e.g. with the same value or with zeros, so that the convolution can also be applied to the border pixels. In that case, the output image will have the same size as the input image.
:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::


#### **Pooling Layers**

The convolutional layers are often intertwined with **Pooling** layers. As opposed to the convolutional layer used in feature extraction, the pooling layer alters the dimensions of the image and reduces it by a scaling factor. It is basically decreasing the resolution of your picture. The rationale behind this is that higher layers of the network should focus on higher-level features of the image. By introducing a pooling layer, the subsequent convolutional layer has a broader 'view' on the original image.

As we saw with convolutional layers, Keras offers several pooling layers and one used for images (2D spatial data) is the `tf.keras.layers.MaxPooling2D` class.

```
# # Pooling layer with input window sized 2,2
# x_intro = keras.layers.MaxPooling2D((2, 2))(x_intro)
```

The instantiation here has a single parameter, pool_size.

The function downsamples the input along its spatial dimensions (height and width) by taking the **maximum** value over an input window (of size defined by pool_size) for each channel of the input. By taking the maximum instead of the average, the most prominent features in the window are emphasized.

A 2x2 pooling size reduces the width and height of the input by a factor of 2. Empirically, a 2x2 pooling size has been found to work well in various for image classification tasks and also strikes a balance between down-sampling for computational efficiency and retaining important spatial information.

:::::::::::::::::::::::::::::::::::::: callout
## Other types of data

Convolutional and Pooling layers are also applicable to different types of data than image data. Whenever the data is ordered in a (spatial) dimension, and translation invariant features are expected to be useful, convolutions can be used. Think for example of time series data from an accelerometer, audio data for speech recognition, or 3d structures of chemical compounds.
::::::::::::::::::::::::::::::::::::::::::::::


##### **Dense layers**

A **dense** layer has a number of neurons, which is a parameter you can choose when you create the layer. When connecting the layer to its input and output layers every neuron in the dense layer gets an edge (i.e. connection) to **all** of the input neurons and **all** of the output neurons.

![](fig/03-neural_network_sketch_dense.png){alt=''}

This layer is called fully connected, because all input neurons are taken into account by each output neuron. It aggregates global information about the features learned in previous layers to make a decision about the class of the input.

In Keras, a densely-connected NN layer is defined by the `tf.keras.layers.Dense` class.

```
# # Dense layer with 64 neurons and ReLU activation
# x_intro = keras.layers.Dense(64, activation='relu')(x_intro)
```

This instantiation has two parameters: the number of neurons and the activation function as we saw in the convolutional layer.

The choice of how many neurons to specify is often determined through experimentation and can impact the performance of our CNN. Too few neurons may not capture complex patterns in the data but too many neurons may lead to overfitting. 


::::::::::::::::::::::::::::::::::::: challenge

Number of parameters

Suppose we create a single Dense (fully connected) layer with 100 hidden units that connect to the input pixels, how many parameters does this layer have?

:::::::::::::::::::::::: solution

Each entry of the input dimensions, i.e. the shape of one single data point, is connected with 100 neurons of our hidden layer, and each of these neurons has a bias term associated to it. So we have 307300 parameters to learn.
```python
width, height = (32, 32)
n_hidden_neurons = 100
n_bias = 100
n_input_items = width * height * 3
n_parameters = (n_input_items * n_hidden_neurons) + n_bias
print(n_parameters)
```
```output
307300
```
We can also check this by building the layer in Keras:
```python
inputs_ex = keras.Input(shape=dim)
outputs_ex = keras.layers.Dense(100)(inputs_ex)
model_ex = keras.models.Model(inputs=inputs_ex, outputs=outputs_ex)
model_ex.summary()
```
```output
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3072)]            0
_________________________________________________________________
dense (Dense)                (None, 100)               307300
=================================================================
Total params: 307,300
Trainable params: 307,300
Non-trainable params: 0
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::


##### **Reshaping Layers: Flatten**

The next type of hidden layer used in our introductory model is a type of reshaping layer defined in Keras by the `tf.keras.layers.Flatten` class. It is necessary when transitioning from convolutional and pooling layers to fully connected layers.

```
# # Flatten layer to convert 2D feature maps into a 1D vector
# x_intro = keras.layers.Flatten()(x_intro)
```

The **Flatten** layer converts the output of the previous layer into a single one-dimensional vector that can be used as input for a dense layer.


#### CNN Part 3. Output Layer

Recall for the outputs we will need to look at what we want to identify from the data. If we are performing a classification problem then typically we will have one output for each potential class. We need to finish with a Dense layer to connect the output cells of the convolutional layer to the outputs for our 10 classes.

```
# # Output layer with 10 units (one for each class) and softmax activation
# outputs_intro = keras.layers.Dense(10, activation='softmax')(x_intro))
```


## Putting it all together

```python
#### Define the Model

# CNN Part 1
# Input layer of 32x32 images with three channels (RGB)
inputs_intro = keras.Input(shape=train_images.shape[1:])

# CNN Part 2
# Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
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
model_intro = keras.Model(inputs=inputs_intro, outputs=outputs_intro, name="cifar_model_intro")

# view the model summary
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
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 6, 6, 32)          0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 1152)              0         
                                                                 
 dense (Dense)               (None, 64)                73792     
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 79530 (310.66 KB)
Trainable params: 79530 (310.66 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

:::::::::::::::::::::::::::::::::::::: callout

## How to choose an architecture?

Even for this neural network, we had to make a choice on the number of hidden neurons. Other choices to be made are the number of layers and type of layers. You might wonder how you should make these architectural choices. Unfortunately, there are no clear rules to follow here, and it often boils down to a lot of trial and error. However, it is recommended to look what others have done with similar datasets and problems. Another best practice is to start with a relatively simple architecture. Once running start to add layers and tweak the network to see if performance increases. 

::::::::::::::::::::::::::::::::::::::::::::::

## We have a model now what?

This CNN should be able to run with the CIFAR-10 dataset and provide reasonable results for basic classification tasks. However, do keep in mind that this model is relatively simple, and its performance may not be as high as more complex architectures. The reason it's called deep learning is because in most cases, the more layers we have, ie, the deeper and more sophisticated CNN architecture we use, the better the performance.

How can we tell? We can look at a couple metrics during the training process to detect whether our model is underfitting or overfitting. To do that, we first need to continue with the next steps in our Deep Learning workflow, **Step 5. Choose a loss function and optimizer** and **Step 6. Train model**. 

::::::::::::::::::::::::::::::::::::: keypoints 

- Artificial neural networks (ANN) are a machine learning technique based on a model inspired by groups of neurons in the brain.
- Convolution neural networks (CNN) are a type of ANN designed for image classification and object detection
- The filter size determines the size of the receptive field where information is extracted and the kernel size changes the mathematical structure
- A CNN can consist of many types of layers including convolutional, pooling, flatten, and dense (fully connected) layers
- Convolutional layers are responsible for learning features from the input data
- Pooling layers are often used to reduce the spatial dimensions of the data
- The flatten layer is used to convert the multi-dimensional output of the convolutional and pooling layers into a flat vector
- Dense layers are responsible for combining features learned by the previous layers to perform the final classification

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->
[CC BY-SA 3.0]: https://creativecommons.org/licenses/by-sa/3.0
[original source]: https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg
[Image kernels explained]: https://setosa.io/ev/image-kernels/
[convolutional neural network cheat sheet]: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
[Layers API]: https://keras.io/api/layers/
