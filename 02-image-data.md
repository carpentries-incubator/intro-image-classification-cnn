---
title: 'Introduction to Image Data'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How are images represented in digital format?
- Where can I find image data to train my model?
- How do I plot image data in python?
- How do I prepare image data for use in a convolutional neural network (CNN)?
- What is one hot encoding?
- How much data do you need for Deep Learning?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify sources of image data
- Write code to plot image data
- Understand the properties of image data
- Prepare an image data set to train a convolutional neural network (CNN)
- Know how to perform one-hot encoding

::::::::::::::::::::::::::::::::::::::::::::::::

## Deep Learning Workflow

Let's start over with the first steps in our workflow.

### Step 1. Formulate/ Outline the problem

Firstly we must decide what it is we want our Deep Learning system to do. This lesson is all about image classification and our aim is to put an image into one of ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck

### Step 2. Identify inputs and outputs

Next we need to identify what the inputs and outputs of the neural network will be. In our case, the data is images and the inputs could be the individual pixels of the images. 

We are performing a classification problem and we want to output one category for each image.

### Step 3. Prepare data

Deep Learning requires extensive training using example data which shows the network what output it should produce for a given input. In this workshop our network will be trained by being “shown” a series of images and told what they contain. Once the network is trained it should be able to take another image and correctly classify its contents.

We will use an existing image dataset known as CINIC-10 that we saw in the introduction. Let's explore this dataset in more detail and talk about steps you would take to make your own dataset.

#### Pre-existing image data

In some cases you will be able to download an image dataset that is already labelled and can be used to classify a number of different object like we see with the CINIC-10 dataset. Other examples include:

- [MNIST database] - 60,000 training images of handwritten digits (0-9)
- [ImageNet] - 14 million hand-annotated images indicating objects from more than 20,000 categories. ImageNet sponsors an [annual software contest] where programs compete to achieve the highest accuracy. When choosing a pretrained network, the winners of these sorts of competitions are generally a good place to start.
- [MS COCO] - >200,000 labelled images used for object detection, instance segmentation, keypoint analysis, and captioning

Where labelled data exists, in most cases the data provider or other users will have created functions that you can use to load the data. We already saw an example of this in the introduction:

```
# load the CIFAR-10 dataset included with the keras packages
#from tensorflow import keras

#(train_images, train_labels), (val_images, val_labels) = #keras.datasets.cifar10.load_data()
```

In this instance the data is likely already prepared for use in a CNN. However, it is always a good idea to first read any associated documentation to find out what steps the data providers took to prepare the images and second to take a closer at the images once loaded and query their attributes.

#### Custom image data

In other cases, you will need to create your own set of labelled images. 

**Custom data i. Data collection and Labeling:**

For image classification the label applies to the entire image; object detection requires bounding boxes around objects of interest, and instance or semantic segmentation requires each pixel to be labelled.

There are a number of open source software that can be used to label your dataset, including:

- (Visual Geometry Group) [VGG Image Annotator] (VIA)
- [ImageJ] can be extended with plugins for annotation
- [COCO Annotator] is designed specifically for creating annotations compatible with Common Objects in Context (COCO) format

:::::::::::::::::::::::::::::::::::::: callout

How much data do you need for Deep Learning?

The rise of Deep Learning is partially due to the increased availability of very large datasets. But how much data do you actually need to train a Deep Learning model? Unfortunately, this question is not easy to answer. It depends, among other things, on the complexity of the task (which you often do not know beforehand), the quality of the available dataset and the complexity of the network. For complex tasks with large neural networks, we often see that adding more data continues to improve performance. However, this is also not a generic truth: if the data you add is too similar to the data you already have, it will not give much new information to the neural network.

In case you have too little data available to train a complex network from scratch, it is sometimes possible to use a pretrained network that was trained on a similar problem. Another trick is data augmentation, where you expand the dataset with artificial data points that could be real. An example of this is mirroring images when trying to classify cats and dogs. An horizontally mirrored animal retains the label, but exposes a different view.
:::::::::::::::::::::::::::::::::::::::::::::::

**Custom data ii. Data preprocessing:**

This step involves various tasks to enhance the quality and consistency of the data:

- **Resizing**: Resize images to a consistent resolution to ensure uniformity and reduce computational load.

- **Normalization**: Scale pixel values to a common range, often between 0 and 1 or -1 and 1. Normalization helps the model converge faster during training.

- **Data Augmentation**: Apply random transformations (e.g., rotations, flips, shifts) to create new variations of the same image. This helps improve the model's robustness and generalization by exposing it to more diverse data.

- **Color Channels**: Depending on the model and library you use, you might need to handle different color channel orders (RGB, BGR, etc.).

- **Data Formats**: Convert image data to a format suitable for your deep learning framework (e.g., NumPy arrays or TensorFlow tensors).

Before we look at some of these tasks in more detail we need to understand that the images we see on hard copy, view with our electronic devices, or process with our programs are represented and stored in the computer as numeric abstractions, or approximations of what we see with our eyes in the real world. And before we begin to learn how to process images with Python programs, we need to spend some time understanding how these abstractions work.

### Pixels

It is important to realise that images are stored as rectangular arrays of hundreds, thousands, or millions of discrete "picture elements," otherwise known as pixels. Each pixel can be thought of as a single square point of coloured light.

For example, consider this image of a Jabiru, with a square area designated by a red box:

![](fig/02_Jabiru_TGS_marked.jpg){alt='Original size image of a Jabiru with a red square surrounding an area to zoom in on'}

Now, if we zoomed in close enough to see the pixels in the red box, we would see something like this:

![](fig/02_Jabiru_TGS_marked_zoom_enlarged.jpg){alt='Enlarged image area of Jabiru'}

Note that each square in the enlarged image area - each pixel - is all one colour, but that each pixel can have a different colour from its neighbors. Viewed from a distance, these pixels seem to blend together to form the image we see.

### Working with Pixels

As noted, in practice, real world images will typically be made up of a vast number of pixels, and each of these pixels will be one of potentially millions of colours. In python, an image can be represented as a multidimensional array, also known as a `tensor`, where each element in the array corresponds to a pixel value in the image. In the context of images, these arrays often have dimensions for height, width, and color channels (if applicable).

::::::::::::::::::::::::::::::::::::::::: callout

Matrices, arrays, images and pixels

The matrix is mathematical concept - numbers evenly arranged in a rectangle. This can be a two dimensional rectangle, like the shape of the screen you're looking at now. Or it could be a three dimensional equivalent, a cuboid, or have even more dimensions, but always keeping the evenly spaced arrangement of numbers. In computing, array refers to a structure in the computer's memory where data is stored in evenly-spaced elements. This is strongly analogous to a matrix. A NumPy array is a type of variable (a simpler example of a type is an integer). For our purposes, the distinction between matrices and arrays is not important, we don't really care how the computer arranges our data in its memory. The important thing is that the computer stores values describing the pixels in images, as arrays. And the terms matrix and array can be used interchangeably.

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::: callout

Python image libraries

Two of the most commonly used libraries for image representation and manipulation are NumPy and Pillow (PIL). Additionally, when working with deep learning frameworks like TensorFlow and PyTorch, images are often represented as tensors within these frameworks.

- NumPy is a powerful library for numerical computing in Python. It provides support for creating and manipulating arrays, which can be used to represent images as multidimensional arrays. 
  - `import numpy as np`

- The Pillow library (PIL fork) provides functions to open, manipulate, and save various image file formats. It represents images using its own Image class. 
  - `from PIL import Image`
  - see [PIL Image Module]

- TensorFlow images are often represented as tensors that have dimensions for batch size, height, width, and color channels. This framework provide tools to load, preprocess, and work with image data seamlessly. 
  - `from tensorflow import keras`
  - see [image preprocessing] documentation
  - Note Keras image functions also use PIL 

::::::::::::::::::::::::::::::::::::::::::::::::::

Let us start by looking at the image we used in the introduction.

```python
# load the libraries required
from keras.utils import img_to_array
from keras.utils import load_img

# specify the image path
new_img_path = "../data/Jabiru_TGS.JPG" # path to image

# read in the image with default arguments
new_img_pil = load_img(new_img_path)

# confirm the data class and size
print('The new image is of type :', new_img_pil.__class__, 'and has the size', new_img_pil.size)
```
```output
The new image is of type : <class 'PIL.JpegImagePlugin.JpegImageFile'> and has the size (552, 573)
```

### Image Dimensions - Resizing

Here we see our new image has shape `(573, 552, 3)`, meaning it is much larger in size, 573x552 pixels; a rectangle instead of a square; and consists of 3 colour channels (RGB).

Recall from the introduction that our training data set consists of 50000 images of 32x32 pixels and 3 channels. 

To reduce the computational load and ensure all of our images have a uniform size, we need to choose an image resolution (or size in pixels) and ensure that all of the images we use are resized to that shape to be consistent.

There are a couple of ways to do this in python but one way is to specify the size you want using an argument to the `load_img()` function from `keras.utils`.

```python
# read in the new image and specify the target size to be the same as our training images
new_img_pil_small = load_img(new_img_path, target_size=(32,32))

# confirm the data class and shape
print('The new image is still of type:', new_img_pil_small.__class__, 'but now has the same size', new_img_pil_small.size, 'as our training data')
```
```output
The new image is still of type: <class 'PIL.Image.Image'> but now has the same size (32, 32) as our training data.
```

### Image Colours

RGB Images:

- For image classification tasks, RGB images are used because they capture the full spectrum of colors that human vision can perceive, allowing the model to learn intricate features and patterns present in the images.

- RGB (Red, Green, Blue) images have three color channels: red, green, and blue, with each channel having an intensity value that ranges from 0 to 255. Each channel represents the intensity of the corresponding color for each pixel. This results in a 3D array, where the dimensions are height, width, and color channel. 

While RGB is the most common representation, there are scenarios where other color palettes might be considered, such as:

Grayscale Images:

- Grayscale images have only one channel, representing the intensity of the pixels. Each pixel's intensity is usually represented by a single numerical value that ranges from 0 (black) to 255 (white). The image is essentially a 2D array where each element holds the intensity value of the corresponding pixel.

- In cases where color information isn't critical, you might convert RGB images to grayscale to reduce the computational load.


### Normalization

Image RGB values are between 0 and 255. As input for neural networks, it is better to have small input values. The process of converting the RGB values to be between 0 and 1 is called **normalization**.

::::::::::::::::::::::::::::::::::::::::: callout
ChatGPT

Normalizing the RGB values to be between 0 and 1 is a common pre-processing step in machine learning tasks, especially when dealing with image data. This normalization has several benefits:

1. **Numerical Stability**: By scaling the RGB values to a range between 0 and 1, you avoid potential numerical instability issues that can arise when working with large values. Neural networks and many other machine learning algorithms are sensitive to the scale of input features, and normalizing helps to keep the values within a manageable range.

2. **Faster Convergence**: Normalizing the RGB values often helps in faster convergence during the training process. Neural networks and other optimization algorithms rely on gradient descent techniques, and having inputs in a consistent range aids in smoother and faster convergence.

3. **Equal Weightage for All Channels**: In RGB images, each channel (Red, Green, Blue) represents different color intensities. By normalizing to the range [0, 1], you ensure that each channel is treated with equal weightage during training. This is important because some machine learning algorithms could assign more importance to larger values.

4. **Generalization**: Normalization helps the model to generalize better to unseen data. When the input features are in the same range, the learned weights and biases can be more effectively applied to new examples, making the model more robust.

5. **Compatibility**: Many image-related libraries, algorithms, and models expect pixel values to be in the range of [0, 1]. By normalizing the RGB values, you ensure compatibility and seamless integration with these tools.

The normalization process is typically done by dividing each RGB value (ranging from 0 to 255) by 255, which scales the values to the range [0, 1].

For example, if you have an RGB image with pixel values (100, 150, 200), after normalization, the pixel values would become (100/255, 150/255, 200/255) ≈ (0.39, 0.59, 0.78).

Remember that normalization is not always mandatory, and there could be cases where other scaling techniques might be more suitable based on the specific problem and data distribution. However, for most image-related tasks in machine learning, normalizing RGB values to [0, 1] is a good starting point.
:::::::::::::::::::::::::::::::::::::::::::::::::::

Before we can normalize our image values we must convert the image to an numpy array.

We saw how to do this in the introduction but what you may not have noticed is that the `keras.datasets.cifar10.load_data` function did the conversion for us whereas now we will do it ourselves.

```python
# convert the Image into an array for normalization
new_img_arr = img_to_array(new_img_pil_small)

# confirm the data class and shape
print('The new image is now of type :', new_img_arr.__class__, 'and has the shape', new_img_arr.shape)
```
```output
The new image is now of type : <class 'numpy.ndarray'> and has the shape (32, 32, 3)
```

Now we can normalize the values. Let us also investigate the image values before and after we normalize them.

```python
# extract the min, max, and mean pixel values
print('The min, max, and mean pixel values are', new_img_arr.min(), ',', new_img_arr.max(), ', and', new_img_arr.mean().round(), 'respectively.')

# normalize the RGB values to be between 0 and 1
new_img_arr_norm = new_img_arr / 255.0

# extract the min, max, and mean pixel values AFTER
print('After normalization, the min, max, and mean pixel values are', new_img_arr_norm.min(), ',', new_img_arr_norm.max(), ', and', new_img_arr_norm.mean().round(), 'respectively.')
```
```output
The min, max, and mean pixel values are 0.0 , 255.0 , and 87.0 respectively.
After normalization, the min, max, and mean pixel values are 0.0 , 1.0 , and 0.0 respectively.
```

Of course, if we have a large number of images to process we do not want to perform these steps one at a time. As you might have guessed, `tf.keras.utils` also provides a function to load an entire directories: `image_dataset_from_directory()` 
We will use this function in a moment to create a test dataset but before we do let us talk about data splitting.


## Data Splitting

In the previous episode we saw that the keras installation includes the Cifar-10 dataset and that by using the 'cifar10.load_data()' method the returned data is split into two (train and validations sets). There is no test dataset.

When using a different dataset, or loading your own set of images, you will do the splits yourself.

::::::::::::::::::::::::::::::::::::::::: callout
ChatGPT

Data is typically split into the training, validation, and test data sets using a process called data splitting or data partitioning. There are various methods to perform this split, and the choice of technique depends on the specific problem, dataset size, and the nature of the data. Here are some common approaches:

- **Hold-Out Method:**

  - In the hold-out method, the dataset is divided into two parts initially: a training set and a test set.

  - The training set is used to train the model, and the test set is kept completely separate to evaluate the model's final performance.

  - This method is straightforward and widely used when the dataset is sufficiently large.

- **Train-Validation-Test Split:**

  - The dataset is split into three parts: the training set, the validation set, and the test set.

  - The training set is used to train the model, the validation set is used to tune hyperparameters and prevent overfitting during training, and the test set is used to assess the final model performance.

  - This method is commonly used when fine-tuning model hyperparameters is necessary.

- **K-Fold Cross-Validation:**

  - In k-fold cross-validation, the dataset is divided into k subsets (folds) of roughly equal size.

  - The model is trained and evaluated k times, each time using a different fold as the test set while the remaining k-1 folds are used as the training set.

  - The final performance metric is calculated as the average of the k evaluation results, providing a more robust estimate of model performance.

  - This method is particularly useful when the dataset size is limited, and it helps in better utilizing available data.

- **Stratified Sampling:**

  - Stratified sampling is used when the dataset is imbalanced, meaning some classes or categories are underrepresented.

  - The data is split in such a way that each subset (training, validation, or test) maintains the same class distribution as the original dataset.

  - This ensures that all classes are well-represented in each subset, which is important to avoid biased model evaluation.

It's important to note that the exact split ratios (e.g., 80-10-10 or 70-15-15) may vary depending on the problem, dataset size, and specific requirements. Additionally, data splitting should be performed randomly to avoid introducing any biases into the model training and evaluation process.
:::::::::::::::::::::::::::::::::::::::::::::::::

Here we will load an entire directory of images and create a test dataset. 

We set up our test image directory to have the following structure:
```
main_directory/
...class_a/
......image_1.jpg
......image_2.jpg
...class_b/
......image_1.jpg
......image_2.jpg
```

If we using this structure and the [tf.keras.utils,image_dataset_from_directory()] function, keras will automatically infer the image labels.

To split an image dataset into a different sets we could use a function from sklearn called [train_test_split()]. However, because we already have training and validation sets, we just need a test set so we will skip the data splitting step. 

```python
# load the required libraries
from keras.utils import image_dataset_from_directory 
from sklearn.model_selection import train_test_split

# define the image directory
test_image_dir = 'D:/20230724_CINIC10/test_images'

# read in the images, infer the labels, and resize to match training dataset
test_images = image_dataset_from_directory(test_image_dir, labels='inferred', batch_size=None, image_size=(32,32), shuffle=False)
```

::::::::::::::::::::::::::::::::::::: challenge
TRAINING AND TEST SETS

Take a look at the training and test set we created. - How many samples do the training and test sets have? - Are the classes in the training set well balanced?

:::::::::::::::::::::::: solution 

TODO will depend on data - see ep02 deep learning for ex on penguins

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

## One-hot encoding

A neural network can only take numerical inputs and outputs, and learns by calculating how “far away” the species predicted by the neural network is from the true species. When the target is a string category column as we have here it is very difficult to determine this “distance” or error. Therefore we will transform this column into a more suitable format. Again there are many ways to do this, however we will be using the one-hot encoding. This encoding creates multiple columns, as many as there are unique values, and puts a 1 in the column with the corresponding correct class, and 0’s in the other columns.

TBC

## Image augmentation

TBC

# Preexisting datasets

TBD

Now that our dataset is ready to go, let us move on to how to build an architecture.

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: instructor

Inline instructor notes can help inform instructors of timing challenges
associated with the lessons. They appear in the "Instructor View"

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Use `.md` files for episodes when you want static content
- Use `.Rmd` files for episodes when you need to generate output
- Run `sandpaper::check_lesson()` to identify any issues with your lesson
- Run `sandpaper::build_lesson()` to preview your lesson locally

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->
[MNIST database]: https://en.wikipedia.org/wiki/MNIST_database
[ImageNet]: https://www.image-net.org/
[annual software contest]: https://www.image-net.org/challenges/LSVRC/#:~:text=The%20ImageNet%20Large%20Scale%20Visual,image%20classification%20at%20large%20scale.
[MS COCO]: https://cocodataset.org/#home

[VGG Image Annotator]: https://www.robots.ox.ac.uk/~vgg/software/via/
[ImageJ]: https://imagej.net/
[COCO Annotator]: https://github.com/jsbroks/coco-annotator

[PIL Image Module]: https://pillow.readthedocs.io/en/latest/reference/Image.html
[image preprocessing]: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image


[tf.keras.utils.image_dataset_from_directory]:  https://keras.io/api/data_loading/image/
[train_test_split()]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
