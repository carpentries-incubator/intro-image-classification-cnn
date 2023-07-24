---
title: 'Introduction to Image Data'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- Where can I find image data to train my model?
- How much data do you need for Deep Learning?
- How do I plot image data in python?
- How do I prepare image data for use in a convolutional neural network (CNN)?
- What is one hot encoding?

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

### 1. Formulate/ Outline the problem
Firstly we must decide what it is we want our Deep Learning system to do. This lesson is all about image classification so our aim is to put an image into one of a few categories.

### 2. Identify inputs and outputs
Next we need to identify what the inputs and outputs of the neural network will be. In our case, the data is images and the inputs could be the individual pixels of the images. We are performing a classification problem and we will have one output for each potential class.

### 3. Prepare data
For this lesson, we will be using an existing image dataset known as CIFAR-10 that we saw in the last episode. Let's explore this dataset in more detail and talk about steps you would take to make your own dataset.

# Preexisting datasets

## Where can I find image data?

Deep Learning requires extensive training using example data which shows the network what output it should produce for a given input. One common application of Deep Learning is classifying images. In this workshop our network will be trained by being “shown” a series of images and told what they contain. Once the network is trained it should be able to take another image and correctly classify its contents.

In some cases you will be able to download an image dataset that is already labelled and can be used to classify a number of different object like we saw with the CIFAR dataset. Other examples include:

- [MNIST database] - 60,000 training images of handwritten digits (0-9)
- [ImageNet] - 14 million hand-annotated images indicating objects from more than 20,000 categories. ImageNet sponsors an [annual software contest] where programs compete to achieve the highest accuracy. When choosing a pretrained network, the winners of these sorts of competitions are generally a good place to start.
- [MS COCO] - >200,000 labelled images used for object detection, instance segmentation, keypoint analysis, and captioning

In other cases, you will need to create your own set of labelled images. For image classification the label applies to the entire image; object detection requires bounding boxes, and instance or semantic segmentation require each pixel to be labelled.

There are a number of different software that can be used to label your dataset, including:

- (Visual Geometry Group) [VGG Image Annotator] (VIA)

:::::::::::::::::::::::::::::::::::::: callout

How much data do you need for Deep Learning?

The rise of Deep Learning is partially due to the increased availability of very large datasets. But how much data do you actually need to train a Deep Learning model? Unfortunately, this question is not easy to answer. It depends, among other things, on the complexity of the task (which you often do not know beforehand), the quality of the available dataset and the complexity of the network. For complex tasks with large neural networks, we often see that adding more data continues to improve performance. However, this is also not a generic truth: if the data you add is too similar to the data you already have, it will not give much new information to the neural network.

In case you have too little data available to train a complex network from scratch, it is sometimes possible to use a pretrained network that was trained on a similar problem. Another trick is data augmentation, where you expand the dataset with artificial data points that could be real. An example of this is mirroring images when trying to classify cats and dogs. An horizontally mirrored animal retains the label, but exposes a different view.
:::::::::::::::::::::::::::::::::::::::::::::::

In cases where the data exists, you can simply load it into memory:


### Load data

```python
# load the cifar dataset included with the keras packages
from tensorflow import keras

(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()
```


## Plotting image data in python

## Image Dimensions

## RGB vs Greyscale

## Split data into training and validation set

In the previous episode we saw that the keras installation includes the Cifar-10 dataset and that by using the cifar10.load_data() method the returned data is already split into two. In this instance, there is no test data.

When using a different dataset, or loading a your own, you will need to do the split yourself. Keep in mind you will also need to a validation set.

::::::::::::::::::::::::::::::::::::::::: callout
ChatGPT

Data is typically split into the training, validation, and test data sets using a process called data splitting or data partitioning. There are various methods to perform this split, and the choice of technique depends on the specific problem, dataset size, and the nature of the data. Here are some common approaches:

1. **Hold-Out Method:**

- In the hold-out method, the dataset is divided into two parts initially: a training set and a test set.

- The training set is used to train the model, and the test set is kept completely separate to evaluate the model's final performance.

- This method is straightforward and widely used when the dataset is sufficiently large.

2. **Train-Validation-Test Split:**

- The dataset is split into three parts: the training set, the validation set, and the test set.

- The training set is used to train the model, the validation set is used to tune hyperparameters and prevent overfitting during training, and the test set is used to assess the final model performance.

- This method is commonly used when fine-tuning model hyperparameters is necessary.

3. **K-Fold Cross-Validation:**

- In k-fold cross-validation, the dataset is divided into k subsets (folds) of roughly equal size.

- The model is trained and evaluated k times, each time using a different fold as the test set while the remaining k-1 folds are used as the training set.

- The final performance metric is calculated as the average of the k evaluation results, providing a more robust estimate of model performance.

- This method is particularly useful when the dataset size is limited, and it helps in better utilizing available data.

4. **Stratified Sampling:**

- Stratified sampling is used when the dataset is imbalanced, meaning some classes or categories are underrepresented.

- The data is split in such a way that each subset (training, validation, or test) maintains the same class distribution as the original dataset.

- This ensures that all classes are well-represented in each subset, which is important to avoid biased model evaluation.

It's important to note that the exact split ratios (e.g., 80-10-10 or 70-15-15) may vary depending on the problem, dataset size, and specific requirements. Additionally, data splitting should be performed randomly to avoid introducing any biases into the model training and evaluation process.
:::::::::::::::::::::::::::::::::::::::::::::::::

To split a cleaned dataset into a training and test set we will use a very convenient function from sklearn called `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(your_data, target, test_size=0.2, random_state=0, shuffle=True, stratify=target)
```

TODO need some test data - maybe not use cifar in this section?

This function takes a number of parameters:

- The first two are the dataset and the corresponding targets.

- Next is the named parameter test_size this is the fraction of the dataset that is used for testing, in this case 0.2 means 20% of the data will be used for testing.

- random_state controls the shuffling of the dataset, setting this value will reproduce the same results (assuming you give the same integer) every time it is called.

- shuffle which can be either True or False, it controls whether the order of the rows of the dataset is shuffled before splitting. It defaults to True.

- stratify is a more advanced parameter that controls how the split is done. By setting it to target the train and test sets the function will return will have roughly the same proportions as the dataset.

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
https://cocodataset.org/#home

[VGG Image Annotator]: https://www.robots.ox.ac.uk/~vgg/software/via/
