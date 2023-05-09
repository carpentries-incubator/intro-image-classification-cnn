---
title: "Introduction to AI, ML, and DL"
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- What is artificial intelligence?
- What is machine learning and what is it used for?
- What is deep learning and what are a few common DL algorithms?
- What is a neural network?
- How do I use a neural network for image classification?


::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain the difference between artificial intelligence, machine learning and deep learning
- Explain how machine learning is used for regression and classification tasks
- Understand what algorithms are used for image classification
- Demonstrate a convolutional neural network

::::::::::::::::::::::::::::::::::::::::::::::::

## Concept: Differentiation between classical ML models and Deep Learning models
Traditional ML algorithms can only use one (possibly two layers) of data transformation to calculate an output (shallow models). With high dimensional data and growing feature space (possible set of values for any given feature), shallow models quickly run out of layers to calculate outputs. Deep neural networks (constructed with multiple layers of neurons) are the extension of shallow models with three layers: input, hidden, and outputs layers. The hidden layer is where learning takes place. As a result, deep learning is best applied to large datasets for training and prediction. As observations and feature inputs decrease, shallow ML approaches begin to perform noticeably better. 

![](https://github.com/erinmgraham/icwithcnn/tree/main/episodes/fig/01_AI_ML_DL_differences.png)

## Concept: Why deep learning is possible and what infrastructure is best suited to deep learning
Systems with high quality GPUs and/or HPCs if available. [Comment: I feel this is important to note, in order to make it clear that anyone attempting to run neural networks on a standard laptop will quickly reach the upper limit of capacity. By setting this expectation clearly in the course, it could help prevent people from trying to do everything neural net related on their machines and becoming disenfranchise with ML as a result]

## What is image classification?
![](https://github.com/erinmgraham/icwithcnn/tree/main/episodes/fig/01_Fei-Fei_Li_Justin_Johnson_Serena_Young__CS231N_2017.png)

## Workflow
1. Load data
1. Define Model
1. Fit Model
1. Evaluate Model
1. Predict new data


### Load data
![](https://github.com/erinmgraham/icwithcnn/tree/main/episodes/fig/01_cifar10.png)

```python
from tensorflow import keras
cifar_images = keras.datasets.cifar10.load_data()
```

::::::::::::::::::::::::::::::::::::: challenge 

## Challenge 1: Describe the Cifar10 dataset

What is the output of this command?

```python
cifar_images.dtypes
```

:::::::::::::::::::::::: solution 

## Output
 
```output
[1] "This new lesson looks good"
```

:::::::::::::::::::::::::::::::::
:::

We only want a subset of this data initially and we will want to divide it into training and testing subsets.

```python
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

n = 50
train_images = train_images[:n]
train_labels = train_labels[:n]
```

### Define Model

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: instructor

Inline instructor notes can help inform instructors of timing challenges
associated with the lessons. They appear in the "Instructor View"

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Deep learning is a subset of machine learning, which is a subset of artificial intelligence
- Machine learning is used for regression and classification tasks
- Convolutional neural networks are well suited for image classification

::::::::::::::::::::::::::::::::::::::::::::::::

[r-markdown]: https://rmarkdown.rstudio.com/
