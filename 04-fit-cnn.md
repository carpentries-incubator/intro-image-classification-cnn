---
title: 'Compile and Train (Fit) a Convolutional Neural Network'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you compile a convolutional neural network (CNN)?
- What is a loss function?
- What is an optimizer?
- How do you train (fit) a CNN?
- How do you evaluate a model during training?
- What is overfitting?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain the difference between compiling and training (fitting) a CNN.
- Know how to select a loss function for your model.
- Understand what an optimizer is.
- Define the terms: learning rate, batch size, epoch.
- Understand what loss and accuracy are and how to monitor them during training.
- Explain what overfitting is and what to do about it.

::::::::::::::::::::::::::::::::::::::::::::::::

### Step 5. Choose a loss function and optimizer

We have designed a convolutional neural network (CNN) that in theory we should be able to train to classify images. 

We now need to select an appropriate optimizer and loss function that we will use during training (fitting). 

Recall how we compiled our model in the introduction:
```
## compile the model
#model_intro.compile(optimizer = 'adam', 
#                    loss = keras.losses.CategoricalCrossentropy(), 
#                    metrics = ['accuracy'])
```              

#### Loss function

The **loss function** tells the training algorithm how wrong, or how 'far away' from the true value the predicted value is. The purpose of loss functions is to compute the quantity that a model should seek to minimize during training. Which class of loss functions you choose depends on your task. 

**Loss for classification**

For classification purposes, there are a number of probabilistic losses to choose from. We chose `CategoricalCrossentropy` because we want to compute the crossentropy loss between our one-hot encoded class labels and the model predictions. This loss function is appropriate to use when the data has two or more label classes.

The loss function is defined by the `tf.keras.losses.CategoricalCrossentropy` class.

More information about loss functions can be found in the Keras [loss documentation].


#### Optimizer

Somewhat coupled to the loss function is the **optimizer**. The optimizer here refers to the algorithm with which the model learns to optimize on the provided loss function.

We need to choose which optimizer to use and, if this optimizer has parameters, what values to use for those. Furthermore, we need to specify how many times to show the training samples to the optimizer. In other words, the optimizer is responsible for taking the output of the loss function and then applying some changes to the weights within the network. It is through this process that the “learning” (adjustment of the weights) is achieved.

```
## compile the model
#model_intro.compile(optimizer = 'adam', 
#                    loss = keras.losses.CategoricalCrossentropy(), 
#                    metrics = ['accuracy'])
``` 

**Adam** 

Here we picked one of the most common optimizers that works well for most tasks, the **Adam** optimizer. Similar to activation functions, the choice of optimizer depends on the problem you are trying to solve, your model architecture and your data. Adam is a good starting point though, which is why we chose it. Adam has a number of parameters, but the default values work well for most problems so we will use it with its default parameters.

It is defined by the `keras.optimizers.Adam` class and takes a single parameter `learning_rate=0.01`

There are many optimizers to choose from so check the [optimizer documentation]. A couple more popular or famous ones include:

- **Stochastic Gradient Descent (sgd)**: Stochastic Gradient Descent (SGD) is one of the fundamental optimization algorithms used to train machine learning models, especially neural networks. It is a variant of the gradient descent algorithm, designed to handle large datasets efficiently.

- **Root Mean Square (rms)prop**: RMSprop is widely used in various deep learning frameworks and is one of the predecessors of more advanced optimizers like Adam, which further refines the concept of adaptive learning rates. It is an extension of the basic Stochastic Gradient Descent (SGD) algorithm and addresses some of the challenges of SGD.

  - For example, one of the main issues with the basic SGD is that it uses a fixed learning rate for all model parameters throughout the training process. This fixed learning rate can lead to slow convergence or divergence (over-shooting) in some cases. RMSprop introduces an adaptive learning rate mechanism to address this problem.

::::::::::::::::::::::::::::::::::::::::: spoiler 

### WANT TO KNOW MORE: Learning Rate

ChatGPT

**Learning rate** is a hyperparameter that determines the step size at which the model's parameters are updated during training. A higher learning rate allows for more substantial parameter updates, which can lead to faster convergence, but it may risk overshooting the optimal solution. On the other hand, a lower learning rate leads to smaller updates, providing more cautious convergence, but it may take longer to reach the optimal solution. Finding an appropriate learning rate is crucial for effectively training machine learning models.

The figure below illustrates a small learning rate that will not traverse toward the minima of the gradient descent algorithm in a timely manner, i.e. number of epochs.

![Small learning rate leads to inefficient approach to loss minima](https://developers.google.com/static/machine-learning/crash-course/images/LearningRateTooSmall.svg "Small learning rate leads to inefficient approach to loss minima"){alt='plot of loss over value of weight shows how a small learning rate takes a long time to reach the optimal solution'}

On the other hand, specifying a learning rate that is *too high* will result in a loss value that never approaches the minima. That is, 'bouncing between the sides', thus never reaching a minima to cease learning.

![A large learning rate results in overshooting the gradient descent minima](https://developers.google.com/static/machine-learning/crash-course/images/LearningRateTooLarge.svg){alt='plot of loss over value of weight shows how a large learning rate never approaches the optimal solution because it bounces between the sides'}

Lastly, we can observe below that a modest learning rate will ensure that the product of multiplying the scalar gradient value, and the learning rate does not result in too small steps, nor a chaotic bounce between sides of the gradient where steepness is greatest.

![An optimal learning rate supports a gradual approach to the minima](https://developers.google.com/static/machine-learning/crash-course/images/LearningRateJustRight.svg){alt='plot of loss over value of weight shows how a a good learning rate gets to optimal solution gradually'}

(These images were obtained from [Google Developers Machine Learning Crash Course] and is licenced under the [Creative Commons 4.0 Attribution Licence].)

::::::::::::::::::::::::::::::::::::::::::::::


#### Metrics

After we select the desired optimizer and loss function we want to specify the metric(s) to be evaluated by the model during training and testing. A **metric** is a function that is used to judge the performance of your model.

```
## compile the model
#model_intro.compile(optimizer = 'adam', 
#                    loss = keras.losses.CategoricalCrossentropy(), 
#                    metrics = ['accuracy']) 
```

Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model. Note that you may use any loss function as a metric.

Typically you will use `accuracy` which calculates how often predictions matches labels.

The accuracy function creates two local variables, total and count that are used to compute the frequency with which predictions matches labels. This frequency is ultimately returned as accuracy: an operation that divides the  total by count.

A list of metrics can be found in the Keras [metrics] documentation.

Now that we have decided on which loss function, optimizer, and metric to use we can compile the model using `model.compile`. Compiling the model prepares it for training.


### Step 6. Train (Fit) model

We are ready to train the model.

Training the model is done using the `fit` method. It takes the image data and target (label) data as inputs and has several other parameters for certain options of the training. Here we only set a different number of epochs.

A training **epoch** means that every sample in the training data has been shown to the neural network and used to update its parameters. In general, CNN models improve with more epochs of training, but only to a point.

We want to train our model for 10 epochs:

```
history_intro = model_intro.fit(train_images, train_labels, 
                                epochs = 10, 
                                validation_data = (val_images, val_labels),
                                batch_size = 32)
```

The `batch_size` parameter defaults to 32. The **batch size** is an important hyperparameter that determines the number of training samples processed together before updating the model's parameters during each iteration (or mini-batch) of training.

Note we are also creating a new variable `history_intro` to capture the history of the training in order to extract metrics we will use for model evaluation.

Other arguments used to fit our model can be found in the documentation for the [fit method].
 

::::::::::::::::::::::::::::::::::::::::: spoiler 

### WANT TO KNOW MORE: Batch size

ChatGPT


The choice of batch size can have various implications, and there are situations where using different batch sizes can be beneficial.

**Large Datasets and Memory Constraints**: If you have a large dataset and limited memory, using a smaller batch size can help fit the data into memory during training. This allows you to train larger models or use more complex architectures that might not fit with larger batch sizes.

**Training on GPUs**: Modern deep learning frameworks and libraries are optimized for parallel processing on GPUs. Using larger batch sizes can fully leverage the parallelism of GPUs and lead to faster training times. However, the choice of batch size should consider the available GPU memory.

**Noise in Parameter Updates**: Smaller batch sizes introduce more noise in the gradients, which can help models escape sharp minima and potentially find better solutions. This regularization effect is similar to the impact of stochasticity in Stochastic Gradient Descent (SGD).

**Generalization**: Using smaller batch sizes may improve the generalization of the model. It prevents the model from overfitting to the training data, as it gets updated more frequently and experiences more diverse samples during training.

However, it's essential to consider the trade-offs of using different batch sizes. Smaller batch sizes may require more iterations to cover the entire dataset, which can lead to longer training times. Larger batch sizes can provide more stable gradients but might suffer from generalization issues. There is no one-size-fits-all answer, and you may need to experiment with different batch sizes to find the one that works best for your specific model, architecture, and dataset.

:::::::::::::::::::::::::::::::::::::::::::::::


### Monitor Training Progress (aka Model Evaluation during Training)

Now that we know more about the compilation and fitting of CNN's let us take a inspect the training metrics for our model.

Using seaborn we can plot the training process using the history:

```python
import seaborn as sns
import pandas as pd

# convert the history to a dataframe for plotting 
history_intro_df = pd.DataFrame.from_dict(history_intro.history)

# plot the loss and accuracy from the training process
fig, axes = plt.subplots(1, 2)
fig.suptitle('cifar_model_pool')
sns.lineplot(ax=axes[0], data=history_intro_df[['loss', 'val_loss']])
sns.lineplot(ax=axes[1], data=history_intro_df[['accuracy', 'val_accuracy']])
```

![](fig/04_model_intro_accuracy_loss.png){alt='two panel figure; the figure on the left shows the training loss starting at 1.5 and decreasing to 0.7 and the validation loss decreasing from 1.3 to 1.0 before leveling out; the figure on the right shows the training accuracy increasing from 0.45 to 0.75 and the validation accuracy increasing from 0.53 to 0.65 before leveling off'}

This plot can be used to identify whether the training is well configured or whether there are problems that need to be addressed. The solid blue lines show the training loss and accuracy; the dashed orange lines show the validation loss and accuracy.

::::::::::::::::::::::::::::::::::::: challenge 

## Inspect the Training Curve

Inspect the training curves we have just made and recall the difference between the training and the validation datasets.

1. How does the training progress?

- Does the loss increase or decrease?
- What about the accuracy?
- Do either change fast or slowly?
- Do the graphs lines go up and down frequently (i.e. jitter)?

2. Do you think the resulting trained network will work well on the test set?

:::::::::::::::::::::::: solution 

1. The loss curve should drop quite quickly in a smooth line with little jitter. The accuracy should increase quite quickly in a smooth line also wtih little jitter.
2. The results of the training give very little information on its performance on a test set. You should be careful not to use it as an indication of a well trained network.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

These is evidence of **overfitting** in these plots. If a model is overfitting, it means that the model performs exceptionally well on the training data but poorly on the validation or test data. Overfitting occurs when the model has learned to memorize the noise and specific patterns in the training data instead of generalizing the underlying relationships. As a result, the model fails to perform well on new, unseen data because it has become too specialized to the training set.

Key characteristics of an overfit model include:

- High Training Accuracy, Low Validation Accuracy: The model achieves high accuracy on the training data but significantly lower accuracy on the validation (or test) data.

- Small Training Loss, Large Validation Loss: The training loss is low, indicating that the model's predictions closely match the true labels in the training set. However, the validation loss is high, indicating that the model's predictions are far from the true labels in the validation set.

How to Address Overfitting:

- Reduce the model's complexity by using fewer layers or units to make it less prone to overfitting.
- Collect more training data if possible to provide the model with a diverse and representative dataset.
- Perform data augmentation to artificially increase the size of the training data and introduce variability.

::::::::::::::::::::::::::::::::::::::::: spoiler

### WANT TO KNOW MORE: What is underfitting?

Underfitting occurs when the model is too simple or lacks the capacity to capture the underlying patterns and relationships present in the data. As a result, the model's predictions are not accurate, and it fails to generalize well to unseen data.

Key characteristics of an underfit model include:

- Low Validation Accuracy: This indicates that the model is not learning from the data effectively.
- Large Training Loss: The training loss (error) is high, indicating that the model's predictions are far from the true labels in the training set.
- Increasing validation loss.

How to address underfitting:

- Increase the model's complexity by adding more layers or units to the existing layers.
- Train the model for more epochs to give it more time to learn from the data.
- Perform data augmentation or feature engineering to provide the model with more informative input features.

::::::::::::::::::::::::::::::::::::::::::::::::::

### Improve Model Generalization (avoid Overfitting)

#### Dropout

Note that the training loss continues to decrease, while the validation loss stagnates, and even starts to increase over the course of the epochs. Similarly, the accuracy for the validation set does not improve anymore after some epochs. This means we are overfitting on our training data set.

Techniques to avoid overfitting, or to improve model generalization, are termed **regularization techniques**. One of the most versatile regularization technique is **dropout** (Srivastava et al., 2014). Dropout essentially means that during each training cycle a random fraction of the dense layer nodes are turned off. This is described with the dropout rate between 0 and 1 which determines the fraction of nodes to silence at a time. 

![](fig/04-neural_network_sketch_dropout.png){alt='diagram of two neural networks; the first network is densely connected without dropout and the second network has some of the neurons dropped out of of the network'}

The intuition behind dropout is that it enforces redundancies in the network by constantly removing different elements of a network. The model can no longer rely on individual nodes and instead must create multiple "paths". In addition, the model has to make predictions with much fewer nodes and weights (connections between the nodes). As a result, it becomes much harder for a network to memorize particular features. At first this might appear a quite drastic approach which affects the network architecture strongly. In practice, however, dropout is computationally a very elegant solution which does not affect training speed. And it frequently works very well.

:::::::::::::::::::::::::::::::::::::: callout

Dropout layers will only randomly silence nodes during training! During a predictions step, all nodes remain active (dropout is off). During training, the sample of nodes that are silenced are different for each training instance, to give all nodes a chance to observe enough training data to learn its weights.

::::::::::::::::::::::::::::::::::::::::::::::

Dropout layers are defined by the `tf.keras.layers.Dropout class and have the following definition:

```
tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
```

The `rate` parameter is a float between 0 and 1 and represents the fraction of the input units to drop.

We want to add one Dropout Layer to our network that randomly drops 80 per cent of the input units but where should we put it?

The placement of the dropout layer matters. Adding dropout before or after certain layers can have different effects. For example, it's common to place dropout after convolutional and dense layers but not typically after pooling layers. Let us add a third convolutional layer to our model and then the dropout layer.

```python
# define the inputs, layers, and outputs of a CNN model with dropout

# CNN Part 1
# Input layer of 32x32 images with three channels (RGB)
inputs_dropout = keras.Input(shape=train_images.shape[1:])

# CNN Part 2
# Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
x_dropout = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs_dropout)
# Pooling layer with input window sized 2,2
x_dropout = keras.layers.MaxPooling2D((2, 2))(x_dropout)
# Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
x_dropout = keras.layers.Conv2D(32, (3, 3), activation='relu')(x_dropout)
# Second Pooling layer with input window sized 2,2
x_dropout = keras.layers.MaxPooling2D((2, 2))(x_dropout)
# Second Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
x_dropout = keras.layers.Conv2D(64, (3, 3), activation='relu')(x_dropout) # This is new!
# Dropout layer andomly drops 60 per cent of the input units
x_dropout = keras.layers.Dropout(0.6)(x_dropout) # This is new!
# Flatten layer to convert 2D feature maps into a 1D vector
x_dropout = keras.layers.Flatten()(x_dropout)
# Dense layer with 128 neurons and ReLU activation
x_dropout = keras.layers.Dense(128, activation='relu')(x_dropout)

# CNN Part 3
# Output layer with 10 units (one for each class) and softmax activation
outputs_dropout = keras.layers.Dense(10, activation='softmax')(x_dropout)

# create the dropout model
model_dropout = keras.Model(inputs=inputs_dropout, outputs=outputs_dropout, name="cifar_model_dropout")

model_dropout.summary()
```

```output
Model: "cifar_model_dropout"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 32, 32, 3)]       0         
                                                                 
 conv2d_2 (Conv2D)           (None, 30, 30, 16)        448       
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 15, 15, 16)        0         
 g2D)                                                            
                                                                 
 conv2d_3 (Conv2D)           (None, 13, 13, 32)        4640      
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 6, 6, 32)          0         
 g2D)                                                            
                                                                 
 conv2d_4 (Conv2D)           (None, 4, 4, 64)          18496     
                                                                 
 dropout (Dropout)           (None, 4, 4, 64)          0         
                                                                 
 flatten_1 (Flatten)         (None, 1024)              0         
                                                                 
 dense_2 (Dense)             (None, 128)               131200    
                                                                 
 dense_3 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 156074 (609.66 KB)
Trainable params: 156074 (609.66 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

Note the dropout does not alter the dimensions of the image and has zero parameters.

::::::::::::::::::::::::::::::::::::: challenge

## Does adding a Dropout Layer improve our model?

Write the code to compile and fit our new dropout model using the same arguments we used for our model in the introduction. Then inspect the training metrics to determine whether our model has improved or not by adding a dropout layer.

:::::::::::::::::::::::: solution

```python
# compile the dropout model
model_dropout.compile(optimizer = 'adam',
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

# fit the dropout model
history_dropout = model_dropout.fit(train_images, train_labels, 
                                    epochs=10,
                                    validation_data=(val_images, val_labels),
                                    batch_size = 32)

# save dropout model
model_dropout.save('fit_outputs/model_dropout.keras')

# inspect the training results

# convert the history to a dataframe for plotting 
history_dropout_df = pd.DataFrame.from_dict(history_dropout.history)

# plot the loss and accuracy from the training process
fig, axes = plt.subplots(1, 2)
fig.suptitle('cifar_model_dropout')
sns.lineplot(ax=axes[0], data=history_dropout_df[['loss', 'val_loss']])
sns.lineplot(ax=axes[1], data=history_dropout_df[['accuracy', 'val_accuracy']])

val_loss_dropout, val_acc_dropout = model_dropout.evaluate(val_images, val_labels, verbose=2)
```

![](fig/04_model_dropout_accuracy_loss.png){alt='two panel figure; the figure on the left shows the training loss starting at 1.7 and decreasing to 1.0 and the validation loss decreasing from 1.4 to 0.9 before leveling out; the figure on the right shows the training accuracy increasing from 0.40 to 0.65 and the validation accuracy increasing from 0.5 to 0.67'}

In this relatively uncommon ,  the training loss is higher than our validation loss while the validation accuracy is higher than the training accuracy. Using dropout or other regularization techniques during training can lead to a lower training accuracy.

Dropout randomly "drops out" units during training, which can prevent the model from fitting the training data too closely. This regularization effect may lead to a situation where the model generalizes better on the validation set.

The final accuracy on the validation set is higher than without dropout.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::::::: spoiler

### WANT TO KNOW MORE: Regularization methods for Convolutional Neural Networks (CNNs)

ChatGPT

**Regularization** methods introduce constraints or penalties to the training process, encouraging the model to be simpler and less prone to overfitting. Here are some common regularization methods for CNNs:

**L1 and L2 Regularization**: L1 and L2 regularization are the two most common regularization techniques used in deep learning. They add a penalty term to the loss function during training to restrict the model's weights.

- L1 regularization adds the absolute value of the weights to the loss function. It tends to produce sparse weight vectors, forcing some of the less important features to have exactly zero weights.

- L2 regularization adds the square of the weights to the loss function. It encourages the model to have smaller weights overall, preventing extreme values and reducing the impact of individual features.

The regularization strength is controlled by a hyperparameter, often denoted as lambda (λ), that determines how much weight should be given to the regularization term. A larger λ value increases the impact of regularization, making the model simpler and more regularized.

b. randomly "dropping out" a fraction of neurons during training. This means that during each training iteration, some neurons are temporarily removed from the network. Dropout effectively reduces the interdependence between neurons, preventing the network from relying too heavily on specific neurons and making it more robust.

**Batch Normalization**: While not explicitly a regularization technique, Batch Normalization has a regularizing effect on the model. It normalizes the activations of each layer in the network, reducing internal covariate shift. This can improve training stability and reduce the need for aggressive dropout or weight decay.

**Data Augmentation**: Data augmentation is a technique where the training data is artificially augmented by applying various transformations like rotation, scaling, flipping, and cropping to create new examples. This increases the diversity of the training data and helps the model generalize better to unseen data.

**Early Stopping**: Early stopping is a form of regularization that stops the training process when the model's performance on a validation set starts to degrade. This prevents the model from overfitting by avoiding further training after the point of best validation performance.

By using regularization techniques, you can improve the generalization performance of CNNs and reduce the risk of overfitting. It's essential to experiment with different regularization methods and hyperparameters to find the optimal combination for your specific CNN architecture and dataset.

::::::::::::::::::::::::::::::::::::::::::::::::


## Choose the best model and use it to predict

Based on our evaluation of the loss and accuracy metrics, the `model_dropout` appears to have the best performance **of the models we have examined thus far**. The next step is to use this model to predict the object classes on our test dataset.


::::::::::::::::::::::::::::::::::::: keypoints 

- Use model.compile to compile a CNN.
- The choice of loss function will depend on your data and aim.
- The choice of optimizer often depends on experimentation and empirical evaluation.
- Use model.fit to make a train (fit) a CNN.
- Training/validation loss and accuracy can be used to evaluate a model during training.
- Dropout is one way to prevent overfitting.

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->

[loss documentation]: https://keras.io/api/losses/
[optimizer documentation]: https://keras.io/api/optimizers/
[metrics]: https://keras.io/api/metrics/
[fit method]: https://keras.io/api/models/model_training_apis/

[Google Developers Machine Learning Crash Course]: https://developers.google.com/machine-learning/crash-course/reducing-loss/learning-rate
[Creative Commons 4.0 Attribution Licence]: https://creativecommons.org/licenses/by/4.0/