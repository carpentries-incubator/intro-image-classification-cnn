---
title: 'Evaluate a Convolutional Neural Network and Make Predictions (Classifications)'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you use a model to make a prediction?
- How do you measure model prediction accuracy?
- What is a hyperparameter?
- What can you do to improve model performance?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Use a convolutional neural network (CNN) to make a prediction (i.e. classify an image).
- Explain how to measure the performance of a CNN.
- Explain hyperparameter tuning.
- Be familiar with advantages and disadvantages of different optimizers.
- Understand what steps to take to improve model accuracy.

::::::::::::::::::::::::::::::::::::::::::::::::

### Step 7. Perform a Prediction/Classification

After you fully train the network to a satisfactory performance on the training and validation sets, we use it to perform predictions on a special hold-out set, the **test** set. The prediction accuracy of the model on new images will be used in **Step 8. Measuring performance** to measure the performance of the network.

#### Prepare test dataset

Recall in [Episode 02 Introduction to Image Data](episodes/02-image-data.md) we discussed how to split your data into training and test datasets and why. In most cases, that means you already have a test set on hand. For example, we are using `keras.models.load_model` to create a training and test set.

When creating and using a test set there are a few things to check:

- It only contains images the model has never seen before.
- It is sufficiently large to provide a meaningful evaluation of model performance. It should include images from every target label and images of classes not in your target set.
- It is processed in the same way as your training set.

Check to make sure you have a model in memory and a test dataset:

```python
# check correct model is loaded
model_best = keras.models.load_model('fit_outputs/model_dropout.h5') # pick your best model
print('We are using', model_best.name)

# load the CIFAR-10 dataset included with the keras library
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# normalize the RGB values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# we do not one hot encode here because our model predicts a class label

# check correct model is loaded
print('We are using', model_best.name)

# check test image dataset is loaded - images and labels
print('The number and shape of images in our test dataset is:', test_images.shape)
print('The number of labels in our test dataset is:', len(test_labels))
```
```output
We are using cifar_model_dropout
The number and shape of images in our test dataset is:  (10000, 32, 32, 3)
The number of labels in our test dataset is:  10000
```


::::::::::::::::::::::::::::::::::::: challenge 

How big should our test data set be?

:::::::::::::::::::::::: solution 

It depends! Recall in an [Episode 02 Introduction to Image Data](episodes/02-image-data.md) Callout we talked about the different ways to partition the data into training, validation and test data sets. For example, using the **Stratified Sampling** technique, we might split the data using these rations: 80-10-10 or 70-15-15.

The test set should be sufficiently large to provide a meaningful evaluation of your model's performance. Smaller datasets might not provide a reliable estimate of how well your model generalizes.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

#### Predict

Armed with a test dataset, we will use our CNN to predict their class labels using the `predict` function and then use these predictions in Step 8 to measure the performance of our trained network.

Recall our model will return a vector of probabilities, one for each class. By finding the class with the highest probability, we can select the most likely class name of the object.


```python
# use our current best model to predict probability of each class on new test set
predictions = model_best.predict(test_images)

# create a list of classnames
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# convert probability predictions to table using class names for column names
prediction_df = pd.DataFrame(predictions, columns=class_names)

# inspect 
print(prediction_df.head())

# convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)

```

```output
airplane  automobile      bird  ...     horse      ship     truck
0  0.165748   -0.118394  0.062156  ...  0.215477  0.013811 -0.047446
1  0.213530   -0.126139  0.052813  ...  0.264517  0.009097 -0.091710
2  0.211900   -0.099055  0.047890  ...  0.242345 -0.014492 -0.073153
3  0.187883   -0.085144  0.044609  ...  0.217864  0.007502 -0.055209
4  0.190110   -0.118892  0.054869  ...  0.252434 -0.030064 -0.061485

[5 rows x 10 columns]

Out  [5]: array([3, 8, 8, ..., 5, 1, 7], dtype=int64)
```


### Step 8. Measuring performance

Once we trained the network we want to measure its performance. There are many different methods available for measuring performance and which one to use depends on the type of task we are attempting. These metrics are often published as an indication of how well our network performs.

An easy way to check the accuracy of our model on the test set is to use the `accuracy_score()` from `sklearn.metrics`:

```python
from sklearn.metrics import accuracy_score

# evaluate the model on the test data set
test_acc = accuracy_score(test_labels, predicted_labels)
print('Accuracy:', round(test_acc,2))
```
```output
Accuracy: 0.67
```

To understand a bit more about how this accuracy is obtained, we create a confusion matrix.


#### Confusion matrix

In the case of multiclass classifications, each cell value (C~i,j~) is equal to the number of observations known to be in group _i_ and predicted to be in group _j_. The diagonal cells in the matrix are where the true class and predicted class match.

![](fig/05_confusion_matrix_explained.png){alt='for ten classes an example confusion matrix has 10 rows and 10 columns where the value in each cell is the number of observations predicted in that class and known to be in that class. The diagonal cells are where the true and predicted classes match.'}

To create a confusion matrix, we use another convenient function from sklearn called `confusion_matrix`. This function takes as a first parameter the true labels of the test set. The second parameter is the predicted labels from our model.

```python
from sklearn.metrics import confusion_matrix

# create a confusion matrix
conf_matrix = confusion_matrix(test_labels_values, predicted_labels)
print(conf_matrix)
```
```output
[[682  36  67  13  15   7  27  11 108  34]
 [ 12 837   3   2   6   3  32   0  29  76]
 [ 56   4 462  37 137  72 184  26  13   9]
 [ 10  13  48 341  87 217 221  27  17  19]
 [ 23   4  38  34 631  23 168  63  13   3]
 [  8   9  59 127  74 550 103  51  10   9]
 [  3   3  18  23  18  12 919   2   1   1]
 [ 14   8  24  28  98  75  34 693   2  24]
 [ 56  39  11  17   4   6  22   2 813  30]
 [ 23 118   6  12   6   5  33  15  35 747]]
 ```

Unfortunately, this matrix is kinda hard to read. It's not clear which column and which row corresponds to which class. So let's convert it to a pandas dataframe with its index and columns set to the class labels as follows:

```python
# Convert to a pandas dataframe
confusion_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Set the names of the x and y axis, this helps with the readability of the heatmap.
confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'
```

We can then use the `heatmap` function from seaborn to create a nice visualization of the confusion matrix.

```python
sns.heatmap(confusion_df, annot=True, fmt='3g')
```

- The `annot=True` parameter here will put the numbers from the confusion matrix in the heatmap.
- The `fmt=3g` will display the values with three significant digits.

![](fig/05_pred_v_true_confusion_matrix.png){alt='Confusion matrix of model predictions where the colour scale goes from black to light to represent values from 0 to the total number of test observations in our test set of 1000. The diagonal has much lighter colours, indicating our model is predicting well, but a few non-diagonal cells also have a lighter colour to indicate where the model is making a large number of prediction errors.'}


::::::::::::::::::::::::::::::::::::: challenge 

Confusion Matrix

Measure the performance of the neural network you trained and visualized as a confusion matrix.

Q1. Did the neural network perform well on the test set?

Q2. Did you expect this from the training loss plot?

Q3. What could we do to improve the performance?

:::::::::::::::::::::::: solution 

Q1. The confusion matrix illustrates that the predictions are not bad but can be improved.

Q2. I expected the performance to be better than average because the accuracy of the model I chose was 67 per cent on the validation set.

Q3. We can try many things to improve the performance from here. One of the first things we can try is to change the network architecture. However, in the interest of time, and given we already learned how to build a CNN, we will now change the training parameters.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::


### Step 9. Tune hyperparameters

Recall the following from [Episode 01 Introduction to Deep Learning](episodes/01-introduction.md):

#### What are hyperparameters? 

Hyperparameters are the parameters set by the person configuring the model instead of those learned by the algorithm itself. Like the dials on a radio which are *tuned* to the best frequency, hyperparameters can be *tuned* to the best combination for a given model and context.

These hyperparameters can include the learning rate, the number of layers in the network, the number of neurons per layer, and many more. The tuning process is systematic searching for the best combination of hyperparameters to  optimize the model's performance.

In some cases, it might be necessary to adjust these and re-run the training many times before we are happy with the result.

Table 1. List of some of the hyperparameters to tune and when.

| During Build          | When Compiling    | During Training   |
|-----------------------|-------------------|-------------------|
| number of neurons     | loss function     | epoch             |
| activation function   | optimizer         | batch size        |
| dropout rate          | learning rate     |                   |
|                       | batch size        |                   |


There are a number of techniques used to tune hyperparameters. Let us explore a few of them.

One common method for hyperparameter tuning is by using a `for` loop to change a particular parameter.

::::::::::::::::::::::::::::::::::::: challenge 

## Tune Dropout Rate using a For Loop

Q1. What do you think would happen if you lower the dropout rate? Write some code to vary the dropout rate and investigate how it affects the model training.

Q2. You are varying the dropout rate and checking its effect on the model performance, what is the term associated to this procedure?

:::::::::::::::::::::::: solution 

Q1. Varying the dropout rate

The code below instantiates and trains a model with varying dropout rates. The resulting plot indicates the ideal dropout rate in this case is around 0.45. This is where the validation loss is lowest.

- NB1: It takes a while to train these five networks.
- NB2: You should do this with a test set and not with the validation set!

```python
# one-hot encode labels
train_labels = keras.utils.to_categorical(train_labels, len(class_names))

# split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, 
																	  test_size = 0.2, 
																	  random_state = 42)

# specify range of dropout rates
dropout_rates = [0.15, 0.3, 0.45, 0.6, 0.75]

# create empty list to hold losses
val_losses_vary = [] 

for dropout_rate in dropout_rates:
    
    # Input layer of 32x32 images with three channels (RGB)
    inputs_vary = keras.Input(shape=train_images.shape[1:])
    
    # CNN Part 2
    # Convolutional layer with 16 filters, 3x3 kernel size, and ReLU activation
    x_vary = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs_vary)
    # Pooling layer with input window sized 2,2
    x_vary = keras.layers.MaxPooling2D((2, 2))(x_vary)
    # Second Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    x_vary = keras.layers.Conv2D(32, (3, 3), activation='relu')(x_vary)
    # Second Pooling layer with input window sized 2,2
    x_vary = keras.layers.MaxPooling2D((2, 2))(x_vary)
    # Second Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    x_vary = keras.layers.Conv2D(64, (3, 3), activation='relu')(x_vary)
    # Dropout layer randomly drops x per cent of the input units
    x_vary = keras.layers.Dropout(dropout_rate)(x_vary) # This is new!
    # Flatten layer to convert 2D feature maps into a 1D vector
    x_vary = keras.layers.Flatten()(x_vary)
    # Dense layer with 128 neurons and ReLU activation
    x_vary = keras.layers.Dense(128, activation='relu')(x_vary)
    
    # CNN Part 3
    # Output layer with 10 units (one for each class) and softmax activation
    outputs_vary = keras.layers.Dense(10, activation='softmax')(x_vary)

    model_vary = keras.Model(inputs = inputs_vary, outputs = outputs_vary, 
                             name ="cifar_model_vary_dropout")

    model_vary.compile(optimizer = 'adam',
                       loss = keras.losses.CategoricalCrossentropy(),
                       metrics = ['accuracy'])

    model_vary.fit(train_images, train_labels, 
                   epochs = 20,
                   validation_data = (val_images, val_labels),
                   batch_size = 32)

    val_loss_vary, val_acc_vary = model_vary.evaluate(val_images, val_labels)
    val_losses_vary.append(val_loss_vary)
    
loss_df = pd.DataFrame({'dropout_rate': dropout_rates, 'val_loss_vary': val_losses_vary})

sns.lineplot(data=loss_df, x='dropout_rate', y='val_loss_vary')
```
![](fig/05_vary_dropout_rate.png){alt='test loss plotted against five dropout rates ranging from 0.15 to 0.75 where the minimum test loss appears to occur between 0.4 and 0.5'}

Q2. Term associated to this procedure

This is called hyperparameter tuning.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

Another common method for hyperparameter tuning is **grid search**. 

#### What is Grid Search?

Grid Search or `GridSearchCV` (as per the library function call) is foundation method for hyperparameter tuning. The aim of hyperparameter tuning is to define a grid of possible values for each hyperparameter you want to tune. GridSearch will then evaluate the model performance for each combination of hyperparameters in a brute-force manner, iterating through every possible combination in the grid.

For instance, suppose you're tuning two hyperparameters:

- Learning rate: with possible values [0.01, 0.1, 1]

- Batch size: with possible values [10, 50, 100]

- GridSearch will evaluate the model for all 3*3 = 9 combinations (e.g., {0.01, 10}, {0.01, 50}, {0.1, 10}, and so on).

::::::::::::::::::::::::::::::::::::: challenge 

## Tune Optimizer using Grid Search

In [Episode 04 Compile and Train a Convolutional Neural Network](episodes/04-fit-cnn.md) we talked briefly about the `Adam` optimizer used in our `model.compile` discussion. Recall the optimizer refers to the algorithm with which the model learns to optimize on the provided loss function.

Here we will use our introductory model to demonstrate how GridSearch is expressed in code to search for an optimizer.

:::::::::::::::::::::::: solution 

First, we will define a **build function** to use during GridSearch. This function will compile the model for each combination of parameters prior to evaluation.

```python
def create_model():

    # Input layer of 32x32 images with three channels (RGB)
    inputs = keras.Input(shape=train_images.shape[1:])

    # Convolutional layer with 50 filters, 3x3 kernel size, and ReLU activation
     = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    # Second Convolutional layer
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    # Flatten layer to convert 2D feature maps into a 1D vector
    x = keras.layers.Flatten()(x)

    # Output layer with 10 units (one for each class)
    outputs = keras.layers.Dense(10)(x)

    # create the model
    mode = keras.Model(inputs=inputs, outputs=outputs)
    
    # compile the pooling model
    model.compile(optimizer = 'adam', 
                  loss = keras.losses.CategoricalCrossentropy(), 
                  metrics=['accuracy'])
    
    return model
```

Secondly, we can define our GridSearch parameters and assign fit results to a variable for output.

```python
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Wrap the model
model = KerasClassifier(build_fn=create_model, epochs=2, batch_size=32, verbose=0)  # epochs, batch_size, verbose can be adjusted as required. Using low epochs to save computation time and demonstration purposes only

# Define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adam']
param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(train_images, train_labels)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
```
```output
Best: 0.586660 using {'optimizer': 'RMSprop'}
```

Thus, we can interpret from this output that our best tested optimiser is the **root mean square propagation** optimiser, or RMSprop.

Curious about RMSprop? [RMSprop in Keras] and [RMSProp, Cornell University]

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

A third way to tune hyperparameters is brute force.

::::::::::::::::::::::::::::::::::::: challenge 

## Tune Activation Function using Brute Foce

In [Episode 03 Build a Convolutional Neural Network](episodes/03-build-cnn.md) we talked briefly about the `relu` activation function passed as an argument to our `Conv2D` hidden layers.

An activation function is like a switch, or a filter, that we use in artificial neural networks, inspired by how our brains work. These functions play a crucial role in determining whether a neuron (a small unit in the neural network) should "fire" or become active. 

Think of an activation function as a tiny decision-maker for each neuron in a neural network. It helps determine whether the neuron should 'fire', or pass on information, or stay 'off' and remain silent, much like a light switch controls whether the light should be ON or OFF. Activation functions are crucial because they add non-linearity to the neural network. Without them, the network would be like a simple linear model, unable to learn complex patterns in data. 

### How do you know what activation function to choose?

Neural networks can be tuned to leverage many different types of activation functions. In fact, it is a crucial decision as the choice of activation function will have a direct impact on the performance of the model.

Table 2. Description of each activation function, its benefits, and drawbacks.

| Activation Function | Positives                                                        | Negatives                                  |
|---------------------|------------------------------------------------------------------|--------------------------------------------|
| ReLU                | - Addresses vanishing gradient problem <br/> - Computationally efficient | - Can cause "dying neurons" <br/> - Not zero-centered |
| Leaky ReLU          | - Addresses the "dying ReLU" problem <br/> - Computationally efficient | - Empirical results can be inconsistent <br/> - Not zero-centered |
| Sigmoid             | - Outputs between 0 and 1 <br/> - Smooth gradient               | - Can cause vanishing gradient problem <br/> - Computationally more expensive |
| Tanh                | - Outputs between -1 and 1 <br/> - Zero-centered                | - Can still suffer from vanishing gradients to some extent |
| Softmax             | - Used for multi-class classification <br/> - Outputs a probability distribution | - Used only in the output layer for classification tasks |
| SELU                | - Self-normalizing properties <br/> - Can outperform ReLU in deeper networks | - Requires specific weight initialization <br/> - May not perform well outside of deep architectures |

Write some code to assessing activation function performance.

:::::::::::::::::::::::: solution 

The code below serves as a practical means for exploring activation performance on an image dataset.

```python
# Define a function to create a model with a given activation function
def create_model(activation_function):

    # Input layer of 32x32 images with three channels (RGB)
    inputs = keras.Input(shape=train_images.shape[1:])

    # Convolutional layer with 50 filters, 3x3 kernel size, and ReLU activation
    x = keras.layers.Conv2D(50, (3, 3), activation=activation_function)(inputs)
    # Second Convolutional layer
    x = keras.layers.Conv2D(50, (3, 3), activation=activation_function)(x)
    # Flatten layer to convert 2D feature maps into a 1D vector
    x = keras.layers.Flatten()(x)

    # Output layer with 10 units (one for each class)
    outputs = keras.layers.Dense(10)(x)

    # create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # create the model
    model.compile(optimizer = 'adam', 
                  loss = keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    return model

# List of activation functions to try
activations = ['relu', 'sigmoid', 'tanh', 'selu', keras.layers.LeakyReLU()]

history_data = {}

# Train a model with each activation function and store the history
for activation in activations:
    model = create_model(activation)
    history = model.fit(train_images, train_labels, 
                        epochs=10, 
                        validation_data=(val_images, val_labels),
                        batch_size = 32)
    history_data[str(activation)] = history

# Plot the validation accuracy for each activation function
plt.figure(figsize=(12, 6))

for activation, history in history_data.items():
    plt.plot(history.history['val_accuracy'], label=activation)

plt.title('Validation accuracy for different activation functions')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
```

![](fig/05_tune_activation_results.png){alt='Validation accuracy plotted against ten epochs for five different activations functions. relu and Leaky relu have the highest accuracy atound 0.60; sigmoid and selu are next with accuracy around 0.45 and tanh has the lowest accuracy of 0.35'}

In this figure, after 10 epochs, the `ReLU` and `Leaky ReLU` activation functions appear to converge around 0.60 per cent validation accuracy. We recommend when tuning your model to ensure you use enough epochs to be confident in your results.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge 

## Open question: What could be next steps to further improve the model?

With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results. Usually models are "well behaving" in the sense that small chances to the architectures also only result in small changes of the performance (if any). It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist? Applying common sense is often a good first step to make a guess of how much better could results be.

- What changes to the model architecture might make sense to explore?
- Ignoring changes to the model architecture, what might notably improve the prediction quality?

:::::::::::::::::::::::: solution 

This is an open question.

Regarding the model architecture:

- In the present case we do not see a magical silver bullet to suddenly boost the performance. But it might be worth testing if deeper networks do better (more layers).

Other changes that might impact the quality notably:

- The most obvious answer here would be: more data! Even this will not always work (e.g., if data is very noisy and uncorrelated, more data might not add much).
- Related to more data: use data augmentation. By creating realistic variations of the available data, the model might improve as well.
- More data can mean more data points and also more features!

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::

By now you should have a well-trained, finely-tuned model that makes accurate predictions and are ready to share the model with others.

::::::::::::::::::::::::::::::::::::: keypoints 

- Use model.predict to make a prediction with your model.
- Model accuracy must be measured on a test dataset with images your model has not seen before.
- There are many hyperparameters to choose from to improve model performance.
- Fitting separate models with different hyperparameters and comparing their performance is a common and good practice in deep learning.

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->

[RMSprop in Keras]: https://keras.io/api/optimizers/rmsprop/
[RMSProp, Cornell University]: https://optimization.cbe.cornell.edu/index.php?title=RMSProp

