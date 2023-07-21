---
title: 'Evaluate a Convolutional Neural Network and Make Predictions (Classifications)'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do you monitor the training process?
- How do you detect overfitting?
- How do you avoid overfitting?
- How do you measure model accuracy?
- How do you use a model to make a prediction?
- How to you improve model performance?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain what overfitting is
- Expain how to measure the performance of model fitting (loss and accuracy) compared to test accuracy
- Understand what steps to take to improve model accuracy

::::::::::::::::::::::::::::::::::::::::::::::::

### 7. Perform a Prediction/Classification

After training the network we can use it to perform predictions. This is the mode you would use the network in after you have fully trained it to a satisfactory performance. Doing predictions on a special hold-out set is used in the next step to measure the performance of the network.

We will use our convolutional neural network to predict the class names of the test set using the predict function. We will be using these predictions in the next step to measure the performance of our trained network. 

This will return a vector of probabilities, one for each class. By finding the highest probability we can select the most likely class name of the object.

```
y_pred = model.predict(testimages)
prediction = pd.DataFrame(y_pred, columns=target.columns)
prediction
```

TODO Maybe first we do this on the test samples to see it do well and then on a new image.
This will let us get a list of predicted species we can then use to demonstrate how to calculate confusion matrix values.


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
print('Result: ', result) # probability for each class
print('Classes: ', class_names) # original list of names
print('Class name: ', class_names[result.argmax()]) # class with highest probability
```

```output
Result:  [[ 8.610077  10.412511   8.109443   8.799986   1.3238649  5.4381804
  16.520676   7.8476925 10.562257   2.2948816]]
Classes:  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Class name:  frog
```

TODO modify all of this section for our example

### 8. Measuring performance
Once we trained the network we want to measure its performance. To do this we use some additional data that was not part of the training, this is known as a test set. There are many different methods available for measuring performance and which one is best depends on the type of task we are attempting. These metrics are often published as an indication of how well our network performs.

Now that we have a trained neural network it is important to assess how well it performs. We want to know how well it will perform in a realistic prediction scenario, measuring performance will also come back when tuning the hyperparameters.

We have created a test set during the data preparation stage which we will use now to create a confusion matrix.

TODO should we do this in ep02 or create our own (maybe with everyone's images?)

#### Confusion matrix

With the predicted species we can now create a confusion matrix and display it using seaborn. To create a confusion matrix we will use another convenient function from sklearn called confusion_matrix. This function takes as a first parameter the true labels of the test set. We can get these by using the idxmax method on the y_test dataframe. The second parameter is the predicted labels which we did above.

```python
from sklearn.metrics import confusion_matrix

true_species = y_test.idxmax(axis="columns")

matrix = confusion_matrix(true_species, predicted_species)
print(matrix)
```
```output
[[22  0  8]
 [ 5  0  9]
 [ 6  0 19]]
 ```

Unfortunately, this matrix is kinda hard to read. Its not clear which column and which row corresponds to which species. So let's convert it to a pandas dataframe with its index and columns set to the species as follows:

```python
# Convert to a pandas dataframe
confusion_df = pd.DataFrame(matrix, index=y_test.columns.values, columns=y_test.columns.values)

# Set the names of the x and y axis, this helps with the readability of the heatmap.
confusion_df.index.name = 'True Label'
confusion_df.columns.name = 'Predicted Label'
```

We can then use the heatmap function from seaborn to create a nice visualization of the confusion matrix. The annot=True parameter here will put the numbers from the confusion matrix in the heatmap.

```python
sns.heatmap(confusion_df, annot=True)
```

::::::::::::::::::::::::::::::::::::: challenge 

## Challenge Confusion Matrix

Looking at the training curve we have just made.

Measure the performance of the neural network you trained and visualize a confusion matrix.

- Did the neural network perform well on the test set?
- Did you expect this from the training loss you saw?
- What could we do to improve the performance?

:::::::::::::::::::::::: solution 

The confusion matrix shows that the predictions for Adelie and Gentoo are decent, but could be improved. However, Chinstrap is not predicted ever.

The training loss was very low, so from that perspective this may be surprising. But this illustrates very well why a test set is important when training neural networks.

We can try many things to improve the performance from here. One of the first things we can try is to balance the dataset better. Other options include: changing the network architecture or changing the training parameters

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

### 9. Tune hyperparameters
Hyperparameters are all the parameters set by the person configuring the machine learning instead of those learned by the algorithm itself. It might be necessary to adjust these and re-run the training many times before we are happy with the result.

The hyperparameters include:

- number of neurons
- activation function

TODO how to choose activation function - here or back in build with a callout?

- optimizer
- learning rate
- batch size
- epoch


TODO Add a challenge to change the loss or optimizer

#### Set expectations: How difficult is the defined problem?
Before we dive deeper into handling overfitting and (trying to) improving the model performance, let us ask the question: How well must a model perform before we consider it a good model?

Now that we defined a problem (classify an image into one of 10 different classes), it makes sense to develop an intuition for how difficult the posed problem is. Frequently, models will be evaluated against a so called **baseline**. A baseline can be the current standard in the field or if such a thing does not exist it could also be an intuitive first guess or toy model. The latter is exactly what we would use for our case.

TODO might be able to do something like this

#### Watch your model training closely

As we saw when comparing the predictions for the training and the test set, deep learning models are prone to overfitting. Instead of iterating through countless cycles of model trainings and subsequent evaluations with a reserved test set, it is common practice to work with a second split off dataset to monitor the model during training. This is the validation set which can be regarded as a second test set. As with the test set, the datapoints of the validation set are not used for the actual model training itself. Instead, we evaluate the model with the validation set after every epoch during training, for instance to stop if we see signs of clear overfitting. Since we are adapting our model (tuning our hyperparameters) based on this validation set, it is very important that it is kept separate from the test set. If we used the same set, we would not know whether our model truly generalizes or is only overfitting.


TODO add new model with validation data
see this section deep-learning 03 weather

::::::::::::::::::::::::::::::::::::: challenge 

## Exercise: plot the training progress

1. Is there a difference between the training and validation data? And if so, what would this imply?
1. (Optional) Take a pen and paper, draw the perfect training and validation curves. (This may seem trivial, but it will trigger you to think about what you actually would like to see)

:::::::::::::::::::::::: solution 

The difference between training and validation data shows that something is not completely right here. The model predictions on the validation set quickly seem to reach a plateau while the performance on the training set keeps improving. That is a common signature of overfitting.

Optional: Ideally you would like the training and validation curves to be identical and slope down steeply to 0. After that the curves will just consistently stay at 0.

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::

#### Counteract model overfitting

Overfitting is a very common issue and there are many strategies to handle it. Most similar to classical machine learning might to **reduce the number of parameters**.

TODO revisit this section deep-learning 03 weather
TODO might need to break this out into new episode


::::::::::::::::::::::::::::::::::::: challenge 

## Open question: What could be next steps to further improve the model?

With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results. Usually models are "well behaving" in the sense that small chances to the architectures also only result in small changes of the performance (if any). It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist? Applying common sense is often a good first step to make a guess of how much better could results be. In the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision. But how much better our model could be exactly, often remains difficult to answer.

- What changes to the model architecture might make sense to explore?
- Ignoring changes to the model architecture, what might notably improve the prediction quality?

:::::::::::::::::::::::: solution 

This is on open question. And we don't actually know how far one could push this sunshine hour prediction (try it out yourself if you like! We're curious!). But there is a few things that might be worth exploring.

Regarding the model architecture:

- In the present case we do not see a magical silver bullet to suddenly boost the performance. But it might be worth testing if deeper networks do better (more layers).

Other changes that might impact the quality notably:

- The most obvious answer here would be: more data! Even this will not always work (e.g. if data is very noisy and uncorrelated, more data might not add much).
- Related to more data: use data augmentation. By creating realistic variations of the available data, the model might improve as well.
- More data can mean more data points (you can test it yourself by taking more than the 3 years we used here!)
- More data can also mean more features! What about adding the month?
- The labels we used here (sunshine hours) are highly biased, many days with no or nearly no sunshine but few with >10 hours. Techniques such as oversampling or undersampling might handle such biased labels better. Another alternative would be to not only look at data from one day, but use the data of a longer period such as a full week. This will turn the data into time series data which in turn might also make it worth to apply different model architectures....

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::


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

