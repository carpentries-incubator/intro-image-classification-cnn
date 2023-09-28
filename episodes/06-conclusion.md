---
title: 'Conclusion'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- What sort of problems can Deep Learning solve?
- What sort of problems should Deep Learning not be used for?
- How do I share my convolutional neural network (CNN)?
- Where can I find pre-trained models?
- What is a GPU?
- What other problems can be solved with a CNN?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain when to use a CNN and when not to
- Understand Github
- Use a pre-trained model for your data
- Know what a GPU is and what it can do for you

::::::::::::::::::::::::::::::::::::::::::::::::

## What sort of problems can Deep Learning solve?

* Pattern/object recognition
* Segmenting images (or any data)
* Translating between one set of data and another, for example natural language translation.
* Generating new data that looks similar to the training data, often used to create synthetic datasets, art or even "deepfake" videos.
    * This can also be used to give the illusion of enhancing data, for example making images look sharper, video look smoother or adding colour to black and white images. But beware of this, it is not an accurate recreation of the original data, but a recreation based on something statistically similar, effectively a digital imagination of what that data could look like.

## What sort of problems can Deep Learning not solve?

* Any case where only a small amount of training data is available.
* Tasks requiring an explanation of how the answer was arrived at.
* Classifying things which are nothing like their training data.

## Step 10. Share model

Now that we have a trained network that performs at a level we are happy with and can maintain high prediction accuracy on a test dataset we might want to consider publishing a file with both the architecture of our network and the weights which it has learned (assuming we did not use a pre-trained network). This will allow others to use it as as pre-trained network for their own purposes and for them to (mostly) reproduce our result.

```python
model.save('my_first_model')
```

This saved model can be loaded again by using the load_model method as follows:

```python
pretrained_model = keras.models.load_model('my_first_model')
```

This loaded model can be used as before to predict.

```python
# use the pretrained model here
y_pretrained_pred = pretrained_model.predict(X_test)
pretrained_prediction = pd.DataFrame(y_pretrained_pred, columns=target.columns.values)

# idxmax will select the column for each row with the highest value
pretrained_predicted_species = pretrained_prediction.idxmax(axis="columns")
print(pretrained_predicted_species)
```
TODO modify above for our example

## Choosing a pretrained model

If your data and problem is very similar to what others have done, you can often use a pretrained network. Even if your problem is different, but the data type is common (for example images), you can use a pretrained network and finetune it for your problem. A large number of openly available pretrained networks can be found in the [Model Zoo], [pytorch hub] or [tensorflow hub].

## Deep Learning Libraries

There are many software libraries available for Deep Learning including:

### TensorFlow

[TensorFlow](https://www.tensorflow.org/) was developed by Google and is one of the older Deep Learning libraries, ported across many languages since it was first released to the public in 2015. It is very versatile and capable of much more than Deep Learning but as a result it often takes a lot more lines of code to write Deep Learning operations in TensorFlow than in other libraries. It offers (almost) seamless integration with GPU accelerators and Google's own TPU (Tensor Processing Unit) chips that are built specially for machine learning.

### PyTorch

[PyTorch](https://pytorch.org/) was developed by Facebook in 2016 and is a popular choice for Deep Learning applications. It was developed for Python from the start and feels a lot more "pythonic" than TensorFlow. Like TensorFlow it was designed to do more than just Deep Learning and offers some very low level interfaces. [PyTorch Lightning](https://www.pytorchlightning.ai/) offers a higher level interface to PyTorch to set up experiments. Like TensorFlow it is also very easy to integrate PyTorch with a GPU. In many benchmarks it outperforms the other libraries.

### Keras

[Keras](https://keras.io/) is designed to be easy to use and usually requires fewer lines of code than other libraries. We have chosen it for this workshop for that reason. Keras can actually work on top of TensorFlow (and several other libraries), hiding away the complexities of TensorFlow while still allowing you to make use of their features.

The performance of Keras is sometimes not as good as other libraries and if you are going to move on to create very large networks using very large datasets then you might want to consider one of the other libraries. But for many applications the performance difference will not be enough to worry about and the time you will save with simpler code will exceed what you will save by having the code run a little faster.

Keras also benefits from a very good set of [online documentation](https://keras.io/guides/) and a large user community. You will find that most of the concepts from Keras translate very well across to the other libraries if you wish to learn them at a later date.



::::::::::::::::::::::::::::::::::::: keypoints 

- "Deep Learning is well suited to classification and prediction problems such as image recognition."
- "To use Deep Learning effectively we need to go through a workflow of: defining the problem, identifying inputs and outputs, preparing data, choosing the type of network, choosing a loss function, training the model, tuning Hyperparameters, measuring performance before we can classify data."
- "Keras is a Deep Learning library that is easier to use than many of the alternatives such as TensorFlow and PyTorch."

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->
[Model Zoo]: https://modelzoo.co/
[pytorch hub]: https://pytorch.org/hub/
[tensorflow hub]: https://pytorch.org/hub/


