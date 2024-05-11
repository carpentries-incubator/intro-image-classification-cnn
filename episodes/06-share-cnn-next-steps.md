---
title: 'Share a Convolutional Neural Network and Next Steps'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do I share my convolutional neural network (CNN)?
- Where can I find pre-trained models?
- Is Keras the best library to use?
- What is a GPU?
- What else can I do with a CNN?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Learn how to save and load models.
- Know where to search for pretrained models.
- know about other Deep Learning Libraries.
- Understand what a GPU is and what it can do for you.
- Explain when to use a CNN and when not to.
::::::::::::::::::::::::::::::::::::::::::::::::


### Step 10. Share model

We now have a trained network that performs at a level we are happy with and maintains high prediction accuracy on a test dataset. We should consider publishing a file with both the architecture of our network and the weights which it has learned (assuming we did not use a pre-trained network). This will allow others to use it as as pre-trained network for their own purposes and for them to (mostly) reproduce our result.

The Keras method to save is found in the [Model training APIs] 'Saving & serialization' section of the documentation and has the following definition:

```
Model.save(filepath, overwrite=True, **kwargs)
```
 - **filepath** tells Keras where to save the model
 
We can use this method to save out model.

```python
# save best model
model_best.save('fit_outputs/model_best.keras')
```

This saved model can be loaded again by using the `load_model` method:

```python
# load a saved model
pretrained_model = keras.models.load_model('fit_outputs/model_best.keras')
```

This loaded model can be used as before to predict.

```python
# use the pretrained model to predict the class name of the first test image
result_pretrained = pretrained_model.predict(test_images[0].reshape(1,32,32,3))

print('The predicted probability of each class is: ', result_pretrained.round(4))
print('The class with the highest predicted probability is: ', class_names[result_pretrained.argmax()])
```
```output
cat
```

The saved .keras file contains:

- The model's configuration (architecture).
- The model's weights.
- The model's optimizer's state (if any).

Note that saving the model does not save the training history (i.e. training and validation loss and accuracy). For that, you save the model history dataframe we used for plotting.

The Keras documentation for [Saving and Serialization] explains other ways to save your model.

To share your model with a wider audience it is recommended you create git repository, such as [GitHub], and upload your code, images, and model outputs to the cloud. In some cases, you may be able to offer up your model to an online repository of pretrained models.

#### Choosing a pretrained model

If your data and problem is very similar to what others have done, a pre-trained network might be preferable. Even if your problem is different, if the data type is common (for example images), you can use a pre-trained network and fine-tune it for your problem. A large number of openly available pre-trained networks can be found online, including:
- [Model Zoo]
- [pytorch hub] 
- [tensorflow hub]
- [GitHub]

### What else do I need to know?

#### How to choose a Deep Learning Library

In this lesson we chose to use [Keras] because it was designed to be easy to use and usually requires fewer lines of code than other libraries. Keras can actually work on top of TensorFlow (and several other libraries), hiding away the complexities of TensorFlow while still allowing you to make use of their features.

The performance of Keras is sometimes not as good as other libraries and if you are going to move on to create very large networks using very large datasets then you might want to consider one of the other libraries. But for many applications the performance difference will not be enough to worry about and the time you will save with simpler code will exceed what you will save by having the code run a little faster.

Keras also benefits from a very good set of [online documentation] and a large user community. You will find most of the concepts from Keras translate very well across to the other libraries if you wish to learn them at a later date.

A couple of those libraries include:

- [TensorFlow] was developed by Google and is one of the older Deep Learning libraries, ported across many languages since it was first released to the public in 2015. It is very versatile and capable of much more than Deep Learning but as a result it often takes a lot more lines of code to write Deep Learning operations in TensorFlow than in other libraries. It offers (almost) seamless integration with GPU accelerators and Google's own TPU (Tensor Processing Unit) chips specially built for machine learning.

- [PyTorch] was developed by Facebook in 2016 and is a popular choice for Deep Learning applications. It was developed for Python from the start and feels a lot more "pythonic" than TensorFlow. Like TensorFlow it was designed to do more than just Deep Learning and offers some very low level interfaces. [PyTorch Lightning] offers a higher level interface to PyTorch to set up experiments. Like TensorFlow it is also very easy to integrate PyTorch with a GPU. In many benchmarks it outperforms the other libraries.

- NEW [Keras Core] In Fall 2023, this library will become Keras 3.0. Keras Core is a full rewrite of the Keras codebase that rebases it on top of a modular backend architecture. It makes it possible to run Keras workflows on top of arbitrary frameworks — starting with TensorFlow, JAX, and PyTorch.


#### What is a GPU and do I need one?

A **GPU**, or **Graphics Processing Unit**, is a specialized electronic circuit designed to accelerate graphics rendering and image processing in a computer. In the context of deep learning and machine learning, GPUs have become essential due to their ability to perform parallel computations at a much faster rate compared to traditional central processing units (CPUs). This makes them well-suited for the intensive matrix and vector operations that are common in deep learning algorithms.

As you have experienced in this lesson, training CNN models can take a long time. If you follow the steps presented here you will find you are training multiple models to find the one best suited to your needs, particularly when fine tuning hyperparameters. However you have also seen that running on CPU only machines can be done! So while a GPU is not an absolute requirement for deep learning, it can significantly accelerate your deep learning work and make it more efficient, especially for larger and more complex tasks. 

If you don't have access to a powerful GPU locally, there are cloud services that provide GPU instances for deep learning. This may be the most cost-effective option for many users.

#### It this the best/only way to code up CNNs for image classification?

Absolutely not! The code we used in today's workshop might be considered old fashioned. A lot of the data preprocessing we did by hand can now be done by adding different layer types to your model. The [preprocessing layers] section fo the Keras documentation provides several examples.

The point is that this technology, both hardware and software, is dynamic and changing at exponentially increasing rates. It is essential to stay curious and open to learning and follow up with continuous education and practice. Other strategies to stay informed include:

 - Online communications and forums, such as the Reddit's [r/MachineLearning] and [Data Science Stack Exchange]
   - watch out for outdated threads!
 - Academic journals and conferences
   - Unlike other sciences, computer science digital libraries like [arXiv] enable researchers to publish their preprints in advance and disseminates recent advances more quickly than traditional methods of publishing
 - [GitHub] repositories
 - Practice
   - like any other language, you must use it or lose it!

#### What other uses are there for neural networks?

In addition to image classification, [Episode 01 Introduction to Deep Learning](episodes/01-introduction.md) introduced other computer vision tasks, including object detection and instance and semantic segmentation. These can all be done with CNNs and are readily transferable to videos. Also included in these tasks is medical imaging for diagnoses of disease and, of course, facial recognition. 

However, there are many other tasks which CNNs are not well suited for:

- Data where input size varies
    - Natural Language Processing (NLP) for text classification (sentiment analysis, spam detection, topic classification)
  - Speech Recognition for speech to text conversion
- Sequential data and Time-series analysis
    - sensor readings, financial data, health monitoring
    - Use Recurrent Nueral Networks (RNNs) or Long Short-Term Memory networks (LTSMs)
- Applications where interpretability and explainability is crucial
    - Use simpler models, e.g., decision trees
- Situations where you lack sufficient training data

::::::::::::::::::::::::::::::::::::: keypoints 

- To use Deep Learning effectively, go through a workflow of: defining the problem, identifying inputs and outputs, preparing data, choosing the type of network, choosing a loss function, training the model, tuning hyperparameters, and measuring performance.
- Use Model.save() and share your model with others.
- Keras is a Deep Learning library that is easier to use than many of the alternatives such as TensorFlow and PyTorch.
- Graphical Processing Units are useful, though not essential, for deep learning tasks.
- CNNs work well for a variety of tasks, especially those involving grid-like data with spatial relationships, but not for time series or variable sized input data.

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->

[Saving and Serialization]: https://keras.io/api/saving/
[GitHub]: https://github.com/
[Model Zoo]: https://modelzoo.co/
[pytorch hub]: https://pytorch.org/hub/
[tensorflow hub]: https://pytorch.org/hub/
[Keras]: https://keras.io/
[online documentation]: https://keras.io/guides/
[TensorFlow]: https://www.tensorflow.org/
[PyTorch]: https://pytorch.org/
[PyTorch Lightning]: https://www.pytorchlightning.ai/
[Keras Core]: https://keras.io/keras_core/announcement/?utm_source=ADSA&utm_campaign=60c8d8b6cb-EMAIL_CAMPAIGN_2022_10_04_06_04_COPY_01&utm_medium=email&utm_term=0_5401c7226a-60c8d8b6cb-461545621
[preprocessing layers]: https://keras.io/guides/preprocessing_layers/
[r/MachineLearning]: https://www.reddit.com/r/MachineLearning/?rdt=58875
[Data Science Stack Exchange]: https://datascience.stackexchange.com/
[arXiv]: https://arxiv.org/
