---
title: 'Conclusion'
teaching: 10
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do I share my convolutional neural network (CNN)?
- Where can I find pre-trained models?
- What is a GPU?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Learn how to save and load models
- Know where to look for pretrained models
- Understand what a GPU is and what it can do for you
- Explain when to use a CNN and when not to
::::::::::::::::::::::::::::::::::::::::::::::::


### Step 10. Share model

Now that we have a trained network that performs at a level we are happy with and can maintain high prediction accuracy on a test dataset we might want to consider publishing a file with both the architecture of our network and the weights which it has learned (assuming we did not use a pre-trained network). This will allow others to use it as as pre-trained network for their own purposes and for them to (mostly) reproduce our result.

We have already seen how to save a model with `model.save`:
```
#model.save('model_final.h5')
```

The `save` method is actually an alias for `tf.keras.saving.save_model()` where the default `save_format=NONE`. By adding the extension **.h5** to our filename, keras will save the model in the legacy HDF5 format.

This saved model can be loaded again by using the `load_model` method as follows:

```python
# load a saved model
pretrained_model = keras.models.load_model('model_final.h5')
```

This loaded model can be used as before to predict.

```python
# use the pretrained model here
from icwithcnn_functions import prepare_image_icwithcnn

new_img_path = "../data/Jabiru_TGS.JPG" # path to image
new_img_prepped = prepare_image_icwithcnn(new_img_path)

# predict the class name
y_pretrained_pred = pretrained_model.predict(new_img_prepped)
pretrained_predicted_class = class_names[y_pretrained_pred.argmax()]
print(pretrained_predicted_class)
```
```output
frog
```

The HDF5 file format contains:

- configuration (architecture)
- weights
- optimizer's state (if any)
  - allows you to continue training; useful for checkpointing

Note that saving the model does not save the training history (ie training and validation loss and accuracy). For that you will need to save the model history dataframe we created for plotting.

To find out more about other file formats you can use to save your model see the Keras documentation for [Saving and Serialization].

To share your model with a wider audience it is recommended you create git repository, such as [GitHub], and upload your code, images, and model outputs to the cloud. In some cases, you may be able to offer up your model to an online repository of pretrained models.

#### Choosing a pretrained model

If your data and problem is very similar to what others have done, you can often use a pretrained network. Even if your problem is different, but the data type is common (for example images), you can use a pretrained network and finetune it for your problem. A large number of openly available pretrained networks can be found in the [Model Zoo], [pytorch hub] or [tensorflow hub].

### What else do I need to know?

#### How to choose a Deep Learning Library

In this lesson we chose to use [Keras] because it was designed to be easy to use and usually requires fewer lines of code than other libraries. Keras can actually work on top of TensorFlow (and several other libraries), hiding away the complexities of TensorFlow while still allowing you to make use of their features.

The performance of Keras is sometimes not as good as other libraries and if you are going to move on to create very large networks using very large datasets then you might want to consider one of the other libraries. But for many applications the performance difference will not be enough to worry about and the time you will save with simpler code will exceed what you will save by having the code run a little faster.

Keras also benefits from a very good set of [online documentation] and a large user community. You will find that most of the concepts from Keras translate very well across to the other libraries if you wish to learn them at a later date.


A couple of those libraries include:

- [TensorFlow] was developed by Google and is one of the older Deep Learning libraries, ported across many languages since it was first released to the public in 2015. It is very versatile and capable of much more than Deep Learning but as a result it often takes a lot more lines of code to write Deep Learning operations in TensorFlow than in other libraries. It offers (almost) seamless integration with GPU accelerators and Google's own TPU (Tensor Processing Unit) chips that are built specially for machine learning.

- [PyTorch] was developed by Facebook in 2016 and is a popular choice for Deep Learning applications. It was developed for Python from the start and feels a lot more "pythonic" than TensorFlow. Like TensorFlow it was designed to do more than just Deep Learning and offers some very low level interfaces. [PyTorch Lightning] offers a higher level interface to PyTorch to set up experiments. Like TensorFlow it is also very easy to integrate PyTorch with a GPU. In many benchmarks it outperforms the other libraries.

- NEW [Keras Core] In Fall 2023, this library will become Keras 3.0. Keras Core is a full rewrite of the Keras codebase that rebases it on top of a modular backend architecture. It makes it possible to run Keras workflows on top of arbitrary frameworks — starting with TensorFlow, JAX, and PyTorch.


#### What is a GPU and do I need one?

A **GPU**, or **Graphics Processing Unit**, is a specialized electronic circuit designed to accelerate graphics rendering and image processing in a computer. In the context of deep learning and machine learning, GPUs have become essential due to their ability to perform parallel computations at a much faster rate compared to traditional central processing units (CPUs). This makes them well-suited for the intensive matrix and vector operations that are common in deep learning algorithms.

As you have seen in this lesson, training CNN models can take a long time. If you follow the steps presented here you will find you are training multiple models to find the one best suited to your needs, particularly when fine tuning hyperparameters. However you have also seen that running on CPU only machines can be done! So while a GPU is not an absolute requirement for deep learning, it can significantly accelerate your deep learning work and make it more efficient, especially for larger and more complex tasks. 

If you don't have access to a powerful GPU locally, you can use cloud services that provide GPU instances for deep learning. This can be a cost-effective option for many users.

#### It this the best/only way to code up CNN's for image classification?

Absolutely not! The code we used in today's workshop might today be considered old fashioned. A lot of the data preprocessing we did by hand can now be done by simply adding different layer types to your model. See, for example, the [preprocessing layers] available with keras.

The point is that this technology, both hardware and software, is dynamic and changing at exponentially increasing rates. It is essential to stay curious and open to learning and follow up with continuous education and practice. Other strategies to stay informed include:

 - Online communications and forums, such as the Reddit's [r/MachineLearning] and [Data Science Stack Exchange]
   - watch out for outdated threads!
 - Academic journals and conferences
   - Unlike other sciences, computer science digital libraries like [arXiv] enable researchers to publish their preprints in advance and disseminates recent advances more quickly than traditional methods of publishing
 - [GitHub] repositories
 - Practice
   - like any other language, you must use it or lose it!

#### What other uses are there for neural networks?

In addition to image classification, we saw in the introduction other computer vision tasks including object detection and instance and semantic segmentation. These can all be done with CNN's and are readily transferable to videos. Also included in these tasks is medical imaging for diagnoses of disease and, of course, facial recognition. 

However, there are many other tasks which CNNs are well suited for:

- Language tasks
  - Natural Language Processing (NLP) for text classification (sentiment analysis, spam detection, topic classification)
  - Speech Recognition for speech to text conversion
- Drug Discovery
- Time-series analysis (sensor readings, financial data, health monitoring)
- Robotics
- Self-driving cars


::::::::::::::::::::::::::::::::::::: keypoints 

- Deep Learning is well suited to classification and prediction problems such as image recognition.
- To use Deep Learning effectively we need to go through a workflow of: defining the problem, identifying inputs and outputs, preparing data, choosing the type of network, choosing a loss function, training the model, tuning Hyperparameters, measuring performance before we can classify data.
- Keras is a Deep Learning library that is easier to use than many of the alternatives such as TensorFlow and PyTorch.
- Graphical Processing Units are useful, though not essential, for deep learning tasks

::::::::::::::::::::::::::::::::::::::::::::::::

<!-- Collect your link references at the bottom of your document -->
[Saving and Serialization]: https://keras.io/api/saving/
[GitHub]: https://github.com/
[Model Zoo]: https://modelzoo.co/
[pytorch hub]: https://pytorch.org/hub/
[tensorflow hub]: https://pytorch.org/hub/
[TensorFlow]: https://www.tensorflow.org/
[Keras]: https://keras.io/
[PyTorch]: https://pytorch.org/
[PyTorch Lightning]: https://www.pytorchlightning.ai/
[online documentation]: https://keras.io/guides/
[Keras Core]: https://keras.io/keras_core/announcement/?utm_source=ADSA&utm_campaign=60c8d8b6cb-EMAIL_CAMPAIGN_2022_10_04_06_04_COPY_01&utm_medium=email&utm_term=0_5401c7226a-60c8d8b6cb-461545621
[preprocessing layers]: https://keras.io/guides/preprocessing_layers/
[arXiv]: https://arxiv.org/
[r/MachineLearning]: https://www.reddit.com/r/MachineLearning/?rdt=58875
[Data Science Stack Exchange]: https://datascience.stackexchange.com/