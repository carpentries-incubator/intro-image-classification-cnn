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

## 10. Share model

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
[Model Zoo]: https://modelzoo.co/
[pytorch hub]: https://pytorch.org/hub/
[tensorflow hub]: https://pytorch.org/hub/


