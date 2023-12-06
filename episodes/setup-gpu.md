---
title: "Setup - GPU"
---

This lesson is designed for Software Carpentry users who have completed [Plotting and Programming in Python] and are looking to jump straight into image classification. We recognize that this jump is quite large and have done our best to provide the content and code to perform these types of analyses.

The default [Setup](../learners/setup.md) is for CPU only environments.

These instructions are for setting up tensorflow in a **GPU** capable environment. Because this is a more advanced topic and installation varies depending on your computer's architecture, please make sure you **set up and test** your installation before the workshop begins. We will not be able to spend class time assisting on GPU setups.

## Software Setup

::::::::::::::::::::::::::::::::::::: challenge
## Install Python using Anaconda

[Python] is a popular language for scientific computing, and a frequent choice for machine learning as well. Installing all of its scientific packages
individually can be a bit difficult, however, so we recommend the installer [Anaconda] which includes most (but not all) of the software you will need. Make sure you install the latest Python version 3.xx.

Also, please set up your python environment **at least** a day in advance of the workshop. If you encounter problems with the installation procedure *for Anaconda*, ask your workshop organizers via e-mail for assistance so you are ready to go as soon as the workshop begins.

:::::::::::::::::::::::: solution
### Windows

Check out the [Windows - Video tutorial] or:

1. Open [https://www.anaconda.com/products/distribution] with your web browser.

2. Download the Python 3 installer for Windows.

3. Double-click the executable and install Python 3 using _MOST_ of the default settings. The only exception is to check the **Make Anaconda the default Python** option. (Note this may already be checked.)
::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::: solution
### MacOS

Check out the [Mac OS X - Video tutorial] or:

1. Open [https://www.anaconda.com/products/distribution] with your web browser.

2. Download the Python 3 installer for Mac.

3. Install Python 3 using all of the defaults for installation.
::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::: solution 
### Linux

Note that the following installation steps require you to work from the shell.
If you run into any difficulties, please request help before the workshop begins.

1.  Open [https://www.anaconda.com/products/distribution] with your web browser.

2.  Download the Python 3 installer for Linux.

3.  Install Python 3 using all of the defaults for installation.

    a.  Open a terminal window.

    b.  Navigate to the folder where you downloaded the installer

    c.  Type

    ```bash
    bash Anaconda3-
    ```

    and press tab.  The name of the file you just downloaded should appear.

    d.  Press enter.

    e.  Follow the text-only prompts.  When the license agreement appears (a colon
        will be present at the bottom of the screen) hold the down arrow until the
        bottom of the text. Type `yes` and press enter to approve the license. Press
        enter again to approve the default location for the files. Type `yes` and
        press enter to prepend Anaconda to your `PATH` (this makes the Anaconda
        distribution the default Python).

:::::::::::::::::::::::::::::::::: 
::::::::::::::::::::::::::::::::::::::::::::::::

## Install the required packages

[Conda] is the package management system associated with [Anaconda] and runs on Windows, macOS and Linux.

Conda should already be available in your system once you installed Anaconda successfully. Conda thus works regardless of the operating system. Make sure you have an up-to-date version of Conda running. See [these instructions] for updating Conda if required.

The easiest way to create a conda environment for this lesson is to use the Anaconda Prompt. You can search for "anaconda prompt" using the Windows search function (Windows Logo Key) or Spotlight on macOS (Command + spacebar).

![](fig/00_anaconda_prompt_search.png){alt='Screenshot of what the Anaconda Prompt application looks like'}

A terminal window will open with the title 'Anaconda Prompt' that looks like this:

![](fig/00_anaconda_prompt_window.png){alt='Screenshot of the terminal window that opens when you launch the Anaconda Prompt application'}

Note the notation of the prompt inside the terminal window. The name inside the parentheses refers to which conda environment you are working inside of, and 'base' is the name given to the default environment that comes with every Anaconda distribution.

To create a new environment for this lesson, the command starts with the conda keywords `conda create`, followed by a name for the new environment and the package(s) to install:

```code
(base) C:\Users\Lab> conda create --name cnn_workshop_gpu python=3.9 spyder seaborn  scikit-learn pandas
```

After the environment is created we tell Anaconda to use the new environment with the conda keywords `conda activate` followed by the environment name:

```code
(base) C:\Users\Lab> conda activate cnn_workshop_gpu
(cnn_workshop_gpu) C:\Users\Lab>
```

You will know that you are in the right environment because the prompt changes from (base) to (cnn_workshop_gpu). 

::::::::::::::::::::::::::::::::::::::::: callout
To set up a GPU environment you need to make sure that you have the appropriate hardware, system, and software necessary for GPU support. Here we are following the [Windows TensorFlow installation instructions] starting at **Step 5. GPU setup** but using Anaconda instead of Miniconda. Specific instructions can also be found there for [MacOS] and [Linux] environments.
:::::::::::::::::::::::::::::::::::::::::::::::::

### NVIDIA GPU

First install NVIDIA GPU driver [https://www.nvidia.com/download/index.aspx] if you have not.

Then install the CUDA, cuDNN with conda.

```code
(cnn_workshop_gpu) C:\Users\Lab> conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
::::::::::::::::::::::::::::::::::::::::: spoiler
### AMD GPU 

First install AMD GPU driver [https://www.amd.com/en/support] if you have not.

TODO Finish these instructions

:::::::::::::::::::::::::::::::::::::::::::::::::

There are two other packages we need to install that we could not install at the same time that we created the environment, `tensorflow` and `scikeras`.

To install these two packages we have to use a different package manager called `pip`.

[pip] is the package management system for Python software packages. It is integrated into your local Python installation and runs regardless of your operating system too.

```code
(cnn_workshop_gpu) C:\Users\Lab>pip install --upgrade pip

# Anything above 2.10 is not supported on the GPU on Windows Native
(cnn_workshop_gpu) C:\Users\Lab>pip install "tensorflow<2.11"
(cnn_workshop_gpu) C:\Users\Lab>pip install scikeras
```

## Start Spyder

We teach this lesson using Python in [Spyder] (Scientific Python Development Environment), a free integrated development environment (IDE) that comes with Anaconda. Editing, interactive testing, debugging, and introspection tools are all included in Spyder.

To start Spyder, type the command `spyder`, making sure you are still in the workshop environment:

```conda
(cnn_workshop) C:\Users\Lab> spyder
```

![](fig/00_spyder_ide_layout.png){alt='Screenshot of the Spyder IDE annotated with boxes and labels for the Editor; Help, Variable Explorer, Plots, Files; and IPython Console areas'}

## Check your setup

To check whether all packages installed correctly, go to the interactive `IPython Console` in Spyder (lower right hand side panel) and type in the following commands:

```python
import sklearn
print('sklearn version: ', sklearn.__version__)

import seaborn
print('seaborn version: ', seaborn.__version__)

import pandas
print('pandas version: ', pandas.__version__)

from tensorflow import keras
print('Keras version: ', keras.__version__)

import tensorflow
print('Tensorflow version: ', tensorflow.__version__)
```

This should output the versions of all required packages without giving errors.
Most versions will work fine with this lesson, but:

- For Keras and Tensorflow, the maximum version is 2.11.
- For sklearn, the minimum version is 0.22.

## Download the exercise python template file

The aim for this workshop is to create a python script that you can used as a "base python program" that can be used for future projects.

In an effort to not clutter the scripts developed in the workshop with episode exercise/challenge code, this workshop will use an exercises python script for all of the exercises completed throughout the workshop.

This file can be downloaded from [exercises.py](../episodes/scripts/exercises.py).


## Get the data

This lesson uses the CIFAR-10 image data that comes prepackaged with Keras.


<!-- Collect your link references at the bottom of your document -->

[Plotting and Programming in Python]: https://swcarpentry.github.io/python-novice-gapminder/
[Conda]: https://docs.conda.io/projects/conda/en/latest/
[Anaconda]: https://www.anaconda.com/products/individual
[anaconda-distribution]: https://www.anaconda.com/products/distribution
[Spyder]: https://www.spyder-ide.org/
[python]: https://python.org
[Mac OS X - Video tutorial]: https://www.youtube.com/watch?v=TcSAln46u9U
[Windows - Video tutorial]: https://www.youtube.com/watch?v=xxQ0mzZ8UvA
[The CIFAR-10 dataset]: https://www.cs.toronto.edu/~kriz/cifar.html
[pip]: (https://pip.pypa.io/en/stable/)
[these instructions]: https://docs.anaconda.com/anaconda/install/update-version/
[Windows TensorFlow installation instructions]: https://www.tensorflow.org/install/pip#windows-native_1
[MacOS]: https://www.tensorflow.org/install/pip#macos_1
[Linux]: https://www.tensorflow.org/install/pip#linux_1
[NVIDIA GPU driver]: https://www.nvidia.com/Download/index.aspx
