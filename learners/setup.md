---
title: "Setup - CPU"
---

These instructions are for setting up tensorflow in a **CPU** only environment.

## Installing Anaconda

[Python] is a popular language for scientific computing, and a frequent choice for machine learning as well. Installing all of its scientific packages individually can be a bit difficult, however, so we recommend the installer [Anaconda] which includes most (but not all) of the software you will need.

Regardless of how you choose to install it, please make sure you install Python version 3.x (e.g., 3.9 is fine).

Also, please set up your python environment at least a day in advance of the workshop. If you encounter problems with the installation procedure, ask your workshop organizers via e-mail for assistance so you are ready to go as soon as the workshop begins.

### [Windows - Video tutorial]

1. Open [https://www.anaconda.com/products/distribution] with your web browser.

2. Download the Python 3 installer for Windows.

3. Double-click the executable and install Python 3 using _MOST_ of the default settings. The only exception is to check the **Make Anaconda the default Python** option.

### [Mac OS X - Video tutorial]

1. Open [https://www.anaconda.com/products/distribution] with your web browser.

2. Download the Python 3 installer for Mac/OS X.

3. Install Python 3 using all of the defaults for installation.

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

## Installing the required packages

::::::::::::::::::::::::::::::::::::::::: callout
[Conda] is the package management system associated with [Anaconda] and runs on Windows, macOS and Linux.

Conda should already be available in your system once you installed Anaconda successfully. Conda thus works regardless of the operating system. Make sure you have an up-to-date version of Conda running. See [these instructions] for updating Conda if required.
:::::::::::::::::::::::::::::::::::::::::::::::::

To create a conda environment called `cnn_workshop` with the required packages, open a terminal and type the command:

```code
conda create --name cnn_workshop python spyder seaborn scikit-learn pandas
```

Activate the newly created environment:

```code
conda activate cnn_workshop
```

Install tensorflow using [pip] (python's package manager):

```code
pip install tensorflow
```

Note that modern versions of Tensorflow make Keras available as a module.


### Troubleshooting for Windows

It is possible that Windows users will run into version conflicts. If you are on Windows and get errors running the command, you can try installing the packages using pip within a conda environment:

```code
conda create -n cnn_workshop python spyder
conda activate cnn_workshop
pip install tensorflow>=2.5 seaborn scikit-learn pandas
```

[pip] is the package management system for Python software packages.
It is integrated into your local Python installation and runs regardless of your operating system too.

### Troubleshooting for Macs with Apple silicon chip

Newer Macs (from 2020 onwards) often have a different kind of chip, manufactured by Apple instead of Intel. This can lead to problems installing Tensorflow .
If you get errors running the installation command or conda hangs endlessly,
you can try installing Tensorflow for Mac with pip:

```conda
pip install tensorflow-macos
```

## Starting Spyder

We will teach using Python in [Spyder] (Scientific Python Development Environment) , a free integrated development environment (IDE) written in Python that comes with Anaconda.Editing, interactive testing, debugging, and introspection tools are all included in Spyder. If you installed Python using Anaconda, Spyder should already be on your system. If you did not use Anaconda, use the Python package manager pip (see the [Spyder website] for details.)

To start Spyder, open a terminal and type the command:

```conda
spyder
```

## Check your setup

To check whether all packages installed correctly, start a jupyter notebook in jupyter lab as explained above. Run the following lines of code:

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

- For Keras and Tensorflow, the minimum version is 2.2.4
- For sklearn, the minimum version is 0.22.

## Fallback option: cloud environment

TODO

## Downloading the required datasets

Download the [The CIFAR-10 dataset].

TODO cifar comes with keras; need to work out if we want to provide ahead of time - might need to be munged

<!-- Collect your link references at the bottom of your document -->

[Conda]: https://docs.conda.io/projects/conda/en/latest/
[Anaconda]: https://www.anaconda.com/products/individual
[anaconda-distribution]: https://www.anaconda.com/products/distribution
[Spyder]: https://www.spyder-ide.org/
[Spyder website]: https://docs.spyder-ide.org/current/installation.html
[python]: https://python.org
[Mac OS X - Video tutorial]: https://www.youtube.com/watch?v=TcSAln46u9U
[Windows - Video tutorial]: https://www.youtube.com/watch?v=xxQ0mzZ8UvA
[The CIFAR-10 dataset]: https://www.cs.toronto.edu/~kriz/cifar.html
[pip]: (https://pip.pypa.io/en/stable/)
[these instructions]: https://docs.anaconda.com/anaconda/install/update-version/
[Google colab]: https://colab.research.google.com/

