---
title: "Setup - CPU"
---

This lesson is designed for Software Carpentry users who have completed [Plotting and Programming in Python] and want to jump straight into image classification. We recognize this jump is quite large and have done our best to provide the content and code to perform these types of analyses.

It uses the [Miniforge] software installer and conda-forge package repository to download the required Python packages, including the Spyder IDE. 

These instructions are for setting up tensorflow in a **CPU** only environment.

## Software Setup

::::::::::::::::::::::::::::::::::::: challenge
## Install Python Using Miniforge

[Python] is a popular language for research computing, and a frequent choice for machine learning as well. Installing multiple scientific packages into a single environment individually can be difficult, however, so we recommend using the [Miniforge] installer. 

:::::::::::::::::::::::: solution
### Windows

1. Open [https://conda-forge.org/download/] with your web browser.

2. Download the latest release of the Miniforge for Windows Installer.

3. Double-click the executable and install Python 3 using _MOST_ of the default settings. The only exception is to check the **Make Anaconda the default Python** option.
::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::: solution
### MacOS

1. Open [https://conda-forge.org/download/] with your web browser.

1. Download the appropriate Miniforge installer for macOS.

1. Open a terminal window and navigate to the directory where the executable is downloaded (e.g., cd ~/Downloads).

1. Type `bash Miniforge3-` and then press `Tab` to autocomplete the full file name. The name of file you just downloaded should appear. Press `Enter` (or `Return` depending on your keyboard).

1. Install Python 3 using all of the defaults for installation.
::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::: solution 
### Linux

1. Open [https://conda-forge.org/download/] with your web browser.

2. Download the appropriate Miniforge installer for Linux.

3. Open a terminal window and navigate to the directory where the executable is downloaded (e.g., cd ~/Downloads).

4. Type `bash Miniforge3-` and then press `Tab` to autocomplete the full file name. The name of file you just downloaded should appear. Press `Enter` (or `Return` depending on your keyboard).

5. Install Python 3 using all of the defaults for installation.
	
:::::::::::::::::::::::::::::::::: 
::::::::::::::::::::::::::::::::::::::::::::::::

## Download workshop files

Download the [data, files, and scripts].

Create a project folder on your local device for this course and move the downloaded materials to the project folder. Extract all of the download contents here. 

Check your folder and file structures looks like this:

![](fig/00_download_zip_folder_files.png){alt='Screenshot of the contents of the intro-image-classification-cnn.zip folder.'}

:::::::::::::::::::::::::::::::::::::: callout

In Spyder, when you execute a script in its entirety (Run File F5), the working directory will automatically be set to the directory that contains that script file. 

For this lesson, after launching Spyder, make sure to set the working directory to the 'scripts' folder included in the download. This will help to ensure that all of the scripts we provide run as intended.

::::::::::::::::::::::::::::::::::::::::::::::

## Install the required packages

::::::::::::::::::::::::::::::::::::::::: challenge
## Using Conda to install packages

[Conda] is an open-source package and environment manager that runs on Windows, macOS and Linux.

If you installed Miniforge successfully, Conda should already be available in your system.

:::::::::::::::::::::::: solution
## Windows

From the Start menu, search for "Miniforge Prompt" using the Windows search function (Windows Logo Key) and a terminal window will open with the title `Miniforge Prompt`: 

![](fig/00_miniforge_prompt_search.png){alt='Screenshot of the Miniforge Prompt application'}

![](fig/00_miniforge_prompt_window.png){alt='Screenshot of the terminal window that opens when you launch the Miniforge Prompt application'}

Note the prompt notation inside the terminal window. The name inside the parentheses refers to which conda environment you are working in. `base` is the name given to the default environment included with every conda distribution.

To create a new environment, the command starts with the conda keywords `conda env create` followed by a name for the new environment and the package(s) to install.

To make things easier for this lesson, use the `cnn_workshop_environment.yml` file inside the `files` folder of the [workshop files](#download-workshop-files).

```code
(base) C:\Users\Lab> miniforge env create --file cnn_workshop_environment.yml
```

If the yml is not in your current directory, you can specify the full path to the file, eg:

```code
(base) C:\Users\Lab> conda env create --file C:\Users\Lab\intro-image-classification-cnn\files\cnn_workshop_environment.yml
```

Be patient because it might take a while (15-20 min) for conda to work out all of the dependencies.

After the environment is created we tell Conda to use the new environment with the keywords `conda activate` followed by the name of the environment:

```code
(base) C:\Users\Lab> conda activate cnn_workshop
(cnn_workshop) C:\Users\Lab>
```

You will know you are in the right environment because the prompt changes from `(base)` to `(cnn_workshop)`.

:::::::::::::::::::::::::::::::::
:::::::::::::::::::::::: solution
## MacOS
Start a terminal session:

![](fig/00_starting-terminal-on-a-mac.png){alt='Screenshot of starting a new terminal session on a Mac.'}

Note the prompt notation inside the terminal window. The name inside the parentheses refers to which conda environment you are working in. `base` is the name given to the default environment included with every conda distribution.

To create a new environment, the command starts with the conda keywords `conda env create` followed by a name for the new environment and the package(s) to install.

To make things easier for this lesson, use the `cnn_workshop_MACOS_environment.yml` file inside the `files` folder of the [workshop files](#download-workshop-files).

```code
(base) Mac$ conda env create --file cnn_workshop_MACOS_environment.yml
```

If the yml is not in your current directory, you can specify the full path to the file, eg:

```code
(base) Mac$ conda env create --file intro-image-classification-cnn/files/cnn_workshop_MACOS_environment.yml
```

![](fig/00_mac-create-cnn-workshop-conda-enviroment.png){alt='Screenshot of create conda enviroment on a Mac.'}

Be patient because it might take a while (15-20 min) for conda to work out all of the dependencies.  

![](fig/00_mac-creating-cnn-workshop-conda-enviroment-1.png){alt='Screenshot of creating conda enviroment on a Mac.'}

![](fig/00_mac-creating-cnn-workshop-conda-enviroment-1.png){alt='Screenshot of creating conda enviroment on a Mac.'}

If it installs correctly, you should get the following output on the screen.

```code
done
#
# To activate this environment, use
#
#     $ conda activate cnn_workshop
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

![](fig/00_mac-cnn-workshop-conda-enviroment-created.png){alt='Screenshot of conda enviroment that has been created on a Mac.'}

After the environment is created we tell Conda to use the new environment with the keywords `conda activate` followed by the name of the environment:

```code
(base) Mac:cnn-workshop ace$ conda activate cnn_workshop
(cnn_workshop) Mac:cnn-workshop ace$ 
```

You will know you are in the right environment because the prompt changes from `(base)` to `(cnn_workshop)`.

![](fig/00_mac-activate-conda-enviroment.png){alt='Screenshot of activate conda enviroment on a Mac.'}

## Macos - Silicon (M1 and M2) - This sections needs to be tested and confirmed

After creating the MACOS environment with the yml, M1 and M2 computers (Apple Silicon) require an additional package to be installed.

TODO check if this is because of the integrated GPU and if so, can Intel version also use the graphics card and how?

```code
(cnn_workshop_macos) C:\Users\Lab> pip install tensorflow-metal
```

:::::::::::::::::::::::::::::::::
:::::::::::::::::::::::: solution
## Linux

Start a terminal session:

Note the prompt notation inside the terminal window. The name inside the parentheses refers to which conda environment you are working in. `base` is the name given to the default environment included with every conda distribution.

To create a new environment, the command starts with the conda keywords `conda env create` followed by a name for the new environment and the package(s) to install.

To make things easier for this lesson, use the `cnn_workshop_MACOS_environment.yml` file inside the `files` folder of the [workshop files](#download-workshop-files).

```code
$ conda env create --file cnn_workshop_environment.yml
```

If the yml is not in your current directory, you can specify the full path to the file, eg:

```code
$ conda env create --file intro-image-classification-cnn/files/cnn_workshop_environment.yml
```
Be patient because it might take a while (15-20 min) for conda to work out all of the dependencies.  

***Note, in the screen shots below, I haven't initilised conda, so I had to give it the path to where Miniforge was installed.***

![](fig/00_linux-create-conda-enviroment-using-yml-file-1.png){alt='Screenshot of creating conda enviroment on a Linux.'}

![](fig/00_linux-create-conda-enviroment-using-yml-file-2.png){alt='Screenshot of creating conda enviroment on a Linux.'}

![](fig/00_linux-create-conda-enviroment-using-yml-file-3.png){alt='Screenshot of creating conda enviroment on a Linux.'}

![](fig/00_linux-create-conda-enviroment-using-yml-file-4.png){alt='Screenshot of creating conda enviroment on a Linux.'}

![](fig/00_linux-create-conda-enviroment-using-yml-file-5.png){alt='Screenshot of creating conda enviroment on a Linux.'}

If it installs correctly, you should get the following output on the screen.

```code
done
#
# To activate this environment, use
#
#     $ conda activate cnn_workshop
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

***Note you will need to initialise conda to be able to activate the conda enviroment.***

Additonally, you will need to install scikeras and tensorflow within the conda environment to get everything working.  

To do this, issue the following commands:

```code
$ conda activate cnn_workshop

(base) $ pip install scikeras tensorflow
```

:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::

## Start Spyder

We teach this lesson using Python in [Spyder] (Scientific Python Development Environment), a free integrated development environment (IDE) and offers code development, interactive testing, debugging, and introspection tools.

To start Spyder, type the command `spyder`, making sure you are still in the workshop environment:

```
(cnn_workshop) C:\Users\Lab> spyder
```

### Spyder IDE components

![](fig/00_spyder_ide_layout.png){alt='Screenshot of the Spyder IDE annotated with boxes and labels for the Editor; Help, Variable Explorer, Plots, Files; and IPython Console areas'}

::::::::::::::::::::::::::::::::::::::::: callout
If you are using Linux, you might need to install qt5-qtbase and qt5-qtbase-gui for spyder to work.

On the Rocky 8 Distro, this can be done by issuing the command (as root):

```code
yum install qt5-qtbase qt5-qtbase-gui
```
:::::::::::::::::::::::::::::::::::::::::::::::::

## Check your setup

To check that all packages installed correctly, go to the interactive `IPython Console` in Spyder (lower right hand side panel) and type in the following commands:

```python
import matplotlib
print('matplotlib version: ', matplotlib.__version__)

import numpy
print('numpy version: ', numpy.__version__)

import pandas
print('pandas version: ', pandas.__version__)

import seaborn
print('seaborn version: ', seaborn.__version__)

import sklearn
print('sklearn version: ', sklearn.__version__)

import scikeras
print('scikeras version: ', scikeras.__version__)

import tensorflow
print('Tensorflow version: ', tensorflow.__version__)
```

Your package versions may vary from the screenshot below but this is similar to what your output will look like. The important thing is that there are no errors.

![](fig/00_package_check_output.png){alt='Screenshot of the IPython Console in Spyder wtih list of package versions and no error messages.'}

## Set the working directory

To set the working directory in Spyder, click on the folder icon (upper right hand side of the toolbar) and navigate to `.../intro-image-classification-cnn/scripts` where '...' is your project folder. 

Verify you are in the right place by selecting the `Files` pane just below the toolbar and check its contents.

![](fig/00_spyder_workingdir_scripts.png){alt='Screenshot of the contents of the scripts folder in the Files tab of the Spyder window after using the folder icon to set the working directory.'}

## Get the data

This lesson uses a subset of the CIFAR-10 image dataset. The data is included in the [workshop files](#download-workshop-files).

<!-- Collect your link references at the bottom of your document -->
[Plotting and Programming in Python]: https://swcarpentry.github.io/python-novice-gapminder/
[Miniforge]: https://conda-forge.org/download/
[Conda]: https://docs.conda.io/projects/conda/en/latest/
[Python]: https://python.org
[Anaconda]: https://www.anaconda.com/products/individual
[Windows - Video tutorial]: https://www.youtube.com/watch?v=xxQ0mzZ8UvA
[Mac OS X - Video tutorial]: https://www.youtube.com/watch?v=TcSAln46u9U
[these instructions]: https://docs.anaconda.com/anaconda/install/update-version/
[pip]: https://pip.pypa.io/en/stable/
[Spyder]: https://www.spyder-ide.org/
[data, files, and scripts]: https://drive.google.com/file/d/1SpcusVYomhukFKWuUcK7LwF7RtrKB8Z_/view?usp=drive_link


