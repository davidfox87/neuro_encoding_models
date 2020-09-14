# neuro and behavior encoding models

This package implements a variety of supervised learning algorithms for building encoding models. Supervised learning is where you have input variables (X) and an output variable (y) and you use an algorithm to learn the mapping function from the input to the output.
Here we are learning the mapping from various olfactory stimuli to neural spike trains or fly olfactory navigation behavior data. 

Basically we can learn what temporal features of inputs are being encoded by an output. Once a model has been fit, it can be used to predict how a biological neuron or a fly will behave to stimuli they have not seen before.


**Description:**
-
This package computes estimates for the parameters of encoding models. These parameters are the stimulus temporal filters and nonlinear functions.

Encoding models characterize the relationship between external or internal regressors and a set of
measured outputs (neural spike trains or behavior). Usually linear functions of inputs are nonlinearly transformed to produce estimates of outputs.
The goals is to understand what features of the input are being encoded by the outputs
 
Currently, the package allows to fit three types of encoding models:
1. a Poisson GLM using maximum liklihood fitting to specifically model spike train responses to stimuli
2. an L2-regularized linear regression model (Ridge) - with the option of using raised cosine basis functions to fit 1D filters - using the Scipy framework. The regularization coefficient of ridge regression can be optimized by using 5-fold cross-validation.
3. Convolutional neural networks (CNN) using Keras and Tensorflow framework. The CNN consists of an input layer, several hidden layers (convolutional layer, pooling layer, or fully-connected layer) and the output layer. Grid search with 5-fold cross-validation (from the Scikit-learn framework) is used to tune the hyperparameters of the Keras CNN model. 

**Installation:**
-
To install the modules, all you have to do is the following:
   1) download dist/NeuroGLM-1.0.tar.gz
   2) unpack it
   3) from the NeuroGLM-1.0 directory run python setup.py install

**System Requirements**
-
1. Python 3.8 or  above
2. Numpy version 1.19 or above
3. Scipy version 1.5.1 or above
4. Matplotlib version 3.3 or above
5. Scikit-learn version 0.23.1 or above
6. Keras 2.4.3
7. Tensorflow 2.3.0

**Usage:**
-
Examine test scripts in sub-directory tests/ to see simple scripts illustrating how to fit stimulus temporal and 
spike history filters to spike train data.

Anyone can use this package within the google colab environment. Below you will find some tutorials to get you started. After you have played around with the notebooks, you can install the package following the steps in th installation section of this document.

|   | View | Run |
| - | --- | ---- |
| Tutorial 1: fitting a Poisson GLM to spike train data | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neuroGLM/master/HowTo_fit_filters.ipynb?flush_cache=true) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Foxy1987/neuroGLM/blob/master/HowTo_fit_filters.ipynb) |
| Tutorial 2: using the GLM to predict responses to novel stimuli | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neuroGLM/master/HowTo_fit_filters.ipynb?flush_cache=true) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Foxy1987/neuroGLM/blob/master/HowTo_fit_filters.ipynb) |
| Tutorial 3: Finding stim filter weights for behavior using ridge-regression and cross-validation | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neuroGLM/master/HowTo_fit_filters.ipynb?flush_cache=true) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Foxy1987/neuroGLM/blob/master/HowTo_fit_filters.ipynb) |
| Tutorial 4: denoising stimulus filters by fitting in a vector space spanned by a basis of raised cosines | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neuroGLM/master/basis_function_fitting.ipynb?flush_cache=true) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Foxy1987/neuroGLM/blob/master/basis_function_fitting.ipynb) |
| Tutorial 5: Compute a histogram-based estimate of the nonlinear function that scales filter outputs to make a predictions of real data | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neuroGLM/master/HowTo_fit_filters.ipynb?flush_cache=true) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Foxy1987/neuroGLM/blob/master/HowTo_fit_filters.ipynb) |
| Tutorial 6: Find stim filter weights for behavior and make predictions to novel stimuli using convolutional neural networks with Keras and Tensorflow | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neuroGLM/master/HowTo_fit_filters.ipynb?flush_cache=true) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Foxy1987/neuroGLM/blob/master/HowTo_fit_filters.ipynb) |
| Tutorial 7: Perform K-fold cross-validation to tune the hyperparameters of a Keras convolutional neural network | [![View the notebook](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/neuroGLM/master/HowTo_fit_filters.ipynb?flush_cache=true) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Foxy1987/neuroGLM/blob/master/HowTo_fit_filters.ipynb) |

**Credits:**
All by me, David Fox

**License:**








