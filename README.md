# neuroGLM
Fitting and simulation of linear guassian and Poisson generalized linear model (GLM) for single neuron spike trains

**Description:**
computes estimates for the parameters of a linear Gaussian GLM spike train model.
The weights on the temporal filter are found using using ridge regression and cross-validation is used to find
 the best ridge penalty. Maximum-likelihood and MAP estimation methods can also estimate filter weights assuming an exponential 
 nonlinearity and poisson spike rate.
 
Parameters of GLMs consist of a stimulus temporal filter, a spike-history filter and a nonlinearity. 
A GLM model with a spike-history filter is a generalization of the "Linear-Nonlinear-Poisson" model that 
incorporates spike-history effects

**Installation:**
To be done

**Usage:**
Examine test scripts in sub-directory tests/ to see simple scripts illustrating how to simulate and fit the GLM to spike train data.
Also to come is a jupyter notebook with embedded code and plots

**Credits:**

**License:**








