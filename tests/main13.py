import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from utils import plot as nmaplt
import os
import pickle

from sklearn.model_selection import train_test_split
from cnn.preprocessing import preprocess
import cnn.utils
from cnn.create_model import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

"""
This script performs gridsearch CV to find the best parameters of a 
Convolutional neural network
"""


if __name__ == "__main__":
	np.random.seed(42)

	# CNN hyperparameters
	batch_size = 64
	epochs = 100
	input_shape = [750, 1]
	print_summary = False

	# dir
	dirs = dict()
	dirs['save'] = 'results/'

	if not os.path.exists(dirs['save']):
		os.makedirs(dirs['save'])
	if not os.path.exists(dirs['save'] + 'weights'):
		os.makedirs(dirs['save'] + 'weights')


	# specify behavior to make a prediction for
	behaviors = ["angvturns", "vmoves", "vymoves"]
	behavior_par = behaviors[1]

	# load the data from MATLAB .mat file
	stim, response = io.load_behavior('../datasets/behavior/control_behavior.mat', 30., 55., behavior_par, 50)

	# preprocess for the CNN to work. This is a VERY important step!
	stim, response = preprocess(stim, response)

	# Get zero-padded vectorized time series of dimension [samples, timesteps]
	expt = Experiment(0.02, 25, stim=stim, response=response)
	expt.registerContinuous('stim')

	# initialize design spec with one trial
	trial_inds = list(range(response.shape[1]))
	dspec = DesignSpec(expt, trial_inds)

	dspec.addRegressorContinuous()
	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	# split into training and test set
	x = X.reshape((X.shape[0], X.shape[1], 1))

	model = KerasRegressor(build_fn=load_model, epochs=100, batch_size=64, verbose=1)

	# define the grid search parameters
	weight_constraint = [1, 2, 3, 4, 5]
	dropout_rate = [0, .1, .2, .3, .4, .5]
	neurons = [1, 2, 4, 8, 16, 32, 64]
	kernel_size = [250, 450, 550, 650, 749]
	#param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
	param_grid = dict(weight_constraint=weight_constraint, dropout_rate=dropout_rate)

	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

	grid_result = grid.fit(x, y)

	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))