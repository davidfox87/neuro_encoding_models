import utils.read as io
import numpy as np
import os
from cnn.preprocessing import preprocess
from cnn.create_model import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


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


	behaviors = ["angvturns", "vmoves", "vymoves"]
	behavior_par = behaviors[2]

	# load the data from MATLAB .mat file
	stim, response = io.load_behavior('../datasets/behavior/control_behavior.mat', 30., 55., behavior_par, 50)
	response = response.mean(axis=1)  # work on the fly-average
	stim = stim[:, 0]
	# preprocess for the CNN to work. This is a VERY important step!
	stim_train, stim_test, resp_train, resp_test = preprocess(stim, response, input_shape)


	model = KerasRegressor(build_fn=load_model, epochs=100, batch_size=64, verbose=1)

	# define the grid search parameters
	weight_constraint = [1, 2, 3, 4, 5]
	dropout_rate = [0, .1, .2, .3, .4, .5]
	neurons = [16, 32, 64]

	param_grid = dict(neurons=neurons)
	#param_grid = dict(kernel_size=kernel_size)
	# for a single time series we want to test our model on time points in the future
	# https: // scikit - learn.org / stable / modules / cross_validation.html
	tscv = TimeSeriesSplit(n_splits=5)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=tscv, scoring='neg_mean_squared_error')

	grid_result = grid.fit(stim_train, resp_train)

	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))