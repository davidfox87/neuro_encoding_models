import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge
import utils.read as io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from glmtools.fit import fit_nlin_hist1d
import pickle
from utils.read import load_ankita_data
from sklearn.model_selection import KFold
from glmtools.make_xdsgn import Experiment, DesignSpec
from sklearn.metrics import r2_score, mean_squared_error
from joblib import Parallel, delayed
"""
Refer to Pillow et al., 2005 for more info.
This script fits model parameters, K, h, dc of a 
Poisson Generalized linear model.

"""

def make_dspec(stim, response, dt, inds):
	# make an Experiment object
	expt = Experiment(dt, len(stim)*dt, stim=stim, response=response)

	# register continuous regressor
	expt.registerContinuous('stim')

	# initialize design spec with number of entries
	return DesignSpec(expt, trialinds=inds)


if __name__ == "__main__":

	dt = 1. / 25.
	stims, responses = load_ankita_data('../datasets/ankita/ConstantAir_glom1.mat')

	inds = np.arange(stims.shape[1])
	train_inds, test_inds = train_test_split(inds, test_size=0.2, random_state=42)
	X_train, X_test = stims[:, train_inds], stims[:, test_inds]
	y_train, y_test = responses[:, train_inds], responses[:, test_inds]

	# now get folds
	# now make the k-fold
	ntrials = X_train.shape[1]
	inds = np.arange(ntrials)  # trial indices used to make splits
	np.random.shuffle(inds)

	folds_xtrain = []
	folds_xtest = []
	folds_ytrain = []
	folds_ytest = []

	kf = KFold(n_splits=5)
	kf.get_n_splits(inds)

	scaler = StandardScaler()
	scaler2 = MinMaxScaler([0, 1])
	X_train = scaler.fit_transform(X_train)
	y_train = scaler2.fit_transform(y_train)

	for train_index, val_index in kf.split(inds):
		# each fold will consist of a design matrix that will have concatenated trials for that fold
		# e.g. the first fold will train on trials 1, 2, 3 and test on 0
		# the second fold will train on 0, 2, 3 and test on 1
		print("TRAIN:", train_index, "VALIDATION:", val_index)

		train_dspec = make_dspec(X_train[:, train_index], y_train[:, train_index], dt, np.arange(len(train_index)))
		train_dspec.addRegressorContinuous()

		# use the inds to take a slice of sps and make a test design matrix
		test_dspec = make_dspec(X_train[:, val_index], y_train[:, val_index], dt, np.arange(len(val_index)))
		test_dspec.addRegressorContinuous()

		dm, Xtrain, ytrain = train_dspec.compileDesignMatrixFromTrialIndices(bias=1)
		dm, Xtest, ytest = test_dspec.compileDesignMatrixFromTrialIndices(bias=1)

		# Xtrain = scaler.fit_transform(Xtrain)
		# Xtest = scaler.transform(Xtest)		# transform Xtest to the same scale as Xtrain

		folds_xtrain.append(Xtrain)
		folds_xtest.append(Xtest)
		folds_ytrain.append(ytrain)
		folds_ytest.append(ytest)


	print('done making folds')


	def ridgefitCV(train, test, parameter):
		msetest_fold = 0
		for train, test in zip(train, test):
			Xtrain, ytrain = train
			Xtest, ytest = test

			model = Ridge(alpha=parameter).fit(Xtrain, ytrain)

			msetest_fold += mean_squared_error(ytest, model.predict(Xtest))

		# take the average mse across folds for this alpha
		return msetest_fold / len(train)


	lamvals = np.logspace(-5, 5, num=20, base=2)


	def _run_search(evaluate_candidates):
		"""Search all candidates in param_grid"""
		evaluate_candidates(lamvals)


	parallel = Parallel(n_jobs=10, verbose=10)

	all_out = []

	with parallel:

		def evaluate_candidates(candidate_params):
			out = parallel(delayed(ridgefitCV)(train=zip(folds_xtrain, folds_ytrain),
											   test=zip(folds_xtest, folds_ytest),
											   parameter=param)
						   for param in candidate_params)

			all_out.extend(out)

	_run_search(evaluate_candidates)

	print(all_out)

	imin = np.argmin(all_out)
	print("best ridge param is {}".format(lamvals[imin]))
	plt.plot(all_out)


	lamvals[imin]

	# run ridge with the best alpha on the average response

	dspec = make_dspec(X_train, y_train, dt, np.arange(X_train.shape[1]))
	dspec.addRegressorContinuous()
	dm, X, y = train_dspec.compileDesignMatrixFromTrialIndices(bias=1)

	model = Ridge(alpha=lamvals[imin]).fit(X, y)

	w = model.coef_
	plt.plot(w)


	d = dm.get_regressor_from_output(w)

	# # convert back to the original basis to get nkt filter weights.
	# # Parameters are returned in a dict
	dc = w[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dt


	X_test = scaler.fit_transform(X_test)
	y_test = scaler2.fit_transform(y_test)

	idx = 10
	input = X_test[:, idx]
	output = y_test[:, idx]
	xx, fnlin, rawfilteroutput = fit_nlin_hist1d(input, output, k, dt, 20)



	Fs = 25
	t = np.arange(0, len(X_test[:, idx])) / Fs
	plt.plot(t, output, 'k', linewidth=2)
	plt.plot(t, fnlin(rawfilteroutput), 'c', linewidth=2)