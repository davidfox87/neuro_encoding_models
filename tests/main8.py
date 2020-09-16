import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from basisFactory.bases import Basis, RaisedCosine
from scipy.optimize import minimize
from glmtools.fit import neg_log_lik, ridgefitCV, fit_nlin_hist1d
import pickle
from glmtools.model import GLM
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from utils import plot as nmaplt

"""
This script uses no basis to represent the stimulus and finds the stimulus temporal filters for behavior using
ridge regression and cross-validation to find the best ridge penalty on the filter weights
"""
if __name__ == "__main__":

	behavior_par = "vymoves"
	fs = 100
	# load behavior from MATLAB .mat file
	stim, response = io.load_behavior('../datasets/behavior/control_behavior.mat', 30., 55., behavior_par, 50)
	#stim, response = io.load_behavior('../datasets/behavior/u13AKD_behavior.mat', 30., 55., behavior_par, fs)
	# stim, response = io.load_behavior('../datasets/behavior/control_PN_to_behavior.mat', 30., 55., behavior_par, fs)
	# stim, response = io.load_behavior('../datasets/behavior/u13AKD_PN_to_behavior.mat', 30., 55., behavior_par, fs)

	# make an Experiment object
	expt = Experiment(0.02, 25, stim=stim, response=response)
	#expt = Experiment(0.01, 25, stim=stim, response=response)

	# register continuous regressor
	expt.registerContinuous('stim')

	# initialize design spec with one trial
	trial_inds = list(range(response.shape[1]))
	dspec = DesignSpec(expt, trial_inds)

	dspec.addRegressorContinuous()

	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	# kfolds cross-validation
	trials = response.shape[1]

	folds_train = []
	folds_test = []
	#
	trial_indices = np.arange(trials)
	kf = KFold(n_splits=5, shuffle=True, random_state=1)
	for train_index, test_index in kf.split(trial_indices):
		print("Train:", train_index, "TEST:", test_index)
		inds_train, inds_test = trial_indices[train_index], trial_indices[test_index]

		# form a train design matrix using the first fold indices
		dspec = DesignSpec(expt, inds_train)
		dspec.addRegressorContinuous()
		_, X_train, y_train = dspec.compileDesignMatrixFromTrialIndices()
		folds_train.append((X_train, y_train))
		dspec = DesignSpec(expt, inds_test)
		dspec.addRegressorContinuous()
		_, X_test, y_test = dspec.compileDesignMatrixFromTrialIndices()
		folds_test.append((X_test, y_test))

	# logic
	# for each value of ridge lambda value,
	# 	calculate an average MSE across folds
	# choose lambda that minimizes average MSE across folds
	lamvals = np.logspace(0, 30, num=20, base=2)
	alpha_ = ridgefitCV(folds_train, folds_test, lamvals)

	# try excluding trials where fly doesn't do anything...there is one where it os mostly standing still
	# model = Ridge(alpha=13479003).fit(X, y)

	model = Ridge(alpha=alpha_).fit(X, y)

	# Get the coefs of the model fit using cross-validation
	w = model.coef_

	fig, ax1 = plt.subplots(1, 1)
	nmaplt.plot_spike_filter(ax1, w[1:], dspec.dt, linewidth=6)
	ax1.set_xlim(-2, 0)

	fs = 50
	stim_ = stim[:, 0]
	response_ = response.mean(axis=1)

	xx, fnlin, rawfilteroutput = fit_nlin_hist1d(stim_, response_, w[1:], dspec.dt, 100)


	fig = plt.figure()
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(212)

	nmaplt.plot_spike_filter(ax1, w[1:], dspec.dt, linewidth=6,color='c')
	ax1.set_xlim(-4, 0)

	ax2.plot(rawfilteroutput, response_, 'ok', markersize=2)
	ax2.plot(xx, fnlin(xx), 'c', linewidth=2)

	t = np.arange(0, 25 * fs) / fs
	ax3.plot(t, response_, 'k', linewidth=2)
	ax3.plot(t, fnlin(rawfilteroutput), 'c', linewidth=1)
	plt.tight_layout()
	plt.show()


	# output filter: k and nonlinearity: nlfun
	data = {'name': behavior_par,
			'k': w[1:],
			'nlfun': (xx, fnlin(xx))}

	# file_name = "../datasets/behavior/ridge_filters/" + behavior_par + "_filter.pkl"
	file_name = "../datasets/behavior/ridge_filters/" + behavior_par + "_filter_PN_to_behavior_control.pkl"
	# file_name = "../datasets/behavior/ridge_filters/" + behavior_par + "_filter_PN_to_behavior_u13AKD.pkl"

	output = open(file_name, 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)

