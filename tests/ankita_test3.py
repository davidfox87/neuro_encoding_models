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
	train_inds, test_inds = train_test_split(inds, test_size=0.5, random_state=42)
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

	for train_index, val_index in kf.split(inds):
		# each fold will consist of a design matrix that will have concatenated trials for that fold
		# e.g. the first fold will train on trials 1, 2, 3 and test on 0
		# the second fold will train on 0, 2, 3 and test on 1
		print("TRAIN:", train_index, "VALIDATION:", val_index)

		train_dspec = make_dspec(X_train[:, train_index], X_train[:, train_index], dt, np.arange(len(train_index)))
		train_dspec.addRegressorContinuous()

		# use the inds to take a slice of sps and make a test design matrix
		test_dspec = make_dspec(X_train[:, val_index], X_train[:, val_index], dt, np.arange(len(val_index)))
		test_dspec.addRegressorContinuous()

		dm, X_train, y_train = train_dspec.compileDesignMatrixFromTrialIndices(bias=1)
		dm, X_test, y_test = test_dspec.compileDesignMatrixFromTrialIndices(bias=1)

		# X_train = scaler_.fit_transform(X_train)
		# X_test = scaler_.transform(X_test)		# transform Xtest to the same scale as Xtrain

		folds_xtrain.append(X_train)
		folds_xtest.append(X_test)
		folds_ytrain.append(y_train)
		folds_ytest.append(y_test)


	print('done making folds')


