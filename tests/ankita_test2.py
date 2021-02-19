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

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
	data = np.hstack(data).reshape(-1, 1)
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg = agg.fillna(0)
		# agg.dropna(inplace=True)
	return agg


# All you need in a Pipeline is any object with a fit and transform method
# that returns an array and takes the right positional arguments
# (array x, one-d optional array y)

class InsertLags(BaseEstimator, TransformerMixin):
	"""
    Automatically Insert Lags
    """

	def __init__(self, lag=1):
		self.lag = lag

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		X = series_to_supervised(X, n_in=self.lag)
		return X.values


if __name__ == "__main__":

	dt = 1. / 25.
	stims, responses = load_ankita_data('../datasets/ankita/ConstantAir_glom1.mat')

	# stims = pd.DataFrame(stims)
	# responses = pd.DataFrame(responses)

	inds = np.arange(stims.shape[1])
	train_inds, test_inds = train_test_split(inds, test_size=0.2, random_state=42)
	X_train, X_test = stims[:, train_inds], stims[:, test_inds]
	y_train, y_test = responses[:, train_inds], responses[:, test_inds]

	inds = np.arange(X_train)

	kf = KFold(n_splits=5)
	kf.get_n_splits(inds)
	scores = []
	cv = KFold(n_splits=5, random_state=42, shuffle=False)
	laggedFeatureTransform = InsertLags()

	for train_index, test_index in cv.split(inds):
		print("Train Index: ", train_index, "\n")
		print("Test Index: ", test_index)

		# best_svr.fit(X_train, y_train)
		# scores.append(best_svr.score(X_test, y_test))


	estimators = [('add_lags', InsertLags()),
				  ('scaler', StandardScaler()),
				  ('model', Ridge(fit_intercept=False))]

	pipe = Pipeline(steps=estimators)
	alphas = np.logspace(0, 20, num=10, base=2)

	fs = 25.
	dt = 1. / 25.
	param_grid = {
		'model__alpha': alphas,
		'add_lags__lag': [int(2 * fs)]
	}

	kf = KFold(n_splits=5)
	search = GridSearchCV(pipe, param_grid=param_grid, cv=kf, verbose=1,
						  scoring='neg_mean_squared_error', n_jobs=-1,
						  return_train_score=True)

	# X_trainTransformed = [series_to_supervised(X_train[:, i].reshape(-1, 1), n_in=2 * 25) for i in
	# 					  range(X_train.shape[1])]

	# scale the y_train between 0 and 1

	search.fit(X_train.T, y_train.T)

	means = search.cv_results_['mean_test_score']
	stds = search.cv_results_['std_test_score']

	for mean, std, params in zip(means, stds, search.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			  % (mean, std * 2, params))

	print("Best: %f using %s" % (search.best_score_, search.best_params_))



	# refit on entire training dataset
	estimators = [('scaler', StandardScaler()),
				  ('model', Ridge(search.best_params_['model__alpha']))]
	pipe = Pipeline(steps=estimators)

	# scale the output
	pipe.fit(X_trainTransformed.r, y_train)
