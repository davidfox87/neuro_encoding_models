import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
import utils.read as io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from glmtools.fit import fit_nlin_hist1d
import pickle


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


	stim, response = io.load_mean_psth('../datasets/neural/control_stim_to_orn.mat', 'control_orn')
	fs = 100
	# consider splitting by 80/20 train/validation and then make a lot of timeseries splits > 10, then test the mse on test
	stim_train, stim_test, resp_train, resp_test = train_test_split(stim, response,
	 																test_size=0.001,
																	shuffle=False,
																	random_state=42)
	scaler = MinMaxScaler([0, 1])
	resp_train_scaled = scaler.fit_transform(resp_train.reshape(-1, 1))

	estimators = [('add_lags', InsertLags()),
				  ('scaler', StandardScaler()),
				  ('model', Ridge(fit_intercept=False))]

	pipe = Pipeline(steps=estimators)
	alphas = np.logspace(0, 20, num=10, base=2)

	param_grid = {
		'model__alpha': alphas,
		'add_lags__lag': [1*fs, 2*fs, 4*fs]
	}
	#
	tscv = TimeSeriesSplit(n_splits=10)
	search = GridSearchCV(pipe, param_grid=param_grid, cv=tscv, verbose=1,
						  scoring='neg_mean_squared_error', n_jobs=-1,
						  return_train_score=True)

	search.fit(stim_train.reshape(-1, 1), resp_train_scaled)

	means = search.cv_results_['mean_test_score']
	stds = search.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, search.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			  % (mean, std * 2, params))

	print("Best: %f using %s" % (search.best_score_, search.best_params_))

	res = [sub['add_lags__lag'] for sub in search.cv_results_['params']]
	# res2 = [sub['model__alpha'] for sub in search.cv_results_['params']]
	df = pd.DataFrame({'rmse': means, 'window_length': res})
	df2 = df.groupby(['window_length']).max()

	# plot the neg_mean_squared_error as a function of window length
	plt.figure()
	plt.plot(df2['rmse'])


	# stim_train_transformed = pipe.named_steps['add_lags'].transform(stim_train.reshape(-1, 1))
	# stim_test_transformed = pipe.named_steps['add_lags'].transform(stim_test.reshape(-1, 1))
	# plt.imshow(stim_train_transformed)

	estimators = [('add_lags', InsertLags(search.best_params_['add_lags__lag'])),
				  ('scaler', StandardScaler()),
				  ('model', Ridge(search.best_params_['model__alpha']))]
	pipe = Pipeline(steps=estimators)

	# scale the output
	pipe.fit(stim_train.reshape(-1, 1), resp_train_scaled)

	plt.figure()
	w = pipe.named_steps['model'].coef_[0]
	d = len(w[1:])
	t = np.arange(-d + 1, 1) * 0.02
	plt.plot(t, w[1:])
	plt.axhline(0, color=".2", linestyle="--", zorder=1)

	plt.figure()

	xx, fnlin, rawfilteroutput = fit_nlin_hist1d(stim_train, resp_train, w, 0.02, 100)
	plt.plot(fnlin(rawfilteroutput))

	plt.plot(resp_train)
	#
	# file_name = "../datasets/behavior/ridge_filters/" + behavior_par + "_filter.pkl"
	# data = {'name': behavior_par,
	#  		'k': w[1:],
	#  		'nlfun': (xx, fnlin(xx)),
	# 		'window_length': search.best_params_['add_lags__lag']}
	#
	# output = open(file_name, 'wb')
	#
	# # pickle dictionary using protocol 0
	# pickle.dump(data, output)
	#
	#
	# # repeat this on average control ORN, PN and unc13A KD ORN, PN
