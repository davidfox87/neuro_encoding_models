import numpy as np
from scipy.linalg import hankel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd


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


def timeseries_from_dataset(stim, bins_before):
	paddedstim2 = np.hstack((np.zeros(bins_before - 1), stim.squeeze()))
	return hankel(paddedstim2[:(-bins_before + 1)], paddedstim2[(
																	-bins_before):])  # needs to be paddedstim because keras model expects inputs to be the same size


def preprocess_resp(resp):
	"""
    preprocess the responses

    """
	# we could also scale with scikit-learn

	# scaler = MinMaxScaler(feature_range=[0, 1])
	# response = scaler.fit_transform(response)
	# trial-average
	# each fly is a trial-average
	# (n_t, n_flies)

	# resp = resp.reshape((-1, 1))
	#
	# # scaling to [0,1]
	# for i in range(resp.shape[1]):
	# 	_resp = resp[:, i]
	# 	if np.max(_resp) == np.min(_resp):
	# 		resp[:, i] = np.zeros(len(_resp))
	# 	else:
	# 		resp[:, i] = (_resp - np.min(_resp)) / (np.max(_resp) - np.min(_resp))
	#
	# # reshape for Keras
	# return resp.reshape((resp.shape[0], resp.shape[1], 1))

	# scaling to [0,1]
	resp = resp.reshape((-1, 1))

	# scaling to [0,1]
	for i in range(resp.shape[1]):
		_resp = resp[:, i]
		if np.max(_resp) == np.min(_resp):
			resp[:, i] = np.zeros(len(_resp))
		else:
			resp[:, i] = (_resp - np.min(_resp)) / (np.max(_resp) - np.min(_resp))

	return resp


def preprocess_stim(stim_train, stim_test, input_shape=None):
	"""
	preprocess the stim to have zero-mean and unit-variance
	"""

	# scaler = StandardScaler()
	# stim = scaler.fit_transform(stim)
	#
	# x = timeseries_from_dataset(stim, input_shape[0])
	#
	# # reshape for Keras in [samples, timesteps, features]
	# return x.reshape((x.shape[0], x.shape[1], 1))

	# scaler = StandardScaler()
	# stim = scaler.fit_transform(stim)
	# return stim
	# stim_train = stim_train.reshape(-1, 1)
	#scalar = StandardScaler()
	scalar = MinMaxScaler()
	scalar.fit(stim_train)
	stim_train = scalar.transform(stim_train)
	stim_test = scalar.transform(stim_test)

	return stim_train, stim_test


def preprocess(stim, response, input_shape=None):

	resp = preprocess_resp(response)
	stim_train, stim_test, resp_train, resp_test = train_test_split(stim, resp,
																	test_size=0.1, shuffle=False, random_state=42)

	scaled_train, scaled_test = preprocess_stim(stim_train, stim_test, input_shape=input_shape)

	stim_train = series_to_supervised(scaled_train.reshape(-1, 1), n_in=input_shape[0], n_out=1)
	stim_test = series_to_supervised(scaled_test.reshape(-1, 1), n_in=input_shape[0], n_out=1)

	stim_train = stim_train.values
	stim_test = stim_test.values

	# keras expects the time series as [samples, timesteps, features]
	stim_train = stim_train.reshape((stim_train.shape[0], stim_train.shape[1], 1))
	stim_test = stim_test.reshape((stim_test.shape[0], stim_test.shape[1], 1))

	return stim_train, stim_test, resp_train, resp_test, scaled_train, scaled_test


def preprocess_groups(stim, response, input_shape):
	for i in range(stim.shape[1]):
		stim = series_to_supervised(stim[:, i].reshape(-1, 1), n_in=input_shape[0] - 1, n_out=1)
