import numpy as np
from scipy.linalg import hankel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def timeseries_from_dataset(stim, bins_before):
	paddedstim2 = np.hstack((np.zeros(bins_before - 1), stim.squeeze()))
	return hankel(paddedstim2[:(-bins_before + 1)], paddedstim2[(-bins_before):]) # needs to be paddedstim because keras model expects inputs to be the same size

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

def preprocess_stim(stim, input_shape=None):
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
	stim = stim.reshape(-1, 1)
	scaler = StandardScaler()
	stim = scaler.fit_transform(stim)

	x = timeseries_from_dataset(stim, input_shape[0])

	# keras expects the time series as [samples, timesteps, features]
	return x.reshape((x.shape[0], x.shape[1], 1))



def preprocess(stim, response, input_shape=None):
	stim_train, stim_test, resp_train, resp_test = train_test_split(stim, response,
																	test_size=0.2, shuffle=False, random_state=42)
	stim_train, stim_test = preprocess_stim(stim_train, input_shape=input_shape), preprocess_stim(stim_test, input_shape=input_shape)
	resp_train, resp_test = preprocess_resp(resp_train), preprocess_resp(resp_test)
	return stim_train, stim_test, resp_train, resp_test

	# stim = preprocess_stim(stim, input_shape=input_shape)
	# resp = preprocess_resp(response)
	# return stim, resp
