import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

	resp = resp.reshape((-1, 1))

	# scaling to [0,1]
	for i in range(resp.shape[1]):
		_resp = resp[:, i]
		if np.max(_resp) == np.min(_resp):
			resp[:, i] = np.zeros(len(_resp))
		else:
			resp[:, i] = (_resp - np.min(_resp)) / (np.max(_resp) - np.min(_resp))

	response = resp.reshape((len(resp), 1))

	return resp

	#response = preprocess_resp(response)


def preprocess_stim(stim):
	"""
	preprocess the stim to have zero-mean and unit-variance
	"""
	scaler = StandardScaler()
	stim = scaler.fit_transform(stim)
	stim = stim[:, 0]
	return stim.reshape((len(stim), 1))


def preprocess(stim, response):
	stim, response = preprocess_stim(stim), preprocess_resp(response)
	return stim, response