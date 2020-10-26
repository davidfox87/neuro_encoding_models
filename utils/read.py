import numpy as np
from scipy import sparse
import scipy.io as io
from sklearn.impute import KNNImputer

def load_spk_times(stim_, response, start, finish):
	'''
	load stimulus and spike raster with only the relevant stimulus part of the trial
	:param stim: column data
	:param response: 3 columns: col, row, data
	:return:
	'''
	stim = np.genfromtxt(stim_, delimiter='\t')
	nt = len(stim)
	spikes = io.loadmat(response)
	# spikes = np.genfromtxt(response, delimiter='\t')

	binfun = lambda t: int(t / 0.001) - (t == start)
	stim = stim[range(binfun(start)+1, binfun(finish))]

	spTimes = spikes['spTimes'].squeeze()

	# chop off pre and post
	sps = [list(filter(lambda num: (num >= start and num < finish), spTimes_.squeeze())) for spTimes_ in spTimes]

	# subtract 5 from every element in sps so every spTime is relative to 0 and not 5
	sps = [list(map(lambda x: x - start, sps_)) for sps_ in sps]

	return stim, sps


def load_behavior(data, start, finish, param, fs):
	dat = io.loadmat(data)
	dat_ = dat['flyData'][param]
	response = dat_[0][0]
	response = response[int(start * fs):int(finish * fs), :]

	imputer = KNNImputer(n_neighbors=2, weights="uniform") # replace nans with an average of the last 2 data points
	response = imputer.fit_transform(response)

	if fs != 50:
		stim = dat['flyData']['PN']
		stim = stim[0][0]
		stim = stim[int(5 * fs):int(30 * fs), :]
	else:
		stim = dat['flyData']['stim']
		stim = stim[0][0]
		stim = stim[int(start * fs):int(finish * fs), :]

	return stim, response

def load_mean_psth(file, cell):
	'''

	:param file: name of the .mat file containing stim and resp
	:param cell: name of the struct ['control_orn', 'control_pn']
	:return: stim and resp traces
	'''
	dat = io.loadmat(file)

	fs = 100
	stim = dat[cell]['stim'][0][0]

	resp = dat[cell]['response'][0][0]
	resp = resp[int(5 * fs):int(30 * fs)]

	return stim, resp

def read_orn_fr(filename, type='pulses'):
	dat = io.loadmat(filename)

	if type == 'pulses':
		responses = np.zeros((30*1000, 7))
		for i in range(7):
			responses[:, i] = dat['orn_responses'][i][0].squeeze()
	elif type == 'chirps':
		responses = np.zeros((35 * 1000, 2))
		responses[:, 0] = dat['orn_responses'][7][0].squeeze()
		responses[:, 1] = dat['orn_responses'][8][0].squeeze()

	return responses

def bin_spikes(x, size, dt):
	# chop off pre and post
	sps = (filter(lambda num: (num >= 4 and num < size*dt), x))
	sps = list(sps)

	# subtract 5 from every element in sps so every spTime is relative to 0 and not 5
	sps = [list(map(lambda x: x - 4, sps_)) for sps_ in sps]

	return np.histogram(sps, np.arange(0.5, size-(4*dt) + 1) * dt - dt)[0]

def load_concatenatedlpvm_spike_data(filename):
	dat = io.loadmat(filename)

	dt = 0.001

	binfun = lambda t: int(t / 0.001) - (t == 4)
	lpvm = dat['data']['lpvm'][0][0][0]
	sptimes = dat['data']['sptimes'][0][0][0]
	input = []
	output = []
	for i, tr in enumerate(lpvm):
		tr = tr[range(binfun(4) + 1, binfun(len(tr)*dt))]

		input = np.concatenate((input, tr), axis=None)

		sps = bin_spikes(sptimes[i], len(tr), dt)

		sps = np.array(sps).T
		output = np.concatenate((output, sps), axis=None)

	return input, output
