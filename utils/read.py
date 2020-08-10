import numpy as np
from scipy import sparse
import scipy.io as io

def load_data(sessionNum):
	'''
	load steinmetz data set. Requires downloading .npz in top-level directory

	:return: data dictionary associated with given animal
	'''

	alldat = np.load('steinmetz_NMA_part1.npz', allow_pickle=True)['dat']
	alldat = np.hstack((alldat, np.load('steinmetz_NMA_part2.npz', allow_pickle=True)['dat']))
	alldat = np.hstack((alldat, np.load('steinmetz_NMA_part3.npz', allow_pickle=True)['dat']))

	# session 11?
	dat = alldat[sessionNum]

	print(dat.keys())

	return dat

def load_spk_times(stim, response, start, finish):
	'''
	load stimulus and spike raster with only the relevant stimulus part of the trial
	:param stim: column data
	:param response: 3 columns: col, row, data
	:return:
	'''
	stim = np.genfromtxt(stim, delimiter='\t')
	nt = len(stim)
	spikes = io.loadmat(response)
	# spikes = np.genfromtxt(response, delimiter='\t')

	binfun = lambda t: (t == start) + int(t // 0.001)
	stim = stim[range(binfun(start), binfun(finish)+1)]

	spTimes = spikes['spTimes'].squeeze()

	# chop off pre and post
	sps = [list(filter(lambda num: (num > start and num <= finish), spTimes_.squeeze())) for spTimes_ in spTimes]

	# subtract 5 from every element in sps so every spTime is relative to 0 and not 5
	sps = [list(map(lambda x: x - start, sps_)) for sps_ in sps]

	return stim, sps

def load_spk_data_neuromatch():
	"""
	Load RGC data used in the neuromatch tutorial
	:return:
	"""
	data = io.loadmat('RGCdata.mat')  # loadmat is a function in scipy.io
	dt_stim = data['dtStim'].item()  # .item extracts a scalar value

	# Extract the stimulus intensity
	stim = data['Stim'].squeeze()  # .squeeze removes dimensions with 1 element

	# Extract the spike counts for one cell
	cellnum = 2
	spikes = data['SpCounts'][:, cellnum]

	# Don't use all of the timepoints in the dataset, for speed
	keep_timepoints = 20000
	stim = stim[:keep_timepoints]
	spikes = spikes[:keep_timepoints]

	return stim, spikes, dt_stim

