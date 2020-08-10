import numpy as np
from scipy import sparse
from scipy.linalg import toeplitz, hankel
from numpy import linalg as LA
from scipy import stats
from scipy import signal

class Regressor:
	def __int__(self):
		name = None
		params = []

	def duration(self, *args, **kwargs):
		pass


class RegressorPoint(Regressor):
	def __init__(self, name, bins_after, bins_before=0):
		self.name = name
		self.bins_after = bins_after
		self.bins_before = bins_before
		self.params = [name + "_time"]

	def duration(self, **kwargs):
		return self.bins_after + self.bins_before

	def matrix(self, params, n_bins, _times, _bintimes, **kwargs):
		time = params[self.name + "_time"]
		M = np.zeros((n_bins, self.bins_before))

		# get the index of the first bin greater than time
		index = next(i for i in range(0, len(_times)) if _bintimes[i] <= time and _bintimes[i + 1] > time)

		M[(range(index - self.bins_before, index),
		   range(0, self.bins_before))] = 1

		return np.fliplr(M)


class RegressorContinuous(Regressor):
	def __init__(self, name, bins_before, bins_after=0):
		self.name = name
		self.bins_after = bins_after
		self.bins_before = bins_before
		self.params = [name + "_time"]

	def duration(self, **kwargs):
		return self.bins_before + self.bins_after

	def matrix(self, params, n_bins, _times, _bintimes, **kwargs):
		time = params[self.name + "_time"]  # time is the time the regressor appears during the trial
		# Xdsgn = np.zeros((n_bins, self.bins_before))
		# index = next(i for i in range(0, len(_times)) if _bintimes[i] <= time and _bintimes[i + 1] > time)

		stim = params[self.name + "_val"]
		#padded_stim = np.concatenate([np.zeros(self.bins_before), stim])
		# n1 = stim.shape[0]+self.bins_before - 1
		# Xdsgn = np.fliplr(toeplitz(padded_stim[range(self.bins_before, n1 + 1)],
		#		padded_stim[range(self.bins_before - 1, -1, -1)]))

		#paddedstim2 = np.concatenate((np.zeros(self.bins_before - 1), stim[range(0, n_bins - self.bins_before + 1)]))
		#Xdsgn2 = hankel(paddedstim2, stim[range(n_bins - self.bins_before, n_bins)])



		padded_stim = np.concatenate([np.zeros(self.bins_before - 1), stim])

		# Construct a matrix where each row has the d frames of
		# the stimulus proceeding and including timepoint t
		T = len(stim)  # Total number of timepoints
		X = np.zeros((T, self.bins_before))
		for t in range(T):
			X[t] = padded_stim[t:t + self.bins_before]

		return X
		#return Xdsgn2


class RegressorSphist(Regressor):
	def __init__(self, name, bins_before, bins_after=0):
		self.name = name
		self.bins_after = bins_after
		self.bins_before = bins_before
		self.params = [name + "_time"]

	# for spike history, at time point ti it is ntfilt -1 to not include the current spike
	def duration(self, **kwargs):
		return self.bins_before + self.bins_after

	def matrix(self, params, n_bins, _times, _bintimes, **kwargs):
		time = params[self.name + "_time"]  # time is the time the regressor appears during the trial

		stim = params[self.name + "_val"]

		paddedstim2 = np.concatenate((np.zeros(self.bins_before), stim[range(0, n_bins-1)])) # everything except the current spike at this time step
		Xdsgn2 = hankel(paddedstim2[range(0, len(paddedstim2)-self.bins_before + 1)], paddedstim2[range(len(paddedstim2) - self.bins_before, len(paddedstim2))])


		return Xdsgn2


binfun = lambda t, dtSp: (t == 5) + int(t // dtSp)

class DesignMatrix:
	def __init__(self, dt=0.001, mintime=-1000, maxtime=1000):
		self._dt = dt
		self._mintime = mintime  # in sec
		self._maxtime = maxtime  # in sec
		self._regressors = []
		self._times = np.arange(mintime, maxtime + dt / 2, dt) + dt / 2
		self._bintimes = np.append(self._times, self._times[-1] + dt / 2)

	def bin_spikes(self, spikes):
		"""
		convert spike times into spike counts
		:param spikes: indices of spikes
		:return: binned spike count
		"""
		return np.histogram(spikes, self._times)[0]

	def empty_matrix(self):
		n_regressor_timepoints = sum([r.duration() for r in self._regressors])
		return np.zeros((0, int(n_regressor_timepoints)))  # this initializes number of column using nfilt

	def n_bins(self):
		return int((self._maxtime - self._mintime) / self._dt)

	def add_regressor(self, regressor):
		self._regressors.append(regressor)

	def build_matrix(self, params=dict(), trial_end=None):
		M = np.zeros((self.n_bins(), 0))  # initialize a matrix with nbins rows and intially 0 columns
		for r in self._regressors:
			# for each regressor (independent variable)
			# construct a design matrix

			# stack each regressor horizontally

			Mr = r.matrix(params=params, n_bins=self.n_bins(), _times=self._times, _bintimes=self._bintimes, \
						  trial_end=trial_end)
			M = np.concatenate([M, Mr], axis=1)

		return M

	def get_regressor_from_output(self, name, output):
		start_index = 0
		for r in self._regressors:
			if r.name == name:
				break
			start_index += r.duration(n_bins=self.n_bins())
		r_length = r.duration(n_bins=self.n_bins())
		y = output[start_index:start_index + r_length]
		# This should be generalized
		x = np.asarray(range(-r.bins_before, r.bins_after)) * self._dt

		return (x, y)

	def get_regressor_from_dm(self, name, output):
		start_index = 0
		for r in self._regressors:
			if r.name == name:
				break
			start_index += r.duration(n_bins=self.n_bins())
		r_length = r.duration(n_bins=self.n_bins())
		dm = output[:, start_index:start_index + r_length]

		return dm

class GLM:
	"""
	% Compute response of glm to stimulus Stim.

	% Dynamics:  Filters the Stimulus with glmprs.k, passes this through a
	% nonlinearity to obtain the point-process conditional intensity.  Add a
	% post-spike current to the linear input after every spike.
	%
	% Input:
	%   glmprs - struct with GLM params, has fields 'k', 'h','dc' for params
	%              and 'dtStim', 'dtSp' for discrete time bin size for stimulus
	%              and spike train (in s).
	%     dm - stimulus matrix, with time running vertically and each
	%              column corresponding to a different pixel / regressor.
	"""
	def __init__(self, dm, dtStim, dtSp):
		"""

		:param dm:
		:param dtStim: bin size for sampling of stimulus
		:param dtSp:
		"""
		self._dm = dm
		self._Y = None
		self._k = None
		self._h = None
		self._dtStim = dtStim
		self._dtSp = dtSp

	def set_k(self, k):
		self._k = k

	def get_filter(self):
		return self.k


	def add_bias_column(self):
		'''
		Add a column of ones as the first column to estimate the bias (DC term)
		:param X:
		:return:
		'''
		nr = self._dm.shape[0]
		self._dm = np.column_stack([np.ones_like(self._Y), self._dm])

# add trial would append to dm design matrix

# get binned spike train
