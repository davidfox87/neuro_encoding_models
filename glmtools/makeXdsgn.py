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

		paddedstim2 = np.concatenate((np.zeros(self.bins_before - 1), stim))
		Xdsgn2 = hankel(paddedstim2[:(len(paddedstim2) - self.bins_before + 1)],
						stim[(len(stim) - self.bins_before):])

		# padded_stim = np.concatenate([np.zeros(self.bins_before - 1), stim])

		# Construct a matrix where each row has the d frames of
		# # the stimulus proceeding and including timepoint t
		# T = len(stim)  # Total number of timepoints
		# X = np.zeros((T, self.bins_before))
		# for t in range(T):
		# 	X[t] = padded_stim[t:t + self.bins_before]
		#
		# return X
		return Xdsgn2


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

		paddedstim2 = np.hstack((np.zeros(self.bins_before), stim[:-1])) # everything except the current spike at this time step
		Xdsgn2 = hankel(paddedstim2[:-self.bins_before + 1],
						paddedstim2[(len(paddedstim2) - self.bins_before):])

		return Xdsgn2


class Experiment:
	"""
	class to hold
	"""
	def __init__(self, stim, sptimes, dtSp, dtStim):
		self._regressortype = {}
		self._sptimes = sptimes
		self._stim = stim
		self._dtSp = dtSp
		self._dtStim = dtStim

	def register_spike_train(self, label):
		self._regressortype[label] = 'spike history'

	def registerContinuous(self, label):
		self._regressortype[label] = 'continuous'

	def register_point(self, label):
		self._regressortype[label] = 'point'

	@property
	def regressortype(self):
		return self._regressortype

	@property
	def dtStim(self):
		return self._dtStim

	@property
	def dtSp(self):
		return self._dtSp

	@property
	def stim(self):
		return self._stim

	@property
	def sptimes(self):
		return self._sptimes


class DesignMatrix:
	def __init__(self, dt=0.001, mintime=-1000, maxtime=1000):
		self._dt = dt
		self._mintime = mintime  # in sec
		self._maxtime = maxtime  # in sec
		self._times = np.arange(mintime, maxtime - dt / 2, dt) + dt / 2
		self._bintimes = np.append(self._times - dt / 2, self._times[-1] + dt / 2)
		self._regressors = []

	def bin_spikes(self, spikes):
		"""
		convert spike times into spike counts
		:param spikes: indices of spikes
		:return: binned spike count
		"""
		return np.histogram(spikes, self._bintimes)[0]

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

#get binned spike train


class DesignSpec:
	def __init__(self, exp: Experiment, trialinds: list):
		self._exp = exp
		self._trialinds = trialinds

		assert(exp.dtStim > exp.dtSp, 'dtStim must be bigger than dtSp')
		self.sampfactor = exp.dtStim / exp.dtSp
		assert (self.sampfactor % 1 == 0, 'dtSp does not evenly divide dtStim: dtStim / dtSp must be an integer')
		# if we make the sampling more coarse, we will get more spikes
		self._stim = signal.resample(exp.stim, len(exp.stim) // int(self.sampfactor))  # 25 gives good results

		self.nt = len(self._stim)
		self.dt_ = exp.dtSp * self.sampfactor

		self._ntfilt = int(2000 / self.sampfactor)
		self._ntsphist = int(2000 / self.sampfactor)

	def compileDesignMatrixFromTrialIndices(self):
		exp_ = self._exp
		dt = self.dt_

		dm = DesignMatrix(dt, 0, self.nt*dt)

		for name in exp_.regressortype:
			if exp_.regressortype[name] == 'continuous':
				dm.add_regressor(RegressorContinuous(name, self._ntfilt))
			if exp_.regressortype[name] == 'spike history':
				dm.add_regressor(RegressorSphist(name, self._ntsphist))

		Xfull = dm.empty_matrix()
		Yfull = np.asarray([])

		for tr in self._trialinds:
			print('forming design matrix from trial indices')
			binned_spikes = dm.bin_spikes(exp_.sptimes[tr])

			# where does the actual stim come from?
			d = {}
			for i in exp_.regressortype:
				if exp_.regressortype[i] == 'spike history':
					d[i + '_time'] = 0	# this time could be used to fetch that part of the stimulus but not really used right now
					d[i + '_val'] = binned_spikes  # this needs to be generalized to any regressor val
				else:
					d[i + '_time'] = 0  # this time could be used to fetch that part of the stimulus but not really used right now
					d[i + '_val'] = self._stim


			X = dm.build_matrix(d)

			Xfull = np.concatenate([Xfull, X], axis=0)
			Yfull = np.concatenate([Yfull, binned_spikes])

		Xfull = stats.zscore(Xfull)
		Xfull = np.column_stack([np.ones_like(Yfull), Xfull])

		return Xfull, Yfull

	@property
	def stim(self):
		return self._stim

	@property
	def ntfilt(self):
		return self._ntfilt

	@property
	def ntsphist(self):
		return self._ntsphist


