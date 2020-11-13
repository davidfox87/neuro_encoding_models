import numpy as np
from scipy.linalg import toeplitz, hankel
from scipy import stats, signal
from basisFactory.bases import Basis, RaisedCosine
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler



class Regressor:
	def __int__(self, basis_):
		name = None
		params = []
		basis_ = None
		self.edim_ = 0

	def duration(self, *args, **kwargs):
		pass

	@property
	def edim(self):
		return self.edim_

	def conv_basis(self, s, bases):
		'''
		Computes the convolutions of X with the selected basis functions
		:param X: has to be an nt, nx ndarray!
		:param bases:
		:return:
		'''
		slen = len(s)
		tb, nkt = bases.shape

		Xstim = np.zeros((slen, nkt))
		# convolve the stim with each column of the basis matrix
		for i in range(nkt):
			Xstim[:, i] = self.sameconv(s, bases[:, i])
		return Xstim

	def sameconv(self, x, f):
		nt = len(x)
		f = np.flipud(f)
		#a = np.concatenate((np.zeros(len(f) - 1), x), axis=None)
		res = np.convolve(x, f, mode='full')
		return res[:nt]


	def spikefilt(self, sps, bases):
		# convolve the spike train with each basis function one at a time
		spklen = len(sps)
		tb, nkt = bases.shape

		Xsp = np.zeros((spklen, nkt))

		# convolve the spikes with each column of the basis matrix
		for i in range(nkt):
			Xsp[:, i] = self.sameconvspike(sps, bases[:, i])
		return Xsp

	def sameconvspike(self, x, f):
		# (B is flipped as in standard convolution).
		nt = len(x)
		#f = np.flipud(f)
		# x = np.concatenate((np.zeros(len(f) - 1), x), axis=None)
		res = np.convolve(x, f, mode='full')  # if x is big, then np.convolve will do convolution in the frequency domain
		return res[:nt]

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
	def __init__(self, name, bins_before, bins_after=0, basis=None):
		self.name = name
		self.params = [name + "_time"]
		self.basis = basis

		if self.basis:
			self.edim_ = basis.nbases
		else:
			self.bins_after = bins_after
			self.bins_before = bins_before
			self.edim_ = self.bins_before + self.bins_after

	def duration(self, **kwargs):
		return self.edim_

	def matrix(self, params, n_bins, _times, _bintimes, **kwargs):
		time = params[self.name + "_time"]  # time is the time the regressor appears during the trial
		# Xdsgn = np.zeros((n_bins, self.bins_before))
		# index = next(i for i in range(0, len(_times)) if _bintimes[i] <= time and _bintimes[i + 1] > time)

		stim = params[self.name + "_val"]

		if self.basis:
			paddedstim2 = np.hstack((np.zeros(1), stim))
			#print("convolving padded stimulus with raised cosine basis functions")
			# convolve stimulus with bases functions
			Xdsgn2 = self.conv_basis(paddedstim2, self.basis.B)
			Xdsgn2 = Xdsgn2[:-1, :]
		else:
			paddedstim2 = np.hstack((np.zeros(self.bins_before - 1), stim))
			Xdsgn2 = hankel(paddedstim2[:(-self.bins_before + 1)], stim[(-self.bins_before):])

		return Xdsgn2


class RegressorSphist(Regressor):
	def __init__(self, name, bins_before, bins_after=0, basis=None):
		# self.name = name
		# self.bins_after = bins_after
		# self.bins_before = bins_before
		# self.params = [name + "_time"]
		#
		# # set super class parameters: every regressor may have a basis (or not)
		# # and associated with it is a dimension
		# self.basis = basis
		# self.edim_ = basis.nbases

		self.name = name
		self.params = [name + "_time"]
		self.basis = basis

		if self.basis:
			self.edim_ = basis.nbases
		else:
			self.bins_after = bins_after
			self.bins_before = bins_before
			self.edim_ = self.bins_before + self.bins_after

	def duration(self, **kwargs):
		return self.bins_before + self.bins_after

	def matrix(self, params, n_bins, _times, _bintimes, **kwargs):
		time = params[self.name + "_time"]  # time is the time the regressor appears during the trial
		stim = params[self.name + "_val"]

		if self.basis:
			#print("convolving padded stimulus with raised cosine basis functions")
			# convolve stimulus with bases functions
			Xdsgn2 = self.spikefilt(stim, self.basis.B)
		else:
			paddedstim2 = np.hstack((np.zeros(self.bins_before), stim[:-1])) # everything except the current spike at this time step
			Xdsgn2 = hankel(paddedstim2[:-self.bins_before + 1], paddedstim2[(-self.bins_before):])

		return Xdsgn2



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
		# n_regressor_timepoints = sum([r.duration() for r in self._regressors])
		# return np.zeros((0, int(n_regressor_timepoints)))  # this initializes number of column using nfilt
		edim = sum([r.edim for r in self._regressors])
		return np.zeros((0, int(edim)))  # this initializes number of column using nfilt

	def n_bins(self):
		return int((self._maxtime - self._mintime) / self._dt)

	def add_regressor(self, regressor):
		self._regressors.append(regressor)

	def build_matrix(self, params=dict(), trial_end=None):
		M = np.zeros((self.n_bins(), 0))  # initialize a matrix with nbins rows and intially 0 columns
		for r in self._regressors:

			Mr = r.matrix(params=params, n_bins=self.n_bins(), _times=self._times, _bintimes=self._bintimes, \
						  trial_end=trial_end)

			M = np.concatenate([M, Mr], axis=1)

		return M

	def get_regressor_from_output(self, output):
		'''

		:param name:
		:param output:
		:return: dictionary containing filter for each regressor
		'''

		d = {}
		start_index = 1 # start from 1 to ignore bias
		for r in self._regressors:
			name = r.name

			out = output[start_index:start_index + r.edim]

			if r.basis:
				y = (r.basis.B * out.T).sum(axis=1) 		# sum across the weights on each basis vector

				if r.name == 'stim':
					x = np.asarray(range(-len(y), 0))
				else:
					x = np.asarray(range(0, len(y)))
				d[name] = (x, y)
				start_index += r.edim
			else:
				y = output[start_index:start_index + r.bins_before]
				x = np.asarray(range(-r.bins_before, r.bins_after)) * self._dt

		return d

	def get_regressor_from_dm(self, name, output):
		start_index = 0
		for r in self._regressors:
			if r.name == name:
				break
			start_index += r.duration(n_bins=self.n_bins())
		r_length = r.duration(n_bins=self.n_bins())
		dm = output[:, start_index:start_index + r_length]

		return dm


class Experiment:
	"""
	class to hold
	"""
	def __init__(self, dtSp, duration, stim=None, sptimes=None, response=None):
		self._regressortype = {}
		self._sptimes = sptimes # if we dealing with behavior, this will be none
		self._stim = stim
		self.duration = duration
		self.response = response # response contains the response variable, which can be either spike times or behavior trace
		# to get the appropriate regressor, we would call dspec.expt.trial[regressor.name]
		# for this to work, regressor must have the name of either 'sptrain' or 'stim' as we access via a
		# dictionary
		self.trial = {}
		self._dtSp = dtSp

	def register_spike_train(self, label):
		# we initialize Experiment object with sptrain and stim
		# registering adds internal/external regressors to a dictionary to be processed
		# not every analysis may use all available regressors
		self.trial['sptrain'] = self.response
		self._regressortype['sptrain'] = label # check this label, probably sptrain needs to be the value not the key

	def registerContinuous(self, label):
		self.trial['stim'] = self._stim
		self._regressortype['stim'] = label

	@property
	def regressortype(self):
		return self._regressortype

	@property
	def dtSp(self):
		return self._dtSp

	@property
	def stim(self):
		return self._stim

	@property
	def sptimes(self):
		return self._sptimes


class DesignSpec:
	def __init__(self, expt: Experiment, trialinds: list):
		self.expt = expt
		self._trialinds = trialinds

		self.dt_ = expt.dtSp
		self._ntfilt = int(2.0/self.expt.dtSp)
		self._ntsphist = 100

		self.regressors = []

	def addRegressorSpTrain(self, basis=None):
		# first make basis to represent the spike history filter
		# basis = RaisedCosine(100, 5, 1, 'sphist')
		#
		# basis.makeNonlinearRaisedCosPostSpike(self.expt.dtSp, [.001, 1], .5)
		# r = RegressorSphist(self.expt.regressortype['sptrain'], self._ntsphist, basis=basis)
		# self.regressors.append(r)
		if basis:
			r = RegressorSphist(self.expt.regressortype['sptrain'], self._ntsphist, basis=basis)
		else:
			r = RegressorSphist(self.expt.regressortype['sptrain'], self._ntsphist)
		self.regressors.append(r)

	def addRegressorContinuous(self, basis=None):
		if basis:
			r = RegressorContinuous(self.expt.regressortype['stim'], self._ntfilt, basis=basis)
		else:
			r = RegressorContinuous(self.expt.regressortype['stim'], self._ntfilt)

		self.regressors.append(r)

	def compileDesignMatrixFromTrialIndices(self, bias=0):
		expt = self.expt
		dt = self.dt_

		# need to fix so i can use dt = 1
		#totalT = np.ceil(expt.duration/ expt.dtSp)
		totalT = expt.duration

		dm = DesignMatrix(dt, 0, totalT)
		for k in self.regressors:
			dm.add_regressor(k)

		Xfull = dm.empty_matrix()
		Yfull = np.asarray([])

		for tr in tqdm(self._trialinds):
			# nT = np.ceil(expt.duration / expt.dtSp)

			# build param dict to pass to dm.buildMatrix(), which contains time that regressor cares about
			# and the regressor values themselves
			d = {}
			for kregressor in self.regressors:

				# gets the stimulus based on the regressor name, the regressor name
				# must match the field in the experiment object
				name = kregressor.name
				stim = self.expt.trial[name][:, tr]

				# print('forming design matrix from trial indices')
				# binned_spikes = dm.bin_spikes(self.expt.trial['sptrain'][tr])
				d[name + '_time'] = 0  # this time could be used to fetch that part of the stimulus but not really used right now
				d[name + '_val'] = stim

			X = dm.build_matrix(d)

			Xfull = np.concatenate([Xfull, X], axis=0)
			Yfull = np.concatenate([Yfull, self.expt.response[:, tr]])



		# if ridge regression and cross-validation, then we need to add intercept column
		if bias:
			Xfull = np.column_stack([np.ones_like(Yfull), Xfull])
		# scaler = StandardScaler()
		# Xfull = scaler.fit_transform(Xfull)
		return dm, Xfull, Yfull



	@property
	def stim(self):
		return self._stim

	@property
	def ntfilt(self):
		return self._ntfilt

	@property
	def ntsphist(self):
		return self._ntsphist

	@property
	def dt(self):
		return self.dt_


