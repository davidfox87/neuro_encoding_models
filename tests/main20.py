
import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from scipy.optimize import minimize
from glmtools.fit import neg_log_lik
import pickle
from glmtools.model import GLM
from basisFactory.bases import RaisedCosine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.sparse import spdiags
from scipy.linalg import block_diag
from sklearn.model_selection import train_test_split
from glmtools.fit import neg_log_lik, mapfit_GLM
from models.synapse import get_psth
from utils.read import load_concatenatedlpvm_spike_data


def make_bases(dur, peaks, stim_nbases, stretch, dt=0.02):
	fs = 1. / dt
	first_peak, last_peak = peaks

	nkt = int(dur * fs)  # length of filter in number of samples
	# stim_nbases = number of vectors that are raised cosines
	stim_basis = RaisedCosine(100, stim_nbases, 1, 'stim')

	# arguments for makeNonlinearRaisedCosStim are
	# dt sample interval
	# peaks = [position of first center, position of last center],
	# stretch = spacing of basis centers (higher value = more linear meaning spread out,
	# lower value = nonlinear more centers near 0)
	# nkt = length of filter. If length of filters exceeds nkt we trim the filters,
	# otherwise we pad with 0's
	# stim_basis.makeNonlinearRaisedCosStim(.1, [10, round(nkt/1.7)], stretch, nkt)  # first and last peak positions,
	stim_basis.makeNonlinearRaisedCosStim(.1, [first_peak / dt, last_peak / dt], stretch, nkt)
	spike_basis = RaisedCosine(100, 7, 1, 'sphist')
	spike_basis.makeNonlinearRaisedCosPostSpike(0.1, [.1, 10], 1, .01)

	return stim_basis, spike_basis

def make_dspec(stim, response, dt, inds):
	# make an Experiment object
	expt = Experiment(dt, len(stim)*dt, stim=stim, response=response)

	# register continuous regressor
	expt.registerContinuous('stim')

	# register spike regressor
	expt.register_spike_train('sptrain')

	# initialize design spec with number of entries
	return DesignSpec(expt, trialinds=inds)


"""
Refer to Pillow et al., 2005 for more info.
This script fits model parameters, K, h, dc of a 
Poisson Generalized linear model.

"""
if __name__ == "__main__":
	# something wrong with cell 6, 8 sometimes the fitting doesn't hone in on the solution...run again
	# check 4
	dt = 0.001
	cell_idx = 1
	stim, sps = load_concatenatedlpvm_spike_data('../datasets/vm_spiking/lpvm_spikes_PN' + str(cell_idx) + '.mat')

	inds = np.random.choice(stim.shape[1], 15, replace=False)
	stim_train = stim[:, inds[:10]]
	sps_train = sps[:, inds[:10]]

	stim_test = stim[:, inds[10:]]
	sps_test = sps[:, inds[10:]]

	# standardize each train and test. Fit to train transform both train and val
	scaler_ = StandardScaler()
	stim_train = scaler_.fit_transform(stim_train)
	stim_test = scaler_.fit_transform(stim_test)

	# stim_basis, spike_basis = make_bases(0.4, [0.005, 0.2], 10, 1, dt=0.001)
	stim_basis, spike_basis = make_bases(0.2, [0.003, 0.1], 10, 1, dt=0.001)
	fig, ax = plt.subplots(1, 2, figsize=[20, 5])
	ax[0].plot(np.arange(-len(stim_basis.B), 0)*dt, stim_basis.B)
	ax[1].plot(spike_basis.B)

	# now make the k-fold
	ntrials = sps_train.shape[1]
	inds = np.arange(ntrials)  # trial indices used to make splits
	np.random.shuffle(inds)

	folds_xtrain = []
	folds_xtest = []
	folds_ytrain = []
	folds_ytest = []

	from sklearn.model_selection import KFold

	kf = KFold(n_splits=5)
	kf.get_n_splits(inds)

	for train_index, test_index in kf.split(inds):
		# each fold will consist of a design matrix that will have concatenated trials for that fold
		# e.g. the first fold will train on trials 1, 2, 3 and test on 0
		# the second fold will train on 0, 2, 3 and test on 1
		print("TRAIN:", train_index, "TEST:", test_index)

		# use the inds to take a slice of sps and make a train design matrix
		train_dspec = make_dspec(stim_train[:, train_index], sps_train[:, train_index], dt, np.arange(len(train_index)))
		train_dspec.addRegressorContinuous(basis=stim_basis)
		train_dspec.addRegressorSpTrain(basis=spike_basis)

		# use the inds to take a slice of sps and make a test design matrix
		test_dspec = make_dspec(stim_train[:, test_index], sps_train[:, test_index], dt, np.arange(len(test_index)))
		test_dspec.addRegressorContinuous(basis=stim_basis)
		test_dspec.addRegressorSpTrain(basis=spike_basis)

		dm, X_train, y_train = train_dspec.compileDesignMatrixFromTrialIndices()
		dm, X_test, y_test = test_dspec.compileDesignMatrixFromTrialIndices()

		# X_train = scaler_.fit_transform(X_train)
		# X_test = scaler_.transform(X_test)		# transform Xtest to the same scale as Xtrain

		folds_xtrain.append(X_train)
		folds_xtest.append(X_test)
		folds_ytrain.append(y_train)
		folds_ytest.append(y_test)


	# Now perform MAP fitting with ridge regression
	neg_log_func = lambda prs, train, test: neg_log_lik(prs, stim_basis.nbases, train, test, 1)

	# regularization
	nlam = 10
	lamvals = np.logspace(-5, 5, num=nlam, base=2)

	Imat = np.eye(stim_basis.nbases + spike_basis.nbases + 1)
	Imat[0, 0] = 0

	prs = np.random.normal(0, 0.01, stim_basis.nbases + spike_basis.nbases + 1)
	res = minimize(neg_log_lik, prs, args=(stim_basis.nbases, folds_xtrain[0], folds_ytrain[0], 1),
				   options={'disp': True, 'maxiter': 500})

	wmap = res['x']

	d = dm.get_regressor_from_output(wmap)

	dc = wmap[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dt
	h = d['sptrain'][1]
	ht = d['sptrain'][0] * dt

	figure, ax = plt.subplots(2, 2)
	ax[0, 0].plot(stim_basis.B)
	ax[0, 1].plot(spike_basis.B)
	ax[1, 0].plot(kt, k, '-ok')
	ax[1, 1].plot(ht, h, '-ok')
	ax[1, 0].axhline(0, color=".2", linestyle="--", zorder=1)
	ax[1, 1].axhline(0, color=".2", linestyle="--", zorder=1)
	ax[1, 0].set_xlabel('Time before spike (s)')
	ax[1, 1].set_xlabel('Time after spike (s)')
	ax[1, 0].set_title('membrane potential Filter')
	ax[1, 1].set_title('post-spike filter Filter')

	figure.suptitle('PN' + str(cell_idx))
	plt.tight_layout()








	# do MAP fitting here
	from joblib import Parallel, delayed


	def _fit_and_score(train, test, wmap, parameter):
		negLTest = []
		for train, test in zip(train, test):
			Xtrain, ytrain = train
			Xtest, ytest = test

			Cinv = parameter * Imat

			wmap_ = mapfit_GLM(wmap, stim_basis.nbases, Xtrain, np.array(ytrain), Cinv, 1);

			negLTest.append(neg_log_func(wmap_, Xtest, ytest))

		return np.mean(negLTest)


	# from tqdm import tqdm


	def _run_search(evaluate_candidates):
		"""Search all candidates in param_grid"""
		evaluate_candidates(lamvals)


	parallel = Parallel(n_jobs=10, verbose=10)

	all_out = []

	with parallel:

		def evaluate_candidates(candidate_params):
			out = parallel(delayed(_fit_and_score)(train=zip(folds_xtrain, folds_ytrain),
												   test=zip(folds_xtest, folds_ytest),
												   wmap=wmap,
												   parameter=param)
						   for param in candidate_params)

			all_out.extend(out)

	_run_search(evaluate_candidates)

	print(all_out)

	# once we find the minimum lambda, fit on all the training data (not folds)
	imin = np.argmin(all_out)
	print("best ridge param is {}".format(lamvals[imin]))
	print(all_out)
	plt.plot(all_out, 'o-')

	# fit on entire dataset with this ridge penalty
	dspec = make_dspec(stim_train, sps_train, dt, np.arange(stim_train.shape[1]))
	dspec.addRegressorContinuous(basis=stim_basis)
	dspec.addRegressorSpTrain(basis=spike_basis)

	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()
	wmap = mapfit_GLM(wmap, stim_basis.nbases, X, y, lamvals[imin] * Imat, 1)

	d = dm.get_regressor_from_output(wmap)
	dc = wmap[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dt
	h = d['sptrain'][1]
	ht = d['sptrain'][0] * dt

	figure, ax = plt.subplots(1, 2, figsize=[20, 10])
	ax[0].plot(kt, k)
	ax[1].plot(ht, h)
	ax[0].axhline(0, color=".2", linestyle="--", zorder=1)
	ax[1].axhline(0, color=".2", linestyle="--", zorder=1)
	ax[0].set_xlabel('Time before spike(s)')
	ax[1].set_xlabel('Time after spike(s)')
	ax[0].set_title('Stimulus Filter')
	ax[1].set_title('post-spike Filter')
	ax[1].set_xlim(0.005, 0.2)
	ax[0].set_xlim(-0.1, 0)


	# and then test on the test set
	# 1) get a design matrix for the test set
	# 2) then get the negative log-likelihood or MSE/R2
	# neg_log_func(wmap_, Xtest, ytest)


	#
	#
	# # output GLM parameters: k, h, dc
	# data = {'k': (kt, k),
	# 		'h': (ht, h),
	# 		'dc': dc}
	# #
	# output = open('../results/vm_to_spiking_filters/new_glmpars_vm_to_spiking_PN' + str(cell_idx) + '.pkl', 'wb')
	# #
	# # # pickle dictionary using protocol 0
	# pickle.dump(data, output)
	# #

	stim_idx = 25
	glm = GLM(dt, k, h, dc)
	stim_ = scaler_.fit_transform(stim[:, stim_idx].reshape(-1, 1))
	#
	nsims = 3
	glm_sims = np.zeros((sps.shape[0], nsims))
	glm_sptimes = np.zeros((sps.shape[0], nsims))
	for i in range(nsims):  # run 5 GLM simulations
		tsp, sps_, itot, istm = glm.simulate(stim_)
		glm_sptimes[:, i] = sps_ * np.arange(len(sps_)) * dt
		glm_sims[:, i] = sps_




	tsp_ = sps[:, stim_idx] * np.arange(len(sps)) * dt
	# #colors1 = ['C{}'.format(1) for i in range(3)]
	#
	fs = 1/dt
	t = np.arange(0, 80 * fs) / fs
	fig, ax = plt.subplots(3, 1, sharex=True)
	ax[1].eventplot(glm_sptimes.T, linewidth=0.5)
	ax[2].eventplot(tsp_.T, linewidth=0.5)
	ax[0].plot(t, stim_)
	ax[0].set_title('Vm')
	ax[1].set_title('GLM Spike raster')
	ax[2].set_title('Actual Spike raster')
	plt.show()
	# plt.plot(np.arange(len(tsp)) * dt, get_psth(tsp, 100, 1000))
	# plt.plot(psth)
	#
	# # compare with actual firing-rate
	#
	# #
	#
	# inds = range(int(70 * fs))
	# t = np.arange(0, 70 * fs) / fs
	# sptimes = resp_train[:int(70*fs)] * t
	#
	# fig, ax = plt.subplots(3, 1, sharex=True)
	# ax[0].plot(t, stim_train[inds])
	#
	# #ax[0].eventplot(sptimes.T, lineoffsets=0.7)
	#
	# ax[1].plot(t, resp_train[inds], linewidth=0.5)
	#
	# smooth_win = np.hanning(100) / np.hanning(100).sum()
	# psth_model = np.convolve(sps, smooth_win)
	# sigrange = np.arange(100 // 2, 100 // 2 + len(sps))
	# psth_model_ = psth_model[sigrange] * 1000
	# #ax[2].plot(np.arange(0, int(70*fs))/1000., psth_model_)
	# ax[2].plot(t, psth_model_, label='GLM')
	# #
	#
	# sps_exp = resp_train[inds, 0]
	# psth_exp = np.convolve(sps_exp, smooth_win, mode='full')
	# sigrange = np.arange(100 // 2, 100 // 2 + len(sps_exp))
	# psth_exp_ = psth_exp[sigrange] * 1000
	# # ax[2].plot(np.arange(0, int(70 * fs - 1/fs)) / 1000., psth_exp_)
	# ax[2].plot(t, psth_exp_, label='biological')

	# ax[2].legend()
