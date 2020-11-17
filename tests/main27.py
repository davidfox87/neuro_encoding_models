import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from scipy.optimize import minimize
from glmtools.model import GLM
from basisFactory.bases import RaisedCosine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from glmtools.fit import neg_log_lik, mapfit_GLM
from utils.read import load_spk_times2

"""
Refer to Pillow et al., 2005 for more info.
This script fits model parameters, K, h, dc of a 
Poisson Generalized linear model.

"""
def make_bases(dt=0.005):
	#dt = 0.001
	fs = 1. / dt
	nkt = int(2. * fs)
	stim_basis = RaisedCosine(100, 6, 1, 'stim')
	stim_basis.makeNonlinearRaisedCosStim(.1, [.1, round(nkt/1.2)], 10, nkt)  # first and last peak positions,

	spike_basis = RaisedCosine(100, 3, 1, 'sphist')
	spike_basis.makeNonlinearRaisedCosPostSpike(0.1, [.1, 10], 1, 0.01)

	return stim_basis, spike_basis

def make_dspec(stim, response, dt):
	# make an Experiment object
	expt = Experiment(dt, len(stim)*dt, stim=stim, response=response)

	# register continuous regressor
	expt.registerContinuous('stim')
	# register spike regressor

	expt.register_spike_train('sptrain')
	# initialize design spec with one trial
	return DesignSpec(expt, np.arange(stim.shape[1]))

if __name__ == "__main__":
	# something wrong with cell 6, 8 sometimes the fitting doesn't hone in on the solution...run again
	# check 4
	cell_idx = 1
	cell = "pn" + str(cell_idx)
	response = '../datasets/spTimesPNControl/{}SpTimes_reverseChirp.mat'.format(cell, '.txt')
	stim, sps, dt = load_spk_times2('../datasets/stim.txt', response, dt=0.001)

	scaler = MinMaxScaler()
	stim = scaler.fit_transform(stim)

	stim_basis, spike_basis = make_bases(dt=dt)
	# data should be imported as a matrix of size [nt, trials]

	#dt = 0.001

	# implement group K-fold Cross-val
	ntrials = sps.shape[1]
	inds = np.arange(ntrials)  # trial indices used to make splits

	folds_xtrain = []
	folds_xtest = []
	folds_ytrain = []
	folds_ytest = []

	from sklearn.model_selection import LeaveOneOut
	loo = LeaveOneOut()
	loo.get_n_splits(inds)
	for train_index, test_index in loo.split(inds):
		# each fold will consist of a design matrix that will have concatenated trials for that fold
		# e.g. the first fold will train on trials 1, 2, 3 and test on 0
		# the second fold will train on 0, 2, 3 and test on 1
		print("TRAIN:", train_index, "TEST:", test_index)

		# use the inds to take a slice of sps and make a train design matrix
		train_dspec = make_dspec(stim[:, train_index], sps[:, train_index], dt)
		train_dspec.addRegressorContinuous(basis=stim_basis)
		train_dspec.addRegressorSpTrain(basis=spike_basis)

		# use the inds to take a slice of sps and make a test design matrix
		test_dspec = make_dspec(stim[:, test_index], sps[:, test_index], dt)
		test_dspec.addRegressorContinuous(basis=stim_basis)
		test_dspec.addRegressorSpTrain(basis=spike_basis)

		dm, X_train, y_train = train_dspec.compileDesignMatrixFromTrialIndices()
		dm, X_test, y_test = test_dspec.compileDesignMatrixFromTrialIndices()

		# plt.imshow(X_train[:, :stim_basis.nbases] @ stim_basis.B.T, extent=[-12, 0, 0, 1250],
		# 		   interpolation='nearest', aspect='auto')
		# plt.imshow(X_train[:, stim_basis.nbases:] @ spike_basis.B.T,
		# 		   aspect='auto')
		folds_xtrain.append(X_train)
		folds_xtest.append(X_test)
		folds_ytrain.append(y_train)
		folds_ytest.append(y_test)

	print('hello')

	# in ridge regression, for each ridge penalty the average negLTest across folds will be saved. The ridge-penalty that will be
	# used is the minimum negative log liklihood (max logli) across folds

	# This is the preferred method for trial-based data because we keep the trials together, instead of concatenating and
	# potentially splitting data in the middle of a trial

	# Now perform MAP fitting with ridge regression
	neg_log_func = lambda prs, train, test: neg_log_lik(prs, stim_basis.nbases, train, test, 1)

	# regularization
	nlam = 10
	lamvals = np.logspace(-5, 5, num=nlam, base=2)

	Imat = np.eye(stim_basis.nbases + spike_basis.nbases + 1)
	Imat[0, 0] = 0

	# negLTrain = np.zeros((nlam, 1))
	negLTest = np.zeros((nlam, len(folds_xtrain)))

	print('====Doing grid search for ridge parameter====\n')
	prs = np.random.normal(0, 0.1, stim_basis.nbases + spike_basis.nbases + 1)
	res = minimize(neg_log_lik, prs, args=(stim_basis.nbases, folds_xtrain[0], folds_ytrain[0], 1),
				   options={'disp': True})

	wmap = res['x']
	d = dm.get_regressor_from_output(wmap)

	dc = wmap[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dt
	h = d['sptrain'][1]
	ht = d['sptrain'][0] * dt

	wmap_vec = np.tile(wmap, (len(folds_xtrain), 1)).T
	for jj in range(nlam):
		print('lambda={:.3f}'.format(lamvals[jj]))
		for kk in range(len(folds_xtrain)):
			wmap = wmap_vec[:, kk]

			Cinv = lamvals[jj] * Imat
			X_train, X_test = folds_xtrain[kk], folds_xtest[kk]
			y_train, y_test = folds_ytrain[kk], folds_ytest[kk]
			wmap_vec[:, kk] = mapfit_GLM(wmap, stim_basis.nbases, X_train, np.array(y_train), Cinv, 1)

			# test negative log liklihood for lambda val jj and fold kk
			negLTest[jj, kk] = neg_log_func(wmap, X_test, y_test)

	print("hello")

	# Select best ridge param and fit on all data (meaning on all trials).
	# make another design matrix which has all the data not just train and test
	imin = np.argmin(negLTest.mean(axis=1))

	# fit on entire dataset with this ridge penalty
	dspec = make_dspec(stim, sps, dt)
	dspec.addRegressorContinuous(basis=stim_basis)
	dspec.addRegressorSpTrain(basis=spike_basis)

	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()
	wmap = mapfit_GLM(wmap, stim_basis.nbases, X, y, lamvals[1]*Imat, 1)

	d = dm.get_regressor_from_output(wmap)

	dc = wmap[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dt
	h = d['sptrain'][1]
	ht = d['sptrain'][0] * dt

	figure, ax = plt.subplots(1, 2)
	ax[0].plot(kt, k)
	ax[1].plot(ht, h)
	ax[0].axhline(0, color=".2", linestyle="--", zorder=1)
	ax[1].axhline(0, color=".2", linestyle="--", zorder=1)
	ax[0].set_xlabel('Time before spike(s)')
	ax[1].set_xlabel('Time after spike(s)')
	ax[0].set_title('Stimulus Filter')
	ax[1].set_title('post-spike Filter')
	ax[1].set_xlim(0.001, 1)
	ax[0].set_xlim(-2, 0)

	# make a prediction on the average response
	glm = GLM(dspec.expt.dtSp, k, h, dc)
	nsims = 10
	glm_sims = np.zeros((sps.shape[0], nsims))
	glm_sptimes = np.zeros((sps.shape[0], nsims))
	for i in range(nsims): # run 5 GLM simulations
		tsp, sps, itot, istm = glm.simulate(stim[:, 0])
		glm_sptimes[:, i] = sps * np.arange(len(sps)) * dt
		glm_sims[:, i] = sps

	t = np.arange(0, len(stim)) * dt
	fig, ax = plt.subplots(3, 1, sharex=True)
	ax[0].plot(t, stim)

	# ax[0].eventplot(sptimes.T, lineoffsets=0.7)

	ax[1].eventplot(glm_sptimes.T, linewidth=0.5)
	smooth_win = np.hanning(100) / np.hanning(100).sum()
	psth_model = np.convolve(np.mean(glm_sims, axis=1), smooth_win)
	sigrange = np.arange(100 // 2, 100 // 2 + len(sps))
	psth_model_ = psth_model[sigrange] * 1000
	# ax[2].plot(np.arange(0, int(70*fs))/1000., psth_model_)
	ax[2].plot(t, psth_model_, label='GLM')
	#

	_, sps_exp = load_spk_times2('../datasets/stim.txt', response)
	psth_exp = np.convolve(np.mean(sps_exp, axis=1), smooth_win, mode='full')
	sigrange = np.arange(100 // 2, 100 // 2 + len(sps_exp))
	psth_exp_ = psth_exp[sigrange] * 1000
	ax[2].plot(t, psth_exp_, label='biological')

	ax[2].legend()




