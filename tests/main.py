import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from glmtools.makeXdsgn import DesignMatrix, RegressorContinuous, RegressorSphist, GLM, Experiment, DesignSpec
from scipy.optimize import minimize
from utils import plot as nmaplt
from scipy import signal
from sklearn.model_selection import train_test_split
from scipy.sparse import spdiags
from scipy.linalg import block_diag
from glmtools.fit import ridge_fit, neg_log_lik, mapfit_GLM, ridgefitCV
from sklearn.model_selection import KFold
import sys


if __name__ == "__main__":
	stimfile = 'stim.txt'
	#stimfile = 'meanORNControlPSTH.txt'
	# orn1SpTimes_reverseChirp.mat
	stim, spikes = io.load_spk_times(stimfile, 'spTimesPNControl/pn1SpTimes_reverseChirp.mat', 5, 30)
	#stim, spikes = io.load_spk_times('stim.txt', 'spTimesPNU13AKD/pn1SpTimes_reverseChirp.mat', 5, 30)
	# stim, spikes = io.load_spk_times('stim.txt', 'spTimesORNControl/orn13SpTimes_reverseChirp.mat', 5, 30)

	dtSp = .001
	dtStim = 0.02 # you can't make this bin too wide because then you will lose the resolution on the stim
					# it's a tradeoff between having enough data and a small bin
					# i think these methods are most useful when you have a lot of data

	# define experiment and register covariates
	exp = Experiment(stim, spikes, dtSp, dtStim)
	exp.registerContinuous('Optostim')
	exp.register_spike_train("PNspikes")

	# kfolds cross-validation
	trials = len(spikes)

	folds_train = []
	folds_test = []

	trial_indices = np.arange(trials)
	kf = KFold(n_splits=trials)
	for train_index, test_index in kf.split(trial_indices):
		print("Train:", train_index, "TEST:", test_index)
		inds_train, inds_test = trial_indices[train_index], trial_indices[test_index]

	# form a train design matrix using the first fold indices
		dspec = DesignSpec(exp, inds_train)
		X_train, y_train = dspec.compileDesignMatrixFromTrialIndices()
		folds_train.append((X_train, y_train))
		dspec = DesignSpec(exp, inds_test)
		X_test, y_test = dspec.compileDesignMatrixFromTrialIndices()
		folds_test.append((X_test, y_test))

	lamvals = np.logspace(1, 6, 30)
	# logic
	# for each value of ridge lambda value,
	# 	calculate an average MSE across folds
	# choose lambda that minimizes average MSE across folds
	alpha_ = ridgefitCV(folds_train, folds_test, lamvals)

	dspec = DesignSpec(exp, trial_indices)
	X, y = dspec.compileDesignMatrixFromTrialIndices()

	model = Ridge(alpha=alpha_).fit(X, y)

	# Get the coefs of the model fit to training data
	w = model.coef_
	t = np.arange(-(len(w[1:])) + 1, 1) * dspec.dt_

	ntfilt = dspec.ntfilt
	nthist = dspec.ntsphist

	fig, (ax1, ax2) = plt.subplots(1, 2)
	nmaplt.plot_spike_filter(ax1, w[1:ntfilt+1, ], dspec.dt_)
	nmaplt.plot_spike_filter(ax2, (w[ntfilt + 1:]), dspec.dt_)
	# save the filter
	# np.savetxt('spTimesORNControl/orn13.txt', np.c_[t, w[1:]])
	# np.savetxt('spTimesPNControl/pn9.txt', np.c_[t, w[1:]])
	# np.savetxt('spTimesPNU13AKD/pn9.txt', np.c_[t, w[1:]])
	# np.savetxt('spTimesPNControl/ORNStimToPN/pn9.txt', np.c_[t, w[1:]])
	#np.savetxt('glmpredPN1.dat', ypred, delimiter='\t')

	plt.plot(np.exp(X[:, 1:ntfilt + 1] @ w[1:ntfilt + 1] + X[:, ntfilt + 1:] @ (w[ntfilt + 1:])) / dspec.dt_)
	# do from glmtools.fit import ridgeFit_linear_Gauss, ML_fit_GLM, MAP_Fit_GLM
	C_values = np.logspace(-6, 6, 20)
#
# 	w, msetest = ridge_fit(X_train, y_train, X_test, y_test, C_values)
#
# 	ypred = (X_test[:, 1:ntfilt+1] @ w[1:ntfilt+1])
# 	np.savetxt('glmpredPN1.dat', ypred, delimiter='\t')
#
# 	t1, filt = dm.get_regressor_from_output("Opto", w[1:])
# 	fig, (ax1, ax2) = plt.subplots(1,2)
# 	ax1.plot(msetest, '-o')
# 	ax2.plot(t1, filt, '-o')
# 	ax2.axhline(0, color=".2", linestyle="--", zorder=1)
#
#
# 	# now use ML fitting with a modified log-likelihood function
# 	Xstim = dm.get_regressor_from_dm('Opto', X_train[:, 1:])
# 	Xstim = X_train[:, :ntfilt + 1]
# 	# initialize stim filter using least squares
# 	theta = np.linalg.inv(Xstim.T @ Xstim) @ Xstim.T @ y_train
# 	# initialize spike - history weights randomly
#
# 	prs = theta
#
# 	res = minimize(neg_log_lik, prs, args=(ntfilt, X_train, y_train, 0))
# 	theta_ml = res['x']
#
# 	# you can call scipy.minimize with multiple arguments
# 	# minimize(f, x0, args=(a, b, c))
#
# 	# now regularize
#
# 	# make grid of lambda (ridge parameter) values
# 	nlam = 30
# 	lamvals = np.logspace(-6, 6, nlam)
#
# 	# identity matrix of size of filter plus const
# 	Imat = np.eye(ntfilt)
# 	# remove penalty on bias term (constant DC offset)
# 	Imat[0, 0] = 0
#
# 	# allocate space for train and test errors
# 	negLTrain = np.zeros((nlam, 1))
# 	negLTest = np.zeros((nlam, 1))
# 	w_ridge = np.zeros((ntfilt+1, nlam))
#
# 	neg_train_func = lambda prs: neg_log_lik(prs, ntfilt, X_train, y_train, 1)
# 	neg_test_func = lambda prs: neg_log_lik(prs, ntfilt, X_test, y_test, 1)
#
#
# 	Dt1 = spdiags((np.ones((ntfilt, 1)) * np.array([-1, 1])).T, np.array([0, 1]), ntfilt-1,
# 				  ntfilt, format="csc").toarray()
# 	Dt = Dt1.T @ Dt1
#
# 	D = block_diag(0, Dt)
#
# 	# identity matrix of size of filter plus const
# 	Imat = np.eye(ntfilt+1)
# 	# remove penalty on bias term (constant DC offset)
# 	Imat[0, 0] = 0
#
#
# 	# find MAP estimate for each value of ridge parameter
# 	print('====Doing grid search for ridge parameter====\n');
# 	wmap = theta_ml
# 	for jj in range(nlam):
# 		print('lambda = {:.3f}'.format(lamvals[jj]))
# 		Cinv = lamvals[jj] * D
#
# 		# Do MAP estimation of model params
# 		wmap = mapfit_GLM(wmap, ntfilt, X_train, y_train, Cinv, 1)
#
# 		negLTrain[jj] = neg_train_func(wmap)
# 		negLTest[jj] = neg_test_func(wmap)
#
# 		w_ridge[:, jj] = wmap
#
# 		plt.plot(wmap[1:])
#
# 	fig, (ax1, ax2) = plt.subplots(1, 2)
# 	ax1.semilogx(negLTest, '-o')
# 	ax1.set_xlabel("lambda")
# 	ax1.set_ylabel("Test log-likelihood")
# 	fig.suptitle("Ridge Prior")
#
# 	nmaplt.plot_spike_filter(ax2, w_ridge[1:ntfilt+1:, np.argmin(negLTest)+2], dt_)
#
#
#
# #
# # 	# estimate the nonlinearity
# # # 	raw_filter_output = clf.intercept_+ clf.predict(X_full)
# # # 	#raw_filter_output[raw_filter_output > 0] = 0
# # # 	#Y_full[Y_full>0] = 0
# # # 	nfbins = 25
# # #
# # # 	bin_means, bin_edges, binnumber = stats.binned_statistic(raw_filter_output, np.arange(len(raw_filter_output)), statistic='mean', bins=nfbins)
# # # 	fx = bin_edges[:-2] + np.diff(bin_edges[0:2]) / 2
# # # 	xx = np.arange(bin_edges[0], bin_edges[-1], 0.1)
# # #
# # # 	fy = np.zeros((nfbins-1, 1))
# # # 	for jj in range(nfbins-1):
# # # 		fy[jj] = np.mean(Y_full[binnumber == jj])
# # #
# # # 	fy = fy.squeeze()
# # # 	fnlin = lambda x: np.interp(x, fx, fy)
# # #
# # # 	plt.plot(raw_filter_output, Y_full)
# # # 	plt.plot(xx, fnlin(xx))
# # #
# # # 	plt.plot(Y_full)
# # # 	plt.plot(fnlin(raw_filter_output))
# # # 	plt.show()
# # #
# # #
# # #
# #
