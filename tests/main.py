import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import RidgeCV
from glmtools.makeXdsgn import DesignMatrix, RegressorContinuous, RegressorSphist, GLM
from scipy.optimize import minimize
from utils import plot as nmaplt
from scipy import signal
from sklearn.model_selection import train_test_split
from scipy.sparse import spdiags
from scipy.linalg import block_diag
from glmtools.fit import ridge_fit, neg_log_lik, mapfit_GLM

if __name__ == "__main__":

	stim, spikes = io.load_spk_times('stim.txt', 'spTimesControl/pn1SpTimes_reverseChirp.mat', 5, 30)
	# stim, spikes = io.load_spk_times('stimPlume.txt', 'pn1SpTimesPlume.mat', 5, 75)
	nt = len(stim)

	dtSp = .001
	dtStim = 0.04
	sampFactor = dtStim / dtSp

	ntfilt = int(2000/sampFactor)

	# Divide data into "training" and "test" sets for cross-validation

	sampfactor = dtStim / dtSp

	assert (sampfactor % 1 == 0, 'dtStim / dtSp must be an integer')
	stim = stim + np.abs(stim[0])
	# if we make the sampling more coarse, we will get more spikes
	stim_ = signal.resample(stim, len(stim) // int(sampfactor))  # 25 gives good results

	nt = len(stim_)
	dt_ = dtSp * sampfactor

	trials = len(spikes)

	# make Xtrain and Ytrain
	dm = DesignMatrix(dt=dt_, mintime=0, maxtime=25)
	dm.add_regressor(RegressorContinuous("Opto", ntfilt))

	X_train = dm.empty_matrix()  # init Xdsgn matrix
	y_train = np.asarray([])  # init response vector

	# iterate trials and concatenate horizontally each design matrix
	# the design matrix X_full with have nt*trials rows and ntfilt columns
	for tr in range(trials-1):
		print('converting spike times to counts\n')

		# sps = np.sum(np.reshape(spikes[:, tr], (-1, len(stim))), axis=0)  # reshape to the size of the stimlus
		binned_spikes = dm.bin_spikes(spikes[tr])

		X = dm.build_matrix({"Opto_time": 0,
								   # start time: this will be 0 because we are using the entire spike train from beginning to end
								   "Opto_val": stim_  # actual stim
								   })

		X_train = np.concatenate([X_train, X], axis=0)
		y_train = np.concatenate([y_train, binned_spikes])

	# z-score design matrix since we have two regressors
	X_train = stats.zscore(X_train)

	X_train = np.column_stack([np.ones_like(y_train), X_train])

	# make Xtest and Ytest
	dm2 = DesignMatrix(dt=dt_, mintime=0, maxtime=25)
	dm2.add_regressor(RegressorContinuous("Opto", ntfilt))

	X_test = dm2.empty_matrix()

	y_test = dm2.bin_spikes(spikes[-1])

	X_test = dm2.build_matrix({"Opto_time": 0,
						 # start time: this will be 0 because we are using the entire spike train from beginning to end
						 "Opto_val": stim_  # actual stim
						 })

	X_test = stats.zscore(X_test)

	X_test = np.column_stack([np.ones_like(y_test), X_test])

	# do from glmtools.fit import ridgeFit_linear_Gauss, ML_fit_GLM, MAP_Fit_GLM
	C_values = np.logspace(-6, 6, 20)

	w, msetest = ridge_fit(X_train, y_train, X_test, y_test, C_values)

	ypred = (X_test[:, 1:ntfilt+1] @ w[1:ntfilt+1])
	np.savetxt('glmpredPN1.dat', ypred, delimiter='\t')

	t1, filt = dm.get_regressor_from_output("Opto", w[1:])
	fig, (ax1, ax2) = plt.subplots(1,2)
	ax1.plot(msetest, '-o')
	ax2.plot(t1, filt, '-o')
	ax2.axhline(0, color=".2", linestyle="--", zorder=1)


	# now use ML fitting with a modified log-likelihood function
	Xstim = dm.get_regressor_from_dm('Opto', X_train[:, 1:])
	Xstim = X_train[:, :ntfilt + 1]
	# initialize stim filter using least squares
	theta = np.linalg.inv(Xstim.T @ Xstim) @ Xstim.T @ y_train
	# initialize spike - history weights randomly

	prs = theta

	res = minimize(neg_log_lik, prs, args=(ntfilt, X_train, y_train, 0))
	theta_ml = res['x']

	# you can call scipy.minimize with multiple arguments
	# minimize(f, x0, args=(a, b, c))

	# now regularize

	# make grid of lambda (ridge parameter) values
	nlam = 30
	lamvals = np.logspace(-6, 6, nlam)

	# identity matrix of size of filter plus const
	Imat = np.eye(ntfilt)
	# remove penalty on bias term (constant DC offset)
	Imat[0, 0] = 0

	# allocate space for train and test errors
	negLTrain = np.zeros((nlam, 1))
	negLTest = np.zeros((nlam, 1))
	w_ridge = np.zeros((ntfilt+1, nlam))

	neg_train_func = lambda prs: neg_log_lik(prs, ntfilt, X_train, y_train, 1)
	neg_test_func = lambda prs: neg_log_lik(prs, ntfilt, X_test, y_test, 1)


	Dt1 = spdiags((np.ones((ntfilt, 1)) * np.array([-1, 1])).T, np.array([0, 1]), ntfilt-1,
				  ntfilt, format="csc").toarray()
	Dt = Dt1.T @ Dt1

	D = block_diag(0, Dt)

	# identity matrix of size of filter plus const
	Imat = np.eye(ntfilt+1)
	# remove penalty on bias term (constant DC offset)
	Imat[0, 0] = 0


	# find MAP estimate for each value of ridge parameter
	print('====Doing grid search for ridge parameter====\n');
	wmap = theta_ml
	for jj in range(nlam):
		print('lambda = {:.3f}'.format(lamvals[jj]))
		Cinv = lamvals[jj] * D

		# Do MAP estimation of model params
		wmap = mapfit_GLM(wmap, ntfilt, X_train, y_train, Cinv, 1)

		negLTrain[jj] = neg_train_func(wmap)
		negLTest[jj] = neg_test_func(wmap)

		w_ridge[:, jj] = wmap

		plt.plot(wmap[1:])

	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.semilogx(negLTest, '-o')
	ax1.set_xlabel("lambda")
	ax1.set_ylabel("Test log-likelihood")
	fig.suptitle("Ridge Prior")

	nmaplt.plot_spike_filter(ax2, w_ridge[1:ntfilt+1:, np.argmin(negLTest)+2], dt_)



#
# 	# estimate the nonlinearity
# # 	raw_filter_output = clf.intercept_+ clf.predict(X_full)
# # 	#raw_filter_output[raw_filter_output > 0] = 0
# # 	#Y_full[Y_full>0] = 0
# # 	nfbins = 25
# #
# # 	bin_means, bin_edges, binnumber = stats.binned_statistic(raw_filter_output, np.arange(len(raw_filter_output)), statistic='mean', bins=nfbins)
# # 	fx = bin_edges[:-2] + np.diff(bin_edges[0:2]) / 2
# # 	xx = np.arange(bin_edges[0], bin_edges[-1], 0.1)
# #
# # 	fy = np.zeros((nfbins-1, 1))
# # 	for jj in range(nfbins-1):
# # 		fy[jj] = np.mean(Y_full[binnumber == jj])
# #
# # 	fy = fy.squeeze()
# # 	fnlin = lambda x: np.interp(x, fx, fy)
# #
# # 	plt.plot(raw_filter_output, Y_full)
# # 	plt.plot(xx, fnlin(xx))
# #
# # 	plt.plot(Y_full)
# # 	plt.plot(fnlin(raw_filter_output))
# # 	plt.show()
# #
# #
# #
#
