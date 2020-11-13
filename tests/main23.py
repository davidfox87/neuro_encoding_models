import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from basisFactory.bases import Basis, RaisedCosine
from scipy.optimize import minimize
from glmtools.fit import x_proj, fit_nlin_hist1d, normalize
from utils import plot as nmaplt
import pickle
from numpy import linalg as LA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def make_dspec(stim, response, dt):
	# make an Experiment object
	expt = Experiment(dt, len(stim)*dt, stim=stim, response=response)

	# register continuous regressor
	expt.registerContinuous('stim')

	# initialize design spec with one trial
	return DesignSpec(expt, [0])

if __name__ == "__main__":

	# name of behavior for which we want to extract the temporal stim filter
	behavior_par = "vmoves"
	# behavior_par = "angvturns"
	#behavior_par = "vymoves"
	# load behavior from MATLAB .mat file
	stim, response = io.load_behavior('../datasets/behavior/control_behavior.mat', 30., 55., behavior_par, 50)
	response = response.mean(axis=1)
	stim = stim[:, 0]

	stim = stim.reshape((len(stim), 1))
	response = response.reshape((len(stim), 1))

	fig, ax = plt.subplots(2, 1, sharex=True)
	ax[0].plot(stim)
	ax[1].plot(response)

	# split into train and test
	stim_train, stim_test, resp_train, resp_test = train_test_split(stim, response,
																	test_size=0.001,
																	shuffle=False,
																	random_state=42)
	dt = 0.02
	# make train dspec
	train_dspec = make_dspec(stim_train, resp_train, dt)
	# make test dspec
	test_dspec = make_dspec(stim_test, resp_test, dt)

	# make a set of basis functions
	Fs = 50
	nkt = 14 * Fs
	nkt = 12 * Fs
	stim_basis = RaisedCosine(100, 10, 1, 'stim')
	stim_basis.makeNonlinearRaisedCosStim(0.1, [1, round(nkt / 1.2)], 10, nkt)  # first and last peak positions,
	plt.figure()
	plt.plot(stim_basis.B)

	train_dspec.addRegressorContinuous(basis=stim_basis)
	test_dspec.addRegressorContinuous(basis=stim_basis)

	# compile a design matrix using all trials
	dm, X_train, y_train = train_dspec.compileDesignMatrixFromTrialIndices(bias=1)
	dm, X_test, y_test = train_dspec.compileDesignMatrixFromTrialIndices(bias=1)


	# this is our design matrix
	plt.imshow(X_train[:, 1:] @ stim_basis.B.T, extent=[-12, 0, 0, 1250],
              interpolation='nearest', aspect='auto')

	prs = np.random.normal(0, .01, stim_basis.nbases + 1)

	# pars = 5 + 8
	res = minimize(x_proj, prs, args=(X_train, y_train, stim_basis.nbases),
				   options={'disp': True})

	w = res['x']

	# model = Ridge()
	# alphas = np.logspace(0, 30, num=20, base=2)
	# param_search = [{'alpha': alphas}]
	#
	# tscv = TimeSeriesSplit(n_splits=5)
	# grid_result = GridSearchCV(estimator=model, cv=tscv,
	# 						   param_grid=param_search, scoring='neg_mean_squared_error', n_jobs=-1)
	#
	# grid_result.fit(X_train, y_train)
	# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	#
	# # using this ridge penalty value, get the mse between resp_test and ridge prediction
	# model = Ridge(alpha=grid_result.best_params_['alpha']).fit(X_test, y_test)
	# w = model.coef_

	d = dm.get_regressor_from_output(w)

	# convert back to the original basis to get nkt filter weights.
	# Parameters are returned in a dict
	dc = w[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dt

	# xx, fnlin, rawfilteroutput = fit_nlin_hist1d(stim_test, resp_test, k, dt, 20)
	# basis_score = r2_score(resp_test, fnlin(rawfilteroutput))
	# basis_mse = mean_squared_error(resp_test, fnlin(rawfilteroutput))
	#
	# print('The r2 on the held-out test set is {}'.format(basis_score))
	# print('The mse on the held-out test set is {}'.format(basis_mse))


	# now fit using this ridge on all the data
	# dspec = make_dspec(stim, response, dt)
	# dspec.addRegressorContinuous(basis=stim_basis)
	#
	# # compile a design matrix using all trials
	# dm, X_, y_ = dspec.compileDesignMatrixFromTrialIndices()
	# model = Ridge(alpha=grid_result.best_params_['alpha']).fit(X_, y_)
	#
	# w = model.coef_
	#
	# d = dm.get_regressor_from_output(w)

	# convert back to the original basis to get nkt filter weights.
	# Parameters are returned in a dict
	# dc = w[0]
	# k = d['stim'][1]
	# kt = d['stim'][0] * dt

	# output GLM parameters: k, dc
	data = {'name': behavior_par,
			'k': k,
			'dc': dc}

	plt.figure()
	plt.plot(kt, k)

	file_name = "../results/behavior_filters/" +  behavior_par + "_filter.pkl"
	output = open(file_name, 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)


	# get nonlinearity and make prediction

	fs = 50
	stim_ = stim[:, 0]
	response_ = np.reshape(response, (25*fs, -1)).mean(axis=1)

	xx, fnlin, rawfilteroutput = fit_nlin_hist1d(stim_, response_, k, dt, 20)
	basis_score = r2_score(response_, fnlin(rawfilteroutput))

	fig = plt.figure()
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(212)

	nmaplt.plot_spike_filter(ax1, k/LA.norm(k), dt, linewidth=3, color='c', label="basis")
	# ax1.set_xlim(-5, 0)
	ax1.set_xlabel('Time before movement (s)')
	ax2.scatter(rawfilteroutput, response_, s=20, c='k', alpha=0.2)
	ax2.plot(xx, fnlin(xx), 'c', linewidth=2)

	t = np.arange(0, 25 * fs) / fs
	ax3.plot(t, response_, 'k', linewidth=2)
	ax3.plot(t, fnlin(rawfilteroutput), 'c', linewidth=2, label="basis")

	ax3.set_ylabel('Ground Speed (mm/s)')
	ax3.set_xlabel('Time (s)')
	ax2.set_xlabel('Filter projection')
	ax2.set_ylabel('Ground Speed (mm/s)')
	ax2.set_title('Nonlinearity')

	ax1.set_title('Stimulus Filter, k')

	ax3.set_title("Prediction")
	ax3.legend()

	ridge_score = r2_score(response_, fnlin(rawfilteroutput))
	basis_mse = mean_squared_error(response_, fnlin(rawfilteroutput))

	print('The r2 on the held-out test set is {}'.format(basis_score))
	print('The mse on the held-out test set is {}'.format(basis_mse))

	plt.tight_layout()
	plt.show()

