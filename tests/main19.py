
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
from utils.read import load_concatenatedlpvm_psth
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

"""
Refer to Pillow et al., 2005 for more info.
This script fits model parameters, K, h, dc of a 
Poisson Generalized linear model.

"""
if __name__ == "__main__":
	#stim, sps = load_concatenatedlpvm_spike_data('../datasets/lpvm_spikes.mat')

	stim, psth = load_concatenatedlpvm_psth('../datasets/lpvm_psth.mat')

	dt = 0.01
	psth = np.expand_dims(psth, axis=1)
	stim = np.expand_dims(stim, axis=1)
	stim_train, stim_test, resp_train, resp_test = train_test_split(stim, psth,
																	test_size=0.7,
																	shuffle=False,
																	random_state=42)


	# scaler = MinMaxScaler([0, 1])
	# scaler.fit(resp_train)
	# resp_train = scaler.transform(resp_train)
	# resp_test = scaler.transform(resp_test)

	# make an Experiment object
	expt_train = Experiment(dt, len(stim_train)*dt, stim=stim_train, sptimes=resp_train, response=resp_train)
	expt_test = Experiment(dt, len(stim_test)*dt, stim=stim_test, sptimes=resp_test, response=resp_test)

	# register continuous
	expt_train.registerContinuous('stim')
	expt_train.register_spike_train('sptrain')
	expt_test.registerContinuous('stim')
	expt_test.register_spike_train('sptrain')

	# initialize design spec with one trial
	dspec_train = DesignSpec(expt_train, [0])
	dspec_test = DesignSpec(expt_test, [0])

	fs = 1 / dt
	nkt = int(2 * fs)
	stim_basis = RaisedCosine(100, 4, 1, 'stim')
	# problem here, the number of columns of A in the orthonormal basis is only equal to number of columns before orthogonalization if A is full rank
	# i.e. vectors are linearly independent
	stim_basis.makeNonlinearRaisedCosStim(expt_train.dtSp, [0.01, round(nkt /1.2)], .01,
										  nkt)  # first and last peak positions,
	dspec_train.addRegressorContinuous(basis=stim_basis)
	dspec_test.addRegressorContinuous(basis=stim_basis)

	spike_basis = RaisedCosine(100, 3, 1, 'sphist')
	spike_basis.makeNonlinearRaisedCosPostSpike(expt_train.dtSp, [0.01, .1], .1, 0.01)
	dspec_train.addRegressorSpTrain(basis=spike_basis)
	dspec_test.addRegressorSpTrain(basis=spike_basis)

	dm, X_train, y_train = dspec_train.compileDesignMatrixFromTrialIndices()
	dm, X_test, y_test = dspec_test.compileDesignMatrixFromTrialIndices()


	model = Ridge()
	alphas = np.logspace(0, 30, num=20, base=2)
	param_search = [{'alpha': alphas}]

	tscv = TimeSeriesSplit(n_splits=5)
	grid_result = GridSearchCV(estimator=model, cv=tscv,
							   param_grid=param_search, scoring='neg_root_mean_squared_error', n_jobs=-1)
	grid_result.fit(X_train, y_train)
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

	# using this ridge penalty value, get the mse between resp_test and ridge prediction
	model = Ridge(alpha=grid_result.best_params_['alpha']).fit(X_train, y_train)

	d = dm.get_regressor_from_output(model.coef_)

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


	glm = GLM(dspec_train.expt.dtSp, k, h, 0)

	# get prediction of spike count over time
	nsim = 3
	sps_ = np.empty((len(stim), nsim))
	# actual = list(map(sp_count_fun, sptimes))
	# actual_ = list(map(lambda x: x * (np.arange(len(stim)) + 1) * dt, actual))
	# for i in range(5):
	# 	sps_[:, i] = actual_[i].T


	for i in range(nsim):
		tsp, sps, itot, istm = glm.simulate(stim)
		sps_[:, i] = sps * np.arange(len(sps)) * dt
	colors1 = ['C{}'.format(1) for i in range(3)]

	plt.figure()
	plt.eventplot(sps_.T, colors=colors1, linewidth=0.5)
	plt.plot(get_psth(sps_.T, 100, 1000))
	plt.plot(psth)
