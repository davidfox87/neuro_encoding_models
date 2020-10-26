
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

from utils.read import load_concatenatedlpvm_spike_data
"""
Refer to Pillow et al., 2005 for more info.
This script fits model parameters, K, h, dc of a 
Poisson Generalized linear model.

"""
if __name__ == "__main__":
	stim, sps = load_concatenatedlpvm_spike_data('../datasets/lpvm_spikes.mat')

	dt = 0.001
	sps = np.expand_dims(sps, axis=1)
	stim = np.expand_dims(stim, axis=1)
	stim_train, stim_test, resp_train, resp_test = train_test_split(stim, sps,
																	test_size=0.2,
																	shuffle=False,
																	random_state=42)

	scaler = MinMaxScaler()
	scaler.fit(stim_train)
	stim_train = scaler.transform(stim_train)
	stim_test = scaler.transform(stim_test)

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

	# make a set of basis functions
	fs = 1/ dt
	nkt = int(.2 * fs)
	stim_basis = RaisedCosine(100, 4, 1, 'stim')
	stim_basis.makeNonlinearRaisedCosStim(expt_train.dtSp, [0.001, nkt / 2], 1, nkt)  # first and last peak positions,
	dspec_train.addRegressorContinuous(basis=stim_basis)
	dspec_test.addRegressorContinuous(basis=stim_basis)

	spike_basis = RaisedCosine(100, 3, 1, 'sphist')
	spike_basis.makeNonlinearRaisedCosPostSpike(expt_train.dtSp, [.001, .1], 1)
	dspec_train.addRegressorSpTrain(basis=spike_basis)
	dspec_test.addRegressorSpTrain(basis=spike_basis)

	dm, X_train, y_train = dspec_train.compileDesignMatrixFromTrialIndices()
	dm, X_test, y_test = dspec_test.compileDesignMatrixFromTrialIndices()


	prs = np.random.normal(0, .2, 8)
	# pars = 5 + 8
	res = minimize(neg_log_lik, prs, args=(4, X_train, y_train, 1),
				   options={'disp': True})

	theta_ml = res['x']
	d = dm.get_regressor_from_output(theta_ml)


	# make grid of lambda (ridge parameter) values
	nlam = 30
	lamvals = np.logspace(-3, 2, nlam)

	# identity matrix of size of filter plus const
	Imat = np.eye(stim_basis.nbases + spike_basis.nbases)
	# remove penalty on bias term (constant DC offset)
	Imat[0, 0] = 0

	# allocate space for train and test errors
	negLTrain = np.zeros((nlam, 1))
	negLTest = np.zeros((nlam, 1))
	w_ridge = np.zeros((stim_basis.nbases + spike_basis.nbases + 1, nlam))

	neg_train_func = lambda prs: neg_log_lik(prs, stim_basis.nbases, X_train, y_train, 1)
	neg_test_func = lambda prs: neg_log_lik(prs, stim_basis.nbases, X_test, y_test, 1)


	Dt1 = spdiags((np.ones((stim_basis.nbases, 1)) * np.array([-1, 1])).T, np.array([0, 1]), stim_basis.nbases-1,
				  stim_basis.nbases, format="csc").toarray()
	Dt = Dt1.T @ Dt1

	Dt2 = spdiags((np.ones((spike_basis.nbases, 1)) * np.array([-1, 1])).T, np.array([0, 1]), spike_basis.nbases - 1,
				  spike_basis.nbases, format="csc").toarray()
	Dt_ = Dt2.T @ Dt2

	D = block_diag(0, Dt, Dt_)

	# identity matrix of size of filter plus const
	Imat = np.eye(stim_basis.nbases + spike_basis.nbases + 1)
	# remove penalty on bias term (constant DC offset)
	Imat[0, 0] = 0

	# find MAP estimate for each value of ridge parameter
	print('====Doing grid search for ridge parameter====\n')
	wmap = theta_ml
	for jj in range(nlam):
		print('lambda = {:.3f}'.format(lamvals[jj]))
		Cinv = lamvals[jj] * D

		# Do MAP estimation of model params
		wmap = mapfit_GLM(wmap, stim_basis.nbases, X_train, y_train, Cinv, 1)

		negLTrain[jj] = neg_train_func(wmap)
		negLTest[jj] = neg_test_func(wmap)

		w_ridge[:, jj] = wmap

		plt.plot(wmap[1:])

	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.semilogx(negLTest, '-o')
	ax1.set_xlabel("lambda")
	ax1.set_ylabel("Test log-likelihood")
	fig.suptitle("Ridge Prior")

	wmap = w_ridge[:, np.argmin(negLTest)]

	# combine weights across each basis vector
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
	ax[0].set_xlabel('Time before spike (s)')
	ax[1].set_xlabel('Time after spike (s)')
	ax[0].set_title('membrane potential Filter')
	ax[1].set_title('post-spike filter Filter')

	# output GLM parameters: k, h, dc
	data = {'k': k,
			'kt': kt,
			'h': h,
			'ht': ht,
			'dc': dc,
			'v_min': scaler.data_min_,
			'v_max': scaler.data_max_}

	output = open('../models/glmpars_vm_to_spiking.pkl', 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)

	output.close()

	glm = GLM(dspec_train.expt.dtSp, k, h, dc)

	# get prediction of spike count over time
	# nsim = 7
	# sps_ = np.empty((len(stim), nsim))
	# actual = list(map(sp_count_fun, sptimes))
	# actual_ = list(map(lambda x: x * (np.arange(len(stim)) + 1) * dt, actual))
	# for i in range(5):
	# 	sps_[:, i] = actual_[i].T
	#
	#
	# for i in range(5, nsim):
	# 	tsp, sps, itot, istm = glm.simulate(scaler.transform(stim.reshape(-1, 1)))
	# 	sps_[:, i] = sps * np.arange(len(sps)) * dt
	# colors1 = ['C{}'.format(1) for i in range(5)]
	# colors2 = ['C{}'.format(2) for i in range(2)]
	# colors = colors1 + colors2

	# plt.figure()
	# plt.eventplot(sps_.T, colors=colors, linewidth=0.5)
