import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from scipy.optimize import minimize
from glmtools.fit import neg_log_lik
import pickle
from glmtools.model import GLM
from basisFactory.bases import RaisedCosine


"""
Refer to Pillow et al., 2005 for more info.
This script fits model parameters, K, h, dc of a 
Poisson Generalized linear model.
 
"""
if __name__ == "__main__":

	# load spike times
	# t, v, stim, sptimes = izhikevich.simulate()
	stim, sptimes = io.load_spk_times('../datasets/lpVmPN1.txt', '../datasets/spTimesPNControl/pn1SpTimes_reverseChirp.mat', 5, 30)

	test = np.apply_along_axis(lambda x: x - np.mean(x), 0, stim)
	stim = np.apply_along_axis(lambda x: x / np.std(x), 0, test)
	stim = stim * 0.001
	dt = 0.001 				# (size 1 (ms) bins)
	duration = 25

	# bin spikes
	sp_count_fun = lambda x: np.histogram(x, np.arange(0.5, len(stim)+1) * dt - dt)[0]
	sps = list(map(sp_count_fun, sptimes))
	#
	sps = np.array(sps).T

	# make an Experiment object
	expt = Experiment(dt, duration, stim=stim, sptimes=sps, response=sps)

	# register continuous
	expt.registerContinuous('stim')

	# register sptrain regressor
	expt.register_spike_train('sptrain')

	# initialize design spec with one trial
	trial_inds = list(range(len(sptimes)))
	dspec = DesignSpec(expt, [0])
	# make a set of basis functions

	fs = 1/dt
	nkt = int(2 * fs)
	cos_basis = RaisedCosine(100, 5, 1, 'stim')
	cos_basis.makeNonlinearRaisedCosStim(expt.dtSp, [1, nkt/2], 10, nkt)  # first and last peak positions,
	dspec.addRegressorContinuous(basis=cos_basis)

	basis = RaisedCosine(100, 5, 1, 'sphist')
	basis.makeNonlinearRaisedCosPostSpike(expt.dtSp, [.001, 1], .05)
	dspec.addRegressorSpTrain(basis=basis)

	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	prs = np.random.normal(0, .2, 11)
	# pars = 5 + 8
	res = minimize(neg_log_lik, prs, args=(5, X, y, 1),
										options={'maxiter': 10000, 'disp': True})

	theta_ml = res['x']

	# combine weights across each basis vector
	d = dm.get_regressor_from_output(theta_ml)

	dc = theta_ml[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dt
	h = d['sptrain'][1]
	ht = d['sptrain'][0] * dt

	# output GLM parameters: k, h, dc
	data = {'k': k,
			'h': h,
			'dc': dc}

	output = open('../models/glmpars_vm_to_spiking.pkl', 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)

	glm = GLM(dspec.expt.dtSp, k, h, dc)

	# get prediction of spike count over time

	nsim = 10
	sps_ = np.empty((len(stim), nsim))
	actual = list(map(sp_count_fun, sptimes))
	actual_ = list(map(lambda x: x * (np.arange(len(stim)) + 1) * dt, actual))
	for i in range(5):
		sps_[:, i] = actual_[i].T

	for i in range(5, nsim):
		tsp, sps = glm.simulate(stim[:, i-5])
		sps_[:, i] = sps*np.arange(len(sps))*dt
	colors1 = ['C{}'.format(1) for i in range(5)]
	colors2 = ['C{}'.format(2) for i in range(5)]
	colors = colors1 + colors2

	plt.eventplot(sps_.T, colors=colors, linewidth=0.5)