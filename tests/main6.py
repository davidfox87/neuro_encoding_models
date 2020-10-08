import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from scipy.optimize import minimize
from glmtools.fit import neg_log_lik
from glmtools.model import GLM
import pickle
from basisFactory.bases import RaisedCosine

if __name__ == "__main__":

	# load spike times for cell
	cell = "pn1"
	response = '../datasets/spTimesPNControl/{}SpTimes_reverseChirp.mat'.format(cell, '.txt')
	stim, sptimes = io.load_spk_times('../datasets/stim.txt', response, 5, 30)

	test = np.apply_along_axis(lambda x: x - np.mean(x), 0, stim)
	stim = np.apply_along_axis(lambda x: x / np.std(x), 0, test)
	stim = stim * 0.001

	dt = 0.001 				# (size 1 (ms) bins)
	duration = 25

	# bin spikes
	sp_count_fun = lambda x: np.histogram(x, np.arange(0.5, len(stim)+1) * dt - dt)[0]
	sps = list(map(sp_count_fun, sptimes))
	sps = np.array(sps).T

	# just import as 5 columns, then you won't have to do this
	stim_ = np.tile(stim, (len(sptimes), 1)).T  # we need a stim for every trial

	# make an Experiment object
	expt = Experiment(dt, duration, stim=stim_, sptimes=sps, response=sps)

	# register continuous
	expt.registerContinuous('stim')

	# register sptrain regressor
	expt.register_spike_train('sptrain')

	# initialize design spec with one trial
	trial_inds = list(range(len(sptimes)))
	dspec = DesignSpec(expt, trial_inds)

	# add stim and spike history regressors
	fs = 1 / dt
	nkt = 2000
	cos_basis = RaisedCosine(100, 5, 1, 'stim')
	cos_basis.makeNonlinearRaisedCosStim(expt.dtSp, [1, 1000], 50, nkt)  # first and last peak positions,
	dspec.addRegressorContinuous(basis=cos_basis)

	basis = RaisedCosine(100, 8, 1, 'sphist')
	basis.makeNonlinearRaisedCosPostSpike(expt.dtSp, [.001, 1], .05)
	dspec.addRegressorSpTrain(basis=basis)

	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	pars = np.random.normal(0, .2, 14)
	res = minimize(neg_log_lik, pars, args=(cos_basis.nbases, X, y, 1),
										options={'maxiter': 1000, 'disp': True})

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

	outfile = '../datasets/spTimesPNControl/{}.pkl'.format(cell)
	output = open(outfile, 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)
	#
	# # for testing purposes load filters in from MATLAB and
	# # compare the simGLM method with glm.simulate
	# glm = GLM(dspec.expt.dtSp, k, h, dc)
	#
	# stim, sptimes = io.load_spk_times('../datasets/stim.txt', '../datasets/spTimesPNControl/pn1SpTimes_reverseChirp.mat', 0, 35)
	#
	# # get prediction of spike count over time
	#
	# nsim = 10
	# sps_ = np.empty((len(stim), nsim))
	actual = list(map(sp_count_fun, sptimes))
	actual_ = list(map(lambda x: x * (np.arange(len(stim)) + 1) * dt, actual))
	# for i in range(5):
	# 	sps_[:, i] = actual_[i].T
	#
	# for i in range(5, nsim):
	# 	tsp, sps = glm.simulate((stim/max(stim))*0.01)
	# 	sps_[:, i] = sps*np.arange(len(sps))*dt
	# colors1 = ['C{}'.format(1) for i in range(5)]
	# colors2 = ['C{}'.format(2) for i in range(5)]
	# colors = colors1 + colors2
	#
	# plt.eventplot(sps_.T, colors=colors, linewidth=0.5)
	#
	# fig, (ax1, ax2) = plt.subplots(1, 2)
	# ax1.plot(kt, k, '-ok', linewidth=1)
	# ax2.plot(ht, h, '-ok', linewidth=1)
	# ax1.plot([-1, 0], [0, 0], '--k')
	# ax2.plot([0, .5], [0, 0], '--k')
	# ax2.set_xlim(0, 0.5)
	# ax1.set_ylim(-0.1, .8)
	# plt.show()


