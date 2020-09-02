import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from scipy.optimize import minimize
from glmtools.fit import neg_log_lik
import pickle
from glmtools.model import GLM


if __name__ == "__main__":

	# load spike times
	# t, v, stim, sptimes = izhikevich.simulate()
	stim, sptimes = io.load_spk_times('../datasets/lpVmPN1.txt', '../datasets/spTimesPNControl/pn1SpTimes_reverseChirp.mat', 5, 30)
	#stim = (stim / abs(np.max(stim)))*0.01
	test = np.apply_along_axis(lambda x: x - np.mean(x), 0, stim)
	stim = np.apply_along_axis(lambda x: x / np.std(x), 0, test)
	stim = stim * 0.01
	dt = 0.001 				# (size 1 (ms) bins)
	duration = 25

	# bin spikes
	sp_count_fun = lambda x: np.histogram(x, np.arange(0.5, len(stim)+1) * dt - dt)[0]
	sps = list(map(sp_count_fun, sptimes))
	#
	sps = np.array(sps).T

	#stim_ = np.tile(stim, (len(sptimes), 1))  # we need a stim for every trial

	# make an Experiment object
	expt = Experiment(dt, duration, stim=stim, sptimes=sps)


	# get an estimate of STA then project this onto basis as an initial guess for ML

	# register continuous
	expt.registerContinuous('stim')

	# register sptrain regressor
	expt.register_spike_train('sptrain')

	# initialize design spec with one trial
	trial_inds = list(range(len(sptimes)))
	dspec = DesignSpec(expt, trial_inds)

	# add stim and spike history regressors
	dspec.addRegressorContinuous()
	dspec.addRegressorSpTrain()

	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	# use linear regression to feed in an initial guess for minimize

	prs = np.random.normal(0, .2, 14)
	res = minimize(neg_log_lik, prs, args=(5, X, y, 1),
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

	output = open('glmpars_vm_to_spiking.pkl', 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)

	glm = GLM(dspec.expt.dtSp, k, h, dc)

	# this is where we put the model Vm to get a transform Vm to spikes
	stim, sptimes = io.load_spk_times('../datasets/lpVmPN1.txt', '../datasets/spTimesPNControl/pn1SpTimes_reverseChirp.mat', 5, 30)
	test = np.apply_along_axis(lambda x: x - np.mean(x), 0, stim)
	stim = np.apply_along_axis(lambda x: x / np.std(x), 0, test)
	stim *= 0.01
	#stim = (stim / abs(np.max(stim))) * 0.01
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