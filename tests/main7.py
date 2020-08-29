import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import DesignMatrix, RegressorContinuous, RegressorSphist, Experiment, DesignSpec
from scipy.optimize import minimize
from glmtools.fit import ridge_fit, neg_log_lik, mapfit_GLM, ridgefitCV, poisson, poisson_deriv
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

	# prs = np.linalg.inv(X.T @ X) @ X.T @ y
	#prs = np.concatenate((np.asarray([0]), prs), axis=0)

	prs = [0.0, -16.9989, 4.5455, 1.1125, 0.8270, -1.3354]
	ih_pars = np.random.normal(0, .2,  8)
	prs = np.concatenate((prs, ih_pars), axis=0)
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


	#np.savetxt('test.out', (kt, k, ht, h))

	# for testing purposes load filters in from MATLAB and
	# compare the simGLM method with glm.simulate
	# we
	glm = GLM(dspec, k, h, dc)



	# this is where we put the model Vm to get a transform Vm to spikes
	stim, sptimes = io.load_spk_times('../datasets/lpVmPN1.txt', '../datasets/spTimesPNControl/pn1SpTimes_reverseChirp.mat', 5, 30)
	#test = np.apply_along_axis(lambda x: x - np.mean(x), 0, stim)
	#stim = np.apply_along_axis(lambda x: x / np.std(x), 0, test)
	#stim *= 0.1
	stim = (stim / abs(np.max(stim))) * 0.01
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

	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.plot(kt, k, '-ok', linewidth=1)
	ax2.plot(ht, h, '-ok', linewidth=1)
	ax1.plot([-1, 0], [0, 0], '--k')
	ax2.plot([0, .5], [0, 0], '--k')
	ax2.set_xlim(0, 0.5)
	ax1.set_ylim(-0.1, .8)
	plt.show()


