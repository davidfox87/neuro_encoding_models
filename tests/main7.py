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


"""
Refer to Pillow et al., 2005 for more info.
This script fits model parameters, K, h, dc of a 
Poisson Generalized linear model.
 
"""
if __name__ == "__main__":

	# load spike times FOR ALL STIM AND CONCATENATE DESIGN MATRICES
	stim, sptimes = io.load_spk_times('../datasets/lpVmPN1.txt', '../datasets/spTimesPNControl/pn1SpTimes_reverseChirp.mat', 0, 35)
	n = stim.shape[1]

	scaler = MinMaxScaler()
	stim = scaler.fit_transform(stim[:, 0].reshape(-1, 1))

	dt = 0.001 				# (size 1 (ms) bins)
	duration = 35

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
	nkt = int(0.5 * fs)
	cos_basis = RaisedCosine(100, 10, 1, 'stim')
	cos_basis.makeNonlinearRaisedCosStim(expt.dtSp, [1, nkt/2], 1, nkt)  # first and last peak positions,
	dspec.addRegressorContinuous(basis=cos_basis)

	basis = RaisedCosine(100, 4, 1, 'sphist')
	basis.makeNonlinearRaisedCosPostSpike(expt.dtSp, [.01, .5], 1)
	dspec.addRegressorSpTrain(basis=basis)

	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	prs = np.random.normal(0, .2, 15)
	# pars = 5 + 8
	res = minimize(neg_log_lik, prs, args=(10, X, y, 1),
										options={'maxiter': 1000, 'disp': True})

	theta_ml = res['x']

	# combine weights across each basis vector
	d = dm.get_regressor_from_output(theta_ml)

	dc = theta_ml[0]
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
			'h': h,
			'dc': dc,
			'v_min': scaler.data_min_,
			'v_max': scaler.data_max_}

	output = open('../models/glmpars_vm_to_spiking.pkl', 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)

	glm = GLM(dspec.expt.dtSp, k, h, dc)

	# get prediction of spike count over time

	nsim = 7
	sps_ = np.empty((len(stim), nsim))
	actual = list(map(sp_count_fun, sptimes))
	actual_ = list(map(lambda x: x * (np.arange(len(stim)) + 1) * dt, actual))
	for i in range(5):
		sps_[:, i] = actual_[i].T

	for i in range(2, nsim):
		tsp, sps, itot, istm = glm.simulate(stim[:, 0])
		sps_[:, i] = sps*np.arange(len(sps))*dt
	colors1 = ['C{}'.format(1) for i in range(5)]
	colors2 = ['C{}'.format(2) for i in range(2)]
	colors = colors1 + colors2

	plt.figure()
	plt.eventplot(sps_.T, colors=colors, linewidth=0.5)

	output.close()

