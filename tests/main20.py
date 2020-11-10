
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
from utils.read import load_concatenatedlpvm_spike_data
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

"""
Refer to Pillow et al., 2005 for more info.
This script fits model parameters, K, h, dc of a 
Poisson Generalized linear model.

"""
if __name__ == "__main__":
	# something wrong with cell 6, 8 sometimes the fitting doesn't hone in on the solution...run again
	# check 4
	cell_idx = 1
	stim, sps = load_concatenatedlpvm_spike_data('../datasets/vm_spiking/lpvm_spikes_PN' + str(cell_idx) + '.mat')

	#stim, psth = load_concatenatedlpvm_psth('../datasets/lpvm_psth.mat')

	dt = 0.001
	psth = np.expand_dims(sps, axis=1)
	stim = np.expand_dims(stim, axis=1)
	stim_train, stim_test, resp_train, resp_test = train_test_split(stim, psth,
																	test_size=0.1,
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
	fs = 1 / dt
	nkt = int(0.6 * fs)
	stim_basis = RaisedCosine(100, 7, 1, 'stim')
	# problem here, the number of columns of A in the orthonormal basis is only equal to number of columns before orthogonalization if A is full rank
	# i.e. vectors are linearly independent
	stim_basis.makeNonlinearRaisedCosStim(.1, [.1, round(nkt/1.2)], 10, nkt)  # first and last peak positions,
	dspec_train.addRegressorContinuous(basis=stim_basis)
	dspec_test.addRegressorContinuous(basis=stim_basis)

	# this reproduces the weber & pillow post-spike filter. Don't touch the bases.py code
	# adjust the arguments to makeNonlinearRaisedCosPostSpike for my dt
	# examine the basis before using
	# spike_basis = RaisedCosine(100, 7, 1, 'sphist')
	# spike_basis.makeNonlinearRaisedCosPostSpike(0.1, [0.1, 100], 10, 1)

	spike_basis = RaisedCosine(100, 7, 1, 'sphist')
	spike_basis.makeNonlinearRaisedCosPostSpike(0.1, [.1, 10], 1, 0.01)
	dspec_train.addRegressorSpTrain(basis=spike_basis)
	dspec_test.addRegressorSpTrain(basis=spike_basis)

	dm, X_train, y_train = dspec_train.compileDesignMatrixFromTrialIndices()
	dm, X_test, y_test = dspec_test.compileDesignMatrixFromTrialIndices()

	prs = np.random.normal(0, .2, stim_basis.nbases + spike_basis.nbases + 1)

	# pars = 5 + 8
	res = minimize(neg_log_lik, prs, args=(stim_basis.nbases, X_train, y_train, 1),
				   options={'disp': True})

	theta_ml = res['x']
	d = dm.get_regressor_from_output(theta_ml)

	dc = theta_ml[0]
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

	#ax[1, 0].set_xlim([-0.2, 0])
	#ax[1, 1].set_xlim([0, 1])
	figure.suptitle('PN' + str(cell_idx))
	plt.tight_layout()


	# output GLM parameters: k, h, dc
	data = {'k': (kt, k),
			'h': (ht, h),
			'dc': dc,
			'v_min': scaler.data_min_,
			'v_max': scaler.data_max_}

	output = open('../results/glmpars_vm_to_spiking_PN' + str(cell_idx) + '.pkl', 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)

	glm = GLM(dspec_train.expt.dtSp, k, h, dc)

	# #for i in range(nsim):
	tsp, sps, itot, istm = glm.simulate(stim_train[:int(70*fs)])
	sps_ = sps * np.arange(len(sps)) * dt
	#colors1 = ['C{}'.format(1) for i in range(3)]

	# plt.figure()
	# plt.eventplot(sps_.T, linewidth=0.5)
	# plt.plot(np.arange(len(tsp)) * dt, get_psth(tsp, 100, 1000))
	# # plt.plot(psth)

	# compare with actual firing-rate

	#

	inds = range(int(70 * fs))
	t = np.arange(0, 70 * fs) / fs
	sptimes = resp_train[:int(70*fs)] * t

	fig, ax = plt.subplots(3, 1, sharex=True)
	ax[0].plot(t, stim_train[inds])

	#ax[0].eventplot(sptimes.T, lineoffsets=0.7)

	ax[1].plot(t, resp_train[inds], linewidth=0.5)

	smooth_win = np.hanning(100) / np.hanning(100).sum()
	psth_model = np.convolve(sps, smooth_win)
	sigrange = np.arange(100 // 2, 100 // 2 + len(sps))
	psth_model_ = psth_model[sigrange] * 1000
	#ax[2].plot(np.arange(0, int(70*fs))/1000., psth_model_)
	ax[2].plot(t, psth_model_, label='GLM')
	#

	sps_exp = resp_train[inds, 0]
	psth_exp = np.convolve(sps_exp, smooth_win, mode='full')
	sigrange = np.arange(100 // 2, 100 // 2 + len(sps_exp))
	psth_exp_ = psth_exp[sigrange] * 1000
	# ax[2].plot(np.arange(0, int(70 * fs - 1/fs)) / 1000., psth_exp_)
	ax[2].plot(t, psth_exp_, label='biological')

	ax[2].legend()
