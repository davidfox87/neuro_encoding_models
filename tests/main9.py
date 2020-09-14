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
from sklearn.metrics import r2_score

if __name__ == "__main__":

	# name of behavior for which we want to extract the temporal stim filter
	behavior_par = "vymoves"

	# load behavior from MATLAB .mat file
	stim, response = io.load_behavior('../datasets/behavior/control_behavior.mat', 30., 55., behavior_par)

	stim = normalize(stim)
	# make an Experiment object
	expt = Experiment(0.02, 25, stim=stim, response=response)

	# register continuous regressor
	expt.registerContinuous('stim')

	# initialize design spec with one trial
	trial_inds = list(range(response.shape[1]))
	dspec = DesignSpec(expt, trial_inds)

	# make a set of basis functions
	Fs = 50
	nkt = 5*Fs
	cos_basis = RaisedCosine(100, 4, 1, 'stim')
	cos_basis.makeNonlinearRaisedCosStim(expt.dtSp, [5, 50], 1, nkt) # first and last peak positions,
	dspec.addRegressorContinuous(basis=cos_basis)

	# compile a design matrix using all trials
	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	# find weights in the new basis that minimizes the difference between the projection of the stimulus onto
	# this basis and the target time series
	prs = np.random.normal(0, .2, 5)
	res = minimize(x_proj, prs, args=(X, y),
										options={'maxiter': 1000, 'disp': True})

	theta = res['x']

	# convert back to the original basis to get nkt filter weights.
	# Parameters are returned in a dict
	d = dm.get_regressor_from_output(theta[1:])

	dc = theta[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dspec.dt

	# fig, ax1 = plt.subplots(1, 1)
	# nmaplt.plot_spike_filter(ax1, k, dspec.dt, linewidth=6)
	# ax1.set_xlim(-2, 0.1)
	# ax1.set_xlabel('Time before movement (s)')

	# output GLM parameters: k, dc
	data = {'name': behavior_par,
			'k': k,
			'dc': dc}

	file_name = behavior_par + "_filter.pkl"
	output = open(file_name, 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)





	# get nonlinearity and make prediction

	fs = 50
	stim_ = stim[:, 0]
	response_ = np.reshape(response, (25*fs, -1)).mean(axis=1)

	xx, fnlin, rawfilteroutput = fit_nlin_hist1d(stim_, response_, k, dspec.dt, 25)
	basis_score = r2_score(response_, fnlin(rawfilteroutput))

	fig = plt.figure()
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(212)

	nmaplt.plot_spike_filter(ax1, k/LA.norm(k), dspec.dt, linewidth=3, color='c', label="basis")
	ax1.set_xlim(-5, 0)
	ax1.set_xlabel('Time before movement (s)')
	ax2.scatter(rawfilteroutput, response_, s=20, c='k', alpha=0.2)
	ax2.plot(xx, fnlin(xx), 'c', linewidth=2)

	t = np.arange(0, 25 * fs) / fs
	ax3.plot(t, response_, 'k', linewidth=2)
	ax3.plot(t, fnlin(rawfilteroutput), 'c', linewidth=2, label="basis")


	pkl_file = open('../datasets/behavior/ridge_filters/' + behavior_par + "_filter.pkl", 'rb')
	pars = pickle.load(pkl_file)
	k_ridge = pars['k']
	k_ridge = k_ridge/LA.norm(k_ridge)
	nmaplt.plot_spike_filter(ax1, k_ridge / LA.norm(k_ridge), dspec.dt, linewidth=3, color='b', label="ridge")

	xx, fnlin, rawfilteroutput = fit_nlin_hist1d(stim_, response_, k_ridge, dspec.dt, 25)
	ax3.plot(t, fnlin(rawfilteroutput), 'b', linewidth=1, label="ridge")

	ax1.legend()

	ax3.set_ylabel('Angular Velocity (deg/s)')
	ax3.set_xlabel('Time (s)')
	ax2.set_xlabel('Filter projection')
	ax2.set_ylabel('Angular Velocity (deg/s)')
	ax2.set_title('Nonlinearity')

	ax1.set_title('Stimulus Filter, k')

	ax3.set_title("Prediction")
	ax3.legend()
	ridge_score = r2_score(response_, fnlin(rawfilteroutput))


	plt.tight_layout()
	plt.show()