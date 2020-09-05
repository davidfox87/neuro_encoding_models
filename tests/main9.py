import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from basisFactory.bases import Basis, RaisedCosine
from scipy.optimize import minimize
from glmtools.fit import x_proj
from utils import plot as nmaplt
import pickle

if __name__ == "__main__":

	# name of behavior for which we want to extract the temporal stim filter
	behavior_par = "vmoves"

	# load behavior from MATLAB .mat file
	stim, response = io.load_behavior('../datasets/behavior/control_behavior.mat', 30., 55., behavior_par)

	# make an Experiment object
	expt = Experiment(0.02, 85, stim=stim, response=response)

	# register continuous regressor
	expt.registerContinuous('stim')

	# initialize design spec with one trial
	trial_inds = list(range(response.shape[1]))
	dspec = DesignSpec(expt, trial_inds)

	# make a set of basis functions
	Fs = 50
	nkt = 2*Fs
	cos_basis = RaisedCosine(100, 3, 1, 'stim')
	cos_basis.makeNonlinearRaisedCosStim(expt.dtSp, [5, nkt/2-10], 1, nkt)
	dspec.addRegressorContinuous(basis=cos_basis)

	# compile a design matrix using all trials
	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	# find weights in the new basis that minimizes the difference between the projection of the stimulus onto
	# this basis and the target time series
	prs = np.random.normal(0, .2, 4)
	res = minimize(x_proj, prs, args=(X, y),
										options={'maxiter': 1000, 'disp': True})

	theta = res['x']

	# convert back to the original basis to get nkt filter weights.
	# Parameters are returned in a dict
	d = dm.get_regressor_from_output(theta[1:])

	dc = theta[0]
	k = d['stim'][1]

	fig, ax1 = plt.subplots(1, 1)
	nmaplt.plot_spike_filter(ax1, k, dspec.dt, linewidth=6)
	ax1.set_xlim(-2, 0.1)

	dc = theta[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dspec.dt

	# output GLM parameters: k, dc
	data = {'name': behavior_par,
			'k': k,
			'dc': dc}

	file_name = "../datasets/behavior/" + behavior_par + "_filter.pkl"
	output = open(file_name, 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)

