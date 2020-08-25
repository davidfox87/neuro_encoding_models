import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from glmtools.make_xdsgn import DesignMatrix, RegressorContinuous, RegressorSphist, Experiment, DesignSpec
from scipy.optimize import minimize
from utils import plot as nmaplt
from scipy import signal
from sklearn.model_selection import train_test_split
from scipy.sparse import spdiags
from scipy.linalg import block_diag
from glmtools.fit import ridge_fit, neg_log_lik, mapfit_GLM, ridgefitCV, poisson
from sklearn.model_selection import KFold
import sys

from models import izhikevich

if __name__ == "__main__":

	# load spike times
	sps = np.genfromtxt('test.txt', delimiter='\t')
	duration = len(sps)

	# make an Experiment object
	dtSp = 1 	# (ms)
	expt = Experiment(dtSp, duration, sptimes=sps)

	# register sptrain regressor
	expt.register_spike_train('sptrain')

	# initialize design spec with one trial
	dspec = DesignSpec(expt, [0])
	# add spike history regressor with basis

	dspec.addRegressorSpTrain()

	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	# use linear regression to feed in an initial guess for minimize
	prs = np.linalg.inv(X.T @ X) @ X.T @ y

	res = minimize(poisson, prs, args=(X, y), options={'disp': True})
	theta_ml = res['x']

	# combine weights across each basis vector
	xx, yy = dm.get_regressor_from_output('sptrain', theta_ml[1:])


	plt.plot(theta_ml[1:])

