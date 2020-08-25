import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from glmtools.make_xdsgn import DesignMatrix, RegressorContinuous, RegressorSphist, GLM, Experiment, DesignSpec
from scipy.optimize import minimize
from utils import plot as nmaplt
from scipy import signal
from sklearn.model_selection import train_test_split
from scipy.sparse import spdiags
from scipy.linalg import block_diag
from glmtools.fit import ridge_fit, neg_log_lik, mapfit_GLM, ridgefitCV
from sklearn.model_selection import KFold
import sys

from models import izhikevich

if __name__ == "__main__":
	t, v, stim, sptimes = izhikevich.simulate()

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
	ax1.plot(t, stim)
	ax2.eventplot(sptimes)
	ax3.plot(t, v)

	idx = np.where(stim > 0)[0]
	stim = stim[np.where(stim > 0)]
	sptimes = np.array(sptimes) - t[idx[0]]

	exp = Experiment(stim, [sptimes], 0.25, 0.25)

	exp.registerContinuous('stim')
	exp.register_spike_train("spikes")

	dspec = DesignSpec(exp, [0])
	X, y = dspec.compileDesignMatrixFromTrialIndices()

	model = Ridge(alpha=10).fit(X, y)

	plt.plot(model.coef_[500:])


