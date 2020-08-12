import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import RidgeCV
from glmtools.makeXdsgn import DesignMatrix, RegressorContinuous, RegressorSphist, GLM
from scipy.optimize import minimize
from utils import plot as nmaplt
from scipy import signal
from sklearn.model_selection import train_test_split
from scipy.sparse import spdiags
from scipy.linalg import block_diag
from glmtools.fit import ridge_fit, neg_log_lik, mapfit_GLM

if __name__ == "__main__":


	stim, spTimes, dt = io.load_spk_data_neuromatch()
	stim_idx = list(range(len(stim)))

	stim_train, stim_test = train_test_split(stim_idx, test_size=0.2, shuffle=False)

	ntfilt = 25
	ntsphist = 25

	dm = DesignMatrix(dt=dt, mintime=stim_train[0]*dt, maxtime=stim_train[-1]*dt)
	dm.add_regressor(RegressorContinuous("Opto", ntfilt))
	dm.add_regressor(RegressorSphist("Spk_hist", ntsphist))  # spike history filter

	X_train = dm.empty_matrix()  # init Xdsgn matrix
	y_train = np.asarray([])  # init response vector

	binned_spikes = dm.bin_spikes(spTimes)

	# plt.stem(binned_spikes)

	X_train = dm.build_matrix({"Opto_time": 0,
						 "Opto_val": stim[stim_train[0]:stim_train[-1]],  # actual stim
						 "Spk_hist_time": 0,
						 "Spk_hist_val": binned_spikes
						 })

	X_train = np.column_stack([np.ones_like(binned_spikes), X_train])
	X_train = stats.zscore(X_train)

	theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ binned_spikes

	fig, (ax1, ax2) = plt.subplots(1, 2)
	nmaplt.plot_spike_filter(ax1, theta[1:ntfilt+1], dt)
	nmaplt.plot_spike_filter(ax2, np.flipud(theta[ntfilt+1:-2]), dt)


	# make test desgin matrix
	dm = DesignMatrix(dt=dt, mintime=stim_test[0] * dt, maxtime=stim_test[-1] * dt)
	dm.add_regressor(RegressorContinuous("Opto", ntfilt))
	dm.add_regressor(RegressorSphist("Spk_hist", ntsphist))  # spike history filter

	X_test = dm.empty_matrix()  # init Xdsgn matrix
	y_test = np.asarray([])  # init response vector

	binned_spikes = dm.bin_spikes(spTimes)

	# plt.stem(binned_spikes)

	X = dm.build_matrix({"Opto_time": 0,
							   "Opto_val": stim[stim_test[0]:stim_test[-1]],  # actual stim
							   "Spk_hist_time": 0,
							   "Spk_hist_val": binned_spikes
							   })

	X_test = np.column_stack([np.ones_like(binned_spikes), X_test])
	X_test = stats.zscore(X_test)

	# do from glmtools.fit import ridgeFit_linear_Gauss, ML_fit_GLM, MAP_Fit_GLM
	C_values = np.logspace(-6, 6, 20)

	w, msetest = ridge_fit(X_train, y_train, X_test, y_test, C_values)


