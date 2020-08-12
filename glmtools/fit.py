from scipy.optimize import minimize
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from glmtools.makeXdsgn import DesignMatrix


def ridge_fit(Xtrain, ytrain, Xtest, ytest, alphavals):
	ntfilt = Xtrain.shape[1]

	msetrain = np.zeros(len(alphavals))
	msetest = np.zeros(len(alphavals))
	wridge = np.zeros((ntfilt, len(alphavals)))

	for i, alpha_ in enumerate(alphavals):

		model = Ridge(alpha=alpha_).fit(Xtrain, ytrain)

		# Get the coefs of the model fit to training data
		w = model.coef_

		msetrain[i] = mean_squared_error(ytrain, model.predict(Xtrain))
		msetest[i] = mean_squared_error(ytest, model.predict(Xtest))

		#print("ridge training score:", train_score)
		print("ridge test mse: ", msetrain[i])

		wridge[:, i] = w
		plt.plot(w[1:])

	imin = np.argmin(msetest)
	plt.plot(wridge[1:, imin], linewidth=4)
	return wridge[:, imin], msetest


def ridgefitCV(folds_train, folds_test, alphavals):
	"""
	# logic
	# for each value of ridge lambda value,
	# 	calculate an average MSE across folds
	# choose lambda that minimizes average MSE across folds
	:param folds:
	:param alphavals:
	:return:
	"""
	msetest = np.zeros(len(alphavals))
	for i, alpha_ in enumerate(alphavals):
		print(i)
		msetest_fold = 0
		for train, test in zip(folds_train, folds_test):
			X_train, X_test = train[0], test[0]
			y_train, y_test = train[1], test[1]

			model = Ridge(alpha=alpha_).fit(X_train, y_train)

			w = model.coef_
			#plt.plot(w[1:])
			msetest_fold += mean_squared_error(y_test, model.predict(X_test))

		# take the avereage mse across folds for this alpha
		msetest[i] = msetest_fold / len(folds_train)

	plt.plot(msetest, '-ob')

	return alphavals[np.argmin(msetest)]

def neg_log_lik(theta: np.ndarray, ntfilt: int,
				X: np.ndarray, y: np.ndarray, flag: int) -> float:
	"""
	Compute negative log-likelihood of data under an LNP model with a single
	filter and fixed nonlinearity (exp), as a function of filter parameters

	Return -loglike for the Poisson GLM model.
	Args:
		theta (1D array): Parameter vector.
		X (2D array): Full design matrix.
		y (1D array): Data values.
	Returns:
		number: Negative log likelihood.
	"""

	# extract some stuff I will use
	k = theta[:ntfilt+1]
	XStim = X[:, :ntfilt+1]
	# k = dm.get_regressor_from_output('Opto', theta)[1]
	# XStim = dm.get_regressor_from_dm('Opto', X)

	if flag:
		h = theta[ntfilt+1:]
		XSp = X[:, ntfilt+1:]
		# XSp = dm.get_regressor_from_dm('Spk_hist', X)
		# h = dm.get_regressor_from_output('Spk_hist', theta)[1]

		itot = XStim @ k + XSp @ h
	else:
		itot = XStim @ k

	# Compute GLM filter output and conditional intensity

	rate = np.exp(itot)
	log_lik = y @ np.log(rate) - rate.sum()
	return -log_lik

def neg_log_posterior(prs, negloglifun, Cinv):
	"""
	compute the negative log liklihood function and zero mean gaussian prior with
	inverse covariance Cinv
	# for an explanation of MAP estimation:
	see https://wiseodd.github.io/techblog/2017/01/05/bayesian-regression/

	:param prs: prs: parameter vector or current estimate of the filter coefficients
	:param negloglifun: handle for negative log-likelihood function
	:param Cinv:
	:return:
	"""

	return negloglifun(prs) + .5 * prs.T@Cinv@prs


def mapfit_GLM(prs, ntfilt, stim, sps, Cinv, flag):
	"""
	Computes the MAP estimate for GLM params, using grad and hessians under
	a zero-mean Gaussian prior with inverse covariance Cinv.

	Minimizes negative log-likelihood plus a penalty of the form
    0.5*x'*Cinv*x, where x is the parameter vector

	:param prs:
	:param stim:
	:param sps:
	:param Cinv:
	:return:
	"""
	# set log-likelihood function
	lfunc = lambda prs : neg_log_lik(prs, ntfilt, stim, sps, flag)

	# set log-posterior function
	lpost = lambda prs : neg_log_posterior(prs, lfunc, Cinv)

	# Find parameters that minimize the negative log likelihood function
	res = minimize(lpost, prs, options={'disp': True})

	return res['x']
