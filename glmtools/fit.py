from scipy.optimize import minimize
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from glmtools.make_xdsgn import DesignMatrix
from scipy import stats, signal
from scipy import interpolate

def normalize(data):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return (data - data_mean) / data_std



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
		# plt.plot(w[1:])

	imin = np.argmin(msetest)
	plt.plot(wridge[1:, imin], linewidth=4)
	return wridge[:, imin], msetest, alphavals[imin]


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

			# msetest_fold += r2_score(y_test, model.predict(X_test))
			msetest_fold += mean_squared_error(y_test, model.predict(X_test))
			#plt.plot(w[1:])
		# take the average mse across folds for this alpha
		msetest[i] = msetest_fold / len(folds_train)

	plt.plot(msetest, '-ob')

	return alphavals[np.argmin(msetest)]

def neg_log_lik(theta: np.ndarray, ntfilt: int,
				X: np.ndarray, y: np.ndarray, flag: int):
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
	k = theta[1:ntfilt+1]				# stim filter
	XStim = X[:, :ntfilt]				# stim convolved with basis functions
	dc = theta[0]  						# dc current

	if flag:
		h = theta[ntfilt+1:]			# post-spike filter
		XSp = X[:, ntfilt:]				# spike response convolved with basis functions

		itot = XStim @ k + XSp @ h + dc
	else:
		itot = XStim @ k + dc


	# Compute GLM filter output and conditional intensity
	nzidx = np.nonzero(y)[0]
	rate = np.exp(itot[nzidx])

	return -(np.dot(y[nzidx], np.log(rate)) - rate.sum())


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

def poisson(wts, x, y):
	'''
	negative log-likelihood of data under the Poisson model
	:param ws: 	[m x 1] regression weights
	:param x: 	[N x m] regressors
	:param y: 	[N x 1] output (binary vector of 1s and 0s
	:return:
	'''

	xproj = x@wts

	f = np.exp(xproj)

	nzidx = np.nonzero(f)[0]

	negloglik = -y[nzidx].T@np.log(f[nzidx]) + np.sum(f)

	return negloglik


def x_proj(wts, x, y, ntstim):

	xproj = x @ wts

	return mean_squared_error(y, xproj)



def sameconv(x, f):
	slen = len(x)
	x = np.convolve(x.squeeze(), np.flipud(f.squeeze()), mode='full')
	return x[:slen]


	a = np.concatenate((np.zeros(nf - 1), x), axis=None)
	b = np.rot90(f, k=2)
	res = signal.convolve2d(np.asarray([a]), b, mode='valid')
	return res.squeeze()


def fit_nlin_hist1d(stim, response, filt, dt, nfbins):
	"""
	Compute histogram-based estimate for piece-wise constant
	nonlinearity by binning filter output and computing, for
	each bin, the mean spike rate
	:param stim: stimulus
	:param response: response vector
	:param filt: stimulus filter
	:param dt: sample interval
	:param nbins: number of bins
	:return:
	"""

	# Filter stimulus with model filter
	rawfilteroutput = sameconv(stim, filt)

	# bin the filtered stimulus into nbins
	counts, bin_edges = np.histogram(rawfilteroutput, bins=nfbins)

	# Return the indices of the bins to which each value in input array belongs.
	inds = np.digitize(rawfilteroutput, bin_edges)

	# use the centers of the bins
	fx = 0.5*(bin_edges[1:]+bin_edges[:-1])

	fy = np.zeros(nfbins)
	for i in range(nfbins):
		fy[i] = np.nanmean(response[inds == i+1])

	fnlin = lambda x: np.interp(x, fx, fy)

	xx = np.linspace(bin_edges[0], bin_edges[-1], 100)

	#plt.plot(xx, fnlin(xx), linewidth=2)

	return xx, fnlin, rawfilteroutput