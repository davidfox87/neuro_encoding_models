import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from basisFactory.bases import RaisedCosine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.read import load_ankita_data
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
from glmtools.fit import fit_nlin_hist1d, fit_mean_nlfn, sameconv


def make_bases(dur, peaks, stim_nbases, stretch, dt=0.04):
	fs = 1. / dt
	first_peak, last_peak = peaks

	nkt = int(dur * fs)  # length of filter in number of samples
	# stim_nbases = number of vectors that are raised cosines
	stim_basis = RaisedCosine(100, stim_nbases, 1, 'stim')

	# arguments for makeNonlinearRaisedCosStim are
	# dt sample interval
	# peaks = [position of first center, position of last center],
	# stretch = spacing of basis centers (higher value = more linear meaning spread out,
	# lower value = nonlinear more centers near 0)
	# nkt = length of filter. If length of filters exceeds nkt we trim the filters,
	# otherwise we pad with 0's
	# stim_basis.makeNonlinearRaisedCosStim(.1, [10, round(nkt/1.7)], stretch, nkt)  # first and last peak positions,
	stim_basis.makeNonlinearRaisedCosStim(.1, [first_peak / dt, last_peak / dt], stretch, nkt)

	return stim_basis


def make_dspec(stim, response, dt, inds):
	# make an Experiment object
	expt = Experiment(dt, len(stim) * dt, stim=stim, response=response)

	# register continuous regressor
	expt.registerContinuous('stim')

	# initialize design spec with number of entries
	return DesignSpec(expt, trialinds=inds)


"""
This script fits a temporal filter to odor-calcium response data
Courtesy of Ankita Gumaste (Yale)

"""
if __name__ == "__main__":
	dt = 1. / 25.
	stims, responses = load_ankita_data('../datasets/ankita/ConstantAir_glom1.mat')

	print('done loading')

	# lets plot the data
	t = np.arange(0, len(stims)) * dt
	fig, ax = plt.subplots(2, 1, sharex=True, figsize=[20, 5])
	ax[0].plot(t, stims[:, 1])
	ax[1].plot(t, responses[:, 1])

	# define train and test
	stims_train = stims[:, :50]
	responses_train = responses[:, :50]

	stims_test = stims[:, 50:]
	responses_test = responses[:, 50:]

	# standardize each train and test.
	scaler_ = StandardScaler()
	stim_train = scaler_.fit_transform(stims_train)
	stim_test = scaler_.fit_transform(stims_test)

	# scale responses to between 0 and 1
	scaler2 = MinMaxScaler()
	responses_train = scaler2.fit_transform(responses_train)
	responses_test = scaler2.fit_transform(responses_test)

	# the data is sampled at 25 Hz, do i need to upsample or not?
	# what did i do with my own behavior data?

	# define basis
	stim_basis = make_bases(0.2, [0.001, 0.1], 10, 1, dt=0.001)
	plt.figure(figsize=[20, 5])
	plt.plot(np.arange(-len(stim_basis.B), 0) * 0.001, stim_basis.B)

	# make train-validation folds
	ntrials = responses_train.shape[1]
	inds = np.arange(ntrials)  # trial indices used to make splits
	np.random.shuffle(inds)

	folds_xtrain = []
	folds_xtest = []
	folds_ytrain = []
	folds_ytest = []

	from sklearn.model_selection import KFold

	kf = KFold(n_splits=5)
	kf.get_n_splits(inds)

	for train_index, test_index in kf.split(inds):
		# each fold will consist of a design matrix that will have concatenated trials for that fold
		# e.g. the first fold will train on trials 1, 2, 3 and test on 0
		# the second fold will train on 0, 2, 3 and test on 1
		print("TRAIN:", train_index, "TEST:", test_index)

		# use the inds to take a slice of sps and make a train design matrix
		train_dspec = make_dspec(stims_train[:, train_index], responses_train[:, train_index], dt,
								 np.arange(len(train_index)))
		train_dspec.addRegressorContinuous(basis=stim_basis)

		# use the inds to take a slice of sps and make a test design matrix
		test_dspec = make_dspec(stims_train[:, test_index], responses_train[:, test_index], dt,
								np.arange(len(test_index)))
		test_dspec.addRegressorContinuous(basis=stim_basis)

		dm, X_train, y_train = train_dspec.compileDesignMatrixFromTrialIndices(bias=1)
		dm, X_test, y_test = test_dspec.compileDesignMatrixFromTrialIndices(bias=1)

		folds_xtrain.append(X_train)
		folds_xtest.append(X_test)
		folds_ytrain.append(y_train)
		folds_ytest.append(y_test)

	print('Done making folds')


	# do ridge regression
	def ridgefitCV(train, test, parameter):

		msetest_fold = 0
		for train, test in zip(train, test):
			Xtrain, ytrain = train
			Xtest, ytest = test

			model = Ridge(alpha=parameter).fit(Xtrain, ytrain)

			msetest_fold += mean_squared_error(ytest, model.predict(Xtest))

		# take the average mse across folds for this alpha
		return msetest_fold / len(train)


	lamvals = np.logspace(0, 20, num=20, base=2)


	def _run_search(evaluate_candidates):
		"""Search all candidates in param_grid"""
		evaluate_candidates(lamvals)


	parallel = Parallel(n_jobs=10, verbose=10)

	all_out = []

	with parallel:

		def evaluate_candidates(candidate_params):
			out = parallel(delayed(ridgefitCV)(train=zip(folds_xtrain, folds_ytrain),
											   test=zip(folds_xtest, folds_ytest),
											   parameter=param)
						   for param in candidate_params)

			all_out.extend(out)

	_run_search(evaluate_candidates)

	print(all_out)

	imin = np.argmin(all_out)
	print("best ridge param is {}".format(lamvals[imin]))
	plt.plot(all_out)

	# run ridge with the best alpha on the average response

	dspec = make_dspec(stims_train, responses_train, dt, np.arange(responses_train.shape[1]))
	dspec.addRegressorContinuous(basis=stim_basis)
	dm, X, y = dspec.compileDesignMatrixFromTrialIndices(bias=1)

	model = Ridge(alpha=lamvals[imin]).fit(X, y)

	w = model.coef_

	d = dm.get_regressor_from_output(w)

	# # convert back to the original basis to get nkt filter weights.
	# # Parameters are returned in a dict
	dc = w[0]
	k = d['stim'][1]
	kt = d['stim'][0] * dt

	# plot the filter
	fig, ax = plt.subplots(figsize=[20, 5])
	ax.plot(kt, k, 'o-')
	ax.set_xlabel('Time lag (s)')
	ax.set_title('Stimulus temporal filter for ground speed')
	ax.axhline(0, color=".2", linestyle="--", zorder=1)
	ax.set_xlim([-2, 0])

	# test the model on the held-out test set data

	#
	# ridge_score = r2_score(response_, fnlin(rawfilteroutput))
	# basis_mse = mean_squared_error(response_, fnlin(rawfilteroutput))
	#
	# print('The r2 on the held-out test set is {}'.format(basis_score))
	# print('The mse on the held-out test set is {}'.format(basis_mse))

	fig10 = plt.figure(constrained_layout=True)
	gs0 = fig10.add_gridspec(1, 2)

	gs00 = gs0[0].subgridspec(3, 1)
	gs01 = gs0[1].subgridspec(4, 1)

	ax = gs00.subplots()
	ax[0].plot(np.arange(-len(stim_basis.B), 0) * dt, stim_basis.B)
	ax[1].plot(kt, k)
	ax[1].set_xlim(-2, 0)

# ax[2].scatter(rawfilteroutput, response_, s=20, c='k', alpha=0.2)
# ax[2].plot(xx, fnlin(xx), 'c', linewidth=2)
#
#


	xx_vec = []
	nlinfn = []
	for i in range(stim_train.shape[1]):
		stim_ = stims_train[:, i]
		response_ = responses_train[:, i]
		xx, fnlin, rawfilteroutput = fit_nlin_hist1d(stim_, response_, k, dt, 40)
		#plt.plot(xx, fnlin(xx), 'c', linewidth=2)
		xx_vec.append(xx)
		nlinfn.append(fnlin(xx))

	xx_vec = np.array(xx_vec)
	nlinfn = np.array(nlinfn)

	xx, fnlin = fit_mean_nlfn(xx_vec, nlinfn, 20)
	ax[2].plot(xx, fnlin(xx))

	# t = np.arange(0, len(stim_)) / 25
	# plt.plot(t, response_, 'k', linewidth=2)
	# plt.plot(t, fnlin(rawfilteroutput), 'c', linewidth=2, label="basis")
	# plt.scatter(rawfilteroutput, response_, s=20, c='k', alpha=0.2)

	# Filter stimulus with model filter
	Fs = 1000

	axs = gs01.subplots()  # Create all subplots for the inner grid.
	for i, ax in enumerate(axs):
		test_idx = i
		rawfilteroutput = sameconv(stim_test[:, test_idx], k)
		ax.plot(scaler2.fit_transform(stim_test[:, test_idx].reshape(-1, 1)))
		prediction = scaler2.fit_transform(fnlin(rawfilteroutput).reshape(-1, 1))
		ax.plot(prediction - np.mean(prediction[1 * Fs:2 * Fs]))
		ax.set_xlim(0, 600)

	plt.tight_layout()