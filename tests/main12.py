import utils.read as io
import numpy as np
from glmtools.make_xdsgn import Experiment, DesignSpec
from sklearn.model_selection import KFold
from cnn.preprocessing import preprocess_resp
# import regularizer
# instantiate regularizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from sklearn.preprocessing import StandardScaler


def create_model():
	model = Sequential()

	# how to determine the number filters and size of filters?
	model.add(Conv1D(filters=64, kernel_size=499, activation='relu'
					 , input_shape=(500, 1)))
	model.add(MaxPooling1D())
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='mse', metrics='mse')
	model.summary()

	return model


if __name__ == "__main__":


	behavior_par = "vmoves"
	fs = 50
	# load behavior from MATLAB .mat file
	stim, response = io.load_behavior('../datasets/behavior/control_behavior.mat', 30., 55., behavior_par, fs)


	# StandardScaler
	scaler = StandardScaler()
	stim = scaler.fit_transform(stim)
	stim = stim[:, 0]
	stim = stim.reshape((len(stim), 1))

	response = preprocess_resp(response)
	response = response.mean(axis=1) # work on the fly-average
	response = response.reshape((len(stim), 1))
	# make an Experiment object
	expt = Experiment(0.02, 25, stim=stim, response=response)

	# register continuous regressor
	expt.registerContinuous('stim')

	# initialize design spec with one trial
	trial_inds = list(range(response.shape[1]))
	dspec = DesignSpec(expt, trial_inds)

	dspec.addRegressorContinuous()

	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()


	dirs = dict()
	dirs['save'] = 'results/'

	# keras expects the time series as [samples, timesteps, features]



	# w = model.get_weights()[0][:, 0, :]
	# fig, ax = plt.subplots()
	# nmaplt.plot_spike_filter(ax, w.mean(axis=1), 0.02, linewidth=4, color='k')
	# ax.set_xlim(-4, 0)
	#


	filepath = dirs['save'] + 'weights/weights_best_' + str(1) + '.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
								 save_best_only=True, mode='auto')

	x = X.reshape((X.shape[0], X.shape[1], 1))

	kfold = KFold(n_splits=5, shuffle=False)
	cv_mse = []

	for train, test in kfold.split(x, y):

		model = create_model()
		# fit the model
		history = model.fit(x[train], y[train], epochs=30, batch_size=10, verbose=1)

		mse = model.evaluate(x[test], y[test], verbose=1)
		print("%s: %.5f" % (model.metrics_names[1], mse[1]))
		cv_mse.append(mse)
	print("%.5f% (+/- %.5f%)" % (np.mean(cv_mse), np.std(cv_mse)))