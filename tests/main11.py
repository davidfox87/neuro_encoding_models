import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from utils import plot as nmaplt
import os
import pickle

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
# custom R2-score metrics for keras backend
from keras import backend as K
from keras.utils import plot_model
from sklearn.linear_model import LinearRegression
from cnn.preprocessing import preprocess
import cnn.utils
from cnn.create_model import load_model




if __name__ == "__main__":
	np.random.seed(42)

	# CNN hyperparameters
	batch_size = 64
	epochs = 100
	input_shape = [750, 1]
	print_summary = False

	# dir
	dirs = dict()
	dirs['save'] = 'results/'

	if not os.path.exists(dirs['save']):
		os.makedirs(dirs['save'])
	if not os.path.exists(dirs['save'] + 'weights'):
		os.makedirs(dirs['save'] + 'weights')


	# specify behavior to make a prediction for
	behaviors = ["angvturns", "vmoves", "vymoves"]
	behavior_par = behaviors[2]

	# load the data from MATLAB .mat file
	stim, response = io.load_behavior('../datasets/behavior/control_behavior.mat', 30., 55., behavior_par, 50)
	response = response.mean(axis=1)  # work on the fly-average

	# preprocess for the CNN to work. This is a VERY important step!
	stim, response = preprocess(stim, response)

	# Get zero-padded vectorized time series of dimension [samples, timesteps]
	expt = Experiment(0.02, 25, stim=stim, response=response)
	expt.registerContinuous('stim')

	# initialize design spec with one trial
	trial_inds = list(range(response.shape[1]))
	dspec = DesignSpec(expt, trial_inds)

	dspec.addRegressorContinuous()
	dm, X, y = dspec.compileDesignMatrixFromTrialIndices()

	# split into training and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

	# keras expects the time series as [samples, timesteps, features]
	x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
	x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


	# construct the CNN model
	model = load_model(input_shape)
	if print_summary:
		model.summary()
		plot_model(model, dirs['save'] + 'model.png', show_shapes=True)

	# callbacks
	filepath = dirs['save'] + 'weights/weights_best_' + behavior_par + '.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
								 save_best_only=True, mode='auto')
	# training
	history = model.fit(x_train, y_train,
						batch_size=batch_size,
						epochs=epochs,
						validation_data=(x_test, y_test),
						callbacks=[checkpoint],
						verbose=1)

	# predict and evaluate
	_pred_train = model.predict(x_train).T[0]
	_pred_test = model.predict(x_test).T[0]

	train_accuracy = cnn.utils.evaluator(y_train, _pred_train)
	test_accuracy = cnn.utils.evaluator(y_test, _pred_test)

	pickle.dump(train_accuracy, open(dirs['save'] + behavior_par + '_train_accuracy.pkl', 'wb'), 2)
	pickle.dump(test_accuracy, open(dirs['save'] + behavior_par + '_test_accuracy.pkl', 'wb'), 2)








	w = model.get_weights()[0][:, 0, :]
	fig, ax = plt.subplots()
	nmaplt.plot_spike_filter(ax, w.mean(axis=1), 0.02, linewidth=4, color='k')
	ax.set_xlim(-5, 0)

	X = X.reshape((X.shape[0], X.shape[1], 1))

	# here we could do some prediction of responses to other stimuli
	plt.figure()
	plt.plot(response.mean(axis=1))
	_pred_train = model.predict(X).T[0]
	plt.plot(_pred_train)

	plt.figure()
	plt.plot(history.history['val_r_square'])
	plt.plot(history.history['r_square'])
	plt.title('model R^2')
	plt.ylabel('R^2')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	plt.figure()
	# plot training curve for rmse
	plt.plot(history.history['mse'])
	plt.plot(history.history['val_mse'])
	plt.title('mse')
	plt.ylabel('mse')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	# print the linear regression and display datapoints
	y_pred = model.predict(x_test).T[0]
	regressor = LinearRegression()
	regressor.fit(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
	y_fit = regressor.predict(y_pred.reshape(-1, 1))

	reg_intercept = round(regressor.intercept_[0], 4)
	reg_coef = round(regressor.coef_.flatten()[0], 4)
	reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

	plt.figure()
	plt.scatter(y_test, y_pred, color='blue', label='data')
	plt.plot(y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
	plt.title('Linear Regression')
	plt.legend()
	plt.xlabel('observed')
	plt.ylabel('predicted')
	plt.show()

	K.clear_session()


	# code to measure performance of CNN using K-fold cross-validation
	# estimators = []
	# # estimators.append(('standardize', StandardScaler()))
	# estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=20, batch_size=256, verbose=1)))
	# pipeline = Pipeline(estimators)
	# kfold = KFold(n_splits=5, shuffle=False)
	# results = cross_val_score(pipeline, x, y, cv=kfold, scoring='r2')
	# print("Visible: %.5f (%.5f)" % (results.mean(), results.std()))