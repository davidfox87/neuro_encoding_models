import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from utils import plot as nmaplt
import os
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
# custom R2-score metrics for keras backend
from keras import backend as K
from keras.utils import plot_model
from sklearn.linear_model import LinearRegression
from cnn.preprocessing import preprocess
import cnn.utils
from keras.wrappers.scikit_learn import KerasRegressor
from cnn.create_model import r_square
import cnn.create_model
from keras.models import load_model


"""
this is a script to run a CNN on fly-averaged behavior time series
"""

if __name__ == "__main__":
	np.random.seed(42)

	# CNN hyperparameters
	batch_size = 64
	epochs = 60
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
	behavior_par = behaviors[1]

	# load the data from MATLAB .mat file
	stim, response = io.load_behavior('../datasets/behavior/control_behavior.mat', 30., 55., behavior_par, 50)
	response = response.mean(axis=1)  # work on the fly-average
	stim = stim[:, 0]
	# preprocess for the CNN to work. This is a VERY important step!
	stim_train, stim_test, resp_train, resp_test = preprocess(stim, response, input_shape)

	# construct the CNN model
	model = cnn.create_model.load_model(input_shape)
	if print_summary:
		model.summary()
		plot_model(model, dirs['save'] + 'model.png', show_shapes=True)

	es = EarlyStopping(monitor='val_loss', patience=20)

	# callbacks
	filepath = dirs['save'] + 'weights/weights_best_' + behavior_par + '.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
								 save_best_only=True, mode='auto')

	# training
	history = model.fit(stim_train, resp_train,
						batch_size=batch_size,
						epochs=epochs,
						validation_data=(stim_test, resp_test),
						callbacks=[checkpoint, es],
						verbose=1)

	saved_model = load_model(filepath=filepath,
							 custom_objects={'r_square': r_square})

	# predict and evaluate
	_pred_train = model.predict(stim_train).T[0]
	_pred_test = model.predict(stim_test).T[0]

	train_accuracy = cnn.utils.evaluator(resp_train, _pred_train)
	test_accuracy = cnn.utils.evaluator(resp_test, _pred_test)

	pickle.dump(train_accuracy, open(dirs['save'] + behavior_par + '_train_accuracy.pkl', 'wb'), 2)
	pickle.dump(test_accuracy, open(dirs['save'] + behavior_par + '_test_accuracy.pkl', 'wb'), 2)

	fig, ax = plt.subplots()
	cnn.utils.plot_weights(ax, saved_model, 0.02, linewidth=4, color='k')
	ax.set_xlim(-4, 0)


	# keras expects the time series as [samples, timesteps, features]
	# X = cnn.preprocessing.preprocess_stim(stim, input_shape)
	# resp = cnn.preprocessing.preprocess_resp(response)
	# predict and evaluate
	_pred = saved_model.predict(stim_train)
	plt.plot(resp_train)
	plt.plot(_pred)
	# #
	# #
	# # #
	# # # #model = KerasRegressor(build_fn=load_model, epochs=100, batch_size=64, verbose=1)
	# # # #accuracies = cross_val_score(estimator=model, X=x, y=y, scoring='r2', cv=5, n_jobs=-1)
	# # # #mean = accuracies.mean()
	# # # #variance = accuracies.std()
	# # #
	# # # # callbacks
	# # # filepath = dirs['save'] + 'weights/weights_best_' + behavior_par + '.hdf5'
	# # # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
	# # # 							 save_best_only=True, mode='auto')
	# # #
	# # # model = load_model(input_shape)
	# # #
	# # # # training
	# # # history = model.fit(X, y,
	# # # 					batch_size=batch_size,
	# # # 					epochs=epochs,
	# # # 					callbacks=[checkpoint],
	# # # 					verbose=1)
	# # #
	# # # # predict and evaluate
	# # # _pred = model.predict(x)
	# # # plt.plot(response.mean(axis=1))
	# # # plt.plot(_pred[:1250])
	# # #
	# # #
	# # #
	# # # w = model.get_weights()[0][:, 0, :]
	# # # fig, ax = plt.subplots()
	# # # nmaplt.plot_spike_filter(ax, w.mean(axis=1), 0.02, linewidth=4, color='k')
	# # # ax.set_xlim(-5, 0)
	# # #
	# # # #
	# # # #
	# # # # # construct the CNN model
	# # # # model = load_model(input_shape)
	# # # # if print_summary:
	# # # # 	model.summary()
	# # # # 	plot_model(model, dirs['save'] + 'model.png', show_shapes=True)
	# # # #
	# # # # # callbacks
	# # # # filepath = dirs['save'] + 'weights/weights_best_' + behavior_par + '.hdf5'
	# # # # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
	# # # # 							 save_best_only=True, mode='auto')
	# # # # # training
	# # # # history = model.fit(x_train, y_train,
	# # # # 					batch_size=batch_size,
	# # # # 					epochs=epochs,
	# # # # 					validation_data=(x_test, y_test),
	# # # # 					callbacks=[checkpoint],
	# # # # 					verbose=1)
	# # # #
	# # # # predict and evaluate
	# _pred_train = model.predict(x_train)
	# _pred_test = model.predict(x_test)
	#
	# train_accuracy = cnn.utils.evaluator(y_train, _pred_train)
	# test_accuracy = cnn.utils.evaluator(y_test, _pred_test)
	# # #
	# # # pickle.dump(train_accuracy, open(dirs['save'] + behavior_par + '_train_accuracy.pkl', 'wb'), 2)
	# # # pickle.dump(test_accuracy, open(dirs['save'] + behavior_par + '_test_accuracy.pkl', 'wb'), 2)
	# # #
	# # #
	# #
	# #
	# #
	# #
	# #
	# #
	# # w = model.get_weights()[0][:, 0, :]
	# # fig, ax = plt.subplots()
	# # nmaplt.plot_spike_filter(ax, w.mean(axis=1), 0.02, linewidth=4, color='k')
	# # ax.set_xlim(-5, 0)
	# #
	# X = X.reshape((X.shape[0], X.shape[1], 1))
	# #
	# # # here we could do some prediction of responses to other stimuli
	# plt.figure()
	# plt.plot(response.mean(axis=1))
	# _pred_train = model.predict(X).T[0]
	# plt.plot(_pred_train)
	#
	# plt.figure()
	# plt.plot(history.history['val_r_square'])
	# plt.plot(history.history['r_square'])
	# plt.title('model R^2')
	# plt.ylabel('R^2')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# plt.show()
	#
	# plt.figure()
	# # plot training curve for rmse
	# plt.plot(history.history['mse'])
	# plt.plot(history.history['val_mse'])
	# plt.title('mse')
	# plt.ylabel('mse')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# plt.show()
	#
	# # print the linear regression and display datapoints
	# y_pred = model.predict(x_test).T[0]
	# regressor = LinearRegression()
	# regressor.fit(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
	# y_fit = regressor.predict(y_pred.reshape(-1, 1))
	#
	# reg_intercept = round(regressor.intercept_[0], 4)
	# reg_coef = round(regressor.coef_.flatten()[0], 4)
	# reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)
	#
	# plt.figure()
	# plt.scatter(y_test, y_pred, color='blue', label='data')
	# plt.plot(y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
	# plt.title('Linear Regression')
	# plt.legend()
	# plt.xlabel('observed')
	# plt.ylabel('predicted')
	# plt.show()
	#
	# K.clear_session()
	#
	#
	# # code to measure performance of CNN using K-fold cross-validation
	# # estimators = []
	# # # estimators.append(('standardize', StandardScaler()))
	# # estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=20, batch_size=256, verbose=1)))
	# # pipeline = Pipeline(estimators)
	# # kfold = KFold(n_splits=5, shuffle=False)
	# # results = cross_val_score(pipeline, x, y, cv=kfold, scoring='r2')
	# # print("Visible: %.5f (%.5f)" % (results.mean(), results.std()))