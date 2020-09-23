import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from utils import plot as nmaplt
import os
from cnn.preprocessing import preprocess_groups
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
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from glmtools.fit import neg_log_lik, ridgefitCV, fit_nlin_hist1d


"""
this is a script to run a CNN on fly-averaged behavior time series
"""

if __name__ == "__main__":
	np.random.seed(42)

	# CNN hyperparameters
	batch_size = 64
	epochs = 200
	input_shape = [850, 1]
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

	#
	#
	#
	#
	#
	#
	# # preprocess for the CNN to work. This is a VERY important step!
	stim_train, stim_test, resp_train, resp_test = preprocess_groups(stim, response, input_shape=input_shape)
	#
	# # show stim train and stim test
	# # plot original stimulus in top subplot
	# # do implot for train in bottom left and implot for test in bottom right
	# fig = plt.figure(constrained_layout=True)
	# spec = fig.add_gridspec(ncols=1, nrows=2)
	#
	# # show the stimulus at the top
	# # make gridspec (2, 2)
	# f_ax1 = fig.add_subplot(spec[0, 0])
	# f_ax2 = fig.add_subplot(spec[1, 0])
	# f_ax1.imshow(stim_train)
	# f_ax2.imshow(stim_test)
	#
	# # construct the CNN model
	# # load model with pretrained weights if already trained
	# filepath = None
	# if filepath is None:
	# 	model = cnn.create_model.load_model(input_shape)
	# else:
	# 	model = cnn.create_model.load_model(input_shape, trained=True, weight_path=filepath)
	#
	# if print_summary:
	# 	model.summary()
	# 	plot_model(model, dirs['save'] + 'model.png', show_shapes=True)
	#
	# es = EarlyStopping(monitor='val_loss', patience=20)
	#
	# # callbacks
	# filepath = dirs['save'] + 'weights/weights_best_' + behavior_par + '.hdf5'
	# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
	# 							 save_best_only=True, mode='auto')
	#
	# # training
	# history = model.fit(stim_train, resp_train,
	# 					batch_size=batch_size,
	# 					epochs=epochs,
	# 					validation_data=(stim_test, resp_test),
	# 					callbacks=[checkpoint, es],
	# 					verbose=1)
	#
	# saved_model = load_model(filepath=filepath,
	# 						 custom_objects={'r_square': r_square})
	#
	# # predict and evaluate
	# _pred_train = model.predict(stim_train).T[0]
	# _pred_test = model.predict(stim_test).T[0]
	#
	# train_accuracy = cnn.utils.evaluator(resp_train, _pred_train)
	# test_accuracy = cnn.utils.evaluator(resp_test, _pred_test)
	#
	# pickle.dump(train_accuracy, open(dirs['save'] + behavior_par + '_train_accuracy.pkl', 'wb'), 2)
	# pickle.dump(test_accuracy, open(dirs['save'] + behavior_par + '_test_accuracy.pkl', 'wb'), 2)
	#
	# fig, ax = plt.subplots()
	# cnn.utils.plot_weights(ax, saved_model, 0.02, linewidth=4, color='k')
	# ax.set_xlim(-5, 0)
	#
	#
	# # predict and evaluate
	# nt_train, nt = len(stim_train), len(stim)
	# time_train = np.arange(nt_train) * 0.02
	# time_test = np.arange(nt_train, nt) * 0.02
	#
	# plt.figure()
	# _pred_train = saved_model.predict(stim_train)
	# _pred_test = saved_model.predict(stim_test)
	# plt.plot(time_train, resp_train.squeeze())
	# plt.plot(time_test, resp_test.squeeze())
	# plt.plot(time_train, _pred_train)
	# plt.plot(time_test, _pred_test)
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
	#
	# # compare CNN with ridge regression first on the mean trace. We will deal with multiple flis later
	# stim_train, stim_test = stim_train.squeeze(), stim_test.squeeze()
	# model = Ridge()
	# alphas = np.logspace(0, 30, num=20, base=2)
	# param_search = [{'alpha': alphas}]
	#
	# tscv = TimeSeriesSplit(n_splits=5)
	# grid_result = GridSearchCV(estimator=model, cv=tscv,
    #                     param_grid=param_search, scoring='neg_mean_squared_error')
	# grid_result.fit(stim_train, resp_train)
	# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	#
	# # using this ridge penalty value, get the mse between resp_test and ridge prediction
	# model = Ridge(alpha=grid_result.best_params_['alpha']).fit(stim_train, resp_train)
	#
	#
	#
	#
	#
	#
	# # the mse has to be evaluated not on the held out test set but on the entire prediction
	# # and the actual measurement
	# print("Ridge mse on train test set is: ", mean_squared_error(resp_train, model.predict(stim_train)))
	# print("Ridge mse on held out test set is: ", mean_squared_error(resp_test, model.predict(stim_test)))
	#
	# print("CNN mse on train set is: ", train_accuracy[1])
	# print("CNN mse on held out test set is: ", test_accuracy[1])
	#
	#
	# w = model.coef_[0]
	# d = len(w)
	# t = np.arange(-d + 1, 1) * 0.02
	#
	# # compare ridge and CNN filters
	# fig, ax = plt.subplots()
	# cnn.utils.plot_weights(ax, saved_model, 0.02, linewidth=4, color='k')
	# ax.set_xlim(-5, 0)
	#
	# ax.plot(t, w, 'b', linewidth=5)
	#
	# xx, fnlin, rawfilteroutput = fit_nlin_hist1d(stim, response, w, 0.02, 100)
	#
	# # compare with CNN
	# plt.plot(_pred_train)
	# plt.plot(fnlin(rawfilteroutput))
	# # compare with actual
	# plt.plot(response)
	#
	# # compare mse
	#
	# # #
	# # # #
	# # # # #model = KerasRegressor(build_fn=load_model, epochs=100, batch_size=64, verbose=1)
	# # # # #accuracies = cross_val_score(estimator=model, X=x, y=y, scoring='r2', cv=5, n_jobs=-1)
	# # # # #mean = accuracies.mean()
	# # # # #variance = accuracies.std()
	# # # #
	# # # # # callbacks
	# # # # filepath = dirs['save'] + 'weights/weights_best_' + behavior_par + '.hdf5'
	# # # # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
	# # # # 							 save_best_only=True, mode='auto')
	# # # #
	# # # # model = load_model(input_shape)
	# # # #
	# # # # # training
	# # # # history = model.fit(X, y,
	# # # # 					batch_size=batch_size,
	# # # # 					epochs=epochs,
	# # # # 					callbacks=[checkpoint],
	# # # # 					verbose=1)
	# # # #
	# # # # # predict and evaluate
	# # # # _pred = model.predict(x)
	# # # # plt.plot(response.mean(axis=1))
	# # # # plt.plot(_pred[:1250])
	# # # #
	# # # #
	# # # #
	# # # # w = model.get_weights()[0][:, 0, :]
	# # # # fig, ax = plt.subplots()
	# # # # nmaplt.plot_spike_filter(ax, w.mean(axis=1), 0.02, linewidth=4, color='k')
	# # # # ax.set_xlim(-5, 0)
	# # # #
	# # # # #
	# # # # #
	# # # # # # construct the CNN model
	# # # # # model = load_model(input_shape)
	# # # # # if print_summary:
	# # # # # 	model.summary()
	# # # # # 	plot_model(model, dirs['save'] + 'model.png', show_shapes=True)
	# # # # #
	# # # # # # callbacks
	# # # # # filepath = dirs['save'] + 'weights/weights_best_' + behavior_par + '.hdf5'
	# # # # # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
	# # # # # 							 save_best_only=True, mode='auto')
	# # # # # # training
	# # # # # history = model.fit(x_train, y_train,
	# # # # # 					batch_size=batch_size,
	# # # # # 					epochs=epochs,
	# # # # # 					validation_data=(x_test, y_test),
	# # # # # 					callbacks=[checkpoint],
	# # # # # 					verbose=1)
	# # # # #
	# # # # # predict and evaluate
	# # _pred_train = model.predict(x_train)
	# # _pred_test = model.predict(x_test)
	# #
	# # train_accuracy = cnn.utils.evaluator(y_train, _pred_train)
	# # test_accuracy = cnn.utils.evaluator(y_test, _pred_test)
	# # # #
	# # # # pickle.dump(train_accuracy, open(dirs['save'] + behavior_par + '_train_accuracy.pkl', 'wb'), 2)
	# # # # pickle.dump(test_accuracy, open(dirs['save'] + behavior_par + '_test_accuracy.pkl', 'wb'), 2)
	# # # #
	# # # #
	# # #
	# # #
	# # #
	# # #
	# # #
	# # #
	# # # w = model.get_weights()[0][:, 0, :]
	# # # fig, ax = plt.subplots()
	# # # nmaplt.plot_spike_filter(ax, w.mean(axis=1), 0.02, linewidth=4, color='k')
	# # # ax.set_xlim(-5, 0)
	# # #
	# # X = X.reshape((X.shape[0], X.shape[1], 1))
	# # #
	# # # # here we could do some prediction of responses to other stimuli
	# # plt.figure()
	# # plt.plot(response.mean(axis=1))
	# # _pred_train = model.predict(X).T[0]
	# # plt.plot(_pred_train)
	# #
	# #
	#
	# #
	# # # print the linear regression and display datapoints
	# # y_pred = model.predict(x_test).T[0]
	# # regressor = LinearRegression()
	# # regressor.fit(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
	# # y_fit = regressor.predict(y_pred.reshape(-1, 1))
	# #
	# # reg_intercept = round(regressor.intercept_[0], 4)
	# # reg_coef = round(regressor.coef_.flatten()[0], 4)
	# # reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)
	# #
	# # plt.figure()
	# # plt.scatter(y_test, y_pred, color='blue', label='data')
	# # plt.plot(y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
	# # plt.title('Linear Regression')
	# # plt.legend()
	# # plt.xlabel('observed')
	# # plt.ylabel('predicted')
	# # plt.show()
	# #
	# # K.clear_session()
	#