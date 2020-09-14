import pickle

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.metrics import r2_score

def evaluator(y_data, y_pred):
    """
    evaluate the regression performance by Explained Variance, Mean Squared Error,
    R^2, and Pearson Correlation.

    y_data: true values (1D)
    y_pred: predicted values (1D)
    """

    evs = explained_variance_score(y_data, y_pred)
    mse = mean_squared_error(y_data, y_pred)
    r2 = r2_score(y_data, y_pred)
    pr = stats.pearsonr(y_data, y_pred)[0]
    return evs, mse, r2, pr

def plot_history(history, save_dir, cell):
    """
    plot the training curve (epoch vs. loss)

    history (history object): training history
    save_dir (str): directory to save the figure
    cell (int): cell ID
    """

    plt.figure()
    plt.plot(history.history['loss'], 'b')
    plt.plot(history.history['val_loss'], 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'])
    plt.savefig(save_dir + 'history.png')
    plt.close()
    fname = save_dir + 'history_' + str(cell) + '.pkl'
    pickle.dump(history.history, open(fname, 'wb'), 2)

def plot_accuracy(acc, save_dir):
    """
    plot the distribution of regression performances

    acc (numpy array): accuracies to plot, (n_cells, 4)
    save_dir (str): directory to save the figure
    """

    fig = plt.figure(figsize=(20, 3))
    x_labels = ['explained variance', 'mean squared error', 'r2', 'pearson r']
    for j in range(4):
        tmp = acc[:, j]
        ax = plt.subplot(1, 4, j + 1)
        ax.hist(tmp[~np.isnan(tmp)], bins=20)
        ax.set_xlabel(x_labels[j])
    plt.savefig(save_dir + 'accuracy_distribution.png')
    plt.savefig(save_dir + 'accuracy_distribution.pdf')
    plt.close()

def plot_prediction(_y_test, _pred_test, save_dir, cell):
    """
    scatter plot the regression performance

    _y_test: true values (1D)
    _pred_test: predicted values (1D)
    save_dir (str): directory to save the figure
    cell (int): cell ID
    """

    m, M = np.min(_y_test), np.max(_y_test)
    plt.figure()
    plt.scatter(_y_test, _pred_test)
    plt.plot([m, M], [m, M], 'k--', lw=4)
    plt.xlabel('measured response')
    plt.ylabel('predicted response')
    plt.savefig(save_dir + 'plot_prediction_' + str(cell) + '.png')
    plt.savefig(save_dir + 'plot_prediction_' + str(cell) + '.pdf')
    plt.close()