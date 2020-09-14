import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
from basisFactory.bases import Basis, RaisedCosine
from scipy.optimize import minimize
from glmtools.fit import x_proj
from utils import plot as nmaplt
import pickle
from numpy import linalg as LA

if __name__ == "__main__":


	pkl_file = open('../datasets/behavior/ridge_filters/vymoves_filter_PN_to_behavior_control.pkl', 'rb')
	vymoves_pars = pickle.load(pkl_file)
	k_vymove = vymoves_pars['k'] / (LA.norm(vymoves_pars['k']))
	xx, nlfun_vymoves = vymoves_pars['nlfun']

	pkl_file = open('../datasets/behavior/ridge_filters/vymoves_filter_PN_to_behavior_u13AKD.pkl', 'rb')
	vymoves_pars = pickle.load(pkl_file)
	k_vymove_u13AKD = vymoves_pars['k'] / (LA.norm(vymoves_pars['k']))
	xx_, nlfun_vymoves_u13AKD = vymoves_pars['nlfun']

	fig = plt.figure()
	ax2 = plt.subplot(121)
	nmaplt.plot_spike_filter(ax2, k_vymove, 0.01, linewidth=3, label="Control", color='k')
	nmaplt.plot_spike_filter(ax2, k_vymove_u13AKD, 0.01, linewidth=3, label="Unc13A KD", color='m')
	ax2.set_ylim(-0.05, 0.2)
	ax2.set_xlim(-3, 0)


	ax5 = plt.subplot(122)
	ax5.plot(xx, nlfun_vymoves, 'k')
	ax5.plot(xx_, nlfun_vymoves_u13AKD, 'm')



