import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple
import pickle
from glmtools.model import GLM
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


def get_psth(sptrain, winsize, samprate):
	smooth_win = np.hanning(winsize) / np.hanning(winsize).sum()
	# for multiple trials, average spikes across time bins and then convolve
	psth = np.convolve(np.mean(sptrain, axis=1), smooth_win)
	sigrange = np.arange(winsize // 2, winsize // 2 + len(sptrain))
	psth_ = psth[sigrange]*samprate
	return psth_

def default_pars(**kwargs) -> Dict[str, float]:
	pars = {}

	# synaptic parameters
	pars['f1'] = 0.0023     		# depletion rate (/ms)
	pars['tau_d1'] = 1006.  	# depletion recovery time (ms)
	#pars['g1'] = 20.     		# conductance gain
	pars['tau_c1'] = 9.3       	# conductance decay time constant (ms)

	pars['f2'] = 0.0073 		# depletion rate (/ms)
	pars['tau_d2'] = 33247.  	# depletion recovery time (ms)
	pars['g2'] = 1.97			# conductance gain
	pars['tau_c2'] = 280.  		# conductance decay time constant (ms)

	# passive membrane equation parameters
	pars['rm'] = 800. / 1000.   # input resistance [GOhm]
	pars['e_syn'] = -10.      	# synaptic reversal potential [mV]
	pars['e_leak'] = -70.       # leak reversal potential [mV]
	pars['tau_m'] = 5.			# membrane time constant [ms]
	pars['v_init'] = -70.  		# initial potential [mV]

	# simulation parameters
	#pars['T'] = 400. # Total duration of simulation [ms]
	#pars['dt'] = .1  # Simulation time step [ms]

	# external parameters if any #
	for k in kwargs:
		pars[k] = kwargs[k]

	pars['t'] = np.arange(0, pars['T'], pars['dt'])   # time points [ms]

	return pars


def run_passive_cell(pars: dict, gsyn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simulate the passive membrane eq with synaptic current

	Args:
	pars       : parameter dictionary
	gsyn       : synaptic conductance resampled to 1 Khz
				The injected current is an ndarray

	Returns:
	Vm 			: membrane potential
	isyn      	: synaptic current
	"""

	# Set parameters
	tau_m, rm = pars['tau_m'], pars['rm']
	v_init, e_leak = pars['v_init'], pars['e_leak']
	e_syn = pars['e_syn']

	dt = pars['dt']
	nt = gsyn.shape[0]

	t = np.arange(0, nt) * dt

	# Initialize voltage and current
	v = np.zeros(nt)
	isyn = np.zeros(nt)

	v[0] = v_init
	#plt.plot(gsyn)
	# simulate the passive neuron dynamics
	for i in range(nt - 1):
		isyn[i] = gsyn[i] * rm * (v[i] - e_syn)

		# calculate the increment of the membrane potential
		dv = -(dt / tau_m) * (v[i] - e_leak + isyn[i])

		# update the membrane potential
		v[i + 1] = v[i] + dv

	return v, isyn


def get_gsyn(fr: np.ndarray, pars: Dict[str, float]):

	s = fr / 1000				# spike rate in spikes / ms

	f1 = pars['f1']
	tau_d1 = pars['tau_d1']
	g1 = pars['g1']
	tau_c1 = pars['tau_c1']

	f2 = pars['f2']
	tau_d2 = pars['tau_d2']
	g2 = pars['g2']
	tau_c2 = pars['tau_c2']

	a1 = np.zeros_like(s)
	c1 = np.zeros_like(s)
	a2 = np.zeros_like(s)
	c2 = np.zeros_like(s)

	a1[0] = 1. / (tau_d1 * f1 * s[0] + 1)
	c1[0] = a1[0] * s[0] * g1 * tau_c1
	a2[0] = 1. / (tau_d2 * f2 * s[0] + 1)
	c2[0] = a2[0] * s[0] * g2 * tau_c2

	for i in range(len(s)-1):
		a1[i + 1] = a1[i] + (-f1 * s[i] * a1[i] + (1 - a1[i]) / tau_d1) 	# depletion
		c1[i + 1] = c1[i] + (g1 * a1[i] * s[i] - c1[i] / tau_c1) 			# conductance

		a2[i + 1] = a2[i] + (-f2 * s[i] * a2[i] + (1 - a2[i]) / tau_d2) 	# depletion
		c2[i + 1] = c2[i] + (g2 * a2[i] * s[i] - c2[i] / tau_c2) 			# conductance

	# down-sample
	#c1, c2 = c1[::10], c2[::10]

	return c1, c2


def obj_fun(theta, target_psth: np.ndarray):
	"""

	:param theta: parameter vector that contains the values of g1 and g2 that we want to optimize
	:param target: the target PSTH that we would like to fit
	:return: rmse
	"""
	g1_ = theta[0]
	#g2_ = theta[1]
	#dc = theta[2]
	print('g1 is {}'.format(g1_))

	fr = np.genfromtxt('../datasets/ornfr_reverse.txt')
	#fr = fr[::10]
	pars = default_pars(T=35.0, dt=0.1, g1=g1_)

	g1, g2 = get_gsyn(fr, pars)
	g_syn = g1

	v, i_syn = run_passive_cell(pars, g_syn)

	# read in pars for GLM using pickle
	pkl_file = open('../tests/glmpars_vm_to_spiking.pkl', 'rb')
	glmpars = pickle.load(pkl_file)
	k = glmpars['k']
	h = glmpars['h']
	dc = 2.13

	# simulate a GLM using these parameters
	glm = GLM(pars['dt'], k, h, dc)

	# need to scale the output Vm for it to work properly
	v = np.apply_along_axis(lambda x: x - np.mean(x), 0, v)
	v = np.apply_along_axis(lambda x: x / np.std(x), 0, v)
	v *= 0.01

	# simulate GLM with model Vm and output a prediction of spikes
	t = np.arange(0, pars['T'], 0.001)
	sptrain = np.zeros((len(t), 5))

	for i in range(5):
		# seed random state
		np.random.seed(i)
		_, sps = glm.simulate(v)
		sptrain[:, i] = sps

	model_psth = get_psth(sptrain, 100, 100)

	model_psth_ = model_psth[5000:10000]
	target_psth_ = target_psth[5000:10000]
	model_psth_ = model_psth_[::10]
	target_psth_ = target_psth_[::10]
	# get rmse using model psth and target psth
	rmse = np.sqrt(mean_squared_error(target_psth_, model_psth_))
	print('The rmse is {} '.format(rmse))

	return rmse  # goal is to minimize this


if __name__ == "__main__":
	# read in firing-rate
	# fr = np.genfromtxt('../datasets/ornfr_reverse.txt')
	# #fr = fr[::10]
	# pars = default_pars(T=35.0, dt=0.1, g1=20, g2=1.8)
	#
	# g1, g2 = get_gsyn(fr, pars)
	# g_syn = g1 + g2
	#
	# v, i_syn = run_passive_cell(pars, g_syn)
	#
	# # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
	# t = np.arange(0, pars['T'], 0.001)
	# # ax1.plot(t, g1, 'm', linewidth=0.5)
	# # ax1.plot(t, g2, 'c', linewidth=0.5)
	# # ax2.plot(t, i_syn, 'k', linewidth=0.5)
	# # ax3.plot(t, v, 'k', linewidth=0.5)
	# # ax4.plot(t, fr, 'k', linewidth=0.5)
	#
	# # read in pars for GLM using pickle
	# pkl_file = open('../tests/glmpars_vm_to_spiking.pkl', 'rb')
	# glmpars = pickle.load(pkl_file)
	# k = glmpars['k']
	# h = glmpars['h']
	# dc = glmpars['dc']
	#
	# # simulate a GLM using these parameters
	# glm = GLM(pars['dt'], k, h, dc)
	#
	# # need to scale the output Vm for it to work properly
	# # v = (v / abs(np.max(v))) * 0.01
	# v = np.apply_along_axis(lambda x: x - np.mean(x), 0, v)
	# v = np.apply_along_axis(lambda x: x / np.std(x), 0, v)
	# v *= 0.01
	#
	# # simulate GLM with model Vm and output a prediction of spikes
	# sptimes = np.zeros((len(t), 5))
	# sptrain = np.zeros((len(t), 5))
	# for i in range(5):
	# 	np.random.seed(i)
	# 	_, sps = glm.simulate(v)
	# 	sptimes[:, i] = sps * t
	# 	sptrain[:, i] = sps
	#
	# #plt.eventplot(sptimes.T, linewidth=0.5)
	# psth = get_psth(sptrain, 100, 100)
	#
	# # output GLM parameters: k, h, dc
	# data = {'model_psth': psth}
	#
	# output = open('model_psth.pkl', 'wb')
	#
	# # pickle dictionary using protocol 0
	# pickle.dump(data, output)
	#
	#
	# # normalize
	# #psth /= np.max(np.abs(psth), axis=0)
	# #plt.plot(t, (5*psth)-0.5, 'k', linewidth=0.5)
	#
	# # see if we can recover the original parameters of 40 and 1.8.
	# # x0 = [2.]
	# # res = minimize(obj_fun, x0, args=psth, method='COBYLA', tol=1e-3)
	# # theta = res['x']

	#target_psth = np.genfromtxt("../datasets/pnfr_reverse_control.txt", delimiter='\t')
	target_psth = np.genfromtxt("../datasets/pnfr_reverse_u13AKD.txt", delimiter='\t')
	#x0 = [18.05932, 1.9742, 3.] # f1, tau_d1, g1, tau_c1
	x0 = [0.03]
	#bounds_ = [(0., None), (0., None), (0., 5.)]
	bounds_ = [(0., None)]
	res = minimize(obj_fun, x0, args=target_psth, method='COBYLA',
				   tol=1e-6, options={'maxiter': 1000})
	theta = res['x']




