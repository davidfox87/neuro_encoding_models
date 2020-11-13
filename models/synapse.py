import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple
import pickle
from glmtools.model import GLM
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy import signal
from scipy.signal import chirp, find_peaks

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
	pars['f1'] = 0.23     		# depletion rate (/ms)
	pars['tau_d1'] = 1006.  	# depletion recovery time (ms)
	#pars['g1'] = 20.     		# conductance gain
	pars['tau_c1'] = 9.3       	# conductance decay time constant (ms)

	pars['f2'] = 0.0073 			# depletion rate (/ms)
	pars['tau_d2'] = 33250.  	# depletion recovery time (ms)
	#pars['g2'] = 0.2		# conductance gain
	pars['tau_c2'] = 80.  		# conductance decay time constant (ms)

	# passive membrane equation parameters
	pars['rm'] = 800. / 1000.   # input resistance [GOhm]
	pars['e_syn'] = 10.      	# synaptic reversal potential [mV]
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


def run_passive_cell(pars: dict, gsyn: np.ndarray, iapp_: np.float) -> Tuple[np.ndarray, np.ndarray]:
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
	iapp = iapp_
	# simulate the passive neuron dynamics
	for i in range(nt - 1):
		isyn[i] = gsyn[i] * rm * (v[i] - e_syn)
		# calculate the increment of the membrane potential
		dv = (dt / tau_m) * (rm*iapp - ((v[i] - e_leak) + isyn[i]))

		# update the membrane potential
		v[i + 1] = v[i] + dv

	return v, isyn


def get_gsyn(fr: np.ndarray, pars: Dict[str, float]):

	# implement presynaptic inhibition here

	s = fr / 1000				# spike rate in spikes / ms
	# a = 1 / 400. * np.arange(500) * np.exp(1 - np.arange(500) / 400.)
	# a = a/np.sum(a)
	# #
	# # a = np.expand_dims(a, axis=0)
	# # s = np.expand_dims(s, axis=0)
	# frlen = len(fr)
	# fr = np.convolve(fr, np.flipud(a), mode='full')
	# fr = fr[:frlen]
	# s = s / (1 + 0.1 * fr)

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

	return c1.squeeze(), c2.squeeze()


def obj_fun(theta, target_psth: np.ndarray):
	"""

	:param theta: parameter vector that contains the values of g1 and g2 that we want to optimize
	:param target: the target PSTH that we would like to fit
	:return: rmse
	"""
	g1_ = theta[0]
	print('g1 is {}'.format(g1_))
	fr = np.genfromtxt('../datasets/ornfr_reverse.txt')
	pars = default_pars(T=35.0, dt=0.1, g1=g1_)

	g1, g2 = get_gsyn(fr, pars)
	g_syn = g1 + g2

	v, i_syn = run_passive_cell(pars, g_syn)

	# read in pars for GLM using pickle
	pkl_file = open('glmpars_vm_to_spiking.pkl', 'rb')
	glmpars = pickle.load(pkl_file)
	k = glmpars['k']
	h = glmpars['h']
	dc = glmpars['dc']

	# simulate a GLM using these parameters
	glm = GLM(pars['dt'], k, h, dc)

	# need to scale the output Vm for it to work properly
	v = np.apply_along_axis(lambda x: x - np.mean(x), 0, v)
	v = np.apply_along_axis(lambda x: x / np.std(x), 0, v)
	v *= 0.001

	# simulate GLM with model Vm and output a prediction of spikes
	t = np.arange(0, pars['T'], 0.001)
	sptrain = np.zeros((len(t), 5))

	for i in range(5):
		# seed random state
		np.random.seed(i)
		_, sps = glm.simulate(v)
		sptrain[:, i] = sps

	model_psth = get_psth(sptrain, 100, 100)

	model_psth_ = model_psth[5000:8000]
	target_psth_ = target_psth[5000:8000]
	model_psth_ = model_psth_[::10]
	target_psth_ = target_psth_[::10]
	# get rmse using model psth and target psth
	rmse = np.sqrt(mean_squared_error(target_psth_, model_psth_))
	print('The rmse is {} '.format(rmse))

	return rmse  # goal is to minimize this


def run(orn_fr, g1, g2, iapp, dc, dur=30):
	"""

	:param theta: parameter vector that contains the values of g1 and g2 that we want to optimize
	:param target: the target PSTH that we would like to fit
	:return: rmse
	"""
	pars = default_pars(T=dur, dt=0.1, g1=g1, g2=g2)

	g1, g2 = get_gsyn(orn_fr, pars)
	g_syn = g1 + g2

	v, i_syn = run_passive_cell(pars, g_syn, iapp)

	# read in pars for GLM using pickle
	pkl_file = open('../results/vm_to_spiking_filters/average_GLM_pars_PN.pkl', 'rb')
	glmpars = pickle.load(pkl_file)
	k = glmpars['k'][1]
	h = glmpars['h'][1]
	#dc = glmpars['dc']
	vmin = glmpars['v_min']
	vmax = glmpars['v_max']

	# simulate a GLM using these parameters
	glm = GLM(pars['dt'], k, h, dc)

	# need to scale the output Vm for it to work properly
	vm = (v - vmin) / (vmax - vmin)

	# simulate GLM with model Vm and output a prediction of spikes
	t = np.arange(0, pars['T'], 0.001)
	sptrain = np.zeros((len(t), 5))

	for i in range(5):
		# seed random state
		np.random.seed(i)
		_, sps, istm, hcurr = glm.simulate(vm)
		sptrain[:, i] = sps

	model_psth = get_psth(sptrain, 100, 100)

	return v, model_psth, istm, hcurr, sptrain, g1, g2, i_syn


def freqz(tr):
	t = np.linspace(0, 25, int(25/0.001))

	w = (0-1)/2 + (0-1)/2.*chirp(t, f0=5., f1=.1, t1=23.85, method='log') + 1
	# plt.plot(t, w)

	peaks, _ = find_peaks(-w)
	peaks = np.concatenate((0, peaks, len(w)-1), axis=None)
	# plt.plot(t[peaks], np.zeros((len(peaks), 1)), 'ok')

	frequency = 1 / (t[peaks[range(1, len(peaks))]] - t[peaks[(range(len(peaks) - 1))]])
	peaks = peaks + 5000

	amp_vec = np.zeros_like(frequency)
	max_vec = np.zeros_like(frequency)
	tpeaks_vec = np.zeros_like(frequency)

	for j in range(len(peaks)-2):
		index1, index2 = peaks[j], peaks[j+1]

		tmp = tr[index1:index2]

		order = 3
		framelen = round(len(tmp) * 0.1)
		if (framelen % 2) == 0:
			framelen = framelen + 1

		tmp = signal.savgol_filter(tmp, framelen, order)  # window size 51, polynomial order 3

		max_, min_ = np.argmax(tmp), np.argmin(tmp)
		amp_vec[j] = tmp[max_] - tmp[min_]
		max_vec[j] = tmp[max_]
		tpeaks_vec[j] = t[max_ + index1]

	return frequency, amp_vec, max_vec, tpeaks_vec


if __name__ == "__main__":
	target_psth = np.genfromtxt("../datasets/pnfr_reverse_control.txt", delimiter='\t')
	# target_psth = np.genfromtxt("../datasets/pnfr_reverse_u13AKD.txt", delimiter='\t')
	x0 = [20]
	bounds_ = [(0., None)]
	res = minimize(obj_fun, x0, args=target_psth, method='COBYLA',
				   tol=1e-6, options={'maxiter': 1000})
	theta = res['x']

	# pickle the parameters




