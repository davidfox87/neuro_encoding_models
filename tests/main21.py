

import utils.read as io
import numpy as np
from matplotlib import pyplot as plt
from glmtools.make_xdsgn import Experiment, DesignSpec
import pickle
import numpy as np
from matplotlib import pyplot as plt

"""
Refer to Pillow et al., 2005 for more info.
This script fits model parameters, K, h, dc of a 
Poisson Generalized linear model.

"""
if __name__ == "__main__":

	stim_filter = []
	postspike_filter = []
	dc = []
	v_min = []
	v_max = []

	cells = [1, 2, 3, 4, 5, 7, 9]
	for i in cells:
		pkl_file = open('../results/glmpars_vm_to_spiking_PN' + str(i) + '.pkl', 'rb')
		glmpars = pickle.load(pkl_file)
		stim_filter.append(glmpars['k'][1])
		postspike_filter.append(glmpars['h'][1])
		dc.append(glmpars['dc'])
		v_max.append(glmpars['v_max'])
		v_min.append(glmpars['v_min'])


	v_min = np.array(v_min)
	v_max = np.array(v_max)
	stim_filter = np.array(stim_filter)
	kt = glmpars['k'][0]

	mu = stim_filter.mean(axis=0)
	sigma = stim_filter.std(axis=0)

	fig, ax = plt.subplots(1)
	ax.plot(kt, mu, lw=2, label='mean population 1', color='blue')
	ax.fill_between(kt, mu + sigma, mu - sigma, facecolor='blue', alpha=0.5)



	# post-spike filter
	postspike_filter = np.array(postspike_filter)
	ht = glmpars['h'][0]

	mu2 = postspike_filter.mean(axis=0)
	sigma2 = postspike_filter.std(axis=0)

	fig, ax = plt.subplots(1)
	ax.plot(ht, mu2, lw=2, label='mean population 1', color='blue')
	ax.fill_between(ht, mu2 + sigma2, mu2 - sigma2, facecolor='blue', alpha=0.5)

	plt.rcParams['font.size'] = 16

	n = len(cells)
	figure, ax = plt.subplots(2, 2)
	ax[0, 0].plot(np.tile(kt, (n, 1)).T, stim_filter.T, 'k', alpha=0.5)
	ax[0, 1].plot(np.tile(ht, (n, 1)).T, postspike_filter.T, 'k', alpha=0.5)
	ax[0, 0].axhline(0, color=".2", linestyle="--", zorder=1)
	ax[0, 1].axhline(0, color=".2", linestyle="--", zorder=1)

	ax[1, 0].plot(kt, mu, lw=2, label='mean population 1', color='blue')
	ax[1, 0].fill_between(kt, mu + sigma, mu - sigma, facecolor='blue', alpha=0.5)
	ax[1, 0].axhline(0, color=".2", linestyle="--", zorder=1)

	ax[1, 1].plot(ht, mu2, lw=2, label='mean population 1', color='blue')
	ax[1, 1].fill_between(ht, mu2 + sigma2, mu2 - sigma2, facecolor='blue', alpha=0.5)
	ax[1, 1].axhline(0, color=".2", linestyle="--", zorder=1)

	figure.suptitle('GLM parameters fit to 9 different PN recordings across all trials/stimuli')

	ax[0, 0].set_xlim(-0.15, 0)
	ax[1, 0].set_xlim(-0.15, 0)
	ax[0, 1].set_xlim(0, 0.3)
	ax[1, 1].set_xlim(0, 0.3)
	ax[0, 0].set_xticklabels([])
	ax[0, 1].set_xticklabels([])

	ax[1, 0].set_xlabel('Time before spike (s)')
	ax[1, 1].set_xlabel('Time after spike (s)')
	ax[0, 0].set_title('membrane potential Filter across cells')
	ax[0, 1].set_title('post-spike Filter across cells')
	ax[1, 0].set_title('Average membrane potential Filter \n(mean +/- sd)')
	ax[1, 1].set_title('Average post-spike Filter \n(mean +/- sd)')

	plt.tight_layout()

	dc = np.array(dc)
	mu_dc = dc.mean(axis=0)
	sigma_dc = dc.std(axis=0)

	plt.figure()
	# plt.plot(np.tile(1, (9, 1)), dc, 'o', markersize=16)
	plt.plot([0.98, 1.02], [mu_dc, mu_dc], '-k')
	plt.plot([1, 1], [mu_dc - sigma_dc, mu_dc + sigma_dc], 'k')
	plt.errorbar(np.tile(1, (n, 1)), dc, yerr=sigma_dc, fmt='ok', markersize=16)
	plt.suptitle('Distribution of dc values')
	plt.tight_layout()



	# save average filter
	data = {'k': (kt, mu),
			'h': (ht, mu2),
			'dc': mu_dc,
			'v_min': v_min.mean(),
			'v_max': v_max.mean()}

	output = open('../results/vm_to_spiking_filters/average_GLM_pars_PN.pkl', 'wb')

	# pickle dictionary using protocol 0
	pickle.dump(data, output)

