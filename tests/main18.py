from models import synapse
from utils.read import read_orn_fr
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat
from models.synapse import freqz
import matplotlib.gridspec as gridspec
import pickle

def plot_model_output(orn_responses, g1_, g2_, iapp, dur, dc):

	# run each orn_fr through the synaptic model and save the output pn firing rate
	model_pn_responses = np.zeros_like(orn_responses)
	vm = np.zeros_like(orn_responses)
	g1 = np.zeros_like(orn_responses)
	g2 = np.zeros_like(orn_responses)
	i_syn = np.zeros_like(orn_responses)

	cmap = plt.get_cmap('Greys', 8)

	fig, ax = plt.subplots(5, 1, sharex=True)
	for i in range(orn_responses.shape[1]):
		# fig, ax = plt.subplots(5, 1, sharex=True)
		vm[:, i], model_pn_responses[:, i], istm, hcurr, sps, g1[:, i], g2[:, i], i_syn[:, i] = synapse.run(
		 	orn_responses[:, i], g1_, g2_, dc=dc, iapp=iapp, dur=dur)
		ax[0].plot(orn_responses[:, i], color=cmap(i+1))
		ax[0].set_title('ORN Firing-rate input to synaptic model')
		ax[0].set_ylabel('(Hz)')
		ax[1].set_title('Conductances')
		ax[1].plot(g1[:, i], 'b', label='g1')
		ax[1].plot(g2[:, i], 'r', label='g2')
		# ax[2].plot(istm, 'b', label='k output')
		# ax[2].plot(hcurr, 'r', label='h output')
		# ax[2].set_ylabel('Filter outputs')
		if i == 0:
			ax[1].legend(loc='upper right')
		#ax[2].axhline(0, color=".2", linestyle="--", zorder=1)


		#ax[4].legend(loc='upper right')
		ax[2].plot(vm[:, i], color=cmap(i+1))
		ax[2].set_ylabel('Vm (mV)')
		ax[2].set_title('Membrane potential')

		# ax[3].set_title('Synaptic Current')
		# ax[3].plot(i_syn[:, i], color=cmap(i))

		ax[3].eventplot(sps[:, 0] * np.arange(len(sps[:, 0])), linewidths=0.5)
		ax[4].set_title('firing-rate')
		ax[4].plot(orn_responses[:, i], color=[0.82, 0.82, 0.82], label='ORN')
		ax[4].plot(model_pn_responses[:, i], 'k', label='Model PN')
		ax[4].set_ylabel('(Hz)')
		fig.tight_layout()

	return model_pn_responses


if __name__ == "__main__":

	fs = 1000
	orn_responses = read_orn_fr('../datasets/neural/meanORNFr.mat', 'pulses')
	g2 = 1.8

	control_g1 = 25.
	iapp_control = -27.
	iapp_unc13 = -3.
	control_dc = -8.5
	unc13_dc = -8
	control_pn_responses = plot_model_output(orn_responses, control_g1, g2, iapp=iapp_control, dur=30., dc=control_dc)

	unc13_g1 = 0
	unc13_pn_responses = plot_model_output(orn_responses, unc13_g1, g2, iapp=iapp_unc13, dur=30., dc=unc13_dc)

	mean_orn_responses = np.ones((7, 1))
	mean_pn_responses = np.ones((7, 1))
	mean_pn_responses_unc13 = np.ones((7, 1))
	for i in range(7):
		mean_orn_responses[i] = np.mean(orn_responses[int(5 * fs): int(7 * fs), i])
		mean_pn_responses[i] = np.mean(control_pn_responses[int(5 * fs): int(7 * fs), i])
		mean_pn_responses_unc13[i] = np.mean(unc13_pn_responses[int(5 * fs): int(7 * fs), i])

	fig2 = plt.figure(constrained_layout=True)
	spec2 = gridspec.GridSpec(ncols=4, nrows=6, figure=fig2)

	f2_ax0_1 = fig2.add_subplot(spec2[2:4, 0])
	f2_ax0_2 = fig2.add_subplot(spec2[4:, 0])
	pkl_file = open('../results/vm_to_spiking_filters/average_GLM_pars_PN.pkl', 'rb')
	glmpars = pickle.load(pkl_file)
	k = glmpars['k'][1]
	h = glmpars['h'][1]
	kt = glmpars['k'][0]
	ht = glmpars['h'][0]
	f2_ax0_1.plot(kt, k)
	f2_ax0_2.plot(ht, h)
	f2_ax0_1.plot([kt[0], kt[-1]], [0, 0], '--k')
	f2_ax0_2.plot([ht[0], ht[-1]], [0, 0], '--k')
	f2_ax0_1.set_title('Stimulus Filter')
	f2_ax0_2.set_title('Post-spike Filter')

	f2_ax1 = fig2.add_subplot(spec2[:3, 1])
	f2_ax2 = fig2.add_subplot(spec2[3:, 1])


	f2_ax3 = fig2.add_subplot(spec2[0, 2])
	f2_ax4 = fig2.add_subplot(spec2[1, 2])
	f2_ax5 = fig2.add_subplot(spec2[2, 2])

	f2_ax6 = fig2.add_subplot(spec2[0, 3])
	f2_ax7 = fig2.add_subplot(spec2[1, 3])
	f2_ax8 = fig2.add_subplot(spec2[2, 3])

	f2_ax9 = fig2.add_subplot(spec2[3:, 2])
	f2_ax10 = fig2.add_subplot(spec2[3:, 3])

	cmap = plt.get_cmap('Greys', 8)
	colors = [cmap(i) for i in range(7, 0, -1)]

	t = np.arange(orn_responses.shape[0]) / fs
	f2_ax1.plot(t, orn_responses)
	f2_ax1.set_xlim([4.5, 7.5])

	for i, j in enumerate(f2_ax1.lines):
		j.set_color(colors[i])

	f2_ax1.set_title('ORN firing-rate')

	f2_ax2.plot(mean_orn_responses, mean_pn_responses, '-ok', label='Control')
	f2_ax2.plot(mean_orn_responses, mean_pn_responses_unc13, '-om', label='Unc13A KD')
	f2_ax2.plot(mean_orn_responses, mean_orn_responses, '--k')
	f2_ax2.axis('equal')
	f2_ax2.legend()
	f2_ax2.set_xlabel('ORN Firing-rate (Hz)')
	f2_ax2.set_ylabel('PN Firing-rate (Hz)')

	f2_ax2.set_title('Gain')


	## chirps
	orn_responses = read_orn_fr('../datasets/neural/meanORNFr.mat', 'chirps')
	control_pn_responses = plot_model_output(orn_responses, control_g1, g2, iapp=iapp_control, dur=35., dc=control_dc)

	orn_responses = read_orn_fr('../datasets/neural/meanORNFr.mat', 'chirps')
	unc13_pn_responses = plot_model_output(orn_responses, unc13_g1, g2, iapp=iapp_unc13, dur=35., dc=unc13_dc)

	_, amp_orn, max_orn, latency_orn = freqz(orn_responses[:, 1])
	_, amp_pn_control, max_pn_control, latency_pn_control = freqz(control_pn_responses[:, 1])
	freq, amp_pn_unc13, max_pn_unc13, latency_pn_unc13 = freqz(unc13_pn_responses[:, 1])

	# freq, amp_orn, max_orn, amp_pn_control, max_pn_control = freq[1:], amp_orn[1:], max_orn[1:], amp_pn_control[1:], max_pn_control[1:]
	# amp_pn_unc13, max_pn_unc13 = amp_pn_unc13[1:], max_pn_unc13[1:]

	t = np.arange(orn_responses.shape[0]) / fs

	f2_ax3.plot(t, orn_responses[:, 1], linewidth=0.5)
	f2_ax4.plot(t, control_pn_responses[:, 1], 'k', linewidth=0.5)
	f2_ax5.plot(t, unc13_pn_responses[:, 1], 'm',  linewidth=0.5)

	f2_ax3.plot(t, orn_responses[:, 1], color=[0.82, 0.82, 0.82], linewidth=0.5)
	f2_ax4.plot(t, orn_responses[:, 1], color=[0.82, 0.82, 0.82], linewidth=0.5)
	f2_ax5.plot(t, orn_responses[:, 1], color=[0.82, 0.82, 0.82], linewidth=0.5)

	f2_ax6.plot(t, orn_responses[:, 1], color=[0.82, 0.82, 0.82], linewidth=0.5)
	f2_ax7.plot(t, orn_responses[:, 1], color=[0.82, 0.82, 0.82], linewidth=0.5)
	f2_ax8.plot(t, orn_responses[:, 1], color=[0.82, 0.82, 0.82], linewidth=0.5)

	f2_ax7.plot(t, control_pn_responses[:, 1], 'k', linewidth=0.5)
	f2_ax8.plot(t, unc13_pn_responses[:, 1], 'm', linewidth=0.5)


	f2_ax6.set_xlim([6.1, 7])
	f2_ax7.set_xlim([6.1, 7])
	f2_ax8.set_xlim([6.1, 7])

	f2_ax9.plot(freq, max_pn_control / max_orn, 'k')
	f2_ax9.plot(freq, max_pn_unc13 / max_orn, 'm')
	#plt.xscale('log')
	f2_ax9.set_xlabel('Frequency (Hz)')
	f2_ax9.set_ylabel('PN mX/ORN max')
	f2_ax9.set_title('Amplitude Filtering')

	f2_ax10.scatter(freq, (latency_pn_control - latency_orn) * (2 * np.pi * freq), 100, 'k')
	f2_ax10.scatter(freq, (latency_pn_unc13 - latency_orn) * (2 * np.pi * freq), 100, 'm')
	f2_ax10.set_ylim([-np.pi/2, 2.])
	f2_ax10.plot([0, 5], [0, 0], '--k')
	f2_ax10.set_title('Phase shift')
	f2_ax10.set_xlabel('Frequency (Hz)')
	f2_ax10.set_ylabel('phase shift (radians)')

	np.savetxt('reverseChirpModel.out', (orn_responses[:, 1], control_pn_responses[:, 1], unc13_pn_responses[:, 1]))
	# # run each orn_fr through the synaptic model and save the output pn firing rate
	# model_pn_responses_control = np.zeros_like(orn_responses)
	# vm_control = np.zeros_like(orn_responses)
	#
	# for i in range(orn_responses.shape[1]):
	# 	vm_control[:, i], model_pn_responses_control[:, i], istm, hcurr, sps, g1[:, i], g2[:, i], i_syn[:, i] = synapse.run(orn_responses[:, i], control_g1, dur=35.)
	#
	# 	fig, ax = plt.subplots(5, 1, sharex=True)
	# 	ax[0].plot(orn_responses[:, i], 'k')
	# 	ax[0].set_title('ORN Firing-rate input to synaptic model')
	# 	ax[0].set_ylabel('(Hz)')
	# 	ax[1].plot(vm_control[:, i], 'k')
	# 	ax[1].set_ylabel('Vm (mV)')
	# 	ax[2].plot(istm, 'b', label='k output')
	# 	ax[2].plot(hcurr, 'r', label='h output')
	# 	ax[2].set_ylabel('Filter outputs')
	# 	ax[2].legend(loc='upper right')
	# 	ax[2].axhline(0, color=".2", linestyle="--", zorder=1)
	#
	# 	ax[3].eventplot(sps[:, 0] * np.arange(len(sps[:, 0])), linewidths=0.5)
	# 	ax[4].set_title('firing-rate')
	# 	ax[4].plot(model_pn_responses_control[:, i], 'k', label='Model PN')
	# 	ax[4].set_ylabel('(Hz)')
	#
	# 	ax[4].plot(orn_responses[:, i], color=[0.82, 0.82, 0.82], label='ORN')
	# 	ax[4].legend(loc='upper right')
