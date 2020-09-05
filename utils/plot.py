import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore

def plot_raster2(ax, spikes):
	avg_spks = zscore(spikes.mean(axis=1), axis=1)
	plt.imshow(avg_spks, aspect='auto', vmax=2, vmin=-2, cmap='gray')
	ax.set(xlabel='binned time', ylabel='neurons')


def plot_raster(ax1, ax2, stim, spks, time, offset):
	nt = spks.shape[-1]
	dt = time[1]-time[0]
	time = time - offset
	spk_times = (spks * time)

	# create a dictionary of spike times where the key is the trial and the value is an array of spike times
	# To get trial 1 spikes = d[1]
	d = {key: np.ma.masked_equal(value, 0).compressed() for (key, value) in enumerate(spk_times)}

	ax1.plot(np.arange(len(stim))*dt, stim, 'k', linewidth=0.5)
	# this plots the vertical lines for each spike in each trial
	for i in range(len(d)):
		ax2.vlines(d[i], ymin=i, ymax=i+1, linewidth=0.5)


def plot_glm_matrices(X, y, nt=50):
	"""Show X and Y as heatmaps.

	Args:
	  X (2D array): Design matrix.
	  y (1D or 2D array): Target vector.

	"""
	from matplotlib.colors import BoundaryNorm
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	Y = np.c_[y]  # Ensure Y is 2D and skinny

	f, (ax_x, ax_y) = plt.subplots(
		ncols=2,
		figsize=(6, 8),
		sharey=True,
		gridspec_kw=dict(width_ratios=(5, 1)),
	)
	norm = BoundaryNorm([-1, -.2, .2, 1], 256)
	#imx = ax_x.pcolormesh(X[:nt], cmap="coolwarm", norm=norm)
	imx = ax_x.pcolormesh(X[:nt], cmap="coolwarm")

	ax_x.set(
		title="X\n(lagged stimulus)",
		xlabel="Time lag (time bins)",
		#xticks=[4, 14, 24],
		#xticklabels=['-20', '-10', '0'],
		ylabel="Time point (time bins)",
	)
	plt.setp(ax_x.spines.values(), visible=True)

	divx = make_axes_locatable(ax_x)
	caxx = divx.append_axes("right", size="5%", pad=0.1)
	cbarx = f.colorbar(imx, cax=caxx)
	cbarx.set_ticks([-.6, 0, .6])
	cbarx.set_ticklabels(np.sort(np.unique(X)))

	# not sure why this normalization doesn't work for my spike count
	# but does for neuromatch. What's the difference
	norm = BoundaryNorm(np.arange(y.max() + 1), 256)
	imy = ax_y.pcolormesh(Y[:nt], cmap="magma")
	ax_y.set(
		title="Y\n(spike count)",
		xticks=[]
	)
	ax_y.invert_yaxis()
	plt.setp(ax_y.spines.values(), visible=True)

	divy = make_axes_locatable(ax_y)
	caxy = divy.append_axes("right", size="30%", pad=0.1)
	cbary = f.colorbar(imy, cax=caxy)
	cbary.set_ticks(np.arange(y.max()) + .5)
	cbary.set_ticklabels(np.arange(y.max()))


def plot_spike_filter(ax, theta, dt, **kws):

	''' Plot estimated weights based on time lag model.
	Args:
		ax (axes):
		theta (1D array): Filter weights, not including DC term.
		dt (number): Duration of each time bin.
		kws: Pass additional keyword arguments to plot() so you can customize the plot
	'''

	d = len(theta)
	t = np.arange(-d + 1, 1) * dt

	ax.plot(t, theta, **kws)
	ax.axhline(0, color=".2", linestyle="--", zorder=1)
	ax.set(
		xlabel="Time before spike (s)",
		ylabel="Filter weight",
	)
