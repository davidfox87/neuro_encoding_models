import numpy as np
import matplotlib.pyplot as plt


def izhikevich_model(iext_=10, a=0.02, b=0.2, c=-65, d=6):
	# Constants
	spike_value = 30  # Maximal Spike Value

	T = 100000  # total simulation length [ms]
	dt = 0.25  # step size [ms]
	time = np.arange(0, T + dt, dt)  # step values [ms]
	# VOLTAGE
	v = np.zeros(len(time))  # array for saving voltage history
	v[0] = -64  # set initial to resting potential

	# Recovery variable
	u = np.zeros(len(time))  # array for saving Recovery history
	u[0] = b * v[0]

	# Input current
	iext = np.zeros(len(time))
	iext[200:90000] = iext_

	sptimes = []
	for t in range(1, len(time)):

		if v[t - 1] < spike_value:
			dV = (0.04 * v[t - 1] + 5) * v[t - 1] + 140 - u[t - 1] + iext[t - 1]
			v[t] = v[t - 1] + dV * dt

			du = a * (b * v[t - 1] - u[t - 1])
			u[t] = u[t - 1] + dt * du
		# spike reached!
		else:
			v[t - 1] = spike_value  # set to spike value
			v[t] = c  # reset membrane voltage
			u[t] = u[t - 1] + d  # reset recovery
			sptimes.append(time[t-1])

	return time, v, iext, sptimes


def simulate():
	t, v, iext, sptimes = izhikevich_model(iext_=14, a=0.02, b=0.2, c=-65, d=6)
	return t, v, iext, sptimes


# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(t, iext)
# ax2.plot(t, v)
# plt.show()
#
# # phasic bursting
# t, v, iext = izhikevich_model(iext_=0.6, a=0.02, b=0.25, c=-55, d=0.05)
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(t, iext)
# ax2.plot(t, v)
# plt.show()

if __name__ == '__main__':
	simulate()
