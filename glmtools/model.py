import numpy as np
from glmtools.make_xdsgn import DesignSpec
from scipy import stats, signal
import numpy.random as rnd

class GLM:
	'''
	compute the response of glm to stimulus Stim

	Dynamics:  Filters the Stimulus with glmprs.k, passes this through a
	nonlinearity to obtain the point-process conditional intensity.  Add a
	post-spike current to the linear input after every spike.

	'''
	def __init__(self, dspec: DesignSpec, k, h, dc):
		'''

		:param k: stimulus filter
		:param h: post-spike filter
		'''
		self.k_ = np.asarray([k.T])
		self.h_ = h
		self.dc_ = dc
		self.dspec_ = dspec

	def simulate(self, stim):

		nlfun = lambda x: np.exp(x)

		nbins_per_eval = 1  # Default number of bins to update for each spike

		dt = 0.001	# bin size for simulation
		slen = len(stim)  		# length of stimulus
		rlen = slen		  		# length of binned spike response

		hlen = len(self.h_)		# length of post-spike filter

		nx = len(stim) 			# must have a second dimension for convolve2d to word
		_, nf = self.k_.shape

		a = np.concatenate((np.zeros(nf - 1), stim), axis=None)
		b = np.rot90(self.k_, k=2)
		istm = signal.convolve2d(np.asarray([a]), b, mode='valid') + self.dc_


		itot = istm.squeeze()			# total filter output

		nsp = 0
		sps = np.zeros(rlen)
		jbin = 1
		tspnext = rnd.exponential(1)
		rprev = 0

		while jbin < rlen:

			iinxt = np.arange(jbin, min(jbin + nbins_per_eval - 1, rlen) + 1) # we have to add 1 because arange does not include upper lim

			rrnxt = nlfun(itot[iinxt]) * dt

			rrcum = np.cumsum(rrnxt) + rprev

			if tspnext >= rrcum[-1]:
				jbin = iinxt[-1] + 1
				rprev = rrcum[-1]
			else:
				ispk = iinxt[np.where(rrcum >= tspnext)[0][0]]
				nsp = nsp + 1
				sps[ispk] = 1

				mxi = min(rlen, ispk + hlen)
				ii_postspk = np.arange(ispk, mxi)

				if len(ii_postspk):
					itot[ii_postspk] = itot[ii_postspk] + self.h_[range(0, mxi - ispk)]

				tspnext = rnd.exponential(1) 	# draw next spike time
				rprev = 0 						# reset integrated intensity
				jbin = ispk + 1 				# Move to next bin

		tsp = np.where(sps > 0)[0]
		return tsp, sps
