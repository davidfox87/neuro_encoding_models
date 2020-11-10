import numpy as np
from scipy.linalg import orth
from matplotlib import pyplot as plt

class Basis:
	def __init__(self, duration_, offset_, _name):
		self.duration = duration_
		self.tr = []
		self.type = None
		self.offset = offset_
		self.name = _name


class RaisedCosine(Basis):
	def __init__(self, duration, nbases, offset, _name):
		Basis.__init__(self, duration, offset, _name)
		self.centers = []
		self.ttb = [] 	# time indices for basis
		self.type = 'Raised Cosine'
		self.nbases = nbases
		self.B = None

	def makeNonlinearRaisedCosStim(self, dt, endpoints, nlOffset, nkt=0):
		kdt = 1

		nlin = lambda x: np.log(x + 1e-20)
		invnl = lambda x: np.exp(x) - 1e-20

		yrnge = nlin(np.array(endpoints) + nlOffset)
		db = np.diff(yrnge) / (self.nbases - 1)
		#ctrs = np.arange(yrnge[0], yrnge[1]+db, db)
		ctrs = np.linspace(yrnge[0], yrnge[1], num=self.nbases)# +db to include yrnge[1] otherwise np.arange excludes upper lim
		mxt = invnl(yrnge[1] + 2 * db) - nlOffset
		kt0 = np.arange(0, mxt, kdt)
		nkt0 = len(kt0)

		a = np.tile(nlin(kt0 + nlOffset), (self.nbases, 1)).T
		b = np.tile(ctrs, (nkt0, 1))

		res = list((map(lambda z: np.minimum(z, np.pi), (a - b) * np.pi / db / 2)))
		res = list(map(lambda z: np.maximum(z, -np.pi), res))
		ktbasis = (np.cos(res) + 1) / 2

		# assign to member variables
		self.centers = invnl(ctrs)

		ktbasis = np.flipud(ktbasis)  # flip so fast timescales are at the end.
		if nkt0 < nkt:
			# else pad with 0s to make up nkt
			ktbasis = np.concatenate((np.zeros((nkt - nkt0, self.nbases)), ktbasis), axis=0)
		else:
			# trim off elements to make nkt points
			ktbasis = ktbasis[-nkt:, :]

		ktbasis = ktbasis / np.array([np.sqrt((ktbasis ** 2).sum(axis=0))])
		self.B = (ktbasis)
		self.tr = kt0

	def makeNonlinearRaisedCosPostSpike(self, dt, endpoints, nlOffset, absref):

		if absref >= dt:
			ncols = self.nbases - 1
		else:
			ncols = self.nbases

		# ncols = self.nbases
		nlin = lambda x: np.log(x + 1e-20)
		invnl = lambda x: np.exp(x) - 1e-20

		yrnge = nlin(np.array(endpoints) + nlOffset)
		db = np.diff(yrnge) / (ncols - 1)
		# ctrs = np.arange(yrnge[0], yrnge[1] + db,
		# 				 db)  # +db to include yrnge[1] otherwise np.arange excludes upper lim
		ctrs = np.linspace(yrnge[0], yrnge[1], num=ncols)
		mxt = invnl(yrnge[1] + 2 * db) - nlOffset
		iht = np.arange(0, mxt, dt)
		nt = len(iht)

		a = np.tile(nlin(iht + nlOffset), (ncols, 1)).T
		b = np.tile(ctrs, (nt, 1))

		res = list((map(lambda z: np.minimum(z, np.pi), (a - b) * np.pi / db / 2)))
		res = list(map(lambda z: np.maximum(z, -np.pi), res))
		ihbasis = (np.cos(res) + 1) / 2

		ii = np.argwhere(iht <= endpoints[0])
		ihbasis[ii, 0] = 1

		if absref >= dt:
			ii = np.argwhere(iht < absref)
			ih0 = np.zeros((ihbasis.shape[0], 1))
			ih0[ii] = 1
			ihbasis[ii, :] = 0
			ihbasis = np.concatenate((ih0, ihbasis), axis=1)

		#ihbasis = orth(ihbasis)
		ihbasis = np.concatenate((np.zeros((1, ihbasis.shape[1])), ihbasis), axis=0)
		# assign to member variables
		self.centers = invnl(ctrs)
		self.B = (ihbasis)
		self.nbases = self.B.shape[1]
		self.tr = iht

