# -*- coding: utf-8 -*-

from functools import reduce

from numpy import apply_along_axis, argmax, array, cumsum, diag, dot, finfo, flatnonzero, int64, outer, sqrt
from numpy.linalg import norm, svd
from pandas import concat, get_dummies
from scipy.linalg import diagsvd
from scipy.sparse import diags
from scipy.sparse.linalg import svds

def process_df(DF, cols, ncols):
	if cols:  # if you want us to do the dummy coding
		K = len(cols)  # the number of categories
		X = dummy(DF, cols)
	else:  # if you want to dummy code it yourself or do all the cols
		K = ncols
		if ncols is None:  # be sure to pass K if you didn't multi-index
			K = len(DF.columns)  # ... it with mca.dummy()
			if not K:
				raise ValueError("Your DataFrame has no columns.")
		elif not isinstance(ncols, int) or ncols <= 0 or \
					ncols > len(DF.columns):  # if you dummy coded it yourself
			raise ValueError("You must pass a valid number of columns.")
		X = DF
	J = X.shape[1]
	return X, K, J


def dummy(DF, cols=None):
	"""Dummy code select columns of a DataFrame."""
	dummies = (get_dummies(DF[col]) for col in 
		(DF.columns if cols is None else cols))
	return concat(dummies, axis=1, keys=DF.columns)


def _mul(*args):
	"""An internal method to multiply matrices."""
	return reduce(dot, args)


class MCA(object):
	"""Run MCA on selected columns of a pd DataFrame.
	
	If the column are specified, assume that they hold
	categorical variables that need to be replaced with
	dummy indicators, otherwise process the DataFrame as is.

	'cols': The columns of the DataFrame to process.
	'ncols': The number of columns before dummy coding. To be passed if cols isn't.
	'benzecri': Perform Benzécri correction (default: True)
	'TOL': value below which to round eigenvalues to zero (default: 1e-4)
	"""

	def __init__(self, DF, cols=None, ncols=None, benzecri=True, TOL=1e-4, 
		sparse=False, approximate=False):

		X, self.K, self.J = process_df(DF, cols, ncols)
		S = X.sum().sum()
		Z = X / S  # correspondence matrix
		self.r = Z.sum(axis=1)
		self.c = Z.sum()
		self.cor = benzecri

		eps = finfo(float).eps
		self.D_r = (diags if sparse else diag)(1/(eps + sqrt(self.r)))
		self.D_c = diag(1/(eps + sqrt(self.c)))  # can't use diags here
		Z_c = Z - outer(self.r, self.c)  # standardized residuals matrix

		product = self.D_r.dot(Z_c).dot(self.D_c)
		if sparse:
			P, s, Q = svds(product, min(product.shape)-1 if ncols is None else ncols)
			# svds and svd use complementary orders
			self.P = P.T[::-1].T
			self.Q = Q[::-1]
			self.s = s[::-1]
			self._numitems = min(product.shape)-1
		else:
			self._numitems = len(DF)
			self.P, self.s, self.Q = svd(product)

		self.E = None
		E = self._benzecri() if self.cor else self.s**2
		self.inertia = sum(E)
		self.rank = argmax(E < TOL)
		if not self.rank: self.rank = len(E)
		self.L = E[:self.rank]

	def _benzecri(self):
		if self.E is None:
			self.E = array([(self.K/(self.K-1.)*(_ - 1./self.K))**2
							  if _ > 1./self.K else 0 for _ in self.s**2])
		return self.E

	def fs_r(self, percent=0.9, N=None):
		"""Get the row factor scores (dimensionality-reduced representation),
		choosing how many factors to retain, directly or based on the explained
		variance.

		'percent': The minimum variance that the retained factors are required
								to explain (default: 90% = 0.9)
		'N': The number of factors to retain. Overrides 'percent'.
				If the rank is less than N, N is ignored.
		"""
		if not 0 <= percent <= 1:
			raise ValueError("Percent should be a real number between 0 and 1.")
		if N:
			if not isinstance(N, (int, int64)) or N <= 0:
				raise ValueError("N should be a positive integer.")
			N = min(N, self.rank)
		self.k = 1 + flatnonzero(cumsum(self.L) >= sum(self.L)*percent)[0]
		#  S = zeros((self._numitems, self.k))
		# the sign of the square root can be either way; singular value vs. eigenvalue
		# fill_diagonal(S, -sqrt(self.E) if self.cor else self.s)
		num2ret = N if N else self.k
		s = -sqrt(self.L) if self.cor else self.s
		S = diagsvd(s[:num2ret], self._numitems, num2ret)
		self.F = self.D_r.dot(self.P).dot(S)
		return self.F

	def fs_c(self, percent=0.9, N=None):
		"""Get the column factor scores (dimensionality-reduced representation),
		choosing how many factors to retain, directly or based on the explained
		variance.

		'percent': The minimum variance that the retained factors are required
								to explain (default: 90% = 0.9)
		'N': The number of factors to retain. Overrides 'percent'.
				If the rank is less than N, N is ignored.
		"""
		if not 0 <= percent <= 1:
			raise ValueError("Percent should be a real number between 0 and 1.")
		if N:
			if not isinstance(N, (int, int64)) or N <= 0:
				raise ValueError("N should be a positive integer.")
			N = min(N, self.rank)  # maybe we should notify the user?
			# S = zeros((self._numitems, N))
		# else:
		self.k = 1 + flatnonzero(cumsum(self.L) >= sum(self.L)*percent)[0]
		#  S = zeros((self._numitems, self.k))
		# the sign of the square root can be either way; singular value vs. eigenvalue
		# fill_diagonal(S, -sqrt(self.E) if self.cor else self.s)
		num2ret = N if N else self.k
		s = -sqrt(self.L) if self.cor else self.s
		S = diagsvd(s[:num2ret], len(self.Q), num2ret)
		self.G = _mul(self.D_c, self.Q.T, S)  # important! note the transpose on Q
		return self.G

	def cos_r(self, N=None):  # percent=0.9
		"""Return the squared cosines for each row."""

		if not hasattr(self, 'F') or self.F.shape[1] < self.rank:
			self.fs_r(N=self.rank)  # generate F
		self.dr = norm(self.F, axis=1)**2
		# cheaper than diag(self.F.dot(self.F.T))?

		return apply_along_axis(lambda _: _/self.dr, 0, self.F[:, :N]**2)

	def cos_c(self, N=None):  # percent=0.9,
		"""Return the squared cosines for each column."""

		if not hasattr(self, 'G') or self.G.shape[1] < self.rank:
			self.fs_c(N=self.rank)  # generate
		self.dc = norm(self.G, axis=1)**2
		# cheaper than diag(self.G.dot(self.G.T))?

		return apply_along_axis(lambda _: _/self.dc, 0, self.G[:, :N]**2)

	def cont_r(self, percent=0.9, N=None):
		"""Return the contribution of each row."""

		if not hasattr(self, 'F'):
			self.fs_r(N=self.rank)  # generate F
		return apply_along_axis(lambda _: _/self.L[:N], 1,
			   apply_along_axis(lambda _: _*self.r, 0, self.F[:, :N]**2))

	def cont_c(self, percent=0.9, N=None):  # bug? check axis number 0 vs 1 here
		"""Return the contribution of each column."""

		if not hasattr(self, 'G'):
			self.fs_c(N=self.rank)  # generate G
		return apply_along_axis(lambda _: _/self.L[:N], 1,
			   apply_along_axis(lambda _: _*self.c, 0, self.G[:, :N]**2))

	def expl_var(self, greenacre=True, N=None):
		"""
		Return proportion of explained inertia (variance) for each factor.

		:param greenacre: Perform Greenacre correction (default: True)
		"""
		if greenacre:
			greenacre_inertia = (self.K / (self.K - 1.) * (sum(self.s**4)
								 - (self.J - self.K) / self.K**2.))
			return (self._benzecri() / greenacre_inertia)[:N]
		else:
			E = self._benzecri() if self.cor else self.s**2
			return (E / sum(E))[:N]

	def fs_r_sup(self, DF, N=None):
		"""Find the supplementary row factor scores.

		ncols: The number of singular vectors to retain.
		If both are passed, cols is given preference.
		"""
		if not hasattr(self, 'G'):
			self.fs_c(N=self.rank)  # generate G

		if N and (not isinstance(N, int) or N <= 0):
			raise ValueError("ncols should be a positive integer.")
		s = -sqrt(self.E) if self.cor else self.s
		N = min(N, self.rank) if N else self.rank
		S_inv = diagsvd(-1/s[:N], len(self.G.T), N)
		# S = diagsvd(s[:N], len(self.tau), N)
		return _mul(DF.div(DF.sum(axis=1), axis=0), self.G, S_inv)[:, :N]

	def fs_c_sup(self, DF, N=None):
		"""Find the supplementary column factor scores.

		ncols: The number of singular vectors to retain.
		If both are passed, cols is given preference.
		"""
		if not hasattr(self, 'F'):
			self.fs_r(N=self.rank)  # generate F

		if N and (not isinstance(N, int) or N <= 0):
			raise ValueError("ncols should be a positive integer.")
		s = -sqrt(self.E) if self.cor else self.s
		N = min(N, self.rank) if N else self.rank
		S_inv = diagsvd(-1/s[:N], len(self.F.T), N)
		# S = diagsvd(s[:N], len(self.tau), N)
		return _mul((DF/DF.sum()).T, self.F, S_inv)[:, :N]
