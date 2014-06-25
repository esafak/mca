# -*- coding: utf-8 -*-

import scipy.linalg, numpy, pandas, functools

# import pdb

def dummy(DF, cols=None):
	"""Dummy code select columns of a DataFrame."""
	return pandas.concat((pandas.get_dummies(DF[col]) for col in (DF.columns if cols is None else cols)), 
						axis=1, keys = DF.columns)

def _mul(*args):
	"""An internal method to multiply matrices."""
	return functools.reduce(numpy.dot, args)

class mca:
	"""Run MCA on selected columns of a pandas DataFrame. 
	If the column are specified, assume that they hold 
	categorical variables that need to be replaced with
	dummy indicators, otherwise process the DataFrame as is.

	'cols': The columns of the DataFrame to process.
	'K': The number of columns before dummy coding. To be passed if cols isn't.
	'benzecri': Perform Benzécri correction (default: True)
	'TOL': value below which to round eigenvalues to zero
	"""

	def __init__(self, DF, cols=None, ncols=None, benzecri=True, TOL=1e-4):

		if cols: # if you want us to do the dummy coding
			K = len(cols) # the number of categories
			X = dummy(DF, cols)
		else: # if you want to dummy code it yourself or do all the cols
			K = ncols
			if ncols is None: # be sure to pass K if you did not multi-index
				K = len(DF.columns) # ... it with mca.dummy()
				if not K: raise ValueError("Your DataFrame has no columns.")
			elif not isinstance(ncols, int) or ncols<=0 or ncols>len(DF.columns): # if you dummy coded it yourself
				raise ValueError("You must pass a valid number of columns.")
			X = DF
		S = X.sum().sum()
		Z = X/S # correspondence matrix
		self.r = Z.sum(axis=1)
		self.c = Z.sum()
		self._numitems = len(DF)
		self.cor = benzecri
		self.D_r = numpy.diag(1/numpy.sqrt(self.r))
		Z_c = Z - numpy.outer(self.r,self.c) # standardized residuals matrix
		self.D_c = numpy.diag(1/numpy.sqrt(self.c))

		# another option, not pursued here, is sklearn.decomposition.TruncatedSVD
		self.P, self.s, self.Q = scipy.linalg.svd(_mul(self.D_r, Z_c, self.D_c))
		
		if benzecri: self.E = numpy.array([(K/(K-1)*(_ - 1/K))**2 
				if _ > 1/K else 0 for _ in self.s**2])
		self.inertia = self.E.sum() if benzecri else sum(self.s**2)
		self.rank = numpy.argmax((self.E if benzecri else self.s**2) < TOL)
		self.L = (self.E if benzecri else self.s**2)[:self.rank]

	def fs_r(self, percent=0.9, N=None):
		"""Get the row factor scores (dimensionality-reduced representation),
		choosing how many factors to retain, directly or based on the explained variance.

		'percent': The minimum variance that the retained factors are required 
					to explain (default: 90% = 0.9)
		'N': The number of factors to retain. Overrides 'percent'.
			If the rank is less than N, N is ignored.
		"""
		if not 0 <= percent <= 1:
			raise ValueError("Percent should be a real number between 0 and 1.")
		if N:
			if not isinstance(N, (int, numpy.int64)) or N<=0:
				raise ValueError("N should be a positive integer.")
			N = min(N, self.rank)
			# S = numpy.zeros((self._numitems, N))
		# else:
		self.k = 1 + numpy.flatnonzero(numpy.cumsum(self.L) >= sum(self.L)*percent)[0]
			# S = numpy.zeros((self._numitems, self.k))		
		# the sign of the square root can be either way; singular value vs. eigenvalue
		# numpy.fill_diagonal(S, -numpy.sqrt(self.E) if self.cor else self.s)
		num2ret = N if N else self.k
		s = -numpy.sqrt(self.L) if self.cor else self.s
		S = scipy.linalg.diagsvd(s[:num2ret], self._numitems, num2ret)		
		self.F = _mul(self.D_r, self.P, S)
		return self.F

	def fs_c(self, percent=0.9, N=None):
		"""Get the column factor scores (dimensionality-reduced representation),
		choosing how many factors to retain, directly or based on the explained variance.

		'percent': The minimum variance that the retained factors are required 
					to explain (default: 90% = 0.9)
		'N': The number of factors to retain. Overrides 'percent'.
			If the rank is less than N, N is ignored.
		"""
		if not 0 <= percent <= 1:
			raise ValueError("Percent should be a real number between 0 and 1.")
		if N:			
			if not isinstance(N, (int, numpy.int64)) or N<=0:
				raise ValueError("N should be a positive integer.")
			N = min(N, self.rank) # maybe we should notify the user?
			# S = numpy.zeros((self._numitems, N))
		# else:
		self.k = 1 + numpy.flatnonzero(numpy.cumsum(self.L) >= sum(self.L)*percent)[0]
			# S = numpy.zeros((self._numitems, self.k))		
		# the sign of the square root can be either way; singular value vs. eigenvalue
		# numpy.fill_diagonal(S, -numpy.sqrt(self.E) if self.cor else self.s)
		num2ret = N if N else self.k
		s = -numpy.sqrt(self.L) if self.cor else self.s
		S = scipy.linalg.diagsvd(s[:num2ret], len(self.Q), num2ret)
		self.G =  _mul(self.D_c, self.Q.T, S) # important! note the transpose on Q
		return self.G

	def cos_r(self, N=None): #percent=0.9, 
		"""Return the squared cosines for each row."""

		if not hasattr(self, 'F') or self.F.shape[1] < self.rank: 
			self.fs_r(N=self.rank) # generate F
		self.dr = numpy.linalg.norm(self.F, axis=1)**2
		# cheaper than numpy.diag(self.F.dot(self.F.T))?

		return numpy.apply_along_axis(lambda _: _/self.dr, 0, self.F[:,:N]**2)

	def cos_c(self, N=None): #percent=0.9, 
		"""Return the squared cosines for each column."""

		if not hasattr(self, 'G') or self.G.shape[1] < self.rank: 
			self.fs_c(N=self.rank) # generate G
		self.dc = numpy.linalg.norm(self.G, axis=1)**2
		# cheaper than numpy.diag(self.G.dot(self.G.T))?

		return numpy.apply_along_axis(lambda _: _/self.dc, 0, self.G[:,:N]**2)		

	def cont_r(self, percent=0.9, N=None):
		"""Return the contribution of each row."""

		if not hasattr(self, 'F'): self.fs_r(N=self.rank) # generate F
		return numpy.apply_along_axis(lambda _: _/self.L[:N], 1, 
			numpy.apply_along_axis(lambda _: _*self.r, 0, self.F[:,:N]**2))

	def cont_c(self, percent=0.9, N=None): # bug? check axis number 0 vs 1 here
		"""Return the contribution of each row."""

		if not hasattr(self, 'G'): self.fs_c(N=self.rank) # generate G
		return numpy.apply_along_axis(lambda _: _/self.L[:N], 1, 
			numpy.apply_along_axis(lambda _: _*self.c, 0, self.G[:,:N]**2))

	def fs_r_sup(self, DF, ncols=None):
		"""Find the supplementary row factor scores.

		ncols: The number of singular vectors to retain.
		If both are passed, cols is given preference.
		"""
		if not hasattr(self, 'G'): self.fs_c(N=self.rank) # generate G

		if ncols and (not isinstance(ncols, int) or ncols<=0):
			raise ValueError("ncols should be a positive integer.")
		s = -numpy.sqrt(self.E) if self.cor else self.s		
		N = min(ncols, self.rank) if ncols else self.rank
		S_inv = scipy.linalg.diagsvd(-1/s[:N], len(self.G.T), N)		
		# S = scipy.linalg.diagsvd(s[:N], len(self.tau), N)
		return _mul(DF.div(DF.sum(axis=1), axis=0), self.G, S_inv)[:,:N]

	def fs_c_sup(self, DF, ncols=None):
		"""Find the supplementary column factor scores.

		ncols: The number of singular vectors to retain.
		If both are passed, cols is given preference.
		"""
		if not hasattr(self, 'F'): self.fs_r(N=self.rank) # generate F

		if ncols and (not isinstance(ncols, int) or ncols<=0):
			raise ValueError("ncols should be a positive integer.")
		s = -numpy.sqrt(self.E) if self.cor else self.s
		N = min(ncols, self.rank) if ncols else self.rank
		S_inv = scipy.linalg.diagsvd(-1/s[:N], len(self.F.T), N)		
		# S = scipy.linalg.diagsvd(s[:N], len(self.tau), N)
		return _mul((DF/DF.sum()).T, self.F, S_inv)[:,:N]