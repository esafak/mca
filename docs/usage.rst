=====
Usage
=====

To use mca in a project:

.. code-block :: python
	
	import mca
	mca_df = mca.mca(dataframe[, cols=None][, ncols=None][, benzecri=True][, TOL=1e-4])

* ``cols``: A list of the pandas DataFrame's columns to encode and process.
* ``ncols``: The number of factors to retain. None to retain all (default).
* ``benzecri``: Perform Benzecri correction to shrink eigenvalues (default).
* ``TOL``: The value below which to round eigenvalues to zero.

The package includes a fairly thorough set of unit tests, which users are invited to inspect (since they will substitute for a formal help file for the time being.)

Example
-------

>>> import mca, pandas, numpy
>>> counts = pandas.read_table('data/burgundies.csv', sep=',', skiprows=1, index_col=0, header=0)
>>> print(counts.shape)
(6, 23)

>>> mca_counts = mca.mca(counts.drop('oak_type', axis=1))
>>> print(mca_counts.fs_r(1)) # 1 = 100%, meaning preserve all variance.
	array([[ 0.87127085,  0.11448396, -0.09250792],
	       [-0.7209849 , -0.22896791, -0.083259  ],
	       [-0.93238083,  0.11448396, -0.02206285],
	       [-0.87127085,  0.11448396,  0.09250792],
	       [ 0.93238083,  0.11448396,  0.02206285],
	       [ 0.7209849 , -0.22896791,  0.083259  ]])

The eigenvalues, or *principal inertias*, of the factors:

>>> print(mca_counts.L)
	array([ 0.71608871,  0.02621315,  0.00532552])

The inertia is simply the sum of the principle inertias:

>>> print(mca_counts.inertia, mca_counts.L.sum())
	0.74762737298514048 0.74762737298514048

If Benzecri correction has been enabled (default), this is less than the the squared sum of the singular values:

>>> print(mca_counts.s)
>>> print(sum(mca_counts.s**2))
	array([  9.23693800e-01,   4.47213595e-01,   3.39283916e-01,
         	1.77978056e-01,   1.71329335e-16,   7.21294550e-17])
	1.2

Benzecri correction plus thresholding has eliminated 3 of the 6 columns. You can adjust the threshold by setting the TOL parameter (default: 1e-4) in the constructor. If we had not set the ``prob`` parameter in ``fs_r()`` to 1, it would have used its default value of 0.9 and we would have eliminated another two columns, leading to a dimensionality reduction ratio of 6:1.

>>> print(mca_counts.fs_r())
	array([[ 0.87127085],
	       [-0.7209849 ],
	       [-0.93238083],
	       [-0.87127085],
	       [ 0.93238083],
	       [ 0.7209849 ]])

The result is identical to the first column of the earlier invocation of ``fs_r(1)``. This holds in general; reducing ``prob`` or ``N`` simply truncates the matrix, exactly as in PCA.
If you want to find the factor score of supplementary data (which has to be `conformable <http://en.wikipedia.org/wiki/Conformable_matrix>`_):

>>> new_counts = pandas.DataFrame(numpy.random.randint(0, 2, (5, len(counts.columns)-1)))

where the decrement is to account for the dropped column ('``oak_types``') in the original ``counts`` DataFrame. As before, we can decide how many columns to keep:

>>> mca_counts.fs_r_sup(new_counts, 2)
	array([[ -3.33523735e-02,   2.27874988e-16],
	       [  3.13116890e-01,  -1.12938488e-01],
	       [ -3.33523735e-02,   3.33829232e-16],
	       [ -5.12296954e-02,   1.21626064e-01],
	       [ -7.71194728e-03,   4.74341649e-01]])
