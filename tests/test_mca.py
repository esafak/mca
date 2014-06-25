#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mca
----------------------------------

Tests for `mca` module.
"""

import unittest, numpy.testing, pandas

from mca import mca, dummy

class TestMca(unittest.TestCase):

	def setUp(self):
		pass

	def test_abdi_valentin(self):
		# Data taken from http://www.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
		# Multiple Correspondence Analysis (Hervé Abdi & Dominique Valentin, 2007)
		# See in particular Table 2,3,4.

		# first we check the eigenvalues and factor scores with Benzecri correction
		df = pandas.read_table('data/burgundies.csv', skiprows=1, sep=',', index_col=0)
		mca_df = mca(df.drop('oak_type', axis=1), ncols = 10)
		numpy.testing.assert_allclose([0.7004, 0.0123, 0.0003], mca_df.E[:3], atol=1e-4)
		true_fs_row = [[0.86, 0.08], [-0.71, -0.16], [-0.92, 0.08], 
					[-0.86, 0.08], [0.92, 0.08], [0.71, -0.16]]
		numpy.testing.assert_allclose(true_fs_row, mca_df.fs_r(N = 2), atol=1e-2)
		true_fs_col = [[.90, -.90, -.97, .00, .97, -.90, .90, .90, -.90, -.90, .90, -.97, .00, .97, -.90, .90, .28, -.28, -.90, .90, -.90, .90, .90, -.90],
					[.00, .00, .18, -.35, .18, .00, .00, .00, .00, .00, .00, .18, -.35, .18, .00, .00, .00, .00, .00, .00, .00, .00, .00, .00]]
		numpy.testing.assert_allclose(numpy.array(true_fs_col).T[:-2], mca_df.fs_c(N = 2), atol=1e-2)

		true_cont_r = [[177, 121, 202, 177, 202, 121], [83, 333, 83, 83, 83, 333]]
		numpy.testing.assert_allclose(true_cont_r, 1000*mca_df.cont_r(N=2).T, atol=1)

		true_cont_c = [[58, 58, 44, 0, 44, 58, 58, 58, 58, 58, 58, 44, 0, 44, 58, 58, 6, 6, 58, 58, 58, 58], 
				[0, 0, 83, 333, 83, 0, 0, 0, 0, 0, 0, 83, 333, 83, 0, 0, 0, 0, 0, 0, 0, 0]]
		numpy.testing.assert_allclose(true_cont_c, 1000*mca_df.cont_c(N=2).T, atol=1)

		# I declined to include a test for the cos_c and cos_r functions because
		# I think the source itself is mistaken. In Abdi-MCA2007-pretty.pdf as in
		# elsewhere the formula for the squared cosine is f**2/d**2. This does not
		# agree with tables 3 and 4. In table 3 the squared cosine is derived from
		# f**2/I where I = 1.2 is the inertia before Benzecri correction. I have no
		# idea how the squared cosines in table 4 were derived. My formula, however
		# does comport with the figures given in (Abdi & Bera, 2014), tested next.

		# oak = pandas.DataFrame([1,2,2,2,1,1], columns=['oak_type'])
		# print(dummy(oak))
		# mca_df.fs_c_sup(dummy(oak))

		# ... then without Benzecri correction
		numpy.testing.assert_allclose([0.8532, 0.2, 0.1151, 0.0317], 
			(mca(df.drop('oak_type', axis=1), ncols = 10, benzecri=False).s**2)[:4], atol=1e-4)       

	def test_abdi_bera(self):
		# Data taken from www.utdallas.edu/~herve/abdi-AB2014_CA.pdf
		# Correspondence Analysis, (Herve Abdi & Michel Bera, 2014)
		# Springer Encyclopedia of Social Networks and Mining.
		df = pandas.read_table('data/music_color.csv', skiprows=0, index_col=0, sep=',')
		mca_df = mca(df, benzecri=False)

		# Table 1, page 13
		numpy.testing.assert_allclose(mca_df.r, [.121, .091, .126, .116, .096, .066, .071, .146, .061, .106], atol=1e-3)
		numpy.testing.assert_allclose(mca_df.c, [.11, .11, .11, .11, .11, .11, .11, .11, .11], atol=1e-2)

		# Table 2, page 14
		numpy.testing.assert_allclose(mca_df.fs_r(N=2), [[-0.026, 0.299], [-0.314, 0.232], [-0.348, 0.202], [-0.044, -0.490], [-0.082, -0.206], [-0.619, 0.475], [-0.328, 0.057], [1.195, 0.315], [-0.57, 0.3], [0.113, -0.997]], atol=1e-3)
		numpy.testing.assert_allclose(mca_df.cont_r(N=2)*1000, [[0, 56], [31, 25], [53, 27], [1, 144], [2, 21], [87, 77], [26, 1], [726, 75], [68, 28], [5, 545]], atol=1)
		numpy.testing.assert_allclose(mca_df.cos_r(N=2)*1000, [[3, 410], [295, 161], [267, 89], [5, 583], [13, 81], [505, 298], [77, 2], [929, 65], [371, 103], [12, 973]], atol=1)

		# Table 3, page 17
		numpy.testing.assert_allclose(mca_df.fs_c(N=2), [[-0.541, 0.386], [-.257, .275], [-.291, -.309], [.991, .397], [-.122, -.637], [-.236, .326], [.954, -.089], [-.427, .408], [-.072, -.757]], atol=1e-3)
		numpy.testing.assert_allclose(mca_df.cont_c(N=2)*1000, [[113, 86], [25, 44], [33, 55], [379, 91], [6, 234], [22, 61], [351, 5], [70, 96], [2, 330]], atol=1)
		numpy.testing.assert_allclose(mca_df.cos_c(N=2)*1000, [[454, 232], [105, 121], [142, 161], [822, 132], [26, 709], [78, 149], [962, 8], [271, 249], [7, 759]], atol=1)

		numpy.testing.assert_allclose(mca_df.L[:2], [.287, .192], atol=2e-3)
		self.assertAlmostEqual(mca_df.inertia, 0.746, 3)

	def test_abdi_williams(self):
		# Data taken from www.utdallas.edu/~herve/abdi-CorrespondenceAnaysis2010-pretty.pdf
		# Correspondence Analysis, (Herve Abdi & Michel Bera, 2010)
		# SAGE Encyclopedia of Research Design. Table 4, page 16.

		df = pandas.read_table('data/french_writers.csv', skiprows=0, index_col=0, sep=',')
		mca_df = mca(df, benzecri=False)

		numpy.testing.assert_allclose(mca_df.c, [.2973, .5642, .1385], atol=1e-4)
		numpy.testing.assert_allclose(mca_df.r, [.0189, .1393, .2522, .3966, .1094, .0835], atol=1e-4)

		true_fs_row = [[0.2398, 0.1895, 0.1033, -0.0918, -0.2243, 0.0475], 
			[0.0741, 0.1071, -0.0297, 0.0017, 0.0631, -0.1963]]
		numpy.testing.assert_allclose(mca_df.fs_r(N=2).T, true_fs_row, atol=1e-4)
		numpy.testing.assert_allclose(mca_df.L, [.0178, .0056], atol=1e-4)

		numpy.testing.assert_allclose(-mca_df.fs_c(N=2).T, [[-0.0489, 0.0973, -0.2914], [.1115, -0.0367, -0.0901]], atol=1e-4)

		true_cont_r = [[0.0611, 0.2807, 0.1511, 0.1876, 0.3089, 0.0106], 
				[0.0186, 0.2864, 0.0399, 0.0002, 0.0781, 0.5767]]
		numpy.testing.assert_allclose(mca_df.cont_r(N=2).T, true_cont_r, atol=1e-4)

		true_cos_r = [[0.9128, 0.7579, 0.9236, 0.9997, 0.9266, 0.0554],
				[0.0872, 0.2421, 0.0764, 0.0003, 0.0734, 0.9446]]
		numpy.testing.assert_allclose(mca_df.cos_r(N=2).T, true_cos_r, atol=1e-4)

		numpy.testing.assert_allclose(mca_df.cont_c(N=2).T, [[0.0399, 0.2999, 0.6601],[0.6628, 0.1359, 0.2014]], atol=1e-4)
		numpy.testing.assert_allclose(mca_df.cos_c(N=2).T, [[0.1614, 0.8758, 0.9128], [0.8386, 0.1242, 0.0872]], atol=1e-4)

		numpy.testing.assert_allclose(mca_df.dc, [0.0148, 0.0108, 0.0930], atol=1e-4)
		numpy.testing.assert_allclose(mca_df.dr, [0.0630, 0.0474, 0.0116, 0.0084, 0.0543, 0.0408], atol=1e-4)

		# abdi = numpy.array([216, 139, 26]) # 
		abdi = pandas.DataFrame([216, 139, 26]).T
		numpy.testing.assert_allclose(mca_df.fs_r_sup(abdi, 2), [[-0.0908, 0.5852]], atol=1e-4)

		supp = pandas.read_table('data/french_writers_supp.csv', skiprows=0, index_col=0, sep=',')

		true_fs_col_sup = [[-0.0596,-0.1991,-0.4695,-0.4008],[0.2318,0.2082,-0.2976,-0.4740]]
		numpy.testing.assert_allclose(mca_df.fs_c_sup(supp).T, true_fs_col_sup, atol=1e-3)

	def test_invalid_inputs(self):
		df = pandas.read_table('data/burgundies.csv', skiprows=1, sep=',')
		self.assertRaises(ValueError, mca, df.iloc[:,2:], ncols = 0)
		self.assertRaises(ValueError, mca, df.iloc[:,2:], ncols = '')

	def tearDown(self):
		pass

if __name__ == '__main__':
	unittest.main()