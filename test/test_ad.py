# -*- coding: utf-8 -*-

import unittest

from sklearn.datasets import make_blobs

from src.mlchemad import *


class TestAD(unittest.TestCase):

    def setUp(self):
        self.X_cent1_sd1, _ = make_blobs(n_samples=1000, n_features=100, centers=[[1] * 100], cluster_std=1, random_state=1234)
        self.X_cent1_sd3, _ = make_blobs(n_samples=1000, n_features=100, centers=[[1] * 100], cluster_std=3, random_state=1234)
        self.X_cent6_sd1, _ = make_blobs(n_samples=1000, n_features=100, centers=[[6] * 100], cluster_std=1, random_state=1234)
        self.X_cent6_sd3, _ = make_blobs(n_samples=1000, n_features=100, centers=[[6] * 100], cluster_std=3, random_state=1234)
        self.mekenyan_veith = data

    def test_minmax_boundingbox(self):
        ad = BoundingBoxApplicabilityDomain(percentiles=None)
        ad.fit(self.X_cent1_sd1)
        self.assertEquals(sum(ad.contains(self.X_cent1_sd1)),
                          len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])


    def test_fixed_minmax_boundingbox(self):
        ad = BoundingBoxApplicabilityDomain(percentiles=None, range_=(-3, 5))
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_percentiles_boundingbox(self):
        ad = BoundingBoxApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.8 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_convexhull(self):
        ad = ConvexHullApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertEquals(ad.contains(self.X_cent1_sd1[:100]).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1[:100]])
        self.assertEquals(ad.contains(self.X_cent6_sd1[:100]).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1[:100]])
        self.assertEquals(ad.contains(self.X_cent1_sd3[:100]).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3[:100]])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_pca_robust_boundingbox(self):
        ad = PCABoundingBoxApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_pca_standard_boundingbox(self):
        ad = PCABoundingBoxApplicabilityDomain(scaling='standard')
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_pca_minmax_boundingbox(self):
        ad = PCABoundingBoxApplicabilityDomain(scaling='minmax')
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                          0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_pca_maxabs_boundingbox(self):
        ad = PCABoundingBoxApplicabilityDomain(scaling='maxabs')
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_leverage(self):
        ad = LeverageApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd3)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_hotellingt2(self):
        ad = HotellingT2ApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd3)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_centroid_distance(self):
        ad = CentroidDistanceApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_knn(self):
        ad = KNNApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))

        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_kerneldensity(self):
        ad = KernelDensityApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_topkat(self):
        ad = TopKatApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertEquals(sum(ad.contains(self.X_cent1_sd1)),
                          len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])

    def test_isolation_forest(self):
        ad = IsolationForestApplicabilityDomain()
        ad.fit(self.X_cent1_sd1)
        self.assertGreater(sum(ad.contains(self.X_cent1_sd1)),
                           0.9 * len(self.X_cent1_sd1))
        self.assertEquals(sum(ad.contains(self.X_cent6_sd1)),
                          0)
        self.assertLess(sum(ad.contains(self.X_cent1_sd3)),
                        len(self.X_cent1_sd3))
        self.assertEquals(ad.contains(self.X_cent1_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd1])
        self.assertEquals(ad.contains(self.X_cent6_sd1).tolist(),
                          [ad.contains(x) for x in self.X_cent6_sd1])
        self.assertEquals(ad.contains(self.X_cent1_sd3).tolist(),
                          [ad.contains(x) for x in self.X_cent1_sd3])
        ad.fit(self.mekenyan_veith.training)
        self.assertGreaterEqual(sum(ad.contains(self.mekenyan_veith.test)), 0)
        self.assertLessEqual(sum(ad.contains(self.mekenyan_veith.test)), self.mekenyan_veith.test.shape[0])
