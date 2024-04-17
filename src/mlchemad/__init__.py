# -*- coding: utf-8 -*-


__version__ = '1.5.2'


from .applicability_domains import (BoundingBoxApplicabilityDomain, CentroidDistanceApplicabilityDomain,
                                    ConvexHullApplicabilityDomain, HotellingT2ApplicabilityDomain,
                                    IsolationForestApplicabilityDomain, KNNApplicabilityDomain,
                                    KernelDensityApplicabilityDomain, LeverageApplicabilityDomain,
                                    PCABoundingBoxApplicabilityDomain, TopKatApplicabilityDomain,
                                    StandardizationApproachApplicabilityDomain, LocalOutlierFactorApplicabilityDomain)
from ._data import data
