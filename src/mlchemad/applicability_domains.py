# -*- coding: utf-8 -*-

"""Definitions of applicability domains."""


from math import floor
from typing import Union, Tuple, Optional

import numpy as np
import scipy
from numpy.random import RandomState
from scipy.spatial.distance import cdist, _METRICS as dist_fns
from scipy.stats import f as Fdistrib
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neighbors._kde import KernelDensity
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.utils.extmath import stable_cumsum

from .base import ApplicabilityDomain


class BoundingBoxApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined as the allowed feature range.
    Samples falling outside at least one range of the fitted features are considered as outliers.
    """

    def __init__(self, percentiles: Optional[Tuple[float, float]] = (0.001, 0.999), range_=('min', 'max')):
        """Instantiate a BoundingBoxApplicabilityDomain.

        :param percentiles: minimum and maximum percentile of features determining the bounding box (default: (0.01, 0.99)).
        :param range_: minimum and maximum values of the bounding box; ignored if percentiles is not None.
        If ('min', 'max'), extremum values of each feature.
        If (int | float, int | float), set the same extremum values for all features.
        If (List[int | float], List[int | float]), set the extremum values per feature.
        (default: ('min', 'max'))
        """
        super().__init__()
        if percentiles is None:
            if not isinstance(range_, tuple):
                raise ValueError('range_ of the bounding box must be a tuple')
            if len(range_) != 2:
                raise ValueError('range_ of the bounding box must be a tuple of 2 values and only 2 values')
            if (isinstance(range_[0], str) and range_[0] != 'min') or (
                    isinstance(range_[1], str) and range_[1] != 'max'):
                raise ValueError('only \'min\' and \'max\' are valid str values for range_')
            if (isinstance(range_[0], (int, float)) and isinstance(range_[1], (int, float)) and range_[0] > range_[1]):
                raise ValueError('the first value for \'range_\' must be less than the second value')
            if (isinstance(range_[0], list) and isinstance(range_[1], list) and len(range_[0]) != len(range_[1])):
                raise ValueError('values of \'range_\' must have the same length')
            if ((isinstance(range_[0], list) and not all(isinstance(x, (int, float)) for x in range_[0]) or
                 isinstance(range_[1], list) and not all(isinstance(x, (int, float)) for x in range_[1]))):
                raise ValueError('list values of \'range_\' must contain only ints or floats')
        elif percentiles is not None:
            if (not isinstance(percentiles, tuple) or len(percentiles) != 2):
                raise ValueError('\'percentiles\' must be a tuple of the lowest and highest percentiles to consider')
            if ((isinstance(percentiles[0], (int, float)) and percentiles[0] < 0 or percentiles[0] > 1) or
                    (isinstance(percentiles[1], (int, float)) and percentiles[1] < 0 or percentiles[1] > 1)):
                raise ValueError('value of \'percentiles\' must be greater than 0.0 and lower than 1.0')
            if (isinstance(percentiles[0], (int, float)) and
                    isinstance(percentiles[1], (int, float)) and percentiles[0] > percentiles[1]):
                raise ValueError('the first value for \'percentiles\' must be less than the second value')
        elif not isinstance(range_[0], (str, int, float, list)) or not isinstance(range_[1], (str, int, float, list)):
            raise ValueError('\'range_\' values must be of the following types: (str, int, float, list)')
        elif (not isinstance(percentiles[0], (str, int, float, list)) or
              not isinstance(percentiles[1], (str, int, float, list))):
                raise ValueError('\'percentiles\' values must be of the following types: (str, int, float, list)')
        self.compute_minmax = (range_ == ('min', 'max'))
        self.constant_value_min = range_[0] if isinstance(range_[0], (int, float, list)) else None
        self.constant_value_max = range_[1] if isinstance(range_[0], (int, float, list)) else None
        self.percentiles_min_max = percentiles

    def _fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        # Percentiles
        if self.percentiles_min_max is not None:
            self.min_, self.max_ = np.percentile(X,
                                                 (self.percentiles_min_max[0] * 100, self.percentiles_min_max[1] * 100),
                                                 axis=0)
            return self
        # Automatic min, max
        if self.compute_minmax:
            self.min_ = X.min(axis=0)
            self.max_ = X.max(axis=0)
            return self
        # Constant values
        if isinstance(self.constant_value_min, (int, float)):
            self.min_ = np.array([self.constant_value_min] * X.shape[1])
            self.max_ = np.array([self.constant_value_max] * X.shape[1])
            return self
        if isinstance(self.constant_value_min, list):
            if len(self.constant_value_min) != X.shape[1] or len(self.constant_value_max) != X.shape[1]:
                raise ValueError('Number of features does not match constant values defined during instantiation')
            self.min_ = np.array(self.constant_value_min)
            self.max_ = np.array(self.constant_value_max)
            return self

    def _contains(self, sample):
        """Determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        if sample.ndim == 1:
            return ((sample >= self.min_) & (sample <= self.max_)).all()
        return ((sample >= self.min_) & (sample <= self.max_)).all(axis=1)


class ConvexHullApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined as the convex hull of the training feature matrix."""

    def __init__(self):
        super().__init__()

    def _fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        # Point matrix transposed
        self.points = np.r_[X.T, np.ones((1, self.num_points))].astype(np.float16)

    def _contains(self, sample):
        """Determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        # Determining if a point belongs to the convex hull of a set of points
        # corresponds to determining if there exists a linear combination of a
        # subset of the points satisfying the following:
        #           - the variable weights of the linear combination are all positive
        #           - the sum of the variable weights equals 1
        #             (this ensures the point does not lie outside the convex hull)
        if sample.ndim == 1:
            sample = np.r_[sample, np.ones(1)].astype(np.float16)  # concatenate 1 at the end of the vector
            # Attempt to solve the inequality
            lp = scipy.optimize.linprog(np.ones(self.num_points, dtype=np.float16), A_eq=self.points, b_eq=sample)
            return lp.success
        else:
            samples = np.c_[sample, np.ones(sample.shape[0])].astype(np.float16)  # concatenate 1 at the end of the vector
            # Attempt to solve the inequality
            return np.array([scipy.optimize.linprog(np.ones(self.num_points, dtype=np.float16),
                                                    A_eq=self.points, b_eq=sample_.reshape((1, -1))
                                                    ).success
                             for sample_ in samples])


class PCABoundingBoxApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined as the bounding box around the principal components of the training feature matrix."""

    def __init__(self, scaling: str = 'robust', explained_var: float = 0.9,
                 random_state: Union[int, RandomState] = 1234,
                 scaler_kwargs=None, pca_kwargs=None):
        """Create the convex hull applicability domain. Transforms the input features
        with principal component analysis (PCA) to ensure a convex hull is found.

        :param scaling: scaling method; must be one of 'robust', 'minmax', 'maxabs' or 'standard' (default: 'robust')
        :param explained_var: minimum value principal components' cumulative variance must reach to be included
        :param random_state: random state of the principal component analysis (PCA)
        :param scaler_kwargs: additional parameters to supply to the scaler
        :param pca_kwargs: additional parameters to supply to the PCA
        """
        super().__init__()
        if pca_kwargs is None:
            pca_kwargs = {}
        if scaler_kwargs is None:
            scaler_kwargs = {}
        scaling_methods = ('robust', 'minmax', 'maxabs', 'standard')
        if scaling not in scaling_methods:
            raise ValueError(f'scaling method must be one of {scaling_methods}')
        if 'random_state' in pca_kwargs:
            raise ValueError('pca_kwargs must not set the following parameters: \'random_state\'')
        # Scaler for input data
        if scaling == 'robust':
            self.scaler = RobustScaler(**scaler_kwargs)
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler(**scaler_kwargs)
        elif scaling == 'maxabs':
            self.scaler = MaxAbsScaler(**scaler_kwargs)
        elif scaling == 'standard':
            self.scaler = StandardScaler(**scaler_kwargs)
        else:
            raise NotImplementedError('scaling methof not implemented')
        self.min_explained_var = explained_var
        # PCA ensuring the simplex is not flat
        self.pca = PCA(random_state=random_state,
                       **pca_kwargs)

    def _fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        # Fit scaler and PCA
        X = self.scaler.fit_transform(X)
        self.pca.fit(X)
        # Determine the number of components
        ratio_cumsum = stable_cumsum(self.pca.explained_variance_ratio_)
        n_components = np.searchsorted(ratio_cumsum, self.pca.n_components_, side="right") + 1
        # Modify the PCA object in place
        self.pca.components_ = self.pca.components_[:n_components]
        self.pca.n_components_ = n_components
        self.pca.explained_variance_ = self.pca.explained_variance_[:n_components]
        self.pca.explained_variance_ratio_ = self.pca.explained_variance_ratio_[:n_components]
        self.pca.singular_values_ = self.pca.singular_values_[:n_components]
        # Determine extremum values
        X = self.pca.transform(X)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.fitted_ = True
        return self

    def _contains(self, sample):
        """Determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        # Scale input features
        if sample.ndim == 1:
            sample = self.pca.transform(self.scaler.transform(sample.reshape((1, len(sample)))))
            return ((sample >= self.min_) & (sample <= self.max_)).all()
        sample = self.pca.transform(self.scaler.transform(sample))
        return ((sample >= self.min_) & (sample <= self.max_)).all(axis=1)


class TopKatApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined as TOPKAT's Optimal Prediction Space (OPS).

    Reference:
    Gombar, Vijay K. (1996). Method and apparatus for validation of model-based predictions (US Patent No. 6-036-349) USPTO.
    """

    def __init__(self):
        """Instantiate a TopKatApplicabilityDomain."""
        super().__init__()

    def _fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        # Keep scaling factors
        self.X_min_, self.X_max_ = X.min(axis=0), X.max(axis=0)
        # Replace extremums for features with no variance
        # Obtain the S-space from the input P-space
        S = (2 * X - self.X_max_ - self.X_min_) / np.where((self.X_max_ - self.X_min_) != 0,
                                                           (self.X_max_ - self.X_min_),1)
        # Add 1-filled column at the beginning of S
        S = np.c_[np.ones(S.shape[0]), S]
        # Obtain eigen values and vectors
        self.eigen_val, self.eigen_vec = np.linalg.eig(S.T.dot(S))
        self.eigen_val, self.eigen_vec = np.real(self.eigen_val), np.real(self.eigen_vec)
        # Determine the OPS
        OPS = S.dot(self.eigen_vec)
        # Determine ranges of the OPS
        self.OPS_min_ = OPS.min(axis=0)
        self.OPS_max_ = OPS.max(axis=0)

    def _contains(self, sample):
        """Determine if a sample is in the OPS applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        # Scale the sample
        Ssample = (2 * sample - self.X_max_ - self.X_min_) / np.where((self.X_max_ - self.X_min_) != 0,
                                                                      (self.X_max_ - self.X_min_),1)
        # Add intercept
        if sample.ndim == 1:
            Ssample = np.c_[1, Ssample.reshape((1, -1))]
        else:
            Ssample = np.c_[np.ones((sample.shape[0], 1)), Ssample]
        # Obtain position of sample in the OPS
        OPS_sample = Ssample.dot(self.eigen_vec)
        # Determine the distance to the OPS
        denom = np.divide(np.ones_like(self.eigen_val, dtype=float),
                          self.eigen_val,
                          out=np.zeros_like(self.eigen_val),
                          where=self.eigen_val!=0)
        dOPS = (OPS_sample * OPS_sample).dot(denom)
        if sample.ndim == 1 and isinstance(dOPS, np.ndarray):
            dOPS = dOPS.item()
        return dOPS < (5 * (self.num_dims)) / (2 * self.num_points)


class LeverageApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined using the leverage approach."""

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def _fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        # Center data
        X = self.scaler.fit_transform(X)
        self.var_covar = np.linalg.inv(X.T.dot(X))
        self.threshold = 3 * (self.num_dims + 1) / self.num_points

    def _contains(self, sample):
        """Determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        # Calculate leverage
        if sample.ndim == 1:
            sample = self.scaler.transform(sample.reshape(1, -1))
            h = sample.dot(self.var_covar).dot(sample.T)
        else:
            sample = self.scaler.transform(sample)
            h = np.diag(sample.dot(self.var_covar).dot(sample.T))
        return h < self.threshold


class HotellingT2ApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined using the Hotelling T² approach."""

    def __init__(self, significance: float = 0.05):
        super().__init__()
        self.alpha = significance

    def _fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        # Determine the Hotelling T² ellipse
        ellipse = (1 / self.num_points) * (X ** 2).sum(axis=0)
        # F statistic
        fs = (self.num_points - 1) / self.num_points * self.num_dims * (self.num_points ** 2 - 1) / (self.num_points * (self.num_points - self.num_dims))
        fs *= Fdistrib.ppf(1 - self.alpha, self.num_dims, self.num_points - self.num_dims)
        # Obtain T² values
        self.t2 = np.sqrt(fs * ellipse)

    def _contains(self, sample):
        """Determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        # Determine volume protrusions
        if sample.ndim == 1:
            return (sample ** 2 / self.t2 ** 2).sum() <= 1
        return (sample ** 2 / self.t2 ** 2).sum(axis=1) <= 1


class KernelDensityApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined by the data kernel-density estimate."""

    def __init__(self, threshold: float = 0.01, kernel: str = 'gaussian',
                 bandwidth: str = 'scott', metric: str = 'euclidean'):
        """Instantiate a BoundingBoxApplicabilityDomain.

        :param threshold: minimum probability for inlier detection
        :param kernel: kernel to be used
        :param bandwidth: bandwidth of the kernel
        :param metric: Metric to use for distance computation
        """
        if 0 > threshold or threshold > 1:
            raise ValueError('threshold value must lie between 0 and 1')
        super().__init__()
        self.kde = KernelDensity(bandwidth=bandwidth,
                                 kernel=kernel,
                                 metric=metric)
        self.threshold = threshold

    def _fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        self.kde.fit(X)
        # Obtain probabilities of the training set
        probs = np.exp(self.kde.score_samples(X))
        # Determine the cut-off probability
        self.cutoff = np.percentile(probs, self.threshold * 100)

    def _contains(self, sample):
        """Determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        if sample.ndim == 1:
            prob = np.exp(self.kde.score_samples(sample.reshape(1, -1)))
            return prob >= self.cutoff
        probs = np.exp(self.kde.score_samples(sample))
        return probs >= self.cutoff


class IsolationForestApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined using isolation forest."""

    def __init__(self):
        super().__init__()
        self.isol = IsolationForest(max_samples=0.7, max_features=0.7,  bootstrap=True, random_state=1234)

    def _fit(self, X):
        self.isol.fit(X)

    def _contains(self, sample):
        if sample.ndim == 1:
            return self.isol.predict(sample.reshape(1, -1)).item() == 1
        else:
            return self.isol.predict(sample) == 1


class CentroidDistanceApplicabilityDomain(ApplicabilityDomain):
    def __init__(self, threshold: float = None, dist: str = 'euclidean'):
        f"""Create the centroid applicability domain.

        :param threshold: distance threshold
        :param dist: kNN distance to be calculated (default: euclidean); one of {list(dist_fns.keys())}
        """
        super().__init__()
        if dist not in dist_fns.keys():
            raise NotImplementedError('distance type is not available')
        else:
            self.dist = dist
        self.scaler = StandardScaler()
        # if threshold is not None and not (threshold > 0 and threshold < 1):
        #     raise ValueError('Threshold must either be None or between 0 and 1')
        self.threshold = threshold

    def _fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        X = self.scaler.fit_transform(X)
        self.centroid = np.mean(X, axis=0).reshape((1, -1))
        # Determine distances to the centroid
        distance = cdist(X, self.centroid, metric=self.dist)
        if self.threshold is None:
            q1, q3 = np.percentile(distance.ravel(), [25, 75])
            self.threshold = q3 + 1.5 * (q3 - q1)
        else:
            self.threshold = np.percentile(distance.ravel(), self.threshold)

    def _contains(self, sample):
        """Determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        if sample.ndim == 1:
            sample = self.scaler.transform(sample.reshape(1, -1))
        else:
            sample = self.scaler.transform(sample)
        distance = cdist(sample, self.centroid, metric=self.dist).ravel()
        if sample.ndim == 1:
            return distance.item() <= self.threshold
        return (distance <= self.threshold)


class KNNApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined using K-nearest neighbours."""

    def __init__(self, k: int = 5,
                 alpha: float = 0.95,
                 hard_threshold: float = None,
                 scaling: Optional[str] = 'robust',
                 dist: str = 'euclidean',
                 scaler_kwargs=None,
                 njobs: int=1):
        f"""Create the k-Nearest Neighbor applicability domain.

        :param k: number of nearest neighbors
        :param alpha: ratio of inlier samples (default: 0.95) calculated from the training set; ignored if hard_threshold is set
        :param hard_threshold: samples with a distance greater or equal to this threshold will be considered outliers
        :param scaling: scaling method; must be one of 'robust', 'minmax', 'maxabs', 'standard' or None (default: 'robust')
        :param dist: kNN distance to be calculated (default: euclidean); one of {list(dist_fns.keys())}
        :param scaler_kwargs: additional parameters to supply to the scaler
        :param njobs: number of parallel processes used to fit the kNN model
        """
        super().__init__()
        if scaler_kwargs is None:
            scaler_kwargs = {}
        if alpha > 1 or alpha < 0:
            raise ValueError('alpha must lie between 0 and 1')
        scaling_methods = ('robust', 'minmax', 'maxabs', 'standard', None)
        if scaling not in scaling_methods:
            raise ValueError(f'scaling method must be one of {scaling_methods}')
        # Scaler for input data
        if scaling == 'robust':
            self.scaler = RobustScaler(**scaler_kwargs)
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler(**scaler_kwargs)
        elif scaling == 'maxabs':
            self.scaler = MaxAbsScaler(**scaler_kwargs)
        elif scaling == 'standard':
            self.scaler = StandardScaler(**scaler_kwargs)
        elif scaling is None:
            self.scaler = None
        else:
            raise NotImplementedError('scaling method not implemented')
        if dist not in dist_fns.keys():
            raise NotImplementedError('distance type is not available')
        else:
            self.dist = dist
        self.k = k
        self.alpha = alpha
        self.hard_threshold = hard_threshold
        self.nn = NearestNeighbors(n_neighbors=k, metric=dist, n_jobs=njobs)

    def _fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        # Normalize the data
        self.X_norm = self.scaler.fit_transform(X) if self.scaler is not None else X
        # Fit the NN
        self.nn.fit(self.X_norm)
        # Find the distance to the kNN
        self.kNN_dist = self.nn.kneighbors(self.X_norm, return_distance=True)[0].mean(axis=1)
        kNN_train_distance_sorted_ = np.trim_zeros(np.sort(self.kNN_dist))
        # Find the confidence threshold
        if self.hard_threshold:
            self.threshold_ = self.hard_threshold
        else:
            self.threshold_ = kNN_train_distance_sorted_[floor(kNN_train_distance_sorted_.shape[0] * self.alpha) - 1]
        return self

    def _contains(self, sample):
        """Determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        # Scale input features
        if self.scaler is not None:
            if sample.ndim == 1:
                sample = self.scaler.transform(sample.reshape((1, len(sample))))
            else:
                sample = self.scaler.transform(sample)
        # Calculate kNN distance to the training set
        kNN_sample_dist = self.nn.kneighbors(sample, return_distance=True)[0].mean(axis=1)
        # Threshold normalized distance
        norm_dist = kNN_sample_dist / self.threshold_
        if self.hard_threshold:
            # Threshold is excluded when given a cutoff value
            return norm_dist < 1
        # Otherwise the threshold is included
        return norm_dist <= 1


class StandardizationApproachApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined using the standardization approach.

    Reference:
    Roy, Kar & Ambure. In: Chemometrics and Intelligent Laboratory Systems, Volume 145, 2015, Pages 22-29.
    """

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def _fit(self, X):
        self.scaler.fit(X)

    def _contains(self, sample):
        if sample.ndim == 1:
            if sample.max() <= 3:
                return True
            elif sample.min() > 3:
                return False
            else:
                return sample.mean() + 1.28 * sample.std() <= 3
        else:
            return np.array([True
                             if s.max() <= 3
                             else (False
                                   if s.min() > 3
                                   else s.mean() + 1.28 * s.std() <= 3)
                             for s in sample])


class LocalOutlierFactorApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined by the local deviation of
    the density of a given sample with respect to its neighbors.

    Reference:
    Breunig et al., In: Proc. 2000 ACM SIGMOD Int. Conf. Manag. Data, ACM, New York, NY, USA, 2000, 93–104.
    """

    def __init__(self, k: int = 5,
                 threshold: float = 1,
                 scaling: str = 'minmax',
                 dist: str = 'euclidean',
                 contamination: float = 0.10,
                 scaler_kwargs=None):
        f"""Create the local outlier factor applicability domain.

        :param k: number of nearest neighbors
        :param threshold: samples with a distance greater or equal to this threshold will be considered outliers
        :param scaling: scaling method; must be one of 'robust', 'minmax', 'maxabs', 'standard' or None (default: 'robust')
        :param dist: kNN distance to be calculated (default: euclidean); one of {list(dist_fns.keys())}
        :param contamination: exprected proportion of outliers in the data set
        :param scaler_kwargs: additional parameters to supply to the scaler
        """
        super().__init__()
        if scaler_kwargs is None:
            scaler_kwargs = {}
        if contamination > 1 or contamination < 0:
            raise ValueError('contamination must lie between 0 and 1')
        scaling_methods = ('robust', 'minmax', 'maxabs', 'standard', None)
        if scaling not in scaling_methods:
            raise ValueError(f'scaling method must be one of {scaling_methods}')
        # Scaler for input data
        if scaling == 'robust':
            self.scaler = RobustScaler(**scaler_kwargs)
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler(**scaler_kwargs)
        elif scaling == 'maxabs':
            self.scaler = MaxAbsScaler(**scaler_kwargs)
        elif scaling == 'standard':
            self.scaler = StandardScaler(**scaler_kwargs)
        elif scaling is None:
            self.scaler = None
        else:
            raise NotImplementedError('scaling method not implemented')
        if dist not in dist_fns.keys():
            raise NotImplementedError('distance type is not available')
        else:
            self.dist = dist
        self.k = k
        self.contamination = contamination
        self.threshold = threshold
        self.lof = LocalOutlierFactor(n_neighbors=k, metric=dist, contamination=contamination,
                                      novelty=True)

    def _fit(self, X):
        X = self.scaler.fit_transform(X)
        self.lof.fit(X)

    def _contains(self, sample):
        if sample.ndim == 1:
            return self.lof.predict(sample.reshape(1, -1)).item() < self.threshold
        else:
            return self.lof.predict(sample) < 1
