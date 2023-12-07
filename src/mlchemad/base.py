# -*- coding: utf-8 -*-

"""Base class of applicability domains."""


from abc import ABC, abstractmethod
from typing import Iterable, Union

from sklearn.utils.validation import check_array, column_or_1d

from .utils import check_is_fitted


class ApplicabilityDomain(ABC):
    """Base class applicability domains inherits from."""

    def __init__(self):
        self.fitted_ = False

    def fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        # Ensure is an array and is finite
        X = check_array(X)
        self.num_points, self.num_dims = X.shape
        # Call underlying _fit method
        self._fit(X)
        self.fitted_ = True

    @abstractmethod
    def _fit(self, X):
        """Method to overload to fit the model

        :param X: feature matrix
        """
        pass

    def contains(self, sample) -> Union[bool, Iterable[bool]]:
        """Determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        # Ensure the AD was fitted
        check_is_fitted(self, 'fitted_')
        try:
            # Ensure is 1D sample
            sample = column_or_1d(sample)
        except ValueError:
            sample = check_array(sample, accept_large_sparse=False)
        # Check dimensions
        if sample.ndim == 1 and sample.shape[0] != self.num_dims:
            raise ValueError('sample must have the same number of features as the applicability domain; '
                             f'{sample.shape[0]} and {self.num_dims} respectively')
        elif sample.ndim == 2 and sample.shape[1] != self.num_dims:
            raise ValueError('sample must have the same number of features as the applicability domain; '
                             f'{sample.shape[1]} and {self.num_dims} respectively')
        return self._contains(sample)

    @abstractmethod
    def _contains(self, sample):
        """Method to overload to determine if a sample is in the applicability domain.

        :param sample: sample to check the applicability domain membership of.
        """
        pass
