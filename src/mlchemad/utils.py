# -*- coding: utf-8 -*-

"""Utility functions."""

from inspect import isclass
from typing import List, Tuple, Union

from sklearn.exceptions import NotFittedError


def check_is_fitted(applicability_domain,
                    attributes: Union[str, List[str], Tuple[str]],
                    msg: str = None,
                    all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the applicability domain is fitted by verifying the presence of
    fitted attributes and otherwise raises a NotFittedError with the given message.
    :param applicability_domain: ApplicabilityDomain instance for which the check is performed.
    :param attributes: attribute name(s) given as string or a list/tuple of strings (default: None)
        e.g.: ``["coef_", "estimator_", ...], "coef_"``
        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.
    :param msg: Message of the Exception to be raised (default: None).
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        applicability domain."  For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "applicability domain, %(name)s, must be fitted before sparsifying".
    :param all_or_any: Specify whether all or any of the given attributes must exist.
    callable, {all, any}, (default all)
    :raise TypeError: If the estimator is a class or not an estimator instance
    :raise NotFittedError: If the attributes are not found.
    """
    if isclass(applicability_domain):
        raise TypeError("{} is a class, not an instance.".format(applicability_domain))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this applicability domain."
        )
    if not hasattr(applicability_domain, "fit"):
        raise TypeError("%s is not an estimator instance." % (applicability_domain))
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
    is_fitted = all_or_any([hasattr(applicability_domain, attr) for attr in attributes])
    if not is_fitted:
        raise NotFittedError(msg % {"name": type(applicability_domain).__name__})
