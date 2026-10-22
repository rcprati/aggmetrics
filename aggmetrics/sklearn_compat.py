# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
scikit-learn interoperability.

``make_scorer`` produces an object usable anywhere scikit-learn accepts a
``scoring`` argument, giving access to combinations the library does not expose ---
macro-MCC, exemplar-Jaccard, weighted informedness --- while keeping the sign
convention that greater is always better.

``sklearn_average`` documents the correspondence with the ``average`` parameter,
which is exact:

===============================  ==================
``sklearn`` ``average``          operator
===============================  ==================
``'micro'``                      ``sigma_1``
``'macro'``                      ``sigma_m`` (uniform weights)
``'weighted'``                   ``sigma_m`` (prevalence weights)
``'samples'``                    ``sigma_n``
===============================  ==================
"""
from __future__ import annotations

from typing import Callable, Optional

from .core import IndicatorMatrix
from .measures import get_measure

__all__ = ["make_scorer", "score", "sklearn_average", "SKLEARN_AVERAGE_MAP"]

SKLEARN_AVERAGE_MAP = {
    "micro": "micro",
    "macro": "macro",
    "weighted": "weighted",
    "samples": "exemplar",
}


def sklearn_average(average: Optional[str]) -> str:
    """Translate a scikit-learn ``average`` value into an operator name."""
    if average is None:
        return "macro"
    if average not in SKLEARN_AVERAGE_MAP:
        raise ValueError(f"unknown average {average!r}; expected one of "
                         f"{sorted(SKLEARN_AVERAGE_MAP)}")
    return SKLEARN_AVERAGE_MAP[average]


def score(y_true, y_pred, measure: str, aggregation: str = "micro",
          *, task: Optional[str] = None, convention: str = "exclude",
          threshold: float = 0.5) -> float:
    """One-shot scoring function, in the style of ``sklearn.metrics``."""
    im = IndicatorMatrix(y_true, y_pred, task=task, threshold=threshold)
    return im.compute(measure, aggregation, convention=convention)


def make_scorer(measure: str, aggregation: str = "micro",
                *, task: Optional[str] = None, convention: str = "exclude",
                threshold: float = 0.5) -> Callable:
    """Build a scikit-learn compatible scorer.

    The returned object follows the ``scorer(estimator, X, y)`` protocol and
    negates measures for which smaller is better, so that a larger score is always
    preferable --- the convention scikit-learn's model-selection tools assume.

    Examples
    --------
    >>> from aggmetrics.sklearn_compat import make_scorer   # doctest: +SKIP
    >>> macro_mcc = make_scorer('mcc', 'macro')              # doctest: +SKIP
    >>> cross_val_score(clf, X, y, scoring=macro_mcc)        # doctest: +SKIP
    """
    meas = get_measure(measure)
    sign = 1.0 if meas.greater_is_better else -1.0

    def _scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        im = IndicatorMatrix(y, y_pred, task=task, threshold=threshold)
        return sign * im.compute(meas, aggregation, convention=convention)

    _scorer.__name__ = f"{aggregation}_{meas.name}"
    _scorer.__doc__ = (f"{aggregation}-{meas.name} scorer "
                       f"({'greater' if meas.greater_is_better else 'smaller'} "
                       f"is better in the underlying measure).")
    return _scorer
