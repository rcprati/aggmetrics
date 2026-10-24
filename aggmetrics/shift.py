# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Evaluating a measure under a different class prior.

A classifier's conditional rates, ``P(predicted k | true j)``, are what a test set
measures. The measures that are not skew-invariant (Definition 11 and Theorem 5 of
the companion paper; Proposition 6 for one-vs-rest rates) also depend on how often
each class occurs, so the same classifier scores differently where the prevalence
differs. ``under_prior`` answers "what would this measure be if the classes occurred
with these frequencies?" by keeping the rates of the data and replacing the class
sizes.

The assumption is the usual one for a change of prior (label shift): the rates of
each class do not change, only how many examples it has. It holds when the
class-conditional distribution of the inputs is the same in both settings. It does
not hold when the shift also changes what the examples look like.
"""
from __future__ import annotations

import numpy as np

from ._rates import Rates
from .measures import BinaryMeasure


class PriorShifted:
    """The data of an :class:`~aggmetrics.core.IndicatorMatrix` at another prior.

    Built by :func:`under_prior`. It offers the parts of the ``IndicatorMatrix``
    interface that depend only on the confusion matrix: ``compute``,
    ``accuracy``, ``counts`` and ``per_class``.
    """

    def __init__(self, model: Rates, priors, task: str):
        self._model = model
        self.priors = np.asarray(priors, float)
        self.task = task

    @property
    def n(self) -> int:
        """Total number of examples, as in the original data."""
        return self._model.n

    def per_class(self) -> np.ndarray:
        """Per-class (or per-label) ``(tp, fp, fn, tn)``, one row per class."""
        return np.column_stack(self._model.cells(self._model.theta))

    @property
    def counts(self):
        """Pooled ``(tp, fp, fn, tn)``."""
        return tuple(self.per_class().sum(axis=0))

    def compute(self, measure: str | BinaryMeasure, aggregation: str = "micro",
                *, convention: str = "exclude") -> float:
        """Value of ``measure`` at the new prior, under the operator ``aggregation``."""
        if aggregation not in ("micro", "macro", "weighted"):
            raise ValueError("aggregation must be 'micro', 'macro' or 'weighted': "
                             "the exemplar operator depends on the examples")
        return self._model.value(measure, aggregation, self._model.theta, convention)

    def accuracy(self) -> float:
        """Accuracy, defined as in ``IndicatorMatrix.accuracy``."""
        if self.task == "multiclass":
            return float(self.counts[0] / self.n)
        return self.compute("label_accuracy", "micro")


def under_prior(im, priors) -> PriorShifted:
    """Re-evaluate ``im`` as if its classes occurred with the frequencies ``priors``.

    Parameters
    ----------
    im
        A binary, multiclass or multilabel :class:`~aggmetrics.core.IndicatorMatrix`
        without a mask or a t-norm.
    priors
        Binary: the prevalence of the positive class, a number in [0, 1].
        Multiclass: the frequency of each class, non-negative and summing to 1.
        Multilabel: the prevalence of each label, one number in [0, 1] per label.

    Returns
    -------
    PriorShifted
        An object with ``compute``, ``accuracy``, ``counts`` and ``per_class``. At
        the prior of the data itself it reproduces ``im`` exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from aggmetrics import IndicatorMatrix, under_prior
    >>> y = np.array([1] * 80 + [0] * 120)
    >>> p = np.array([1] * 60 + [0] * 20 + [1] * 30 + [0] * 90)
    >>> im = IndicatorMatrix(y, p, task="binary")
    >>> round(im.compute("precision"), 3)
    0.667
    >>> shifted = under_prior(im, 0.05)             # 5% positives instead of 40%
    >>> round(shifted.compute("precision"), 3)
    0.136
    >>> round(shifted.compute("recall"), 3) == round(im.compute("recall"), 3)
    True
    """
    model = Rates(im)
    return PriorShifted(model.with_priors(priors), priors, im.task)
