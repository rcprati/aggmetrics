# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Multi-output classification: one indicator matrix per target.

With ``T`` targets, the paper (Section 6) represents the data as a block-structured
indicator matrix ``[Y(1) | ... | Y(T)]`` and adds a fourth level of averaging, the
*target-wise* operator, which averages the measure computed within each block. The
three operators of the single-target case apply unchanged to the full matrix, so
the choice is between averaging over targets and pooling them.

For an aggregation-decomposable measure, pooling every target (``'micro'``) equals the
target-wise average weighted by the denominator of each target (Theorem 1 applied to
the blocks), and the two coincide exactly when the targets have equal denominators
(Corollary 1).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .aggregation import _combine
from .core import IndicatorMatrix
from .measures import BinaryMeasure


def multioutput_aggregate(ims: Sequence[IndicatorMatrix],
                          measure: str | BinaryMeasure, how: str = "target",
                          *, within: str = "micro",
                          convention: str = "exclude") -> float:
    """Aggregate a measure over several targets.

    Parameters
    ----------
    ims
        One :class:`~aggmetrics.core.IndicatorMatrix` per target, on the same
        examples (same number of rows, in the same order). The targets may have
        different numbers of classes.
    how
        ``'target'`` averages the measure computed within each target
        (the operator of the paper); ``'micro'``, ``'macro'`` and ``'weighted'``
        apply that operator to the block-structured matrix of all targets, so that
        ``'micro'`` pools the cells of every target and ``'macro'`` averages over
        every class of every target.
    within
        With ``how='target'``, the operator used inside each target.
    convention
        How an undefined value is treated, as elsewhere in the package.

    Examples
    --------
    >>> import numpy as np
    >>> from aggmetrics import IndicatorMatrix, multioutput_aggregate
    >>> y1 = np.array([0, 1, 1, 0, 1, 0]); p1 = np.array([0, 1, 0, 0, 1, 1])
    >>> y2 = np.array([0, 1, 2, 2, 1, 0]); p2 = np.array([0, 2, 2, 1, 1, 1])
    >>> ims = [IndicatorMatrix(y1, p1, task="multiclass"),
    ...        IndicatorMatrix(y2, p2, task="multiclass")]
    >>> round(multioutput_aggregate(ims, "recall", "target", within="macro"), 3)
    0.583
    """
    ims = list(ims)
    if not ims:
        raise ValueError("at least one target is required")
    if len({im.n for im in ims}) != 1:
        raise ValueError("all targets must be evaluated on the same examples")
    if any(im.mask is not None or im.tnorm is not None for im in ims):
        raise ValueError("masks and t-norms are not supported")
    if how == "target":
        values = [im.compute(measure, within, convention=convention) for im in ims]
        return _combine(np.array(values), convention)
    if how not in ("micro", "macro", "weighted"):
        raise ValueError("how must be 'target', 'micro', 'macro' or 'weighted'")
    Y = np.hstack([np.asarray(im.Y, float) for im in ims])
    P = np.hstack([np.asarray(im.P, float) for im in ims])
    return IndicatorMatrix(Y, P, task="multilabel").compute(
        measure, how, convention=convention)
