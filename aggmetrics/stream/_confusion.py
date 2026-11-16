# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""Measures computed from a (weighted) confusion matrix (internal).

The state of a stream evaluator is a confusion matrix ``C`` whose entry ``C[j, k]``
is the weighted number of instances of true class ``j`` predicted as ``k``. Every
measure of the registry is a function of the four one-vs-rest cells of a class, so
a measure over the state is the measure over those cells, reduced by one of the
aggregation operators (:func:`aggmetrics.aggregation.aggregate` does the same on
indicator matrices).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..aggregation import _combine
from ..measures import BinaryMeasure, get_measure

__all__ = ["Audit", "cells_from_confusion", "evaluate", "denominator_side"]


@dataclass(frozen=True)
class Audit:
    """What a value was computed from, reported with every evaluation.

    ``per_class`` holds the measure of each class (NaN where it is undefined),
    ``defined`` is the support indicator ``u`` (True where the class has a value),
    and ``m_active`` is the number of classes with a value. For a binary state
    there is one entry, the positive class.
    """

    per_class: np.ndarray
    defined: np.ndarray
    m_active: int
    weights: np.ndarray

    def __repr__(self) -> str:                           # pragma: no cover - cosmetic
        return (f"Audit(defined={self.defined.astype(int).tolist()}, "
                f"m_active={self.m_active})")


def cells_from_confusion(C, task: str) -> np.ndarray:
    """One-vs-rest ``(tp, fp, fn, tn)`` per class, one row per class.

    Binary: a single row for the positive class (index 1), as in
    ``IndicatorMatrix(..., task='binary')``.
    """
    C = np.asarray(C, float)
    tot = C.sum()
    if task == "binary":
        return np.array([[C[1, 1], C[0, 1], C[1, 0], C[0, 0]]])
    tp = np.diag(C)
    fn = C.sum(axis=1) - tp
    fp = C.sum(axis=0) - tp
    tn = tot - tp - fn - fp
    return np.column_stack([tp, fp, fn, tn])


def denominator_side(measure: BinaryMeasure) -> str:
    """``'prediction'`` if the denominator is a combination of ``tp + fp`` and
    ``fn + tn``, else ``'label'``.

    Under verification latency the prediction side of the state is exact (its
    column sums do not depend on the interim estimates) and the label side is not.
    Non-decomposable measures have no denominator and are treated as label side.
    """
    if measure.linear is None:
        return "label"
    b = measure.linear[1]
    pred_side = math.isclose(b[0], b[1]) and math.isclose(b[2], b[3])
    return "prediction" if pred_side else "label"


def evaluate(C, task: str, measure, aggregation: str = "micro", *,
             convention: str = "exclude", tol: float = 0.0, support_C=None):
    """Value of ``measure`` over the confusion matrix ``C`` and its audit.

    ``tol`` is the support tolerance: a class whose denominator, taken on
    ``support_C`` (default ``C``), is at most ``tol`` times the total mass has no
    value. It is zero under a sliding window and a small positive number under
    exponential fading, where a cell never reaches exactly zero.
    """
    meas = get_measure(measure)
    if aggregation not in ("micro", "macro", "weighted"):
        raise ValueError("aggregation must be 'micro', 'macro' or 'weighted'")
    cells = cells_from_confusion(C, task)
    sup = cells if support_C is None else cells_from_confusion(support_C, task)
    mass = float(np.sum(C))
    parts = np.array([meas(*c) for c in cells], float)
    if tol > 0 and meas.linear is not None:
        den = np.array([meas.denominator(tuple(c)) for c in sup])
        parts = np.where(den <= tol * mass, np.nan, parts)
    elif support_C is not None and meas.linear is not None:
        den = np.array([meas.denominator(tuple(c)) for c in sup])
        parts = np.where(den <= 0, np.nan, parts)
    defined = ~np.isnan(parts)
    weights = cells[:, 0] + cells[:, 2]
    audit = Audit(parts, defined, int(defined.sum()), weights)
    if aggregation == "micro":
        if tol > 0 or support_C is not None:
            # the pooled value is defined when some class is supported
            if not defined.any():
                return float("nan"), audit
        pooled = cells.sum(axis=0)
        if task == "binary":
            pooled = cells[0]
        return float(meas(*pooled)), audit
    w = weights if aggregation == "weighted" and float(weights.sum()) > 0 else None
    return _combine(parts, convention, w), audit
