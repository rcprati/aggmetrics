# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
The three aggregation operators, plus the conventions they require.

Given indicator matrices :math:`Y, \\hat{Y} \\in \\{0,1\\}^{n \\times m}`, the four
cell matrices are formed entrywise and then reduced in one of three ways:

===============  ==========================  ====================================
Operator         Reduction                   Conventional name
===============  ==========================  ====================================
``sigma_1``      over all entries            micro
``sigma_m``      column-wise, then averaged  macro (or weighted)
``sigma_n``      row-wise, then averaged     exemplar (``average='samples'``)
===============  ==========================  ====================================

Column- and row-wise aggregation divide before averaging, so an individual entry
can be undefined when its denominator vanishes. Because the resolution changes the
reported value, it is an explicit parameter rather than an implicit default. The
package defaults to ``'exclude'``, which averages over the defined entries only.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from .measures import BinaryMeasure, get_measure

__all__ = ["cells", "sigma_1", "sigma_m", "sigma_n", "sigma_k",
           "aggregate", "fold_aggregate",
           "UNDEFINED_CONVENTIONS", "Aggregation"]

Aggregation = Literal["micro", "macro", "weighted", "exemplar"]
UNDEFINED_CONVENTIONS = ("exclude", "zero", "one", "nan")


def _as2d(a) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def cells(Y, P, mask=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Entrywise TP, FP, FN, TN matrices.

    Works for crisp indicator matrices and, unchanged, for soft ones in
    ``[0, 1]`` when they have already been combined by a t-norm (see
    :mod:`aggmetrics.tnorms`).

    ``mask`` is an optional observation mask: entries where ``mask == 0`` are
    treated as unobserved and contribute nothing, which covers partial labels,
    semi-supervised evaluation and masked-out abstentions.
    """
    Y, P = _as2d(Y).astype(float), _as2d(P).astype(float)
    if Y.shape != P.shape:
        raise ValueError(f"shape mismatch: Y is {Y.shape}, predictions are {P.shape}")
    TP, FP, FN, TN = Y * P, (1 - Y) * P, Y * (1 - P), (1 - Y) * (1 - P)
    if mask is not None:
        M = _as2d(mask).astype(float)
        if M.shape != Y.shape:
            raise ValueError(f"mask shape {M.shape} does not match {Y.shape}")
        TP, FP, FN, TN = TP * M, FP * M, FN * M, TN * M
    return TP, FP, FN, TN


def sigma_1(X: np.ndarray) -> float:
    """Global aggregation: sum over all entries."""
    return float(np.sum(X))


def sigma_m(X: np.ndarray) -> np.ndarray:
    """Column-wise aggregation: one value per class or label."""
    return np.asarray(X).sum(axis=0)


def sigma_n(X: np.ndarray) -> np.ndarray:
    """Row-wise aggregation: one value per example."""
    return np.asarray(X).sum(axis=1)


def sigma_k(cell_list: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fold-wise aggregation across K cross-validation splits.

    Sums the four cell matrices (TP, FP, FN, TN) across multiple folds.
    """
    if not cell_list:
        raise ValueError("cell_list must not be empty")
    TP = sum(c[0] for c in cell_list)
    FP = sum(c[1] for c in cell_list)
    FN = sum(c[2] for c in cell_list)
    TN = sum(c[3] for c in cell_list)
    return TP, FP, FN, TN


def _combine(values: np.ndarray, convention: str,
             weights: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=float)
    defined = ~np.isnan(values)
    if convention == "nan":
        return float(np.nan) if not defined.all() else float(np.average(
            values, weights=weights))
    if convention == "exclude":
        if not defined.any():
            return float("nan")
        w = None if weights is None else np.asarray(weights, float)[defined]
        return float(np.average(values[defined], weights=w))
    if convention in ("zero", "one"):
        filled = np.where(defined, values, 0.0 if convention == "zero" else 1.0)
        return float(np.average(filled, weights=weights))
    raise ValueError(f"convention must be one of {UNDEFINED_CONVENTIONS}, "
                     f"got {convention!r}")


def aggregate(Y, P, measure: str | BinaryMeasure, how: Aggregation = "micro",
              *, convention: str = "exclude", mask=None,
              return_parts: bool = False):
    """Apply ``measure`` to ``(Y, P)`` under the operator ``how``.

    Parameters
    ----------
    how
        ``'micro'`` pools all entries before applying the measure; ``'macro'``
        applies it per class and averages; ``'weighted'`` applies it per class and
        averages with weights proportional to the class prevalence; ``'exemplar'``
        applies it per example and averages.
    convention
        Treatment of undefined per-class or per-example entries. ``'exclude'``
        averages over defined entries only. ``'zero'`` reproduces the behaviour of
        ``sklearn``'s ``zero_division=0``.
    return_parts
        If True, also return the vector of per-class or per-example values, which
        is what the diagnostics use to check whether a pooled value lies inside
        their convex hull.
    """
    meas = get_measure(measure)
    TP, FP, FN, TN = cells(Y, P, mask=mask)

    if how == "micro":
        value = meas(sigma_1(TP), sigma_1(FP), sigma_1(FN), sigma_1(TN))
        return (value, None) if return_parts else value

    if how in ("macro", "weighted"):
        red, axis = sigma_m, 0
    elif how == "exemplar":
        red, axis = sigma_n, 1
    else:
        raise ValueError("how must be 'micro', 'macro', 'weighted' or 'exemplar'")

    tp, fp, fn, tn = red(TP), red(FP), red(FN), red(TN)
    parts = np.array([meas(*c) for c in zip(tp, fp, fn, tn)], dtype=float)

    weights = None
    if how == "weighted":
        weights = tp + fn                      # class prevalence
        if float(np.sum(weights)) <= 0:
            weights = None

    value = _combine(parts, convention, weights)
    return (value, parts) if return_parts else value


def fold_aggregate(im_list: Sequence, measure: str | BinaryMeasure,
                   how: Aggregation = "macro",
                   pooling: Literal["pooled", "averaged"] = "pooled",
                   *, convention: str = "exclude") -> float:
    """Aggregate a performance measure across K cross-validation folds.

    Parameters
    ----------
    im_list
        Sequence of :class:`~aggmetrics.core.IndicatorMatrix` instances, one per fold.
    measure
        The performance measure to compute.
    how
        Intra-fold aggregation operator: ``'micro'``, ``'macro'``, or ``'exemplar'``.
    pooling
        ``'pooled'`` concatenates the folds' predictions before computing the
        non-linear ratio, so folds may differ in size; ``'averaged'`` computes the
        measure separately in each fold and returns the sample-weighted mean.
    convention
        Convention for undefined entries (``'exclude'``, ``'zero'``, etc.).
    """
    if not im_list:
        raise ValueError("im_list must not be empty")
    meas = get_measure(measure)

    if pooling == "averaged":
        vals = [im.compute(meas, how, convention=convention) for im in im_list]
        weights = [im.n for im in im_list]
        return float(np.average(vals, weights=weights))

    if pooling == "pooled":
        # Concatenate the folds' cell matrices (one row per test example), so the
        # folds may differ in size. Every example sits in exactly one test fold, hence
        # this is the measure on the pooled predictions, as in the companion paper.
        cell_tuples = [im.cells for im in im_list]
        if len({c[0].shape[1] for c in cell_tuples}) != 1:
            raise ValueError("all folds must have the same number of classes or labels")
        TP, FP, FN, TN = (np.vstack([c[i] for c in cell_tuples]) for i in range(4))
        if how == "micro":
            return float(meas(sigma_1(TP), sigma_1(FP), sigma_1(FN), sigma_1(TN)))
        if how in ("macro", "weighted"):
            tp, fp, fn, tn = sigma_m(TP), sigma_m(FP), sigma_m(FN), sigma_m(TN)
            parts = np.array([meas(*c) for c in zip(tp, fp, fn, tn)], dtype=float)
            weights = (tp + fn) if how == "weighted" else None
            return _combine(parts, convention, weights)
        if how == "exemplar":
            tp, fp, fn, tn = sigma_n(TP), sigma_n(FP), sigma_n(FN), sigma_n(TN)
            parts = np.array([meas(*c) for c in zip(tp, fp, fn, tn)], dtype=float)
            return _combine(parts, convention)
        raise ValueError("how must be 'micro', 'macro', 'weighted' or 'exemplar'")

    raise ValueError("pooling must be 'pooled' or 'averaged'")

