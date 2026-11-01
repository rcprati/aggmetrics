# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Murphy decompositions and the Brier score under a change of prior.

The Murphy decomposition writes the Brier score of a column as

    BS = UNC - RES + REL,

with ``UNC = ybar (1 - ybar)`` (uncertainty, a property of the labels),
``RES`` the spread of the calibration map around ``ybar`` (resolution) and
``REL`` the distance between the scores and the calibration map (reliability).
The calibration map ``f(v)`` is the mean label over the cells with score ``v``.

Applied to columns it describes each label; applied to rows it describes each
example (Theorems ``murphy_col`` and ``murphy_row`` of the companion paper).
When every score in a column is distinct, each level set holds one cell, so
``f`` reproduces the labels, ``REL = BS`` and ``RES = UNC``: the unbinned
decomposition is then uninformative, and :func:`murphy` warns. Grouping the scores
into bins and replacing each score by its bin mean gives a discretised predictor
for which the decomposition is again exact (Proposition ``murphy_binned``); its
Brier score differs from the original one by the within-bin term reported as
``binning_gap``.

``method='isotonic'`` estimates the calibration map by isotonic (non-decreasing)
regression of the labels on the scores, the CORP approach of Dimitriadis, Gneiting
and Jordan (PNAS 2021) used for score decompositions by Fissler, Lorentzen and
Mayer (2022). It needs no bins and does not degenerate when the scores are
distinct. The terms are then score differences,
``REL = BS(p) - BS(f)`` (miscalibration), ``RES = UNC - BS(f)`` (discrimination),
and ``BS = UNC - RES + REL`` holds exactly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .matrix import ScoreMatrix

__all__ = ["Murphy", "murphy", "brier_under_prior", "class_weighted_brier",
           "DegenerateDecompositionWarning"]


class DegenerateDecompositionWarning(UserWarning):
    """The unbinned decomposition has one cell per level set (``REL = BS``)."""


@dataclass(frozen=True)
class Murphy:
    """Per-column (or per-row) Murphy terms; ``brier = unc - res + rel``.

    With ``method='isotonic'`` the terms are also known as miscalibration
    (``rel``), discrimination (``res``) and uncertainty (``unc``).
    """

    brier: np.ndarray
    unc: np.ndarray
    res: np.ndarray
    rel: np.ndarray
    axis: str
    bins: Optional[int]
    binning_gap: Optional[np.ndarray]
    """Original Brier score minus that of the binned predictor (``None`` unbinned)."""
    method: str = "levels"

    @property
    def miscalibration(self) -> np.ndarray:
        """Alias of ``rel``."""
        return self.rel

    @property
    def discrimination(self) -> np.ndarray:
        """Alias of ``res``."""
        return self.res

    @property
    def uncertainty(self) -> np.ndarray:
        """Alias of ``unc``."""
        return self.unc


def _decompose(y: np.ndarray, p: np.ndarray):
    """Exact Murphy terms of one vector of labels and scores (level sets of ``p``)."""
    ybar = y.mean()
    values, inverse = np.unique(p, return_inverse=True)
    counts = np.bincount(inverse)
    f = (np.bincount(inverse, weights=y) / counts)[inverse]
    return (float(np.mean((y - p) ** 2)), float(ybar * (1 - ybar)),
            float(np.mean((f - ybar) ** 2)), float(np.mean((p - f) ** 2)),
            len(values) == len(p))


def _pav(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted non-decreasing least-squares fit (pool adjacent violators)."""
    means, wts, sizes = [], [], []
    for v, w in zip(values, weights):
        means.append(float(v))
        wts.append(float(w))
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2, s2 = means.pop(), wts.pop(), sizes.pop()
            m1, w1, s1 = means.pop(), wts.pop(), sizes.pop()
            means.append((m1 * w1 + m2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            sizes.append(s1 + s2)
    return np.repeat(means, sizes)


def _decompose_isotonic(y: np.ndarray, p: np.ndarray):
    """CORP terms: the calibration map is the isotonic regression of ``y`` on ``p``."""
    ybar = y.mean()
    values, inverse = np.unique(p, return_inverse=True)
    counts = np.bincount(inverse)
    level_means = np.bincount(inverse, weights=y) / counts
    f = _pav(level_means, counts)[inverse]
    bs, bs_f = float(np.mean((y - p) ** 2)), float(np.mean((y - f) ** 2))
    unc = float(ybar * (1 - ybar))
    return bs, unc, unc - bs_f, bs - bs_f


def _bin_means(p: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    """Each score replaced by the mean score of its bin."""
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    else:
        edges = np.unique(np.percentile(p, np.linspace(0, 100, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    sums = np.bincount(idx, weights=p, minlength=len(edges) - 1)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    return sums[idx] / counts[idx]


def murphy(sm: ScoreMatrix, axis: str = "class", bins: Optional[int] = None,
           strategy: str = "uniform", method: str = "levels") -> Murphy:
    """Murphy decomposition of every column (``axis='class'``) or row.

    Parameters
    ----------
    bins
        ``None`` uses the exact level sets of the scores. An integer groups the
        scores into that many bins and decomposes the discretised predictor, whose
        decomposition is exact; ``binning_gap`` then holds the difference between
        the original and the binned Brier scores.
    strategy
        ``'uniform'`` (equal-width bins on ``[0, 1]``) or ``'quantile'``
        (equal-count bins); empty bins are dropped.
    method
        ``'levels'`` (the default) uses the mean label of each level set of the
        scores (or of the bins). ``'isotonic'`` uses the isotonic regression of the
        labels on the scores (CORP) and cannot be combined with ``bins``.
    """
    if axis == "class":
        Y, P = sm.Y, sm.P
    elif axis == "example":
        Y, P = sm.Y.T, sm.P.T
    else:
        raise ValueError("axis must be 'class' or 'example'")
    if bins is not None and (int(bins) != bins or bins < 1):
        raise ValueError("bins must be a positive integer or None")
    if strategy not in ("uniform", "quantile"):
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    if method not in ("levels", "isotonic"):
        raise ValueError("method must be 'levels' or 'isotonic'")
    if method == "isotonic" and bins is not None:
        raise ValueError("method='isotonic' does not use bins")
    if method == "isotonic":
        out = np.array([_decompose_isotonic(Y[:, k], P[:, k])
                        for k in range(Y.shape[1])]).T
        return Murphy(brier=out[0], unc=out[1], res=out[2], rel=out[3], axis=axis,
                      bins=None, binning_gap=None, method="isotonic")

    out = np.empty((4, Y.shape[1]))
    gap = None if bins is None else np.empty(Y.shape[1])
    degenerate = True
    for k in range(Y.shape[1]):
        y, p = Y[:, k], P[:, k]
        if bins is not None:
            pb = _bin_means(p, int(bins), strategy)
            gap[k] = np.mean((y - p) ** 2) - np.mean((y - pb) ** 2)
            p = pb
        *terms, singletons = _decompose(y, p)
        out[:, k] = terms
        degenerate = degenerate and singletons
    if bins is None and degenerate and Y.shape[0] > 1:
        warnings.warn("every score is distinct within its column (row), so the "
                      "unbinned decomposition gives REL = BS and RES = UNC; "
                      "pass bins= for an informative decomposition",
                      DegenerateDecompositionWarning, stacklevel=2)
    return Murphy(brier=out[0], unc=out[1], res=out[2], rel=out[3], axis=axis,
                  bins=None if bins is None else int(bins), binning_gap=gap)


def brier_under_prior(sm: ScoreMatrix, prevalence) -> np.ndarray:
    """Per-class Brier score if each class had the given prevalence.

    Keeps the class-conditional errors ``alpha_j`` and ``beta_j`` of the data and
    replaces the prevalence: ``BS_j = pi_j alpha_j + (1 - pi_j) beta_j``. This is
    the per-class convention of the companion paper (each class reweighted on its
    own); replicating whole multilabel rows also changes the other columns.
    """
    return sm.conditional_errors("squared").mixture(prevalence)


def class_weighted_brier(sm: ScoreMatrix, w_pos, w_neg) -> np.ndarray:
    """Per-class ``w_pos * alpha_j + w_neg * beta_j``.

    With weights that do not depend on the prevalence the score is skew-invariant;
    with prevalence-dependent weights it is not (Remark ``class_weighted_brier``).
    Skew-invariance does not make it proper: reweighting the two outcomes moves the
    minimiser of the expected loss away from the truthful report unless the
    weights are equal.
    """
    ce = sm.conditional_errors("squared")
    return np.asarray(w_pos, float) * ce.alpha + np.asarray(w_neg, float) * ce.beta
