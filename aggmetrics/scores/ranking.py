# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Ranking operators: the DCG family (rows) and the AUC family (columns).

Ranking has two axes. Within a row, the labels of one example are ordered by
score; a positional operator gives the cell ``(i, j)`` the gain
``Y_ij * w(rank_i(j))``, and the DCG family aggregates these cells. Within a
column, the examples are ordered by their score for one label; the concordance
cell of a positive ``(i, j)`` is the share of the column's negatives that it
outranks (ties count one half), and the AUC family aggregates these cells.

Both per-class summaries are conditional means over the positives of the class
(Theorem ``duality`` of the companion paper), of a row-rank quality for
``nDCG_j`` and of a column-rank quality for ``AUC_j``; they can order the labels
in opposite ways. The raw aggregates of either family collapse (micro, class and
exemplar averages of the cells are the same number), while the normalised
summaries need not coincide (Theorems ``dcg_collapse`` and ``auc_collapse``).

The concordance cells cost ``O(n log n)`` per column. The row ranks compare every
pair of labels of an example, ``O(n m^2)`` time and memory, which suits label sets
up to a few hundred labels.

Ties within a row follow ``ties``: ``'min'`` gives tied labels the best of their
ranks (the paper's min-rank convention), ``'average'`` the mid-rank, and
``'ordinal'`` breaks ties by column order.
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np

from .matrix import ScoreMatrix

__all__ = ["row_ranks", "log_weight", "geometric", "top_k", "dcg", "dcg_parts",
           "ndcg", "ndcg_parts", "auc_cells", "auc", "auc_parts", "pooled_auc",
           "precision_at_k"]

Weight = Callable[[np.ndarray], np.ndarray]
_TIES = ("min", "average", "ordinal")


# -- ranks and weights ---------------------------------------------------------------

def _ranks_desc(S: np.ndarray, ties: str) -> np.ndarray:
    """Rank of each entry within its row, 1 for the highest score."""
    if ties not in _TIES:
        raise ValueError(f"ties must be one of {_TIES}")
    higher = (S[:, None, :] > S[:, :, None]).sum(axis=2)
    if ties == "min":
        return higher + 1.0
    equal = (S[:, None, :] == S[:, :, None]).sum(axis=2)
    if ties == "average":
        return higher + (equal + 1) / 2.0
    m = S.shape[1]
    earlier = np.tril(np.ones((m, m), bool), -1)  # column k < j
    ties_before = ((S[:, None, :] == S[:, :, None]) & earlier[None]).sum(axis=2)
    return higher + ties_before + 1.0


def row_ranks(sm: ScoreMatrix, ties: str = "min") -> np.ndarray:
    """Rank of each label within its example (1 for the highest score)."""
    return _ranks_desc(sm.P, ties)


def log_weight(r) -> np.ndarray:
    """The nDCG discount ``1 / log2(r + 1)``."""
    return 1.0 / np.log2(np.asarray(r, float) + 1.0)


def geometric(p: float) -> Weight:
    """The weight ``p ** (r - 1)`` of rank-biased precision (up to normalisation)."""
    if not 0 < p < 1:
        raise ValueError("p must lie in (0, 1)")
    return lambda r: p ** (np.asarray(r, float) - 1.0)


def top_k(k: int) -> Weight:
    """The indicator weight ``1[r <= k]`` (counts the relevant labels in the top k)."""
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    return lambda r: (np.asarray(r, float) <= k).astype(float)


def _weight(weight: Union[str, Weight]) -> Weight:
    if weight == "log":
        return log_weight
    if callable(weight):
        return weight
    raise ValueError("weight must be 'log' or a callable of the rank")


# -- DCG family ----------------------------------------------------------------------

def _dcg_cells(sm, weight, ties) -> np.ndarray:
    return sm.Y * _weight(weight)(row_ranks(sm, ties))


def dcg_parts(sm: ScoreMatrix, axis: str = "class", weight="log",
              ties: str = "min") -> np.ndarray:
    """Raw per-class ``(1/n) sum_i D_ij`` or per-example ``(1/m) sum_j D_ij``."""
    D = _dcg_cells(sm, weight, ties)
    if axis == "class":
        return D.mean(axis=0)
    if axis == "example":
        return D.mean(axis=1)
    raise ValueError("axis must be 'class' or 'example'")


def dcg(sm: ScoreMatrix, aggregation: str = "micro", weight="log",
        ties: str = "min") -> float:
    """Raw DCG; its micro, macro and exemplar averages coincide."""
    D = _dcg_cells(sm, weight, ties)
    if aggregation == "micro":
        return float(D.mean())
    if aggregation == "macro":
        return float(D.mean(axis=0).mean())
    if aggregation == "exemplar":
        return float(D.mean(axis=1).mean())
    raise ValueError("aggregation must be 'micro', 'macro' or 'exemplar'")


def _ideal(k: np.ndarray, w: Weight) -> np.ndarray:
    """``IDCG_i = sum_{r=1}^{k_i} w(r)``."""
    kmax = int(k.max()) if k.size else 0
    cum = np.concatenate([[0.0], np.cumsum(w(np.arange(1, kmax + 1)))])
    return cum[k.astype(int)]


def _apply_empty(values: np.ndarray, empty_mask: np.ndarray, empty: str
                 ) -> np.ndarray:
    if empty not in ("zero", "nan"):
        raise ValueError("empty must be 'zero' or 'nan'")
    return np.where(empty_mask, 0.0 if empty == "zero" else np.nan, values)


def ndcg_parts(sm: ScoreMatrix, axis: str = "class", weight="log",
               ties: str = "min", empty: str = "zero") -> np.ndarray:
    """Per-class ``nDCG_j`` or per-example ``nDCG_i``.

    ``nDCG_j = sum_i D_ij / (n_j^+ w(1))`` is the mean gain of label ``j`` over the
    examples where it is relevant; ``nDCG_i = sum_j D_ij / IDCG_i``. Classes or
    examples without a relevant cell get ``0`` (``empty='zero'``, the paper's
    convention for examples) or ``nan`` (``empty='nan'``, to drop them).
    """
    w = _weight(weight)
    D = sm.Y * w(row_ranks(sm, ties))
    if axis == "class":
        den = sm.Y.sum(axis=0) * float(w(np.array([1.0]))[0])
        num = D.sum(axis=0)
    elif axis == "example":
        den = _ideal(sm.Y.sum(axis=1), w)
        num = D.sum(axis=1)
    else:
        raise ValueError("axis must be 'class' or 'example'")
    with np.errstate(invalid="ignore", divide="ignore"):
        values = num / den
    return _apply_empty(values, den == 0, empty)


def ndcg(sm: ScoreMatrix, aggregation: str = "micro", weight="log",
         ties: str = "min", empty: str = "zero") -> float:
    """Normalised DCG: ``'micro'`` (``sum D / sum IDCG_i``), ``'macro'`` (mean of
    ``nDCG_j``) or ``'exemplar'`` (mean of ``nDCG_i``). The three need not agree.
    """
    if aggregation == "micro":
        w = _weight(weight)
        D = sm.Y * w(row_ranks(sm, ties))
        den = _ideal(sm.Y.sum(axis=1), w).sum()
        return float(D.sum() / den) if den > 0 else 0.0
    if aggregation in ("macro", "exemplar"):
        axis = "class" if aggregation == "macro" else "example"
        return float(np.nanmean(ndcg_parts(sm, axis, weight, ties, empty)))
    raise ValueError("aggregation must be 'micro', 'macro' or 'exemplar'")


def precision_at_k(sm: ScoreMatrix, k: int, ties: str = "ordinal") -> np.ndarray:
    """Per-example Precision@k: relevant labels among the ``k`` best-ranked, over ``k``.

    Ties are broken by column order by default, so that exactly ``k`` positions
    are selected.
    """
    if int(k) != k or not 1 <= k <= sm.m:
        raise ValueError(f"k must be an integer in [1, {sm.m}]")
    return dcg_parts(sm, "example", top_k(k), ties) * sm.m / k


# -- AUC family ----------------------------------------------------------------------

def auc_cells(sm: ScoreMatrix) -> np.ndarray:
    """Concordance cells: for a positive ``(i, j)``, the share of the negatives of
    column ``j`` scored below ``P_ij`` (ties count one half); ``0`` elsewhere.
    Columns without negatives give ``nan`` on their positive cells.
    """
    Y, P = sm.Y, sm.P
    frac = np.full(Y.shape, np.nan)
    for j in range(sm.m):
        neg = np.sort(P[Y[:, j] == 0, j])
        if neg.size:
            below = np.searchsorted(neg, P[:, j], side="left")
            ties = np.searchsorted(neg, P[:, j], side="right") - below
            frac[:, j] = (below + 0.5 * ties) / neg.size
    return np.where(Y == 1, frac, 0.0)


def auc_parts(sm: ScoreMatrix, axis: str = "class", empty: str = "nan"
              ) -> np.ndarray:
    """Per-class ``AUC_j`` (Wilcoxon-Mann-Whitney, half credit for ties) or
    per-example ``AUC_i``, the mean concordance of the example's relevant labels.

    Undefined values (a class without positives or negatives, an example without
    relevant labels) are ``nan`` by default or ``0`` with ``empty='zero'``.
    """
    A = auc_cells(sm)
    Y = sm.Y
    if axis == "class":
        pos = Y.sum(axis=0)
        undefined = (pos == 0) | (pos == sm.n)
        num, den = np.nansum(A, axis=0), pos
    elif axis == "example":
        num, den = np.nansum(A, axis=1), Y.sum(axis=1)
        undefined = (den == 0) | np.isnan(np.where(Y == 1, A, 0.0)).any(axis=1)
    else:
        raise ValueError("axis must be 'class' or 'example'")
    with np.errstate(invalid="ignore", divide="ignore"):
        values = num / den
    return _apply_empty(values, undefined, empty)


def auc(sm: ScoreMatrix, aggregation: str = "macro", empty: str = "nan") -> float:
    """AUC summaries.

    ``'macro'``: mean of ``AUC_j``. ``'micro'``: mean of ``AUC_j`` weighted by the
    number of positives ``n_j^+`` (the paper's micro-AUC, not pooled ROC-AUC; see
    :func:`pooled_auc`). ``'exemplar'``: mean of ``AUC_i``. ``'raw'``: the cell
    aggregate ``sum A / (n m)``, equal to ``(1/m) sum_j pi_j AUC_j``. Undefined
    classes or examples are skipped (``empty='nan'``) or count as ``0``.
    """
    if aggregation == "raw":
        return float(np.nansum(auc_cells(sm)) / (sm.n * sm.m))
    if aggregation == "exemplar":
        return float(np.nanmean(auc_parts(sm, "example", empty)))
    if aggregation in ("macro", "micro"):
        a = auc_parts(sm, "class", empty)
        if aggregation == "macro":
            return float(np.nanmean(a))
        w = sm.Y.sum(axis=0)
        ok = ~np.isnan(a)
        return float((w[ok] * a[ok]).sum() / w[ok].sum())
    raise ValueError("aggregation must be 'macro', 'micro', 'exemplar' or 'raw'")


def pooled_auc(sm: ScoreMatrix) -> float:
    """ROC-AUC of all cells pooled into one ranking (scikit-learn's ``'micro'``).

    This compares a positive cell of one class with a negative cell of another, so
    it is a different statistic from the micro-AUC of :func:`auc`; the two differ
    when the classes' scores are on different scales. Provided for contrast.
    """
    y, p = sm.Y.ravel(), sm.P.ravel()
    pos, neg = p[y == 1], p[y == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("pooled AUC needs positive and negative cells")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    # mid-ranks (ascending) of the pooled scores
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    mid = (start + (counts + 1) / 2.0)[inv]
    ranks = np.empty_like(mid)
    ranks[order] = mid
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
