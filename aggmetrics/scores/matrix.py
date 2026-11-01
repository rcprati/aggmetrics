# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Score matrices: actual labels and predicted scores in matrix form.

:class:`ScoreMatrix` is the counterpart of :class:`aggmetrics.IndicatorMatrix` for
scores in ``[0, 1]`` that are not thresholded. A deviation operator maps every cell
to a loss, and the three aggregation operators of the framework average the cell
matrix over all cells (``micro``), over each column and then across columns
(``macro``), or over each row and then across rows (``exemplar``).

Because every per-class and per-example value divides by a fixed count (``n`` or
``m``), the three averages of a deviation operator are the same number
(Theorem ``brier_collapse`` of the companion paper). Their parts differ, and the
class-conditional errors ``alpha_j`` and ``beta_j`` are the per-class summaries that
do not change when the prevalence of a class changes.

Normalisation: the micro average divides the cell sum by ``n * m``. For one-hot
multiclass labels the usual multiclass Brier score divides by ``n`` only, so it is
``m`` times the value reported here; :meth:`ScoreMatrix.loss` with ``scale="sum"``
returns that convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .deviation import get_deviation

__all__ = ["ScoreMatrix", "ConditionalErrors"]

_AGGREGATIONS = ("micro", "macro", "exemplar")


@dataclass(frozen=True)
class ConditionalErrors:
    """Class-conditional (or example-conditional) mean losses.

    ``alpha`` is the mean loss over the positive cells of each column (or row),
    ``beta`` over the negative cells, and ``prevalence`` the share of positive
    cells. Entries without support are ``nan``.
    """

    alpha: np.ndarray
    beta: np.ndarray
    prevalence: np.ndarray

    def mixture(self, prevalence=None) -> np.ndarray:
        """``pi * alpha + (1 - pi) * beta``, the per-class mean loss at ``pi``.

        With the observed prevalence this reproduces the per-class loss exactly
        (Proposition ``bs_decomp``); with another prevalence it is the per-class
        loss under a change of prior that keeps the class-conditional errors.
        """
        pi = self.prevalence if prevalence is None else np.asarray(prevalence, float)
        if np.shape(pi) != np.shape(self.alpha):
            raise ValueError(f"prevalence must have shape {np.shape(self.alpha)}")
        if np.any((pi < 0) | (pi > 1)):
            raise ValueError("prevalences must lie in [0, 1]")
        a = np.where(pi > 0, self.alpha, 0.0)
        b = np.where(pi < 1, self.beta, 0.0)
        return pi * a + (1 - pi) * b


class ScoreMatrix:
    """Actual labels and predicted scores as two ``(n, m)`` matrices.

    Parameters
    ----------
    y_true
        1-D class indices (binary or multiclass) or a 2-D binary matrix
        (multilabel, or explicit one-hot).
    scores
        Scores in ``[0, 1]``: a 1-D vector for a binary task (the score of the
        positive class) or an ``(n, m)`` matrix.
    task
        ``'binary'``, ``'multiclass'`` or ``'multilabel'``; inferred when omitted
        (1-D labels with at most two values and 1-D scores give ``'binary'``).
    n_classes
        Number of classes for 1-D multiclass labels when ``scores`` is not given
        as a matrix that fixes it.

    Examples
    --------
    >>> import numpy as np
    >>> Y = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0]])
    >>> P = np.array([[.7, .6, .1], [.6, .1, .4], [.8, .2, .7], [.4, .3, .2]])
    >>> sm = ScoreMatrix(Y, P)
    >>> round(sm.loss("squared"), 6)        # micro = macro = exemplar = 13/48
    0.270833
    >>> np.round(sm.conditional_errors().alpha, 4)
    array([0.125, 0.43 , 0.36 ])
    """

    def __init__(self, y_true, scores, task: Optional[str] = None,
                 *, n_classes: Optional[int] = None):
        yt = np.asarray(y_true)
        sp = np.asarray(scores, float)
        if task is None:
            if yt.ndim == 1 and sp.ndim == 1:
                task = "binary"
            elif yt.ndim == 1:
                task = "multiclass"
            else:
                task = "multilabel"
        if task not in ("binary", "multiclass", "multilabel"):
            raise ValueError("task must be 'binary', 'multiclass' or 'multilabel'")
        self.task = task

        if task == "binary":
            Y = yt.reshape(-1, 1).astype(float) if yt.ndim == 1 else yt.astype(float)
            P = sp.reshape(-1, 1) if sp.ndim == 1 else sp
        elif yt.ndim == 1:
            if sp.ndim != 2 and n_classes is None:
                raise ValueError("multiclass scores must be an (n, m) matrix")
            m = sp.shape[1] if sp.ndim == 2 else int(n_classes)
            idx = yt.astype(int)
            if np.any(idx != yt) or np.any((idx < 0) | (idx >= m)):
                raise ValueError(f"class indices must be integers in [0, {m})")
            Y = np.zeros((len(idx), m))
            Y[np.arange(len(idx)), idx] = 1.0
            P = sp
        else:
            Y, P = yt.astype(float), sp

        if Y.ndim != 2 or Y.shape != P.shape:
            raise ValueError(f"shape mismatch: labels {Y.shape}, scores {P.shape}")
        if not np.isin(Y, (0.0, 1.0)).all():
            raise ValueError("labels must be 0/1")
        if not np.isfinite(P).all() or P.min() < 0 or P.max() > 1:
            raise ValueError("scores must be finite and lie in [0, 1]")
        self.Y, self.P = Y, P
        self.n, self.m = Y.shape

    # ------------------------------------------------------------------
    @property
    def prevalence(self) -> np.ndarray:
        """Share of positive cells in each column."""
        return self.Y.mean(axis=0)

    def cells(self, op="squared") -> np.ndarray:
        """The deviation cell matrix ``phi(Y - P)``."""
        return get_deviation(op)(self.Y - self.P)

    def parts(self, op="squared", axis: str = "class") -> np.ndarray:
        """Per-class (``axis='class'``) or per-example mean losses."""
        C = self.cells(op)
        if axis == "class":
            return C.mean(axis=0)
        if axis == "example":
            return C.mean(axis=1)
        raise ValueError("axis must be 'class' or 'example'")

    def loss(self, op="squared", aggregation: str = "micro",
             scale: str = "mean") -> float:
        """Aggregate loss of a deviation operator.

        ``aggregation`` is ``'micro'``, ``'macro'`` or ``'exemplar'``; for a
        deviation operator the three coincide (they average the same cells with
        fixed denominators), and the argument is kept so that the call mirrors
        :meth:`aggmetrics.IndicatorMatrix.compute`. ``scale='sum'`` multiplies by
        ``m``, the per-example sum over labels (the usual multiclass Brier score).
        """
        if aggregation not in _AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {_AGGREGATIONS}")
        if scale not in ("mean", "sum"):
            raise ValueError("scale must be 'mean' or 'sum'")
        C = self.cells(op)
        if aggregation == "micro":
            value = float(C.mean())
        elif aggregation == "macro":
            value = float(C.mean(axis=0).mean())
        else:
            value = float(C.mean(axis=1).mean())
        return value * self.m if scale == "sum" else value

    def brier(self, aggregation: str = "micro") -> float:
        """The Brier score (squared deviation), averaged over all ``n * m`` cells."""
        return self.loss("squared", aggregation)

    def conditional_errors(self, op="squared", axis: str = "class"
                           ) -> ConditionalErrors:
        """Mean loss over the positive and over the negative cells.

        With ``axis='class'`` these are ``alpha_j`` and ``beta_j`` of each column;
        with ``axis='example'`` they are ``alpha_i`` and ``beta_i`` of each row.
        """
        C = self.cells(op)
        if axis == "class":
            Y, C = self.Y, C
        elif axis == "example":
            Y, C = self.Y.T, C.T
        else:
            raise ValueError("axis must be 'class' or 'example'")
        pos = Y.sum(axis=0)
        neg = Y.shape[0] - pos
        with np.errstate(invalid="ignore", divide="ignore"):
            alpha = np.where(pos > 0, np.where(Y == 1, C, 0.0).sum(axis=0) / pos,
                             np.nan)
            beta = np.where(neg > 0, np.where(Y == 0, C, 0.0).sum(axis=0) / neg,
                            np.nan)
        return ConditionalErrors(alpha=alpha, beta=beta, prevalence=pos / Y.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ScoreMatrix(n={self.n}, m={self.m}, task={self.task!r})"
