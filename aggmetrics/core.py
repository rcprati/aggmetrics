# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
The :class:`IndicatorMatrix` object: a pair of label matrices plus everything
derived from them.

Construction accepts the shapes practitioners actually have --- class-index
vectors, binary indicator matrices, or probability matrices --- and converts them
once. The four cell matrices are computed lazily and cached, so a full report over
many measures costs one pass over the data.
"""
from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence

import numpy as np

from .aggregation import Aggregation, aggregate, cells, sigma_1, sigma_m, sigma_n
from .measures import BinaryMeasure, get_measure
from .tnorms import soft_cells

__all__ = ["IndicatorMatrix", "pareto_dominance"]

Task = Literal["multiclass", "multilabel", "binary", "ordinal"]


def _one_hot(idx: Sequence[int], m: Optional[int] = None) -> np.ndarray:
    idx = np.asarray(idx).ravel().astype(int)
    m = int(idx.max()) + 1 if m is None else m
    Z = np.zeros((idx.size, m), dtype=float)
    Z[np.arange(idx.size), idx] = 1.0
    return Z


def _cumulative_ordinal(idx: Sequence[int], m: int) -> np.ndarray:
    """Encode ordinal classes 0..(m-1) into (m-1) cumulative binary thresholds."""
    idx = np.asarray(idx).ravel().astype(int)
    k = max(1, m - 1)
    Z = np.zeros((idx.size, k), dtype=float)
    for j in range(k):
        Z[:, j] = (idx > j).astype(float)
    return Z


class IndicatorMatrix:
    """Actual and predicted labels in indicator-matrix form.

    Parameters
    ----------
    y_true, y_pred
        Either 1-D arrays of class indices (multiclass, binary, or ordinal) or 2-D
        binary matrices of shape ``(n, m)`` (multilabel, or explicit one-hot).
        ``y_pred`` may also be a real-valued score or probability matrix, in which case
        ``threshold`` or ``argmax`` conversion is applied according to ``task``.
    task
        ``'multiclass'`` enforces one prediction per example via argmax;
        ``'multilabel'`` thresholds each column independently; ``'binary'`` is
        multilabel with a single column; ``'ordinal'`` encodes $m$ classes into
        $m-1$ cumulative threshold indicators. Inferred when omitted.
    threshold
        Threshold for multilabel conversion of scores. Ignored for multiclass.
    tnorm
        If given, ``y_true`` and ``y_pred`` are treated as soft in ``[0,1]`` and
        combined with this t-norm instead of Boolean conjunction. ``'product'`` is
        the only marginally consistent choice.
    mask
        Optional observation mask; zero entries are treated as unobserved.
    """

    def __init__(self, y_true, y_pred, task: Optional[Task] = None,
                 *, threshold: float = 0.5, tnorm: Optional[str] = None,
                 mask=None, n_classes: Optional[int] = None):
        yt, yp = np.asarray(y_true), np.asarray(y_pred)

        if task is None:
            if yt.ndim == 1 and len(np.unique(yt)) <= 2 and yp.ndim == 1:
                task = "binary"
            elif yt.ndim == 1:
                task = "multiclass"
            else:
                task = "multilabel"
        self.task: Task = task

        # -- actual labels -------------------------------------------------
        if task == "ordinal":
            top = np.asarray(yp).max() if yp.ndim == 1 else 0
            m_classes = n_classes or (int(max(yt.max(), top)) + 1)
            self.m_classes = m_classes
            if yt.ndim == 1:
                self.Y = _cumulative_ordinal(yt, m_classes)
            else:
                self.Y = yt.astype(float)
            n, m = self.Y.shape
            if yp.ndim == 1:
                self.P = _cumulative_ordinal(yp, m_classes)
            elif yp.shape == self.Y.shape:
                if np.isin(yp, (0.0, 1.0)).all():
                    self.P = yp.astype(float)
                else:
                    self.P = (yp > threshold).astype(float)
            elif yp.shape == (n, m_classes):
                self.P = _cumulative_ordinal(yp.argmax(axis=1), m_classes)
            else:
                self.P = yp.astype(float)
        else:
            if yt.ndim == 1:
                # a score matrix fixes the number of classes even when some of
                # them never occur in y_true
                m = n_classes or (int(max(yt.max(), np.asarray(yp).max() if yp.ndim == 1
                                          else yp.shape[1] - 1)) + 1)
                if task != "binary":
                    self.Y = _one_hot(yt, m)
                else:
                    self.Y = yt.reshape(-1, 1).astype(float)
            else:
                self.Y = yt.astype(float)
            n, m = self.Y.shape

            # -- predictions ---------------------------------------------------
            if yp.ndim == 1:
                self.P = (_one_hot(yp, m) if task != "binary"
                          else yp.reshape(-1, 1).astype(float))
            else:
                P = yp.astype(float)
                is_soft = tnorm is not None or not np.isin(P, (0.0, 1.0)).all()
                if is_soft and tnorm is None:
                    if task == "multiclass":
                        P = _one_hot(P.argmax(axis=1), m)
                    else:
                        P = (P > threshold).astype(float)
                self.P = P

        if self.Y.shape != self.P.shape:
            raise ValueError(f"shape mismatch: Y is {self.Y.shape}, "
                             f"predictions are {self.P.shape}")

        self.tnorm = tnorm
        self.mask = None if mask is None else np.asarray(mask, float)
        self.n, self.m = self.Y.shape
        self._cells = None

    # ------------------------------------------------------------------
    @property
    def cells(self):
        """Cached ``(TP, FP, FN, TN)`` cell matrices."""
        if self._cells is None:
            if self.tnorm is not None:
                c = soft_cells(self.Y, self.P, self.tnorm)
                if self.mask is not None:
                    c = tuple(x * self.mask for x in c)
                self._cells = c
            else:
                self._cells = cells(self.Y, self.P, mask=self.mask)
        return self._cells

    @property
    def counts(self):
        """Pooled ``(tp, fp, fn, tn)`` scalars."""
        return tuple(sigma_1(X) for X in self.cells)

    def per_class(self):
        """Per-class ``(tp, fp, fn, tn)``, one row per class or label."""
        return np.column_stack([sigma_m(X) for X in self.cells])

    def per_example(self):
        """Per-example ``(tp, fp, fn, tn)``, one row per example."""
        return np.column_stack([sigma_n(X) for X in self.cells])

    # ------------------------------------------------------------------
    def compute(self, measure: str | BinaryMeasure,
                aggregation: Aggregation = "micro",
                *, convention: str = "exclude") -> float:
        """Value of ``measure`` under the chosen aggregation operator."""
        return aggregate(self.Y, self.P if self.tnorm is None else self.P,
                         measure, aggregation, convention=convention,
                         mask=self.mask) if self.tnorm is None else \
            self._compute_soft(measure, aggregation, convention)

    def _compute_soft(self, measure, aggregation, convention):
        from .aggregation import _combine
        meas = get_measure(measure)
        TP, FP, FN, TN = self.cells
        if aggregation == "micro":
            return meas(sigma_1(TP), sigma_1(FP), sigma_1(FN), sigma_1(TN))
        red = sigma_m if aggregation in ("macro", "weighted") else sigma_n
        tp, fp, fn, tn = red(TP), red(FP), red(FN), red(TN)
        parts = np.array([meas(*c) for c in zip(tp, fp, fn, tn)], float)
        w = (tp + fn) if aggregation == "weighted" else None
        if w is not None and float(np.sum(w)) <= 0:
            w = None
        return _combine(parts, convention, w)

    def parts(self, measure: str | BinaryMeasure,
              aggregation: Aggregation = "macro") -> np.ndarray:
        """Per-class or per-example values, before averaging."""
        _, p = aggregate(self.Y, self.P, measure, aggregation,
                         mask=self.mask, return_parts=True)
        if p is None:
            raise ValueError("micro-averaging produces a single value, not parts")
        return p

    def accuracy(self) -> float:
        """Standard accuracy.

        For one-hot multiclass this is ``sigma_1(TP) / n``, which is *not* the
        same as ``compute('label_accuracy')``: the latter is ``(tp+tn)/(nm)`` and
        has a floor of ``(m-2)/m`` under one-hot encoding. Keeping them separate
        avoids a conflation that is easy to make and hard to notice.
        """
        if self.task == "multiclass":
            return float(sigma_1(self.cells[0]) / self.n)
        return self.compute("label_accuracy", "micro")

    def subset_accuracy(self) -> float:
        """Proportion of examples whose predicted label set matches exactly."""
        return float(np.all(np.isclose(self.Y, self.P), axis=1).mean())

    # ------------------------------------------------------------------
    def report(self, measures: Optional[Iterable[str]] = None,
               aggregations: Sequence[Aggregation] = ("micro", "macro", "exemplar"),
               *, convention: str = "exclude", as_frame: bool = True):
        """Table of measures against aggregation operators.

        Returns a ``pandas.DataFrame`` when pandas is available and
        ``as_frame`` is True, otherwise a dict of dicts.
        """
        names = list(measures) if measures is not None else [
            "precision", "recall", "f1", "jaccard", "balanced_accuracy",
            "gmean", "mcc", "label_accuracy"]
        table = {n: {a: self.compute(n, a, convention=convention)
                     for a in aggregations} for n in names}
        if as_frame:
            try:
                import pandas as pd
                return pd.DataFrame(table).T[list(aggregations)]
            except ImportError:
                pass
        return table

    @property
    def degrees_of_freedom(self) -> int:
        """Intrinsic independent degrees of freedom of the confusion matrix.

        For binary classification ($m=2$), this is $2(1) = 2$, spanned by the
        canonical orthogonal basis $(\\text{recall}, \\text{specificity})$.
        For $m$-class multiclass classification, this is $m(m-1)$, spanned by the
        directional off-diagonal conditional misclassification rates $R_{jk}$
        ($k \\neq j$). For multilabel classification with $m$ labels, each label has 2
        degrees of freedom ($2m$).
        """
        if self.task == "binary" or self.m == 1:
            return 2
        if self.task == "multilabel":
            return 2 * self.m
        return self.m * (self.m - 1)

    def canonical_coordinates(self) -> np.ndarray:
        """Extract the canonical orthogonal coordinates (conditional error rates).

        For multiclass problems, returns the $m(m-1)$ off-diagonal conditional error
        rates $R_{jk} = P(\\hat{Y}=k \\mid Y=j)$ for $k \\neq j$, which form an
        orthogonal basis in Frobenius inner product.
        For binary problems, returns $(\\text{recall}, \\text{specificity})$.
        """
        if self.task == "binary" or self.m == 1:
            rec = self.compute("recall", "micro")
            spec = self.compute("specificity", "micro")
            return np.array([rec, spec], float)
        C = (self.Y.T @ self.P).astype(float)
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        R = C / row_sums
        coords = []
        for j in range(self.m):
            for k in range(self.m):
                if j != k:
                    coords.append(float(R[j, k]))
        return np.array(coords, float)

    def cost(self, C, aggregation: Aggregation = "micro") -> float:
        """Total or averaged cost under an ``m x m`` cost matrix.

        ``Cost = sigma_1((Y C) o Yhat)``. With ``C[j,k] = |j-k|`` this is
        ``n`` times the mean absolute error, and with ``C[j,k] = (j-k)^2``
        it is ``n`` times the mean squared error.
        """
        C = np.asarray(C, float)
        if C.shape != (self.m, self.m):
            raise ValueError(f"cost matrix must be {(self.m, self.m)}, got {C.shape}")
        W = (self.Y @ C) * self.P
        if self.mask is not None:
            W = W * self.mask
        if aggregation == "micro":
            return float(np.sum(W))
        if aggregation == "macro":
            return float(np.mean(W.sum(axis=0)))
        if aggregation == "exemplar":
            return float(np.mean(W.sum(axis=1)))
        raise ValueError("aggregation must be 'micro', 'macro' or 'exemplar'")

    def dominates(self, other: IndicatorMatrix, strict: bool = False) -> bool:
        """Check whether this classifier dominates another on all monotonic measures.

        By Section 7.11 of the companion paper (which rests on Proposition 3), a
        classifier A dominates B across all
        measures respecting confusion-matrix monotonicity if and only if A dominates B
        elementwise in the canonical coordinate space:

        - In binary: TP_A >= TP_B, FP_A <= FP_B, FN_A <= FN_B, TN_A >= TN_B
        - In multiclass: R_{jk}^{(A)} <= R_{jk}^{(B)} for all off-diagonal errors
          (k != j)
          and R_{jj}^{(A)} >= R_{jj}^{(B)} for all on-diagonal recalls.
        """
        if self.task != other.task or self.m != other.m or self.n != other.n:
            raise ValueError(
                "Both IndicatorMatrix instances must have the same task, m, and n."
            )
        if self.task == "binary" or self.m == 1:
            tp_a, fp_a, fn_a, tn_a = self.counts
            tp_b, fp_b, fn_b, tn_b = other.counts
            ge = (tp_a >= tp_b) and (fp_a <= fp_b) and (fn_a <= fn_b) and (tn_a >= tn_b)
            strict_ge = (tp_a > tp_b) or (fp_a < fp_b) or (fn_a < fn_b) or (tn_a > tn_b)
            return bool((ge and strict_ge) if strict else ge)
        else:
            C_a = (self.Y.T @ self.P).astype(float)
            C_b = (other.Y.T @ other.P).astype(float)
            row_a = C_a.sum(axis=1, keepdims=True)
            row_a[row_a == 0] = 1.0
            row_b = C_b.sum(axis=1, keepdims=True)
            row_b[row_b == 0] = 1.0
            R_a = C_a / row_a
            R_b = C_b / row_b
            diag_ge = np.all(np.diag(R_a) >= np.diag(R_b) - 1e-12)
            off_mask = ~np.eye(self.m, dtype=bool)
            off_le = np.all(R_a[off_mask] <= R_b[off_mask] + 1e-12)
            ge = bool(diag_ge and off_le)
            if not strict:
                return ge
            strict_cond = (np.any(np.diag(R_a) > np.diag(R_b) + 1e-12) or
                           np.any(R_a[off_mask] < R_b[off_mask] - 1e-12))
            return bool(ge and strict_cond)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tp, fp, fn, tn = self.counts
        soft = f", tnorm={self.tnorm!r}" if self.tnorm else ""
        return (f"IndicatorMatrix(task={self.task!r}, n={self.n}, m={self.m}{soft}, "
                f"tp={tp:g}, fp={fp:g}, fn={fn:g}, tn={tn:g})")


def pareto_dominance(im_a: IndicatorMatrix, im_b: IndicatorMatrix) -> str:
    """Determine Pareto dominance between two classifiers.

    Returns:
    - ``'A_dominates_B'`` if A strictly dominates B across all monotonic measures;
    - ``'B_dominates_A'`` if B strictly dominates A;
    - ``'equivalent'`` if both confusion profiles are identical;
    - ``'incomparable'`` if neither dominates the other (the ranking depends on the
      choice of measure).
    """
    a_dom_b = im_a.dominates(im_b, strict=True)
    b_dom_a = im_b.dominates(im_a, strict=True)
    if a_dom_b and not b_dom_a:
        return "A_dominates_B"
    if b_dom_a and not a_dom_b:
        return "B_dominates_A"
    if im_a.dominates(im_b) and im_b.dominates(im_a):
        return "equivalent"
    return "incomparable"

