# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
The cells of a data set as functions of its confusion-matrix rates (internal).

A confusion matrix with fixed class sizes is determined by the conditional rates
``R[j, k] = P(predicted k | true j)`` for ``k != j`` in the multiclass case, and by
``(recall, specificity)`` for each binary column otherwise. Writing the cells as
functions of these rates serves two uses: the Jacobian of a measure with respect
to them (:mod:`aggmetrics.suites`), and the evaluation of a measure when the class
sizes are changed and the rates are kept (:mod:`aggmetrics.shift`).
"""
from __future__ import annotations

import copy

import numpy as np

from .aggregation import _combine
from .measures import get_measure


class Rates:
    """Cells of the data as functions of its free confusion-matrix parameters."""

    def __init__(self, im):
        if im.mask is not None or im.tnorm is not None:
            raise ValueError("masks and t-norms are not supported")
        if im.task not in ("binary", "multiclass", "multilabel"):
            raise ValueError(f"task {im.task!r} is not supported")
        Y = np.asarray(im.Y, float)
        P = np.asarray(im.P, float)
        self.task = "multiclass" if im.task == "multiclass" else "columns"
        self.n = Y.shape[0]
        if self.task == "multiclass":
            self.m = Y.shape[1]
            C = Y.T @ P
            self.nj = C.sum(axis=1)
            rows = C / np.where(self.nj == 0, 1.0, self.nj)[:, None]
            self.pairs = [(j, k) for j in range(self.m)
                          for k in range(self.m) if j != k]
            self.theta = np.array([rows[j, k] for j, k in self.pairs])
            self.names = [f"R[{j},{k}]" for j, k in self.pairs]
        else:
            self.m = Y.shape[1]
            self.pos = Y.sum(axis=0)
            self.neg = self.n - self.pos
            tp = (Y * P).sum(axis=0)
            tn = ((1 - Y) * (1 - P)).sum(axis=0)
            rec = tp / np.where(self.pos == 0, 1.0, self.pos)
            spec = tn / np.where(self.neg == 0, 1.0, self.neg)
            self.theta = np.column_stack([rec, spec]).ravel()
            if im.task == "binary" or self.m == 1:
                self.names = ["recall", "specificity"]
            else:
                self.names = [f"{k}[{j}]" for j in range(self.m)
                              for k in ("recall", "specificity")]

    def with_priors(self, priors):
        """Same rates, class sizes changed to those of ``priors`` (total ``n``).

        Multiclass: ``priors`` is a vector over the classes that sums to 1. Binary:
        the prevalence of the positive class. Multilabel: the prevalence of each
        label, a vector in [0, 1].
        """
        out = copy.copy(self)
        pri = np.asarray(priors, float)
        if self.task == "multiclass":
            if pri.shape != (self.m,) or (pri < 0).any() or abs(pri.sum() - 1) > 1e-8:
                raise ValueError(f"priors must be {self.m} non-negative numbers "
                                 f"that sum to 1")
            if ((self.nj == 0) & (pri > 0)).any():
                raise ValueError("a class absent from the data cannot be given a "
                                 "positive prior: its rates are unknown")
            out.nj = pri * self.n
            return out
        if pri.ndim == 0:
            pri = np.full(self.m, float(pri))
        if pri.shape != (self.m,) or ((pri < 0) | (pri > 1)).any():
            raise ValueError(f"priors must be one prevalence in [0, 1] for each of "
                             f"the {self.m} column(s)")
        if ((self.pos == 0) & (pri > 0)).any() or ((self.neg == 0) & (pri < 1)).any():
            raise ValueError("a column without positives (or without negatives) "
                             "cannot be given a prevalence that needs them")
        out.pos, out.neg = pri * self.n, (1 - pri) * self.n
        return out

    def cells(self, theta):
        """``(tp, fp, fn, tn)`` per class or label for the parameters ``theta``."""
        if self.task == "multiclass":
            W = np.zeros((self.m, self.m))
            for (j, k), v in zip(self.pairs, theta):
                W[j, k] = self.nj[j] * v
            W[np.diag_indices(self.m)] = self.nj - W.sum(axis=1)
            tp = np.diag(W)
            fn = self.nj - tp
            fp = W.sum(axis=0) - tp
            tn = self.n - tp - fn - fp
            return tp, fp, fn, tn
        rec, spec = theta[0::2], theta[1::2]
        return (self.pos * rec, self.neg * (1 - spec),
                self.pos * (1 - rec), self.neg * spec)

    def value(self, measure, how, theta, convention="exclude"):
        """The measure under the operator ``how``, as ``aggregate`` computes it."""
        tp, fp, fn, tn = self.cells(theta)
        meas = get_measure(measure)
        if how == "micro":
            return float(meas(tp.sum(), fp.sum(), fn.sum(), tn.sum()))
        per = np.array([meas(*c) for c in zip(tp, fp, fn, tn)], float)
        weights = None
        if how == "weighted":
            weights = tp + fn
            if float(weights.sum()) <= 0:
                weights = None
        return _combine(per, convention, weights)

    def gradient(self, measure, how, h=1e-6):
        g = np.empty_like(self.theta)
        for i, t in enumerate(self.theta):
            lo, hi = max(t - h, 0.0), min(t + h, 1.0)
            a, b = self.theta.copy(), self.theta.copy()
            a[i], b[i] = lo, hi
            change = self.value(measure, how, b) - self.value(measure, how, a)
            g[i] = change / (hi - lo)
        return g
