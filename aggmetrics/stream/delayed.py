# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Prequential evaluation when the labels arrive late, or never.

An instance is predicted at arrival, and its true label reaches the evaluator after
a latency, if it is requested at all. The evaluator keeps one unified state ``C(t)``
in which every instance first contributes an *interim* term, the product t-norm of a
stand-in label distribution (the model's calibrated class probabilities) and the
prediction, ``y_tilde p^T``. If the label arrives while the instance is still in the
queue, the interim term is swapped for the verified one, ``y p^T``. Because the
product t-norm is marginally consistent, the prediction-side sums of ``C(t)`` do not
depend on the stand-ins; the label-side sums do, so a class can have mass in ``C(t)``
without a verified instance. The evaluator therefore also keeps the verified part of
the state and reports the verified and the interim support and the coverage.

Temporal operators, as in :mod:`aggmetrics.stream.state`:

``alpha``  exponential fading, ``C <- alpha * C + term``; a label arriving after an
           instance has aged ``s`` steps is corrected with weight ``alpha ** s``;
           with ``eps``, queue entries whose normalised weight ``(1 - alpha) alpha^s``
           falls below ``eps`` are evicted (their term stays in ``C`` and keeps
           decaying), which bounds the queue by
           :func:`aggmetrics.stream.sizing.queue_bound`;
``window`` the last ``window`` instances; a correction replaces the term stored for
           the instance, and an instance that leaves the window cannot be corrected;
neither    the cumulative count: nothing is forgotten, so a label can be corrected
           whenever it arrives and the queue grows with the number of instances
           still waiting for a label.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from typing import Optional

import numpy as np

from ..measures import get_measure
from ._confusion import denominator_side, evaluate
from .state import (
    Snapshot,
    _check_operator,
    _default_tol,
    _onehot,
    _task_for,
)

__all__ = ["DelayedEvaluator"]


class DelayedEvaluator:
    """Unified confusion matrix of a stream with delayed and partial labels.

    Parameters
    ----------
    n_classes
        Number of classes.
    alpha, window
        The fading factor, or the window size. Neither: cumulative.
    eps
        Fading only: eviction tolerance on the normalised weight of a queue entry.
        ``None`` keeps every eligible instance in the queue.
    task, tol
        As in :class:`~aggmetrics.stream.state.TemporalState`.

    Examples
    --------
    >>> from aggmetrics.stream import DelayedEvaluator
    >>> ev = DelayedEvaluator(2, window=3)
    >>> k0 = ev.arrive([0.2, 0.8], 1)            # predicted 1, stand-in says 0.8
    >>> k1 = ev.arrive([0.9, 0.1], 0)
    >>> ev.label(k0, 1)                          # the first label arrives
    True
    >>> ev.coverage().round(2).tolist()          # share of each class row verified
    [0.0, 0.91]
    """

    def __init__(self, n_classes: int = 2, *, alpha: Optional[float] = None,
                 window: Optional[int] = None, eps: Optional[float] = None,
                 task: Optional[str] = None, tol: Optional[float] = None):
        _check_operator(alpha, window)
        if eps is not None and alpha is None:
            raise ValueError("eps applies to exponential fading only")
        if eps is not None and eps <= 0:
            raise ValueError("eps must be positive")
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.m = int(n_classes)
        self.alpha = None if alpha is None else float(alpha)
        self.window = None if window is None else int(window)
        self.eps = eps
        self.task = _task_for(self.m, task)
        self.tol = _default_tol(self.alpha, tol)
        self._C = np.zeros((self.m, self.m))          # unified: interim + verified
        self._V = np.zeros((self.m, self.m))          # verified part
        self._mass = 0.0
        self.t = 0
        self.queue: "OrderedDict[int, tuple]" = OrderedDict()   # k -> (y_soft, pred)
        self.n_evicted = 0
        self._terms: dict = {}                        # window: k -> current term
        self._vterms: dict = {}                       # window: k -> verified term
        self._order: deque = deque()                  # window: instances in the window

    # -- arrival and label -------------------------------------------------
    def arrive(self, y_soft, y_pred, *, eligible: bool = True) -> int:
        """Register an instance and its prediction; return its index ``k``.

        ``y_soft`` is the stand-in distribution over the classes for the true
        label (a calibrated probability vector). ``eligible=False`` marks an
        instance whose label will never be requested: it keeps its interim term and
        is not queued (the evaluator cannot know this in a real stream, so use it
        only to replay a recorded one).
        """
        ys = _onehot(y_soft, self.m)
        p = _onehot(y_pred, self.m)
        term = np.outer(ys, p)
        k = self.t
        if self.alpha is not None:
            self._C = self.alpha * self._C + term
            self._V = self.alpha * self._V
            self._mass = self.alpha * self._mass + 1.0
        elif self.window is None:                     # cumulative: nothing is forgotten
            self._C = self._C + term
            self._mass += 1.0
        else:
            self._C = self._C + term
            self._terms[k] = term
            self._order.append(k)
            if len(self._order) > self.window:
                k0 = self._order.popleft()
                self._C = self._C - self._terms.pop(k0)
                v = self._vterms.pop(k0, None)
                if v is not None:
                    self._V = self._V - v
                self.queue.pop(k0, None)              # left the window: no correction
                self._C[np.abs(self._C) < 1e-12] = 0.0
                self._V[np.abs(self._V) < 1e-12] = 0.0
            self._mass = float(len(self._order))
        if eligible:
            self.queue[k] = (ys, p)
        self.t += 1
        if self.alpha is not None and self.eps is not None:
            now = self.t - 1
            while self.queue:
                k0 = next(iter(self.queue))
                if (1 - self.alpha) * self.alpha ** (now - k0) < self.eps:
                    self.queue.popitem(last=False)
                    self.n_evicted += 1
                else:
                    break
        return k

    def label(self, k: int, y_true) -> bool:
        """The label of instance ``k`` arrives now.

        Returns ``True`` if the interim term was replaced by the verified one, and
        ``False`` if the instance is no longer in the queue (evicted, out of the
        window, or never eligible); the state is then unchanged.
        """
        if not 0 <= k < self.t:
            raise ValueError(f"instance {k} has not arrived")
        if k not in self.queue:
            return False
        ys, p = self.queue.pop(k)
        soft = np.outer(ys, p)
        hard = np.outer(_onehot(y_true, self.m), p)
        if self.alpha is not None:
            w = self.alpha ** (self.t - 1 - k)
            self._C = self._C + w * (hard - soft)
            self._V = self._V + w * hard
        else:
            self._C = self._C + (hard - soft)
            self._V = self._V + hard
            if self.window is not None:
                self._terms[k] = hard
                self._vterms[k] = hard
        return True

    # -- the state -----------------------------------------------------------
    @property
    def C(self) -> np.ndarray:
        """Unified confusion matrix: verified terms plus interim terms."""
        return self._C.copy()

    @property
    def C_verified(self) -> np.ndarray:
        """Only the terms of instances whose label has arrived."""
        return self._V.copy()

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def queue_size(self) -> int:
        """Instances awaiting a label that can still be corrected."""
        return len(self.queue)

    def support(self):
        """``(u_ver, u_int)``: classes with a verified instance, and classes with
        mass in the unified state, on the label side (row sums). They coincide
        without latency; with it, ``u_int`` is true for any class the stand-ins
        spread mass over."""
        thr = self.tol * self._mass
        return self._V.sum(axis=1) > thr, self._C.sum(axis=1) > thr

    def coverage(self) -> np.ndarray:
        """``kappa_j``: the share of the label-side mass of class ``j`` that comes
        from verified instances. ``nan`` where the class has no mass."""
        rows_u, rows_v = self._C.sum(axis=1), self._V.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            safe = np.where(rows_u > 0, rows_u, 1.0)
            return np.where(rows_u > 0, rows_v / safe, np.nan)

    def canonical_coordinates(self) -> np.ndarray:
        """Rates of the unified state, in the layout of
        :meth:`aggmetrics.stream.state.TemporalState.canonical_coordinates`."""
        from .state import TemporalState
        ts = TemporalState.__new__(TemporalState)
        ts.m, ts.task, ts._C, ts._mass = self.m, self.task, self._C, self._mass
        return TemporalState.canonical_coordinates(ts)

    def snapshot(self) -> Snapshot:
        if self._mass <= 0:
            raise ValueError("the state is empty")
        return Snapshot.from_confusion(self._C, self.task)

    # -- measures ----------------------------------------------------------------
    def compute(self, measure, aggregation: str = "micro", *,
                convention: str = "exclude", support: str = "auto",
                return_audit: bool = False):
        """Value of ``measure`` on the unified state.

        ``support`` chooses which state decides whether a class has a value:
        ``'auto'`` uses the interim support for a measure whose denominator is on
        the prediction side (precision, accuracy), exact under latency, and the
        verified support for one that involves the label side (recall,
        specificity, F1); ``'verified'`` and ``'interim'`` force one. With a
        verified support a class whose label has not yet reached the evaluator is
        not reported, even if the stand-ins give it an interim value.
        """
        if self._mass <= 0:
            raise ValueError("the state is empty")
        if support not in ("auto", "verified", "interim"):
            raise ValueError("support must be 'auto', 'verified' or 'interim'")
        meas = get_measure(measure)
        use_ver = (support == "verified" or
                   (support == "auto" and denominator_side(meas) == "label"))
        value, audit = evaluate(self._C, self.task, meas, aggregation,
                                convention=convention, tol=self.tol,
                                support_C=self._V if use_ver else None)
        return (value, audit) if return_audit else value
