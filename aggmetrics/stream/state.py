# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Temporal states: a confusion matrix that forgets.

A prequential evaluator keeps one confusion matrix ``C(t)`` and updates it with
each instance. Two temporal operators are supported, as in the companion paper:

``alpha``  exponential fading, ``C <- alpha * C + y p^T``;
``window`` sliding window, the sum of the last ``window`` terms;

and with neither the state is the plain cumulative count. The state is kept
unnormalised (a weighted count): every measure is a ratio of cells and is the same
under the normalised state, which divides by :attr:`TemporalState.mass`.

Because a measure is evaluated from the state, all the measures of the registry,
the three aggregation operators and the zero-division conventions are available on
a stream without a second implementation of any formula.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .._rates import Rates
from ._confusion import Audit, evaluate

__all__ = ["TemporalState", "ClassTimeState", "Snapshot"]


def _onehot(v, m: int) -> np.ndarray:
    """A class index or a length-``m`` vector as a vector of length ``m``."""
    a = np.asarray(v, float)
    if a.ndim == 0:
        k = int(a)
        if not 0 <= k < m or k != a:
            raise ValueError(f"class index must be an integer in [0, {m})")
        out = np.zeros(m)
        out[k] = 1.0
        return out
    if a.shape != (m,) or (a < -1e-12).any() or abs(a.sum() - 1.0) > 1e-8:
        raise ValueError(f"a label vector must be {m} non-negative numbers "
                         f"that sum to 1")
    return a


def _task_for(m: int, task):
    """The evaluation task of a state with ``m`` classes."""
    task = task or ("binary" if m == 2 else "multiclass")
    if task not in ("binary", "multiclass"):
        raise ValueError("task must be 'binary' or 'multiclass'")
    if task == "binary" and m != 2:
        raise ValueError("task 'binary' needs n_classes=2")
    return task


def _default_tol(alpha, tol) -> float:
    """Support tolerance: tiny under fading, where cells never reach zero."""
    if tol is not None:
        return float(tol)
    return 1e-9 if alpha is not None else 0.0


def _check_operator(alpha, window):
    if alpha is not None and window is not None:
        raise ValueError("give alpha or window, not both")
    if alpha is not None and not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if window is not None and (int(window) != window or window < 1):
        raise ValueError("window must be a positive integer")


class Snapshot:
    """The rates and the class sizes of a state at one instant.

    A snapshot is what the attribution compares: the conditional rates (the
    canonical coordinates, which a change of prior does not move) and the prior.
    Build it from a state (:meth:`TemporalState.snapshot`,
    :meth:`ClassTimeState.snapshot`) or from a confusion matrix.
    """

    def __init__(self, model: Rates, prior, task: str):
        self.model = model
        self.prior = np.asarray(prior, float)
        self.task = task

    @classmethod
    def from_confusion(cls, C, task: str = "multiclass") -> "Snapshot":
        C = np.asarray(C, float)
        model = Rates.from_confusion(C, task)
        rows = C.sum(axis=1)
        prior = (rows / rows.sum())[1] if task == "binary" else rows / rows.sum()
        return cls(model, prior, task)

    @property
    def rates(self) -> np.ndarray:
        """Canonical coordinates: ``(recall, specificity)`` for a binary state, the
        off-diagonal entries of the rows of the confusion matrix otherwise."""
        return self.model.theta.copy()

    def compute(self, measure, aggregation: str = "micro", *, prior=None,
                convention: str = "exclude") -> float:
        """The measure at this snapshot's rates and ``prior`` (its own by default)."""
        model = self.model.with_priors(self.prior if prior is None else prior)
        return model.value(measure, aggregation, model.theta, convention)


class _Operator:
    """The forgetting rule shared by the states."""

    def __init__(self, alpha, window):
        _check_operator(alpha, window)
        self.alpha = None if alpha is None else float(alpha)
        self.window = None if window is None else int(window)


class TemporalState(_Operator):
    """Confusion matrix of a stream under exponential fading or a sliding window.

    Parameters
    ----------
    n_classes
        Number of classes (2 for a binary stream).
    alpha, window
        The temporal operator: exponential fading with factor ``alpha`` in (0, 1),
        or a sliding window of the last ``window`` instances. Neither: cumulative.
    task
        ``'binary'`` (the default for two classes) evaluates measures on the
        positive class, index 1, as ``IndicatorMatrix(..., task='binary')`` does.
        ``'multiclass'`` aggregates one-vs-rest tables with ``aggregation``.
    tol
        Support tolerance, relative to the total mass. A class whose denominator
        is at most ``tol * mass`` has no value. Default: ``1e-9`` under fading,
        where cells decay but never reach zero, and ``0`` otherwise.

    Notes
    -----
    ``update`` accepts a class index or a probability vector for the true label.
    A vector credits the instance with the product t-norm term ``y p^T``, which is
    marginally consistent (:func:`aggmetrics.tnorms.marginal_defect`).

    Examples
    --------
    >>> from aggmetrics.stream import TemporalState
    >>> s = TemporalState(2, window=4)
    >>> for y, p in [(1, 1), (0, 0), (1, 0), (0, 0), (1, 1)]:
    ...     s.update(y, p)
    >>> s.C.tolist()
    [[2.0, 0.0], [1.0, 1.0]]
    >>> round(s.compute("recall"), 3), round(s.compute("precision"), 3)
    (0.5, 1.0)
    """

    def __init__(self, n_classes: int = 2, *, alpha: Optional[float] = None,
                 window: Optional[int] = None, task: Optional[str] = None,
                 tol: Optional[float] = None):
        super().__init__(alpha, window)
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.m = int(n_classes)
        self.task = _task_for(self.m, task)
        self.tol = _default_tol(self.alpha, tol)
        self._C = np.zeros((self.m, self.m))
        self._mass = 0.0
        self.t = 0
        self._terms: deque = deque()

    # -- update ----------------------------------------------------------
    def update(self, y_true, y_pred) -> None:
        """Credit one instance with true label ``y_true`` and prediction ``y_pred``."""
        term = np.outer(_onehot(y_true, self.m), _onehot(y_pred, self.m))
        if self.alpha is not None:
            self._C = self.alpha * self._C + term
            self._mass = self.alpha * self._mass + 1.0
        elif self.window is not None:
            self._terms.append(term)
            self._C = self._C + term
            if len(self._terms) > self.window:
                self._C = self._C - self._terms.popleft()
                self._C[np.abs(self._C) < 1e-12] = 0.0
            self._mass = float(len(self._terms))
        else:
            self._C = self._C + term
            self._mass += 1.0
        self.t += 1

    # -- the state -------------------------------------------------------
    @property
    def C(self) -> np.ndarray:
        """The confusion matrix ``C[j, k]``: true class ``j``, predicted ``k``."""
        return self._C.copy()

    @property
    def mass(self) -> float:
        """Total weight of the state: ``sum(C)``. ``(1 - alpha^t) / (1 - alpha)``
        under fading, the number of instances in the window, or ``t``."""
        return self._mass

    @property
    def normalized(self) -> np.ndarray:
        """``C`` divided by its mass (the state of the paper, with ``Tr W(t) = 1``)."""
        if self._mass <= 0:
            return np.zeros_like(self._C)
        return self._C / self._mass

    def per_class(self) -> np.ndarray:
        """One-vs-rest ``(tp, fp, fn, tn)`` per class (the positive class if binary)."""
        from ._confusion import cells_from_confusion
        return cells_from_confusion(self._C, self.task)

    def canonical_coordinates(self) -> np.ndarray:
        """``(recall, specificity)`` for a binary state; the off-diagonal entries of
        the rows of the confusion matrix, row by row, otherwise. Classes without
        support give ``nan``."""
        C = self._C
        if self.task == "binary":
            rp, rn = C[1].sum(), C[0].sum()
            return np.array([C[1, 1] / rp if rp > 0 else np.nan,
                             C[0, 0] / rn if rn > 0 else np.nan])
        rows = C.sum(axis=1)
        out = []
        for j in range(self.m):
            for k in range(self.m):
                if j != k:
                    out.append(C[j, k] / rows[j] if rows[j] > 0 else np.nan)
        return np.array(out)

    def snapshot(self) -> Snapshot:
        """The rates and the observed prior of the state now."""
        if self._mass <= 0:
            raise ValueError("the state is empty")
        return Snapshot.from_confusion(self._C, self.task)

    # -- measures --------------------------------------------------------
    def compute(self, measure, aggregation: str = "micro", *,
                convention: str = "exclude", prior=None,
                return_audit: bool = False):
        """Value of ``measure`` on the state.

        Parameters
        ----------
        aggregation
            ``'micro'``, ``'macro'`` or ``'weighted'`` (identical for a binary
            state, which has one table).
        convention
            Treatment of a class with no support: ``'exclude'`` (average over the
            classes that have it), ``'zero'``, ``'one'`` or ``'nan'``.
        prior
            If given, the value the measure would take with the rates of the state
            at this class prior (the prevalence of the positive class if binary;
            the frequencies of the classes otherwise). Skew-invariant measures do
            not change; the others become comparable across periods with
            different prevalence. A class absent from the state cannot be given a
            positive prior, since its rates are unknown.
        return_audit
            Also return the :class:`~aggmetrics.stream._confusion.Audit` tuple
            ``(per-class values, support u, m_active)``.
        """
        if self._mass <= 0:
            raise ValueError("the state is empty")
        if prior is not None:
            model = Rates.from_confusion(self._C, self.task).with_priors(prior)
            value = model.value(measure, aggregation, model.theta, convention)
            return (value, None) if return_audit else value
        value, audit = evaluate(self._C, self.task, measure, aggregation,
                                convention=convention, tol=self.tol)
        return (value, audit) if return_audit else value

    def audit(self, measure, aggregation: str = "macro", *,
              convention: str = "exclude") -> Audit:
        """The audit tuple of a measure on the state, without the value."""
        return self.compute(measure, aggregation, convention=convention,
                            return_audit=True)[1]


class ClassTimeState(_Operator):
    """Rates estimated with each class forgetting on its own arrivals.

    Time-fading cells mix the class prior and the conditional rates: the
    accumulated recall is a prior-weighted average of past recalls (Remark on
    accumulated coordinates in the paper). Here row ``j`` of the confusion matrix
    decays only when an instance of class ``j`` arrives, so the rates
    ``R[j, :] = C[j, :] / sum(C[j, :])`` are free of the prior, and a class that
    stops arriving keeps its last rates instead of being forgotten. The state
    carries no prior; give one to :meth:`compute`.

    Parameters
    ----------
    alpha, window
        Fading factor applied per arrival of the class, or the last ``window``
        arrivals of each class. Exactly one is required.
    """

    def __init__(self, n_classes: int = 2, *, alpha: Optional[float] = None,
                 window: Optional[int] = None, task: Optional[str] = None):
        super().__init__(alpha, window)
        if (alpha is None) == (window is None):
            raise ValueError("give exactly one of alpha and window")
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.m = int(n_classes)
        self.task = _task_for(self.m, task)
        self._C = np.zeros((self.m, self.m))
        self._arrivals = np.zeros(self.m, int)
        self._wins = [deque() for _ in range(self.m)]

    def update(self, y_true: int, y_pred: int) -> None:
        j = int(y_true)
        p = _onehot(y_pred, self.m)
        if not 0 <= j < self.m:
            raise ValueError(f"y_true must be an integer in [0, {self.m})")
        if self.alpha is not None:
            self._C[j] = self.alpha * self._C[j] + p
        else:
            self._wins[j].append(p)
            self._C[j] = self._C[j] + p
            if len(self._wins[j]) > self.window:
                self._C[j] = self._C[j] - self._wins[j].popleft()
                self._C[j][np.abs(self._C[j]) < 1e-12] = 0.0
        self._arrivals[j] += 1

    @property
    def C(self) -> np.ndarray:
        """Unnormalised rows: each is the faded (or windowed) count of one class."""
        return self._C.copy()

    @property
    def arrivals(self) -> np.ndarray:
        """Number of instances of each class seen so far."""
        return self._arrivals.copy()

    def n_ess(self) -> np.ndarray:
        """Effective number of arrivals each row averages.

        ``min(k_j, (1+alpha)/(1-alpha))`` under fading, the arrivals in the window
        otherwise. The binomial variance of
        a rate is ``R (1 - R) / n_ess``, if the arrivals are independent."""
        if self.alpha is not None:
            return np.minimum(self._arrivals.astype(float),
                              (1 + self.alpha) / (1 - self.alpha))
        return np.array([float(len(w)) for w in self._wins])

    def rates(self) -> np.ndarray:
        """The confusion matrix with each row normalised; a class never seen gives
        a row of ``nan``."""
        rows = self._C.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(rows > 0, self._C / np.where(rows > 0, rows, 1.0), np.nan)

    def canonical_coordinates(self) -> np.ndarray:
        """``(recall, specificity)`` for a binary state; the off-diagonal rates row
        by row otherwise. Same layout as :meth:`TemporalState.canonical_coordinates`."""
        R = self.rates()
        if self.task == "binary":
            return np.array([R[1, 1], R[0, 0]])
        return np.array([R[j, k] for j in range(self.m)
                         for k in range(self.m) if j != k])

    def _confusion_at(self, prior) -> np.ndarray:
        pri = np.asarray(prior, float)
        if self.task == "binary":
            if pri.ndim != 0 or not 0.0 <= float(pri) <= 1.0:
                raise ValueError("prior must be the prevalence of the positive "
                                 "class, a number in [0, 1]")
            pri = np.array([1.0 - float(pri), float(pri)])
        elif pri.shape != (self.m,) or (pri < 0).any() or abs(pri.sum() - 1) > 1e-8:
            raise ValueError(f"prior must be {self.m} non-negative numbers "
                             f"that sum to 1")
        R = self.rates()
        if ((self._arrivals == 0) & (pri > 0)).any():
            raise ValueError("a class never seen cannot be given a positive prior: "
                             "its rates are unknown")
        R = np.where(np.isnan(R), 0.0, R)
        return pri[:, None] * R * 1000.0

    def snapshot(self, prior) -> Snapshot:
        """The rates of the state at the class ``prior`` given."""
        C = self._confusion_at(prior)
        pri = float(prior) if self.task == "binary" else np.asarray(prior, float)
        return Snapshot(Rates.from_confusion(C, self.task), pri, self.task)

    def compute(self, measure, aggregation: str = "micro", *, prior,
                convention: str = "exclude") -> float:
        """The measure with the class-time rates at the class ``prior`` given."""
        return self.snapshot(prior).compute(measure, aggregation,
                                            convention=convention)
