# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Binary performance measures, each carrying the metadata the framework needs to
reason about it.

A measure is not merely a function of the four confusion-matrix counts. Whether
the framework's identities apply to it depends on structural properties that are
recorded here alongside the formula:

``linear``
    Coefficients :math:`(\\mathbf{a}, \\mathbf{b})` such that
    :math:`M = \\langle a, v\\rangle / \\langle b, v\\rangle` with
    :math:`v = (tp, fp, fn, tn)`, or ``None`` if no such representation exists.
    A measure with coefficients is *aggregation-decomposable*: micro-averaging
    equals denominator-weighted macro-averaging. Without them the identity fails,
    and the pooled value need not even lie in the convex hull of the per-class
    values.

``skew_invariant``
    Whether the measure is unchanged when the two classes are rescaled
    independently, equivalently whether it is a function of recall and
    specificity alone.

``tn_dependent_at_zero``
    Whether the value at ``tp == 0`` still depends on ``tn``. This determines
    whether the measure can attain its nominal minimum under one-hot multiclass
    encoding, where an incorrectly classified example still contributes ``m - 2``
    true negatives.

``greater_is_better``
    Direction, needed for ranking and for the scikit-learn scorer interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["BinaryMeasure", "MEASURES", "get_measure", "list_measures"]

Counts = Tuple[float, float, float, float]  # (tp, fp, fn, tn)


def _ratio(num: float, den: float) -> float:
    """Divide, returning NaN when the denominator vanishes.

    NaN is used deliberately rather than 0: an undefined per-class value is not
    a value of zero, and the aggregation layer needs to distinguish them in order
    to apply the exclusion convention.
    """
    return num / den if abs(den) > 1e-12 else float("nan")


@dataclass(frozen=True)
class BinaryMeasure:
    """A binary measure together with its structural properties."""

    name: str
    fn: Callable[[float, float, float, float], float]
    greater_is_better: bool = True
    linear: Optional[Tuple[Sequence[float], Sequence[float]]] = None
    skew_invariant: bool = False
    tn_dependent_at_zero: bool = False
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    # -- structural queries ------------------------------------------------
    @property
    def decomposable(self) -> bool:
        """True iff micro-averaging equals denominator-weighted macro-averaging."""
        return self.linear is not None

    def numerator(self, counts: Counts) -> float:
        if self.linear is None:
            raise ValueError(
                f"{self.name!r} is not aggregation-decomposable, so it has no "
                "numerator that aggregates additively across classes."
            )
        a, _ = self.linear
        return float(np.dot(a, counts))

    def denominator(self, counts: Counts) -> float:
        if self.linear is None:
            raise ValueError(
                f"{self.name!r} is not aggregation-decomposable, so it has no "
                "denominator that aggregates additively across classes."
            )
        _, b = self.linear
        return float(np.dot(b, counts))

    def bayes_threshold(self) -> Optional[float]:
        """Optimal decision threshold on the conditional probability, or None.

        For a measure whose numerator is linear with coefficients
        ``(alpha, -beta, -gamma, delta)`` and whose denominator does not depend on
        the predictions, the population-optimal rule is to predict positive when
        the conditional probability exceeds
        ``(beta + delta) / (alpha + beta + gamma + delta)``.

        Returns None when the measure is not of that form, in which case no such
        closed form exists and an iterative procedure is required.
        """
        if self.linear is None:
            return None
        a, b = self.linear
        # denominator must be prediction-independent: it may combine only
        # (tp + fn) and (fp + tn), the two class totals.
        b = np.asarray(b, dtype=float)
        if not (math.isclose(b[0], b[2]) and math.isclose(b[1], b[3])):
            return None
        alpha, beta, gamma, delta = a[0], -a[1], -a[2], a[3]
        total = alpha + beta + gamma + delta
        if abs(total) < 1e-12:
            return None
        return float((beta + delta) / total)

    def erm_consistent(self) -> Optional[bool]:
        """Whether the measure is maximised by the Bayes rule for 0/1 loss."""
        t = self.bayes_threshold()
        return None if t is None else math.isclose(t, 0.5, abs_tol=1e-9)

    def attainable_floor(self, m: int) -> float:
        """Value attained under one-hot encoding when every prediction is wrong.

        With ``m`` classes and ``n`` examples all misclassified, the pooled counts
        are ``tp = 0``, ``fp = fn = n`` and ``tn = n (m - 2)``. Measures that do
        not depend on ``tn`` at ``tp = 0`` reach their nominal minimum; the rest
        are bounded away from it by an amount that grows with ``m``.
        """
        if m < 2:
            raise ValueError("m must be at least 2")
        n = 1000.0
        return float(self.fn(0.0, n, n, n * (m - 2)))

    def __call__(self, tp: float, fp: float, fn: float, tn: float) -> float:
        return float(self.fn(tp, fp, fn, tn))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tags = ["decomposable" if self.decomposable else "non-decomposable"]
        if self.skew_invariant:
            tags.append("skew-invariant")
        return f"<BinaryMeasure {self.name!r} ({', '.join(tags)})>"


# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------

def _recall(tp, fp, fn, tn):      return _ratio(tp, tp + fn)
def _specificity(tp, fp, fn, tn): return _ratio(tn, tn + fp)
def _precision(tp, fp, fn, tn):   return _ratio(tp, tp + fp)
def _npv(tp, fp, fn, tn):         return _ratio(tn, tn + fn)
def _fpr(tp, fp, fn, tn):         return _ratio(fp, fp + tn)
def _fnr(tp, fp, fn, tn):         return _ratio(fn, tp + fn)
def _fdr(tp, fp, fn, tn):         return _ratio(fp, tp + fp)
def _for(tp, fp, fn, tn):         return _ratio(fn, fn + tn)
def _jaccard(tp, fp, fn, tn):     return _ratio(tp, tp + fp + fn)
def _label_accuracy(tp, fp, fn, tn): return _ratio(tp + tn, tp + fp + fn + tn)
def _error_rate(tp, fp, fn, tn):  return _ratio(fp + fn, tp + fp + fn + tn)


def _fbeta(beta: float):
    b2 = beta * beta

    def f(tp, fp, fn, tn):
        return _ratio((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp)

    return f


def _balanced_accuracy(tp, fp, fn, tn):
    r, s = _recall(tp, fp, fn, tn), _specificity(tp, fp, fn, tn)
    return float("nan") if (math.isnan(r) or math.isnan(s)) else 0.5 * (r + s)


def _gmean(tp, fp, fn, tn):
    r, s = _recall(tp, fp, fn, tn), _specificity(tp, fp, fn, tn)
    if math.isnan(r) or math.isnan(s):
        return float("nan")
    return math.sqrt(max(r * s, 0.0))


def _informedness(tp, fp, fn, tn):
    r, s = _recall(tp, fp, fn, tn), _specificity(tp, fp, fn, tn)
    return float("nan") if (math.isnan(r) or math.isnan(s)) else r + s - 1.0


def _markedness(tp, fp, fn, tn):
    p, n = _precision(tp, fp, fn, tn), _npv(tp, fp, fn, tn)
    return float("nan") if (math.isnan(p) or math.isnan(n)) else p + n - 1.0


def _mcc(tp, fp, fn, tn):
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _ratio(tp * tn - fp * fn, den)


def _kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n <= 0:
        return float("nan")
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    return _ratio(po - pe, 1.0 - pe)


def _fowlkes_mallows(tp, fp, fn, tn):
    p, r = _precision(tp, fp, fn, tn), _recall(tp, fp, fn, tn)
    if math.isnan(p) or math.isnan(r):
        return float("nan")
    return math.sqrt(max(p * r, 0.0))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Coefficient order throughout is (tp, fp, fn, tn).

def _m(*args, **kwargs) -> BinaryMeasure:
    return BinaryMeasure(*args, **kwargs)


_REGISTRY: Dict[str, BinaryMeasure] = {}


def _register(measure: BinaryMeasure) -> BinaryMeasure:
    _REGISTRY[measure.name] = measure
    for alias in measure.aliases:
        _REGISTRY[alias] = measure
    return measure


# -- decomposable, skew-invariant --------------------------------------------
_register(_m("recall", _recall, linear=((1, 0, 0, 0), (1, 0, 1, 0)),
             skew_invariant=True, aliases=("sensitivity", "tpr", "hit_rate")))
_register(_m("specificity", _specificity, linear=((0, 0, 0, 1), (0, 1, 0, 1)),
             skew_invariant=True, aliases=("tnr", "selectivity"),
             tn_dependent_at_zero=True))
_register(_m("fnr", _fnr, greater_is_better=False,
             linear=((0, 0, 1, 0), (1, 0, 1, 0)), skew_invariant=True,
             aliases=("miss_rate",)))
_register(_m("fpr", _fpr, greater_is_better=False,
             linear=((0, 1, 0, 0), (0, 1, 0, 1)), skew_invariant=True,
             tn_dependent_at_zero=True, aliases=("fall_out",),
             note="Lower is better, and its value at tp=0 depends on tn: under "
                  "one-hot encoding a wholly wrong classifier attains 1/(m-1), "
                  "which looks good rather than bad."))

# -- decomposable, not skew-invariant ----------------------------------------
_register(_m("precision", _precision, linear=((1, 0, 0, 0), (1, 1, 0, 0)),
             aliases=("ppv",)))
_register(_m("npv", _npv, linear=((0, 0, 0, 1), (0, 0, 1, 1)),
             tn_dependent_at_zero=True))
_register(_m("fdr", _fdr, greater_is_better=False,
             linear=((0, 1, 0, 0), (1, 1, 0, 0))))
_register(_m("for", _for, greater_is_better=False,
             linear=((0, 0, 1, 0), (0, 0, 1, 1)), tn_dependent_at_zero=True))
_register(_m("f1", _fbeta(1.0), linear=((2, 0, 0, 0), (2, 1, 1, 0)),
             aliases=("dice", "f_measure", "fbeta1")))
_register(_m("f2", _fbeta(2.0), linear=((5, 0, 0, 0), (5, 1, 4, 0))))
_register(_m("f0.5", _fbeta(0.5), linear=((1.25, 0, 0, 0), (1.25, 1, 0.25, 0))))
_register(_m("jaccard", _jaccard, linear=((1, 0, 0, 0), (1, 1, 1, 0)),
             aliases=("iou", "csi", "tanimoto")))
_register(_m("label_accuracy", _label_accuracy,
             linear=((1, 0, 0, 1), (1, 1, 1, 1)), tn_dependent_at_zero=True,
             aliases=("accuracy_labelwise",),
             note="For one-hot multiclass this is not standard accuracy; "
                  "see IndicatorMatrix.accuracy()."))
_register(_m("error_rate", _error_rate, greater_is_better=False,
             linear=((0, 1, 1, 0), (1, 1, 1, 1)), tn_dependent_at_zero=True,
             aliases=("hamming_loss",),
             note="Lower is better, and its value at tp=0 depends on tn: under "
                  "one-hot encoding a wholly wrong classifier attains 2/m "
                  "rather than 1."))

# -- not decomposable --------------------------------------------------------
_register(_m("balanced_accuracy", _balanced_accuracy, skew_invariant=True,
             tn_dependent_at_zero=True, aliases=("ba",)))
_register(_m("gmean", _gmean, skew_invariant=True,
             tn_dependent_at_zero=False,
             note="Complete (uses all four cells) yet attains 0 at tp=0, "
                  "because recall annihilates the product.",
             aliases=("g_mean", "geometric_mean")))
_register(_m("informedness", _informedness, skew_invariant=True,
             tn_dependent_at_zero=True, aliases=("youden_j", "bookmaker")))
_register(_m("markedness", _markedness, tn_dependent_at_zero=True,
             aliases=("deltap",)))
_register(_m("mcc", _mcc, tn_dependent_at_zero=True,
             aliases=("matthews", "phi")))
_register(_m("kappa", _kappa, tn_dependent_at_zero=True,
             aliases=("cohen_kappa",)))
_register(_m("fowlkes_mallows", _fowlkes_mallows, aliases=("fm",)))

MEASURES: Dict[str, BinaryMeasure] = _REGISTRY


def get_measure(name: str | BinaryMeasure) -> BinaryMeasure:
    """Look up a measure by name or alias."""
    if isinstance(name, BinaryMeasure):
        return name
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown measure {name!r}. Available: "
            + ", ".join(sorted({m.name for m in _REGISTRY.values()}))
        )
    return _REGISTRY[key]


def list_measures(decomposable: Optional[bool] = None,
                  skew_invariant: Optional[bool] = None) -> list[str]:
    """Canonical measure names, optionally filtered by structural property."""
    out = sorted({m.name: m for m in _REGISTRY.values()}.values(),
                 key=lambda m: m.name)
    if decomposable is not None:
        out = [m for m in out if m.decomposable is decomposable]
    if skew_invariant is not None:
        out = [m for m in out if m.skew_invariant is skew_invariant]
    return [m.name for m in out]
