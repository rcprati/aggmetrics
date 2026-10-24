# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Diagnostics: warnings the framework can raise about its own measures.

This module is what distinguishes the package from a collection of formulas. Each
check corresponds to a result about when a measure behaves as a practitioner is
likely to assume, and fires when it does not:

* applying a weighted-averaging interpretation to a measure that is not
  aggregation-decomposable, where the pooled value may lie outside the range of
  the per-class values entirely;
* reporting a ``tn``-dependent measure on one-hot multiclass data, where the
  attainable floor is bounded away from the nominal minimum by an amount growing
  with the number of classes;
* selecting an operating point by maximising a measure whose optimal decision
  rule is not the Bayes rule for 0/1 loss --- degenerately so for recall and
  specificity;
* macro-averaging with undefined per-class entries, where the convention chosen
  can shift the reported value materially;
* reporting micro-precision, micro-recall or micro-F1 on single-label data, where
  all three collapse onto accuracy.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .aggregation import aggregate
from .measures import BinaryMeasure, get_measure

__all__ = ["Finding", "diagnose", "check_hull", "micro_macro_gap",
           "uniform_error_deviation", "check_ovr_skew_drift",
           "check_redundancy", "MeasureWarning"]


class MeasureWarning(UserWarning):
    """Raised when a measure is being used outside the range where it behaves
    as commonly assumed."""


@dataclass
class Finding:
    """One diagnostic result."""

    code: str
    severity: str          # 'error' | 'warning' | 'note'
    message: str
    detail: dict

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity}] {self.code}: {self.message}"


# ---------------------------------------------------------------------------

def check_hull(im, measure: str | BinaryMeasure,
               aggregation: str = "macro") -> Optional[Finding]:
    """Check whether the pooled value lies within the per-class range.

    Any weighted average of the per-class values must lie between their minimum
    and maximum. A micro value outside that interval therefore proves that *no*
    weighting reproduces it, so no weighted-averaging interpretation of
    micro-averaging is available for that measure.
    """
    meas = get_measure(measure)
    micro = im.compute(meas, "micro")
    parts = im.parts(meas, aggregation)
    parts = parts[~np.isnan(parts)]
    if parts.size == 0 or np.isnan(micro):
        return None
    lo, hi = float(parts.min()), float(parts.max())
    inside = lo - 1e-12 <= micro <= hi + 1e-12
    if inside:
        return None
    return Finding(
        code="outside_hull",
        severity="warning",
        message=(f"micro-{meas.name} = {micro:.4f} lies outside the range of the "
                 f"per-{'class' if aggregation == 'macro' else 'example'} values "
                 f"[{lo:.4f}, {hi:.4f}], so it is not a weighted average of them. "
                 f"{meas.name!r} is not aggregation-decomposable."),
        detail=dict(measure=meas.name, micro=micro, lo=lo, hi=hi,
                    decomposable=meas.decomposable),
    )


def micro_macro_gap(im, measure: str | BinaryMeasure) -> dict:
    """Decompose the micro--macro gap for a decomposable measure.

    For an aggregation-decomposable measure the gap equals the covariance between
    the per-class denominators and the per-class values, normalised by the mean
    denominator. The identity does not extend beyond that class, and the returned
    dictionary reports ``exact=False`` when it does not apply.
    """
    meas = get_measure(measure)
    micro = im.compute(meas, "micro")
    macro = im.compute(meas, "macro")
    out = dict(measure=meas.name, micro=micro, macro=macro,
               gap=micro - macro, decomposable=meas.decomposable, exact=False)
    if not meas.decomposable:
        out["note"] = ("gap is not a covariance for this measure; see the "
                       "curvature/mismatched-weight decomposition")
        return out
    pc = im.per_class()
    w = np.array([meas.denominator(tuple(row)) for row in pc], float)
    v = np.array([meas(*row) for row in pc], float)
    ok = ~np.isnan(v) & (w > 0)
    w, v = w[ok], v[ok]
    if w.size == 0:
        return out
    wbar, vbar = w.mean(), v.mean()
    cov = float(np.mean((w - wbar) * (v - vbar)))
    out.update(covariance=cov, mean_denominator=float(wbar),
               predicted_gap=cov / wbar if wbar else float("nan"),
               weights=w.tolist(), values=v.tolist(), exact=True)
    return out


def _is_structural(meas) -> bool:
    """True if the denominator depends only on the total count (such as ``n``)."""
    a, b = (1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0)      # the same total
    return meas.denominator(a) == meas.denominator(b)


def denominator_balance(im, measure: str | BinaryMeasure) -> dict:
    """Per-class denominators of a decomposable measure, and whether they are equal.

    By Corollary 1 of the companion paper, micro-averaging equals unweighted
    macro-averaging for every configuration of the per-class values if and only
    if all per-class denominators are equal. For recall the denominator is the
    class size, so micro and macro recall coincide on data whose classes have the
    same size; for precision it is the number of predicted positives.

    Returns ``{'measure', 'denominators', 'balanced', 'structural', 'micro',
    'macro'}``: ``structural`` is true when the denominator is the same for every
    confusion matrix with the same total (``error_rate``, ``label_accuracy``), so the
    balance holds on any data and says nothing about this one.
    """
    meas = get_measure(measure)
    if not meas.decomposable:
        raise ValueError(f"{meas.name!r} is not aggregation-decomposable: Corollary 1 "
                         "applies only to decomposable measures")
    den = np.array([meas.denominator(tuple(row)) for row in im.per_class()], float)
    equal = np.allclose(den, den[0], rtol=1e-9, atol=0)
    balanced = bool(den.size > 1 and den[0] > 0 and equal)
    return dict(measure=meas.name, denominators=den, balanced=balanced,
                structural=_is_structural(meas),
                micro=im.compute(meas, "micro"), macro=im.compute(meas, "macro"))


def uniform_error_deviation(im) -> float:
    """Compute the mean uniform-error deviation across classes in multiclass OvR.

    Proposition 6(2) of the companion paper proves that One-vs-Rest specificity is
    strictly prior-free if and only if the off-diagonal error rates into class j
    are uniform ($R_{kj} = \\text{const}$ for all $k \\neq j$). This function computes
    the mean over classes $j$ of the sample standard deviation of $R_{kj}$ over
    $k \\neq j$.
    """
    if im.task != "multiclass" or im.m <= 2:
        return 0.0
    C = (im.Y.T @ im.P).astype(float)
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    R = C / row_sums
    stds = []
    for j in range(im.m):
        off_diag = [R[k, j] for k in range(im.m) if k != j]
        stds.append(float(np.std(off_diag, ddof=1) if len(off_diag) > 1 else 0.0))
    return float(np.mean(stds))


def check_ovr_skew_drift(im, threshold: float = 0.01) -> Optional[Finding]:
    """Check whether multiclass One-vs-Rest specificity and balanced accuracy
    suffer from non-trivial prior drift due to asymmetric off-diagonal error rates.
    """
    if im.task != "multiclass" or im.m <= 2:
        return None
    dev = uniform_error_deviation(im)
    if dev < threshold:
        return None
    return Finding(
        code="ovr_skew_drift",
        severity="note",
        message=(f"Mean uniform-error deviation is {dev:.4f} >= {threshold:.4f}; "
                 f"multiclass macro-specificity and macro-balanced-accuracy will "
                 f"drift under class prior shifts (with Delta_BA = 0.5 * Delta_spec) "
                 f"due to asymmetric off-diagonal confusion rates into classes."),
        detail=dict(uniform_error_dev=dev, threshold=threshold, m=im.m),
    )


def check_redundancy(
    im, measures: Optional[Sequence[str]] = None
) -> Optional[Finding]:
    """Check whether a suite of measures exceeds the intrinsic degrees of freedom.

    As shown in Proposition 3 and Remark 2 of the companion paper, a confusion
    matrix has strictly d = m(m-1) independent degrees of freedom (or d = 2 in
    binary classification, spanned by the canonical orthogonal basis (recall,
    specificity)). A benchmark suite with k > d measures inevitably contains
    collinear gradients and functional redundancy (e.g. reporting accuracy,
    balanced accuracy, and F1 alongside recall and specificity under fixed
    priors).
    """
    if measures is None:
        return None
    unique_measures = list(dict.fromkeys(measures))
    dof = im.degrees_of_freedom
    if len(unique_measures) > dof:
        return Finding(
            code="redundant_suite",
            severity="note",
            message=(f"The requested suite of {len(unique_measures)} measures "
                     f"exceeds the {dof} independent degrees of freedom of the "
                     f"confusion matrix. Under fixed priors, scalar measures "
                     f"beyond the canonical orthogonal basis have collinear "
                     f"gradients and convey redundant information."),
            detail=dict(measures=unique_measures, count=len(unique_measures), dof=dof),
        )
    return None


# ---------------------------------------------------------------------------

def diagnose(im, measures: Optional[Sequence[str]] = None,
             *, aggregation: str = "macro", emit_warnings: bool = False
             ) -> List[Finding]:
    """Run all checks against an :class:`~aggmetrics.core.IndicatorMatrix`.

    Returns a list of :class:`Finding`. With ``emit_warnings=True`` each finding
    of severity ``'warning'`` is also raised as a :class:`MeasureWarning`, which
    makes the checks usable inside a test suite or a CI pipeline.
    """
    names = list(measures) if measures is not None else [
        "precision", "recall", "f1", "balanced_accuracy", "gmean", "mcc",
        "label_accuracy", "specificity"]
    out: List[Finding] = []
    one_hot = im.task == "multiclass" and np.allclose(im.Y.sum(axis=1), 1.0)

    # 1. pooled value outside the per-class range
    for n in names:
        f = check_hull(im, n, aggregation)
        if f is not None:
            out.append(f)

    # 2. tn-dependent measures on one-hot multiclass data
    if one_hot and im.m > 2:
        for n in names:
            meas = get_measure(n)
            if not meas.tn_dependent_at_zero:
                continue
            floor = meas.attainable_floor(im.m)
            out.append(Finding(
                code="inflated_floor",
                severity="warning",
                message=(f"{meas.name!r} cannot fall below {floor:.4f} on this "
                         f"{im.m}-class one-hot problem even if every prediction "
                         f"is wrong, because a misclassified example still "
                         f"contributes {im.m - 2} true negatives."),
                detail=dict(measure=meas.name, m=im.m, floor=floor),
            ))

    # 3. micro-collapse on single-label data
    if one_hot:
        vals = {n: im.compute(n, "micro") for n in ("precision", "recall", "f1")}
        acc = im.accuracy()
        if max(abs(v - acc) for v in vals.values()) < 1e-9:
            out.append(Finding(
                code="micro_collapse",
                severity="note",
                message=("micro-precision, micro-recall and micro-F1 are all "
                         f"equal to accuracy ({acc:.6f}) on single-label data; "
                         "reporting more than one of them adds no information."),
                detail=dict(accuracy=acc, **vals),
            ))

    # 3b. equal per-class denominators: micro and macro coincide (Corollary 1), unless
    #     that is so for every data set (a denominator that is the total count)
    if im.m > 1:
        for n in names:
            meas = get_measure(n)
            if not meas.decomposable:
                continue
            info = denominator_balance(im, meas)
            if not info["balanced"] or info["structural"]:
                continue
            out.append(Finding(
                code="balanced_denominators",
                severity="note",
                message=(f"all per-class denominators of {meas.name!r} are equal "
                         f"({info['denominators'][0]:.6g}), so micro and macro "
                         f"{meas.name} are the same number ({info['micro']:.6f}) on "
                         "this data (Corollary 1); reporting both adds no "
                         "information."),
                detail=info,
            ))

    # 4. undefined per-class / per-example entries (with a single label the per-example
    #    values are the entries themselves, so only the per-class check is meaningful)
    for n in names:
        for agg in (("macro",) if im.m == 1 else ("macro", "exemplar")):
            try:
                p = im.parts(n, agg)
            except Exception:
                continue
            k = int(np.isnan(p).sum())
            if k == 0:
                continue
            excl = aggregate(im.Y, im.P, n, agg, convention="exclude")
            zero = aggregate(im.Y, im.P, n, agg, convention="zero")
            out.append(Finding(
                code="undefined_entries",
                severity="warning",
                message=(f"{agg}-{get_measure(n).name} has {k} of {p.size} "
                         f"undefined entries; the reported value is "
                         f"{excl:.4f} under the exclusion convention and "
                         f"{zero:.4f} under substitution of zero "
                         f"(difference {abs(excl - zero):.4f})."),
                detail=dict(measure=get_measure(n).name, aggregation=agg,
                            undefined=k, total=int(p.size),
                            exclude=excl, zero=zero),
            ))

    # 5. measures not optimised by the Bayes rule for 0/1 loss
    for n in names:
        meas = get_measure(n)
        cons = meas.erm_consistent()
        if cons is False:
            t = meas.bayes_threshold()
            extra = ""
            if t is not None and (t <= 1e-9 or t >= 1 - 1e-9):
                extra = (" This is degenerate: the optimum is to predict every "
                         "instance " + ("positive." if t <= 1e-9 else "negative."))
                out.append(Finding(
                    code="degenerate_optimum",
                    severity="warning",
                    message=(f"{meas.name!r} is maximised at threshold {t:.3f}."
                             + extra),
                    detail=dict(measure=meas.name, bayes_threshold=t),
                ))
            else:
                out.append(Finding(
                    code="not_erm_consistent",
                    severity="note",
                    message=(f"{meas.name!r} is maximised at threshold "
                             f"{t:.3f}, not 0.5, so standard training does not "
                             f"target it."),
                    detail=dict(measure=meas.name, bayes_threshold=t),
                ))

    # 6. multiclass OvR prior drift (asymmetric off-diagonal errors)
    drifting = ("specificity", "balanced_accuracy", "gmean", "informedness")
    if one_hot and im.m > 2 and any(n in names for n in drifting):
        f_drift = check_ovr_skew_drift(im)
        if f_drift is not None:
            out.append(f_drift)

    # 7. suite redundancy beyond intrinsic degrees of freedom
    if measures is not None:
        f_red = check_redundancy(im, measures)
        if f_red is not None:
            out.append(f_red)

    if emit_warnings:
        for f in out:
            if f.severity == "warning":
                warnings.warn(str(f), MeasureWarning, stacklevel=2)
    return out
