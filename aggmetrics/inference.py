# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Deriving a measure's structural properties from its formula.

The framework's generative claim is that any function of the four
confusion-matrix counts extends to the other settings by substituting an
aggregation operator. Taken alone that is cheap: the extensions are always
*defined*. What makes an extension usable is knowing which identities hold for
it, and this module shows that those properties need not be supplied by hand ---
they are recoverable from the formula itself.

Four properties are inferred:

*Decomposability.* A measure admits the representation
:math:`M = \\langle a, v\\rangle / \\langle b, v\\rangle` iff
:math:`M(v)\\,\\langle b, v\\rangle - \\langle a, v\\rangle = 0` for all count
vectors :math:`v`. Sampling count vectors turns this into a homogeneous linear
system in the eight unknowns :math:`(a, b)`, solved by taking the right singular
vector of smallest singular value. Because a near-null direction can arise by
chance, the recovered coefficients are then validated on count vectors that were
not used to fit them; only if they reproduce the measure there is the measure
reported as decomposable. This both decides the property and returns the
coefficients, which is what the Bayes threshold and the weighted-averaging
identity need.

*Skew-invariance.* Tested directly against the definition, by rescaling the
positive- and negative-class counts independently.

*Dependence on* ``tn`` *at* ``tp = 0``. Decides whether the measure can attain its
nominal minimum under one-hot multiclass encoding.

*Direction.* Whether correcting a prediction increases the value.

Inference is numerical and therefore not a proof; ``verify_inference`` checks it
against the measures whose properties are known analytically, and the test suite
requires exact agreement on all of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["InferredProperties", "infer_properties", "register_measure",
           "verify_inference"]

Formula = Callable[[float, float, float, float], float]


@dataclass
class InferredProperties:
    """Structural properties recovered from a formula."""

    linear: Optional[Tuple[Tuple[float, float, float, float],
                           Tuple[float, float, float, float]]]
    skew_invariant: bool
    tn_dependent_at_zero: bool
    greater_is_better: bool
    residual: float                 # fit residual on held-out count vectors
    diagnostics: Dict[str, float]

    @property
    def decomposable(self) -> bool:
        return self.linear is not None

    def summary(self) -> str:
        lines = [
            f"decomposable          : {self.decomposable}",
            f"skew-invariant        : {self.skew_invariant}",
            f"tn-dependent at tp=0  : {self.tn_dependent_at_zero}",
            f"greater is better     : {self.greater_is_better}",
        ]
        if self.linear is not None:
            a, b = self.linear

            def fmt(v):
                return "(" + ", ".join(f"{x:+.4g}" for x in v) + ")"

            lines += [f"numerator coeffs      : {fmt(a)}",
                      f"denominator coeffs    : {fmt(b)}",
                      f"held-out residual     : {self.residual:.2e}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------

def _sample_counts(n: int, rng: np.random.Generator,
                   low: float = 1.0, high: float = 60.0) -> np.ndarray:
    """Count vectors that keep every marginal strictly positive.

    Degenerate vectors are excluded so that measures with vanishing denominators
    do not contaminate the fit; the properties being inferred are about the
    interior of the domain.
    """
    return rng.uniform(low, high, size=(n, 4))


def _eval(fn: Formula, V: np.ndarray) -> np.ndarray:
    return np.array([fn(*v) for v in V], dtype=float)


def _fit_linear_fractional(fn: Formula, rng: np.random.Generator,
                           n_fit: int = 40, n_check: int = 40,
                           tol: float = 1e-8):
    """Recover ``(a, b)`` with ``M = <a,v>/<b,v>``, or None.

    Returns ``(a, b, residual)``. The system is homogeneous, so the solution is
    fixed only up to scale; it is normalised so that the largest denominator
    coefficient is positive and of unit magnitude, which makes the output
    comparable with hand-written coefficients.
    """
    V = _sample_counts(n_fit, rng)
    y = _eval(fn, V)
    if not np.all(np.isfinite(y)):
        return None

    # rows of  [ -v | y * v ]  annihilate (a, b)
    A = np.hstack([-V, y[:, None] * V])
    *_, Vt = np.linalg.svd(A, full_matrices=False)
    sol = Vt[-1]
    a, b = sol[:4], sol[4:]

    if np.max(np.abs(b)) < 1e-10:  # pragma: no cover - defensive: V has full column
        return None                # rank, so a null vector with b = 0 would force a = 0
    k = b[np.argmax(np.abs(b))]
    a, b = a / k, b / k

    # validate on fresh count vectors: this is what rejects accidental fits
    Vc = _sample_counts(n_check, rng)
    yc = _eval(fn, Vc)
    den = Vc @ b
    ok = np.abs(den) > 1e-9
    if not ok.any() or not np.all(np.isfinite(yc[ok])):
        return None
    pred = (Vc @ a)[ok] / den[ok]
    scale = max(1.0, float(np.max(np.abs(yc[ok]))))
    residual = float(np.max(np.abs(pred - yc[ok])) / scale)
    if residual > tol:
        return None

    a = np.where(np.abs(a) < 1e-9, 0.0, a)
    b = np.where(np.abs(b) < 1e-9, 0.0, b)
    return tuple(float(x) for x in a), tuple(float(x) for x in b), residual


def _test_skew_invariance(fn: Formula, rng: np.random.Generator,
                          trials: int = 60, tol: float = 1e-9) -> Tuple[bool, float]:
    """Is ``M(lam*tp, mu*fp, lam*fn, mu*tn) == M(tp, fp, fn, tn)``?"""
    worst = 0.0
    for _ in range(trials):
        v = _sample_counts(1, rng)[0]
        base = fn(*v)
        if not np.isfinite(base):
            continue
        lam, mu = rng.uniform(0.2, 5.0, 2)
        alt = fn(lam * v[0], mu * v[1], lam * v[2], mu * v[3])
        if not np.isfinite(alt):
            continue
        worst = max(worst, abs(alt - base) / max(1.0, abs(base)))
    return bool(worst <= tol), float(worst)


def _test_tn_dependence_at_zero(fn: Formula, rng: np.random.Generator,
                                trials: int = 40, tol: float = 1e-9
                                ) -> Tuple[bool, float]:
    """At ``tp = 0``, does the value still move when only ``tn`` changes?"""
    worst = 0.0
    for _ in range(trials):
        fp, fn_ = rng.uniform(1.0, 50.0, 2)
        vals = []
        for tn in (1.0, 25.0, 500.0, 5000.0):
            v = fn(0.0, fp, fn_, tn)
            if np.isfinite(v):
                vals.append(v)
        if len(vals) >= 2:
            worst = max(worst, max(vals) - min(vals))
    return bool(worst > tol), float(worst)


def _test_direction(fn: Formula, rng: np.random.Generator,
                    trials: int = 40) -> Tuple[bool, float]:
    """Does correcting a prediction raise the value?

    A correction moves the counts by ``(+1, -1, -1, +1)``, which is the admissible
    move under one-hot encoding; it is favourable in all four coordinates at once.
    """
    ups = 0
    total = 0
    for _ in range(trials):
        v = _sample_counts(1, rng, low=5.0, high=60.0)[0]
        before = fn(*v)
        after = fn(v[0] + 1, v[1] - 1, v[2] - 1, v[3] + 1)
        if np.isfinite(before) and np.isfinite(after) and abs(after - before) > 1e-12:
            total += 1
            ups += after > before
    if total == 0:
        return True, 0.0
    return bool(ups >= total / 2), float(ups / total)


# ---------------------------------------------------------------------------

def infer_properties(fn: Formula, *, seed: int = 0,
                     tol: float = 1e-8) -> InferredProperties:
    """Recover the structural properties of a binary measure from its formula.

    Parameters
    ----------
    fn
        A callable ``fn(tp, fp, fn, tn) -> float``. It may return NaN on
        degenerate inputs; such points are skipped.
    seed
        Seed for the count vectors sampled during inference.
    tol
        Relative tolerance for the held-out validation of the linear-fractional
        fit, and for the invariance tests.

    Examples
    --------
    >>> from aggmetrics.inference import infer_properties
    >>> p = infer_properties(lambda tp, fp, fn, tn: tp / (tp + fn))   # recall
    >>> p.decomposable, p.skew_invariant
    (True, True)
    >>> p = infer_properties(lambda tp, fp, fn, tn:
    ...     0.5 * (tp / (tp + fn) + tn / (tn + fp)))                  # balanced acc.
    >>> p.decomposable, p.skew_invariant
    (False, True)
    """
    rng = np.random.default_rng(seed)
    fit = _fit_linear_fractional(fn, rng, tol=tol)
    skew, skew_worst = _test_skew_invariance(fn, rng, tol=tol)
    tn_dep, tn_worst = _test_tn_dependence_at_zero(fn, rng)
    up, up_frac = _test_direction(fn, rng)

    linear = None if fit is None else (fit[0], fit[1])
    residual = float("nan") if fit is None else fit[2]
    return InferredProperties(
        linear=linear, skew_invariant=skew, tn_dependent_at_zero=tn_dep,
        greater_is_better=up, residual=residual,
        diagnostics=dict(skew_worst=skew_worst, tn_spread=tn_worst,
                         fraction_increasing=up_frac),
    )


def register_measure(name: str, fn: Formula, *, aliases: Sequence[str] = (),
                     note: str = "", overwrite: bool = False,
                     seed: int = 0, **overrides):
    """Add a user-supplied measure, inferring what is not given explicitly.

    This is the generative claim as an operation: supply a binary formula and the
    measure becomes available under all four aggregation operators, with the
    properties that determine which identities apply to it derived automatically.
    Any property may be overridden by keyword if it is known analytically.

    Returns the registered :class:`~aggmetrics.measures.BinaryMeasure`.

    Examples
    --------
    >>> import numpy as np
    >>> from aggmetrics import IndicatorMatrix, register_measure
    >>> # Prevalence-threshold, a measure not in the registry
    >>> def prevalence_threshold(tp, fp, fn, tn):
    ...     r = tp / (tp + fn); s = tn / (tn + fp)
    ...     d = r + s - 1
    ...     return (np.sqrt(r * (1 - s)) + s - 1) / d if abs(d) > 1e-12 else np.nan
    >>> m = register_measure('prevalence_threshold', prevalence_threshold)
    >>> m.decomposable, m.skew_invariant
    (False, True)
    >>> im = IndicatorMatrix([0, 1, 2, 2, 1, 0], [0, 2, 2, 1, 1, 1],
    ...                      task='multiclass')
    >>> isinstance(im.compute('prevalence_threshold', 'macro'), float)
    True
    """
    from .measures import _REGISTRY, BinaryMeasure  # local import: avoid a cycle

    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _REGISTRY and not overwrite:
        raise ValueError(f"measure {name!r} already registered; "
                         "pass overwrite=True to replace it")

    inferred = infer_properties(fn, seed=seed)
    kwargs = dict(
        linear=inferred.linear,
        skew_invariant=inferred.skew_invariant,
        tn_dependent_at_zero=inferred.tn_dependent_at_zero,
        greater_is_better=inferred.greater_is_better,
    )
    unknown = set(overrides) - set(kwargs)
    if unknown:
        raise TypeError(f"unexpected override(s): {sorted(unknown)}")
    kwargs.update(overrides)

    measure = BinaryMeasure(name=key, fn=fn, aliases=tuple(aliases),
                            note=note, **kwargs)
    _REGISTRY[key] = measure
    for a in aliases:
        _REGISTRY[a.strip().lower().replace("-", "_").replace(" ", "_")] = measure
    return measure


def verify_inference(names: Optional[Sequence[str]] = None,
                     *, seed: int = 0) -> Dict[str, dict]:
    """Compare inferred properties against the registry's analytic annotations.

    Returns a mapping from measure name to a record of agreement. Used by the test
    suite: inference must reproduce every hand-derived property, which is a check
    on the inference and on the annotations at once.
    """
    from .measures import get_measure, list_measures

    out: Dict[str, dict] = {}
    for n in (names or list_measures()):
        meas = get_measure(n)
        inf = infer_properties(meas.fn, seed=seed)
        rec = dict(
            decomposable_declared=meas.decomposable,
            decomposable_inferred=inf.decomposable,
            skew_declared=meas.skew_invariant,
            skew_inferred=inf.skew_invariant,
            tn_declared=meas.tn_dependent_at_zero,
            tn_inferred=inf.tn_dependent_at_zero,
            direction_declared=meas.greater_is_better,
            direction_inferred=inf.greater_is_better,
        )
        rec["agrees"] = (
            rec["decomposable_declared"] == rec["decomposable_inferred"]
            and rec["skew_declared"] == rec["skew_inferred"]
            and rec["tn_declared"] == rec["tn_inferred"]
            and rec["direction_declared"] == rec["direction_inferred"]
        )
        if meas.decomposable and inf.decomposable:
            a0, b0 = meas.linear
            a1, b1 = inf.linear
            a0, b0 = np.asarray(a0, float), np.asarray(b0, float)
            k = b0[np.argmax(np.abs(b0))]
            rec["coeff_max_error"] = float(max(
                np.max(np.abs(a0 / k - np.asarray(a1))),
                np.max(np.abs(b0 / k - np.asarray(b1)))))
        out[meas.name] = rec
    return out
