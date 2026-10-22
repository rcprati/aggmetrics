# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
The generative claim, as executable tests.

Two things are checked. First, that inference recovers the properties of every
measure whose properties are known analytically --- which validates the inference
and the hand annotations simultaneously. Second, that a user-supplied measure
becomes fully usable, with its properties derived rather than declared.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aggmetrics import (
    IndicatorMatrix,
    get_measure,
    infer_properties,
    list_measures,
    register_measure,
    verify_inference,
)
from aggmetrics.inference import _fit_linear_fractional

# ---------------------------------------------------------------------------
# Inference agrees with the analytic annotations
# ---------------------------------------------------------------------------

def test_inference_reproduces_every_annotation():
    """The strongest self-check available: no measure may disagree."""
    report = verify_inference()
    bad = {k: v for k, v in report.items() if not v["agrees"]}
    assert not bad, f"inference disagrees with the registry for: {sorted(bad)}"
    assert len(report) == len(list_measures())


def test_recovered_coefficients_match_the_declared_ones():
    report = verify_inference()
    errs = [v["coeff_max_error"] for v in report.values()
            if "coeff_max_error" in v]
    assert len(errs) == len(list_measures(decomposable=True))
    assert max(errs) < 1e-10


@pytest.mark.parametrize("name", list_measures(decomposable=True))
def test_decomposable_measures_are_detected(name):
    p = infer_properties(get_measure(name).fn)
    assert p.decomposable
    assert p.residual < 1e-10


@pytest.mark.parametrize("name", list_measures(decomposable=False))
def test_non_decomposable_measures_are_rejected(name):
    p = infer_properties(get_measure(name).fn)
    assert not p.decomposable
    assert p.linear is None


# ---------------------------------------------------------------------------
# The fit must not accept a measure that only looks linear-fractional
# ---------------------------------------------------------------------------

def test_holdout_validation_rejects_accidental_fits():
    """A quadratic ratio is not linear-fractional and must be rejected.

    Without validation on unseen count vectors, an eight-parameter fit to a
    handful of points can appear to succeed.
    """
    def quadratic(tp, fp, fn, tn):
        return (tp * tn) / ((tp + fp) * (tp + fn)) if (tp + fp) * (tp + fn) else float("nan")

    rng = np.random.default_rng(0)
    assert _fit_linear_fractional(quadratic, rng) is None
    assert infer_properties(quadratic).decomposable is False


def test_a_genuinely_linear_fractional_measure_is_accepted():
    """A measure not in the registry, but of the admissible form."""
    def weird(tp, fp, fn, tn):          # 3tp + tn over the total
        d = tp + fp + fn + tn
        return (3 * tp + tn) / d if d else float("nan")

    p = infer_properties(weird)
    assert p.decomposable
    a, b = p.linear
    # normalised so the largest |b| is 1; here all four b are equal
    assert a[0] == pytest.approx(3 * b[0], rel=1e-9)
    assert a[3] == pytest.approx(b[3], rel=1e-9)
    assert a[1] == pytest.approx(0.0, abs=1e-9)
    assert a[2] == pytest.approx(0.0, abs=1e-9)


def test_constant_measure_is_handled():
    p = infer_properties(lambda tp, fp, fn, tn: 1.0)
    assert p.skew_invariant is True          # trivially


# ---------------------------------------------------------------------------
# Registering a user measure: the generative claim as an operation
# ---------------------------------------------------------------------------

def test_register_a_new_measure_and_use_all_operators():
    """Prevalence threshold: a real measure absent from the registry."""
    def prevalence_threshold(tp, fp, fn, tn):
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        s = tn / (tn + fp) if (tn + fp) else float("nan")
        if math.isnan(r) or math.isnan(s):
            return float("nan")
        d = r + s - 1.0
        return (math.sqrt(r * (1 - s)) + s - 1) / d if abs(d) > 1e-12 else float("nan")

    m = register_measure("prevalence_threshold", prevalence_threshold,
                         aliases=("pt",), overwrite=True)
    # properties were derived, not supplied
    assert m.decomposable is False
    assert m.skew_invariant is True          # a function of recall and specificity
    assert m.bayes_threshold() is None       # no closed form for non-decomposable

    rng = np.random.default_rng(1)
    Y = (rng.random((150, 4)) < 0.4).astype(int)
    P = Y.copy(); flip = rng.random(Y.shape) < 0.2; P[flip] = 1 - P[flip]
    im = IndicatorMatrix(Y, P, task="multilabel")
    for agg in ("micro", "macro", "exemplar"):
        v = im.compute("prevalence_threshold", agg)
        assert np.isfinite(v)
    # reachable by alias, and via the aggregate() entry point
    assert get_measure("pt").name == "prevalence_threshold"


def test_registered_decomposable_measure_satisfies_theorem1():
    """A user measure of the admissible form inherits the identity automatically."""
    def tp_minus_fp_rate(tp, fp, fn, tn):
        d = tp + fp + fn + tn
        return (tp - fp) / d if d else float("nan")

    m = register_measure("tp_minus_fp_rate", tp_minus_fp_rate, overwrite=True)
    assert m.decomposable

    rng = np.random.default_rng(2)
    Y = (rng.random((200, 6)) < 0.35).astype(int)
    P = Y.copy(); flip = rng.random(Y.shape) < 0.25; P[flip] = 1 - P[flip]
    im = IndicatorMatrix(Y, P, task="multilabel")

    pc = im.per_class()
    num = np.array([m.numerator(tuple(r)) for r in pc])
    den = np.array([m.denominator(tuple(r)) for r in pc])
    assert im.compute(m, "micro") == pytest.approx(num.sum() / den.sum(), abs=1e-10)

    # and its Bayes threshold follows from the recovered coefficients:
    # the numerator is tp - fp, so alpha = beta = 1 and gamma = delta = 0,
    # giving t* = (beta + delta) / (alpha + beta + gamma + delta) = 1/2.
    assert m.bayes_threshold() == pytest.approx(0.5, abs=1e-9)
    assert m.erm_consistent() is True


def test_overrides_take_precedence_over_inference():
    m = register_measure("always_half", lambda tp, fp, fn, tn: 0.5,
                         skew_invariant=False, overwrite=True)
    assert m.skew_invariant is False        # inference would have said True


def test_duplicate_registration_is_refused():
    register_measure("dup_probe", lambda tp, fp, fn, tn: tp, overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        register_measure("dup_probe", lambda tp, fp, fn, tn: tp)


def test_unknown_override_is_refused():
    with pytest.raises(TypeError, match="unexpected override"):
        register_measure("bad_probe", lambda tp, fp, fn, tn: tp,
                         not_a_property=True, overwrite=True)


def test_inferred_floor_and_diagnostics_apply_to_user_measures():
    """A registered tn-dependent measure is picked up by the diagnostics."""
    from aggmetrics import diagnose

    def tn_heavy(tp, fp, fn, tn):
        d = tp + fp + fn + tn
        return (tp + 2 * tn) / (2 * d) if d else float("nan")

    m = register_measure("tn_heavy", tn_heavy, overwrite=True)
    assert m.tn_dependent_at_zero is True
    assert m.attainable_floor(10) > 0.5

    y = np.arange(200) % 10
    im = IndicatorMatrix(y, (y + 1) % 10, task="multiclass")
    codes = {f.code for f in diagnose(im, measures=["tn_heavy"])}
    assert "inflated_floor" in codes


def test_summary_is_human_readable():
    p = infer_properties(get_measure("f1").fn)
    text = p.summary()
    assert "decomposable" in text and "numerator coeffs" in text
