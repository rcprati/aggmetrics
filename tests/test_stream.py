"""aggmetrics.stream: temporal states, delayed labels, sizing, attribution, monitor."""
from __future__ import annotations

from fractions import Fraction as F

import numpy as np
import pytest

from aggmetrics import IndicatorMatrix, list_measures, under_prior
from aggmetrics.stream import (
    ClassTimeState,
    DelayedEvaluator,
    RateMonitor,
    Snapshot,
    TemporalState,
    attribute,
    calibrate_threshold,
    sizing,
)

# The eight-instance running example of the data-stream paper (classes A, B, C).
Y = [0, 1, 0, 2, 1, 0, 1, 1]
P = [0, 1, 1, 2, 1, 0, 2, 1]
DELAY = [0, 1, 2, 3, 2, 3, 2, 1]
AUDITED = [1, 1, 1, 1, 1, 1, 0, 1]


def stream(n=300, m=3, seed=0, acc=0.7):
    r = np.random.default_rng(seed)
    y = r.integers(0, m, n)
    p = np.where(r.random(n) < acc, y, r.integers(0, m, n))
    return y, p


# -- temporal state -------------------------------------------------------------

@pytest.mark.parametrize("m,task", [(2, "binary"), (3, "multiclass")])
def test_cumulative_state_is_the_indicator_matrix(m, task):
    y, p = stream(400, m, seed=1)
    s = TemporalState(m)
    for a, b in zip(y, p):
        s.update(a, b)
    im = IndicatorMatrix(y, p, task=task)
    hows = ["micro"] if task == "binary" else ["micro", "macro", "weighted"]
    for name in ("precision", "recall", "f1", "mcc", "gmean", "label_accuracy"):
        for how in hows:
            assert s.compute(name, how) == pytest.approx(im.compute(name, how))


def test_window_state_is_the_last_w_instances():
    y, p = stream(200, 3, seed=2)
    W = 37
    s = TemporalState(3, window=W)
    for a, b in zip(y, p):
        s.update(a, b)
    im = IndicatorMatrix(y[-W:], p[-W:], task="multiclass", n_classes=3)
    for name in ("precision", "recall", "f1"):
        assert s.compute(name, "macro") == pytest.approx(im.compute(name, "macro"))
    assert s.mass == W and s.C.sum() == W


def test_fading_state_is_the_weighted_sum():
    y, p = stream(120, 3, seed=3)
    a = 0.93
    s = TemporalState(3, alpha=a)
    C = np.zeros((3, 3))
    for k, (yy, pp) in enumerate(zip(y, p)):
        s.update(yy, pp)
    t = len(y)
    for k, (yy, pp) in enumerate(zip(y, p)):
        C[yy, pp] += a ** (t - 1 - k)
    np.testing.assert_allclose(s.C, C, atol=1e-12)
    assert s.mass == pytest.approx((1 - a ** t) / (1 - a))
    assert s.normalized.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("kw", [dict(alpha=0.9), dict(window=25), dict()])
def test_theorem_1_pooled_equals_denominator_weighted_macro(kw):
    y, p = stream(200, 4, seed=4)
    s = TemporalState(4, **kw)
    for a, b in zip(y, p):
        s.update(a, b)
    from aggmetrics import get_measure
    for name in list_measures(decomposable=True):
        meas = get_measure(name)
        cells = s.per_class()
        den = np.array([meas.denominator(tuple(c)) for c in cells])
        vals = np.array([meas(*c) for c in cells])
        ok = den > 0
        if not ok.any() or not (den[ok] > 0).all():
            continue
        weighted = float(np.sum(den[ok] * vals[ok]) / den[ok].sum())
        assert s.compute(name, "micro") == pytest.approx(weighted, abs=1e-9), name


def test_running_example_window_state():
    s = TemporalState(3, window=4)
    for y, p in zip(Y, P):
        s.update(y, p)
    # last four instances: (B,B) (A,A) (B,C) (B,B)
    np.testing.assert_array_equal(s.C, [[1, 0, 0], [0, 2, 1], [0, 0, 0]])
    assert s.compute("recall", "micro") == pytest.approx(3 / 4)
    value, audit = s.compute("recall", "macro", return_audit=True)
    assert value == pytest.approx(5 / 6)                       # exclude
    assert audit.defined.tolist() == [True, True, False] and audit.m_active == 2
    assert s.compute("recall", "macro", convention="zero") == pytest.approx(5 / 9)
    assert np.isnan(s.compute("recall", "macro", convention="nan"))
    # pooled recall = weighted macro with weights (1/4, 3/4, 0)
    assert s.compute("recall", "weighted") == pytest.approx(3 / 4)


def test_soft_label_term_is_the_product_tnorm():
    s = TemporalState(2)
    s.update([0.25, 0.75], 1)
    np.testing.assert_allclose(s.C, [[0, 0.25], [0, 0.75]])
    # the column sums are the prediction counts whatever the stand-in
    assert s.C.sum(axis=0).tolist() == [0.0, 1.0]


def test_fading_support_tolerance():
    s = TemporalState(2, alpha=0.5)
    s.update(1, 1)
    for _ in range(30):
        s.update(0, 0)
    assert np.isnan(s.compute("recall"))                     # class 1 faded below tol
    assert s.compute("specificity") == 1.0
    s2 = TemporalState(2, alpha=0.5, tol=0.0)
    s2.update(1, 1)
    for _ in range(30):
        s2.update(0, 0)
    assert not np.isnan(s2.compute("recall"))


def test_prior_argument_matches_under_prior():
    y, p = stream(500, 2, seed=5, acc=0.8)
    y = (y == 1).astype(int)
    p = (p == 1).astype(int)
    s = TemporalState(2)
    for a, b in zip(y, p):
        s.update(a, b)
    im = IndicatorMatrix(y, p, task="binary")
    for name in ("precision", "f1", "label_accuracy", "mcc"):
        assert s.compute(name, prior=0.05) == pytest.approx(under_prior(im, 0.05).compute(name))
    # skew-invariant measures do not move
    assert s.compute("recall", prior=0.05) == pytest.approx(s.compute("recall"))
    assert s.compute("specificity", prior=0.9) == pytest.approx(s.compute("specificity"))
    # at the observed prior the value is the value
    assert s.compute("f1", prior=s.snapshot().prior) == pytest.approx(s.compute("f1"))


def test_multiclass_prior_and_absent_class():
    s = TemporalState(3)
    for y, p in [(0, 0), (0, 1), (1, 1), (1, 1)]:
        s.update(y, p)
    assert s.compute("recall", "macro", prior=[0.5, 0.5, 0.0]) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="absent"):
        s.compute("recall", "macro", prior=[0.4, 0.4, 0.2])


def test_validation_errors():
    with pytest.raises(ValueError):
        TemporalState(2, alpha=0.9, window=5)
    with pytest.raises(ValueError):
        TemporalState(2, alpha=1.0)
    with pytest.raises(ValueError):
        TemporalState(3, task="binary")
    s = TemporalState(2)
    with pytest.raises(ValueError):
        s.update(2, 0)
    with pytest.raises(ValueError):
        s.compute("recall")                                  # empty state
    with pytest.raises(ValueError):
        s.update([0.5, 0.6], 0)


# -- class-time state --------------------------------------------------------------

def test_class_time_rates_do_not_depend_on_the_prior():
    r = np.random.default_rng(6)

    def run(priors):
        ct, tt = ClassTimeState(2, alpha=0.98), TemporalState(2, alpha=0.98)
        for pi in priors:
            y = int(r.random() < pi)
            p = int(r.random() < (0.9 if y else 0.15))          # recall 0.9, FPR 0.15
            ct.update(y, p)
            tt.update(y, p)
        return ct, tt

    ct, tt = run([0.5] * 2000 + [0.05] * 400)                    # prior falls at the end
    rec_ct, spec_ct = ct.canonical_coordinates()
    assert rec_ct == pytest.approx(0.9, abs=0.12) and spec_ct == pytest.approx(0.85, abs=0.12)
    # the class-time recall uses only positives: with few of them it is noisy, but
    # the effective number of arrivals is capped at (1+alpha)/(1-alpha)
    assert (ct.n_ess() <= (1.98 / 0.02) + 1e-9).all()


def test_class_time_keeps_rates_of_a_class_that_stops_arriving():
    ct = ClassTimeState(2, alpha=0.9)
    for _ in range(50):
        ct.update(1, 1)
    before = ct.canonical_coordinates()[0]
    for _ in range(500):
        ct.update(0, 0)
    assert ct.canonical_coordinates()[0] == before == 1.0
    # time-fading forgets it
    tt = TemporalState(2, alpha=0.9, tol=0.0)
    for _ in range(50):
        tt.update(1, 1)
    for _ in range(500):
        tt.update(0, 0)
    assert tt.C[1].sum() < 1e-20


def test_class_time_compute_at_a_prior_and_unseen_class():
    ct = ClassTimeState(2, alpha=0.95)
    for _ in range(100):
        ct.update(1, 1)
        ct.update(0, 0)
    assert ct.compute("f1", prior=0.3) == pytest.approx(1.0)
    ct2 = ClassTimeState(2, alpha=0.95)
    ct2.update(0, 0)
    with pytest.raises(ValueError, match="never seen"):
        ct2.compute("f1", prior=0.3)


def test_class_time_window_variant():
    ct = ClassTimeState(2, window=3)
    for p in (1, 1, 0, 0, 0):
        ct.update(1, p)
    assert ct.rates()[1, 1] == 0.0
    assert ct.n_ess().tolist() == [0.0, 3.0]


# -- delayed labels ----------------------------------------------------------------

def standin(k):
    s = [1 / 8] * 3
    s[P[k]] = 3 / 4
    return s


def replay_running_example(t_end, **kw):
    ev = DelayedEvaluator(3, **kw)
    ids = {}
    for t in range(1, t_end + 1):
        ids[t - 1] = ev.arrive(standin(t - 1), P[t - 1], eligible=bool(AUDITED[t - 1]))
        for k in range(t):
            if AUDITED[k] and k + 1 + DELAY[k] == t:
                ev.label(ids[k], Y[k])
    return ev


def test_running_example_with_latency():
    ev = replay_running_example(8, window=4)
    # verified part: the audited instances of the window whose label arrived by t = 8
    ver_expected = np.zeros((3, 3))
    for k in range(4, 8):
        if AUDITED[k] and k + 1 + DELAY[k] <= 8:
            ver_expected[Y[k], P[k]] += 1
    np.testing.assert_allclose(ev.C_verified, ver_expected)
    u_ver, u_int = ev.support()
    assert u_ver.tolist() == [False, True, False]
    assert u_int.all()                                        # the stand-ins spread mass
    kappa = ev.coverage()
    assert kappa[1] == pytest.approx(0.5) and kappa[0] == 0 and kappa[2] == 0
    # column sums are exact: they do not depend on the stand-ins
    oracle = TemporalState(3, window=4)
    for y, p in zip(Y, P):
        oracle.update(y, p)
    np.testing.assert_allclose(ev.C.sum(axis=0), oracle.C.sum(axis=0))
    # macro recall under the verified support
    assert ev.compute("recall", "macro", support="verified") == pytest.approx(7 / 8)
    assert ev.compute("recall", "macro", support="verified", convention="zero") == pytest.approx(7 / 24)
    assert np.isnan(ev.compute("recall", "macro", support="verified", convention="nan"))
    # auto: recall is label side (verified); precision is prediction side (interim)
    assert ev.compute("recall", "macro") == pytest.approx(7 / 8)
    _, a_prec = ev.compute("precision", "macro", return_audit=True)
    assert a_prec.m_active == 3                               # all three classes were predicted


def test_no_latency_equals_the_temporal_state():
    y, p = stream(150, 3, seed=7)
    for kw in (dict(alpha=0.95), dict(window=40)):
        ev = DelayedEvaluator(3, **kw)
        ts = TemporalState(3, **kw)
        for a, b in zip(y, p):
            k = ev.arrive(np.full(3, 1 / 3), b)
            ev.label(k, a)
            ts.update(a, b)
        np.testing.assert_allclose(ev.C, ts.C, atol=1e-10)
        np.testing.assert_allclose(ev.C_verified, ts.C, atol=1e-10)
        assert ev.compute("f1", "macro") == pytest.approx(ts.compute("f1", "macro"))


def test_fading_correction_weight_and_eviction():
    a = 0.9
    y, p = stream(80, 2, seed=8)
    stand = np.array([0.5, 0.5])
    delays = np.random.default_rng(9).integers(0, 30, 80)
    ev = DelayedEvaluator(2, alpha=a, eps=1e-3)
    pending = []
    verified = {}
    for t in range(80):
        k = ev.arrive(stand, p[t])
        pending.append((t + delays[t], k, y[t]))
        for item in [q for q in pending if q[0] == t]:
            if ev.label(item[1], item[2]):
                verified[item[1]] = item[2]
    for item in pending:                                      # late labels after the end
        if item[0] >= 80 and ev.label(item[1], item[2]):
            verified[item[1]] = item[2]
    t_now = ev.t
    C = np.zeros((2, 2))
    for k in range(t_now):
        w = a ** (t_now - 1 - k)
        if k in verified:
            C[verified[k], p[k]] += w
        else:
            C += w * np.outer(stand, np.eye(2)[p[k]])
    np.testing.assert_allclose(ev.C, C, atol=1e-10)
    assert ev.queue_size <= sizing.queue_bound(a, 1e-3)


def test_eviction_drops_late_labels_and_keeps_the_term():
    a, eps = 0.5, 1 / 32
    ev = DelayedEvaluator(2, alpha=a, eps=eps)
    k0 = ev.arrive([0.5, 0.5], 1)
    for _ in range(sizing.queue_bound(a, eps)):
        ev.arrive([0.5, 0.5], 0)
    assert ev.n_evicted >= 1 and k0 not in ev.queue
    before = ev.C
    assert ev.label(k0, 1) is False
    np.testing.assert_array_equal(ev.C, before)


def test_window_correction_and_leaving():
    ev = DelayedEvaluator(2, window=3)
    k0 = ev.arrive([0.5, 0.5], 1)
    k1 = ev.arrive([0.5, 0.5], 0)
    assert ev.label(k0, 1)
    np.testing.assert_allclose(ev.C, [[0.5, 0.0], [0.5, 1.0]])
    ev.arrive([0.5, 0.5], 0)
    ev.arrive([0.5, 0.5], 0)                                 # k0 leaves the window
    assert k0 not in ev.queue
    assert ev.C_verified.sum() == 0
    assert ev.label(k1, 0) is True                           # k1 is still in the window
    with pytest.raises(ValueError):
        ev.label(99, 0)


def test_delayed_evaluator_validation():
    with pytest.raises(ValueError):
        DelayedEvaluator(2)
    with pytest.raises(ValueError):
        DelayedEvaluator(2, window=5, eps=0.1)


# -- sizing ---------------------------------------------------------------------------

def test_eviction_age_of_the_running_example():
    assert sizing.eviction_age(0.5, 1 / 32) == 5
    assert sizing.eviction_age(0.5, 1 / 32, strict=False) == 4


@pytest.mark.parametrize("alpha,eps", [(0.9, 1e-3), (0.99, 1e-4), (0.5, 0.2), (0.8, 0.5)])
def test_queue_and_eviction_bounds(alpha, eps):
    s = sizing.eviction_age(alpha, eps)
    assert (1 - alpha) * alpha ** s < eps
    assert s == 0 or (1 - alpha) * alpha ** (s - 1) >= eps
    n_eff = sizing.memory_horizon(alpha)
    assert sizing.eviction_error_bound(alpha, eps) < eps * n_eff + 1e-15
    if n_eff * eps < 1:
        assert s <= n_eff * np.log(1 / (n_eff * eps)) + 2


def test_undefined_window_probability_matches_the_paper():
    assert sizing.undefined_window_probability(0.25, 4) == pytest.approx(0.75 ** 4)
    assert sizing.undefined_window_probability(0.25, 4, delay=2) == pytest.approx(0.75 ** 2)
    assert sizing.undefined_window_probability(0.25, 4, delay=4) == 1.0
    assert sizing.undefined_window_probability(0.05, 60, delay=25, audit=0.5) \
        == pytest.approx((1 - 0.05 * 0.5) ** 35)


def test_undefined_window_probability_against_simulation():
    r = np.random.default_rng(10)
    p, W, D, b, T = 0.05, 30, 10, 0.7, 60000
    y = r.random(T) < p
    aud = r.random(T) < b
    ok = y & aud
    empty = []
    for t in range(W + D, T):
        idx = np.arange(t - W + 1, t + 1)
        verified = ok[idx] & (idx + D <= t)
        empty.append(not verified.any())
    assert np.mean(empty) == pytest.approx(
        sizing.undefined_window_probability(p, W, delay=D, audit=b), abs=0.02)


def test_undefined_window_probability_with_a_cdf():
    def F_(s):                                                # uniform delay on 0..10
        return min(1.0, s / 10)

    prob = sizing.undefined_window_probability(0.1, 20, delay=F_)
    assert prob == pytest.approx(np.prod([1 - 0.1 * F_(s) for s in range(20)]))


def test_required_window():
    for p, delta, D, b in [(0.05, 0.01, 0, 1.0), (0.05, 0.01, 25, 1.0), (0.1, 1e-3, 5, 0.5)]:
        W = sizing.required_window(p, delta, delay=D, audit=b)
        assert sizing.undefined_window_probability(p, W, delay=D, audit=b) <= delta
        assert sizing.undefined_window_probability(p, W - 1, delay=D, audit=b) > delta
    # grows by the delay
    assert sizing.required_window(0.05, 0.01, delay=25) == sizing.required_window(0.05, 0.01) + 25


def test_fading_bounds_enclose_the_probability():
    p, a, eps = 0.05, 0.9, 1e-3
    lo, hi = sizing.undefined_fading_bounds(p, a, eps)
    assert 0 < lo <= hi < 1
    r = np.random.default_rng(11)
    y = (r.random(200000) < p).astype(float)
    m = 0.0
    hit = []
    for t, v in enumerate(y):
        m = a * m + (1 - a) * v
        if t > 500:
            hit.append(m <= eps)
    assert lo - 0.01 <= np.mean(hit) <= hi + 0.01


def test_fading_variance_and_kappa():
    a, v = 0.95, 0.21
    t = 50
    direct = v * (1 - a) ** 2 * sum(a ** (2 * s) for s in range(t))
    assert sizing.fading_variance(a, v, t) == pytest.approx(direct)
    assert sizing.fading_variance(a, v) == pytest.approx(v * (1 - a) / (1 + a))
    assert sizing.effective_sample_size(a, 4.0) == pytest.approx(((1 + a) / (1 - a)) / 4.0)
    rho = 0.6
    acf = rho ** np.arange(0, 400)
    assert sizing.variance_inflation(a, acf) == pytest.approx(sizing.kappa_ar1(a, rho), rel=1e-9)
    assert sizing.variance_inflation(a, [1.0, 0.0, 0.0]) == 1.0
    assert sizing.half_life(a) == pytest.approx(np.log(2) / np.log(1 / a))


def test_estimate_kappa_on_ar1():
    r = np.random.default_rng(12)
    n, rho, a = 40000, 0.7, 0.98
    e = r.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
    assert sizing.estimate_kappa(x, a) == pytest.approx(sizing.kappa_ar1(a, rho), rel=0.2)
    assert sizing.estimate_kappa(r.standard_normal(n), a) < 1.5
    assert sizing.estimate_kappa(np.ones(100), a) == 1.0


def test_variance_of_the_faded_mean_with_dependence():
    r = np.random.default_rng(13)
    a, rho, n, reps = 0.9, 0.5, 600, 3000
    ests = []
    for _ in range(reps):
        e = r.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
        w = (1 - a) * a ** np.arange(n)[::-1]
        ests.append(float(w @ x))
    assert np.var(ests) == pytest.approx(
        sizing.fading_variance(a, 1.0, kappa=sizing.kappa_ar1(a, rho)), rel=0.15)


# -- attribution -----------------------------------------------------------------------

CONF = np.array([[3, 1, 0], [0, 3, 1], [1, 0, 3]]) / 4          # cyclic confusion rows


def population(pi, N=1600):
    return N * np.asarray(pi)[:, None] * CONF


def test_attribution_of_the_running_example_is_all_prior():
    ref = Snapshot.from_confusion(population([1 / 2, 1 / 4, 1 / 4]), "multiclass")
    cur = Snapshot.from_confusion(population([1 / 2, 1 / 8, 3 / 8]), "multiclass")
    assert ref.compute("f1", "micro") != 0
    f_ref = ref.model.with_priors(ref.prior)
    # class A one-vs-rest F1: 4/5 -> 24/31
    tp, fp, fn, tn = f_ref.cells(f_ref.theta)
    assert 2 * tp[0] / (2 * tp[0] + fp[0] + fn[0]) == pytest.approx(4 / 5)
    for method in ("counterfactual", "shapley"):
        at = attribute("f1", ref, cur, cls=0, method=method)
        assert at.total == pytest.approx(F(24, 31) - F(4, 5))
        assert at.total == pytest.approx(-4 / 155)
        assert at.concept == pytest.approx(0.0, abs=1e-12)
        assert at.prior == pytest.approx(-4 / 155)
        assert at.triage() == "prior"
    aj = attribute("f1", ref, cur, cls=0, method="jacobian", steps=10)
    assert aj.concept == pytest.approx(0.0, abs=1e-9)
    assert aj.prior == pytest.approx(-4 / 155, abs=1e-4)


def binary_pop(recall, spec, prev, N=10000):
    return np.array([[N * (1 - prev) * spec, N * (1 - prev) * (1 - spec)],
                     [N * prev * (1 - recall), N * prev * recall]])


def test_pure_concept_and_pure_prior_changes():
    ref = Snapshot.from_confusion(binary_pop(0.8, 0.9, 0.2), "binary")
    concept = Snapshot.from_confusion(binary_pop(0.6, 0.9, 0.2), "binary")
    prior = Snapshot.from_confusion(binary_pop(0.8, 0.9, 0.05), "binary")
    a = attribute("f1", ref, concept)
    assert a.prior == pytest.approx(0.0, abs=1e-12) and a.triage() == "concept"
    b = attribute("f1", ref, prior)
    assert b.concept == pytest.approx(0.0, abs=1e-12) and b.triage() == "prior"
    # skew-invariant measure: a prior change attributes nothing
    c = attribute("recall", ref, prior)
    assert c.total == pytest.approx(0.0, abs=1e-12) and c.triage() == "none"


def test_methods_agree_on_a_mixed_change_and_split_the_total():
    ref = Snapshot.from_confusion(binary_pop(0.8, 0.9, 0.2), "binary")
    cur = Snapshot.from_confusion(binary_pop(0.7, 0.85, 0.1), "binary")
    cf = attribute("f1", ref, cur)
    sh = attribute("f1", ref, cur, method="shapley")
    jc = attribute("f1", ref, cur, method="jacobian", steps=50)
    for a in (cf, sh, jc):
        assert a.total == cf.total
        assert a.total == pytest.approx(a.concept + a.prior + a.remainder)
    assert sh.remainder == pytest.approx(0.0, abs=1e-12)
    assert cf.concept != sh.concept or cf.remainder == pytest.approx(0.0, abs=1e-12)
    assert jc.concept == pytest.approx(sh.concept, abs=3e-3)
    assert jc.prior == pytest.approx(sh.prior, abs=3e-3)
    assert abs(jc.remainder) < 1e-3


def test_accuracy_is_bilinear_and_the_midpoint_is_exact():
    ref = Snapshot.from_confusion(binary_pop(0.8, 0.9, 0.2), "binary")
    cur = Snapshot.from_confusion(binary_pop(0.6, 0.7, 0.4), "binary")
    a = attribute("label_accuracy", ref, cur, method="jacobian", steps=1)
    assert abs(a.remainder) < 1e-6


def test_attribution_validation():
    b = Snapshot.from_confusion(binary_pop(0.8, 0.9, 0.2), "binary")
    m = Snapshot.from_confusion(population([0.5, 0.25, 0.25]), "multiclass")
    with pytest.raises(ValueError):
        attribute("f1", b, m)
    with pytest.raises(ValueError):
        attribute("f1", b, b, method="nope")
    with pytest.raises(ValueError):
        attribute("f1", b, b, cls=0)


def test_state_snapshots_feed_the_attribution():
    r = np.random.default_rng(14)

    def fill(prev, recall):
        s = TemporalState(2, window=4000)
        for _ in range(4000):
            y = int(r.random() < prev)
            p = int(r.random() < (recall if y else 0.1))
            s.update(y, p)
        return s.snapshot()

    ref, cur = fill(0.3, 0.8), fill(0.1, 0.8)
    at = attribute("f1", ref, cur)
    assert at.triage() == "prior"
    ref2, cur2 = fill(0.3, 0.8), fill(0.3, 0.5)
    assert attribute("f1", ref2, cur2).triage() == "concept"


# -- monitor -----------------------------------------------------------------------------

def null_run_factory(prior=0.3, tpr=0.8, tnr=0.9, n=1500, alpha=0.95, warmup=300,
                     shift=None, change=None):
    def run(seed):
        r = np.random.default_rng(seed)
        mon = RateMonitor(2, alpha=alpha, warmup=warmup)
        for t in range(n):
            pi = prior if (shift is None or t < n // 2) else shift
            tp_ = tpr if (change is None or t < n // 2) else change
            y = int(r.random() < pi)
            p = int(r.random() < (tp_ if y else 1 - tnr))
            mon.update(y, p)
        return mon.max_statistic
    return run


def test_calibrated_monitor_controls_the_run_level_false_alarm():
    thr = calibrate_threshold(null_run_factory(), n_runs=150, seed=1000)
    fa = np.mean([null_run_factory()(5000 + i) > thr for i in range(150)])
    assert fa <= 0.14


def test_label_shift_does_not_trigger_but_a_recall_change_does():
    thr = calibrate_threshold(null_run_factory(), n_runs=150, seed=1000)
    shifted = np.mean([null_run_factory(shift=0.1)(7000 + i) > thr for i in range(100)])
    changed = np.mean([null_run_factory(change=0.4)(9000 + i) > thr for i in range(100)])
    assert shifted <= 0.16
    assert changed >= 0.9


def test_monitor_is_undefined_until_the_reference_is_frozen():
    mon = RateMonitor(2, alpha=0.9, warmup=50)
    for t in range(49):
        assert np.isnan(mon.update(t % 2, t % 2))
    assert not mon.frozen
    mon.update(0, 0)
    assert mon.frozen
    assert np.isnan(mon.statistic)


def test_multiclass_monitor_runs_and_detects():
    r = np.random.default_rng(15)
    mon = RateMonitor(3, alpha=0.95, warmup=300)
    for t in range(1500):
        y = int(r.integers(0, 3))
        acc = 0.85 if t < 750 else 0.3
        p = y if r.random() < acc else int((y + 1 + r.integers(0, 2)) % 3)
        mon.update(y, p)
    assert mon.max_statistic > 20


def test_calibrate_threshold_validation():
    with pytest.raises(ValueError):
        calibrate_threshold(lambda s: 1.0, n_runs=5)
    with pytest.raises(ValueError):
        calibrate_threshold(lambda s: float("nan"), n_runs=40)


# -- argument checks and corners -------------------------------------------------------

def test_rates_from_confusion_validation():
    from aggmetrics._rates import Rates
    with pytest.raises(ValueError, match="square"):
        Rates.from_confusion(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        Rates.from_confusion(-np.ones((2, 2)))
    with pytest.raises(ValueError, match="2 x 2"):
        Rates.from_confusion(np.ones((3, 3)), "binary")
    with pytest.raises(ValueError, match="task"):
        Rates.from_confusion(np.ones((2, 2)), "multilabel")
    r = Rates.from_confusion(np.array([[0.0, 0.0], [1.0, 3.0]]), "binary")
    assert r.theta.tolist() == [0.75, 0.0]                  # no negatives: placeholder


def test_state_validation_and_empty_states():
    with pytest.raises(ValueError, match="task"):
        TemporalState(2, task="multilabel")
    with pytest.raises(ValueError):
        TemporalState(2, window=0)
    with pytest.raises(ValueError):
        TemporalState(1)
    s = TemporalState(2)
    assert s.normalized.sum() == 0
    with pytest.raises(ValueError, match="empty"):
        s.snapshot()
    with pytest.raises(ValueError, match="aggregation"):
        s.update(1, 1)
        s.compute("recall", "exemplar")
    with pytest.raises(ValueError):
        ClassTimeState(2)
    with pytest.raises(ValueError):
        ClassTimeState(1, alpha=0.9)
    ct = ClassTimeState(3, alpha=0.9)
    with pytest.raises(ValueError):
        ct.update(5, 0)
    with pytest.raises(ValueError, match="prevalence"):
        ClassTimeState(2, alpha=0.9).snapshot(1.5)
    with pytest.raises(ValueError, match="sum to 1"):
        ct.snapshot([0.5, 0.2, 0.2])


def test_canonical_coordinates_of_the_states():
    s = TemporalState(2)
    for y, p in [(1, 1), (1, 0), (0, 0)]:
        s.update(y, p)
    assert s.canonical_coordinates().tolist() == [0.5, 1.0]
    only_neg = TemporalState(2)
    only_neg.update(0, 0)
    assert np.isnan(only_neg.canonical_coordinates()[0])
    m = TemporalState(3)
    for y, p in [(0, 0), (0, 1), (1, 1)]:
        m.update(y, p)
    c = m.canonical_coordinates()
    assert c.shape == (6,) and c[0] == 0.5 and np.isnan(c[4])      # class 2 unseen
    assert m.snapshot().rates.shape == (6,)
    ct = ClassTimeState(3, alpha=0.9)
    for y, p in [(0, 0), (0, 1), (1, 1), (2, 2)]:
        ct.update(y, p)
    assert ct.canonical_coordinates().shape == (6,)
    assert ct.C.shape == (3, 3) and ct.arrivals.tolist() == [2, 1, 1]
    assert s.audit("recall", "macro").m_active == 1


def test_state_snapshot_compute_default_prior():
    snap = Snapshot.from_confusion(binary_pop(0.8, 0.9, 0.2), "binary")
    assert snap.compute("recall") == pytest.approx(0.8)
    assert snap.compute("precision", prior=0.5) > snap.compute("precision")


def test_delayed_evaluator_corners():
    ev = DelayedEvaluator(2, window=2)
    with pytest.raises(ValueError, match="empty"):
        ev.compute("recall")
    with pytest.raises(ValueError, match="empty"):
        ev.snapshot()
    ev.arrive([0.5, 0.5], 1)
    ev.arrive([0.5, 0.5], 0)
    assert ev.mass == 2
    assert ev.canonical_coordinates().shape == (2,)
    assert ev.snapshot().task == "binary"
    with pytest.raises(ValueError, match="support"):
        ev.compute("recall", support="both")
    ev.arrive([0.5, 0.5], 0)                                  # first instance leaves unverified
    assert ev.C_verified.sum() == 0
    with pytest.raises(ValueError):
        DelayedEvaluator(2, alpha=0.9, eps=-1)
    with pytest.raises(ValueError):
        DelayedEvaluator(1, alpha=0.9)
    # an eviction that empties the queue
    ev2 = DelayedEvaluator(2, alpha=0.5, eps=0.9)
    ev2.arrive([0.5, 0.5], 0)
    assert ev2.queue_size == 0 and ev2.n_evicted == 1


def test_support_side_of_a_measure():
    from aggmetrics import get_measure
    from aggmetrics.stream._confusion import denominator_side, evaluate
    assert denominator_side(get_measure("precision")) == "prediction"
    assert denominator_side(get_measure("accuracy" if False else "label_accuracy")) == "prediction"
    assert denominator_side(get_measure("recall")) == "label"
    assert denominator_side(get_measure("mcc")) == "label"
    with pytest.raises(ValueError, match="aggregation"):
        evaluate(np.eye(2), "binary", "recall", "exemplar")


def test_audit_of_a_state_without_verified_instances():
    ev = DelayedEvaluator(2, window=5)
    for p in (1, 0, 1):
        ev.arrive([0.4, 0.6], p)
    assert np.isnan(ev.compute("recall"))                     # nothing verified yet
    assert ev.compute("precision") == pytest.approx(0.6)       # prediction side is exact


def test_sizing_argument_checks():
    for call in (
        lambda: sizing.memory_horizon(1.0),
        lambda: sizing.effective_sample_size(0.9, 0.0),
        lambda: sizing.fading_variance(0.9, 1.0, t=5, kappa=2.0),
        lambda: sizing.fading_variance(0.9, 1.0, t=0),
        lambda: sizing.eviction_age(0.9, 0.0),
        lambda: sizing.undefined_window_probability(0.0, 5),
        lambda: sizing.undefined_window_probability(0.1, 0),
        lambda: sizing.undefined_window_probability(0.1, 5, audit=1.5),
        lambda: sizing.undefined_window_probability(0.1, 5, delay=-1),
        lambda: sizing.required_window(0.1, 1.0),
        lambda: sizing.required_window(0.1, 0.1, audit=0.0),
        lambda: sizing.required_window(0.1, 0.1, delay=-2),
        lambda: sizing.kappa_ar1(0.9, 1.0),
        lambda: sizing.variance_inflation(0.9, [1.0]),
        lambda: sizing.estimate_kappa([1, 0, 1], 0.9),
    ):
        with pytest.raises(ValueError):
            call()
    assert sizing.required_window(1.0, 0.1, delay=3) == 4
    assert sizing.fading_variance(0.9, 1.0, t=1) == pytest.approx(0.1 / 1.9 * (1 - 0.81))
    r = np.random.default_rng(16)
    x = np.cumsum(r.standard_normal(5000)) * 0.01 + r.standard_normal(5000)
    assert sizing.estimate_kappa(x, 0.95, detrend=201) >= 1.0


def test_attribution_corners():
    ref = Snapshot.from_confusion(binary_pop(0.8, 0.9, 0.2), "binary")
    cur = Snapshot.from_confusion(binary_pop(0.6, 0.9, 0.2), "binary")
    a = attribute("f1", ref, cur, method="jacobian")          # no prior change
    assert a.prior == 0.0 and a.concept < 0
    with pytest.raises(ValueError, match="steps"):
        attribute("f1", ref, cur, method="jacobian", steps=0)
    nan_total = attribute("precision", Snapshot.from_confusion(np.array([[5.0, 0], [5.0, 0]]), "binary"), ref)
    assert nan_total.triage() == "none"


def test_monitor_corners():
    with pytest.raises(ValueError):
        RateMonitor(2, warmup=0)
    with pytest.raises(ValueError):
        calibrate_threshold(lambda s: 1.0, n_runs=30, target=1.5)
    # a class that never arrives in the warm-up leaves the reference undefined
    mon = RateMonitor(2, alpha=0.9, warmup=10)
    for _ in range(30):
        assert np.isnan(mon.update(0, 0))
    assert not mon.frozen
    # a perfect classifier has zero variance: the statistic is undefined, not a crash
    perfect = RateMonitor(2, alpha=0.9, warmup=20)
    for t in range(60):
        perfect.update(t % 2, t % 2)
    assert perfect.frozen and np.isnan(perfect.statistic)


def test_multiclass_class_time_compute_at_a_prior():
    ct = ClassTimeState(3, alpha=0.95)
    for _ in range(50):
        for c in range(3):
            ct.update(c, c)
    assert ct.compute("f1", "macro", prior=[0.5, 0.3, 0.2]) == pytest.approx(1.0)
    snap = ct.snapshot([0.5, 0.3, 0.2])
    assert snap.prior.tolist() == [0.5, 0.3, 0.2]
