# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
The paper's results, as executable tests.

Each test below corresponds to a stated result. Running the suite therefore
re-verifies the theory against the implementation, which is the main reason the
package and the paper are worth releasing together: a change that breaks a
theorem breaks a test.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggmetrics import (
    IndicatorMatrix,
    get_measure,
    list_measures,
    marginal_defect,
    micro_macro_gap,
)
from aggmetrics.diagnostics import check_hull

rng = np.random.default_rng(20260728)


def random_multilabel(n=120, m=7, p=0.35, seed=0):
    r = np.random.default_rng(seed)
    Y = (r.random((n, m)) < p).astype(int)
    P = Y.copy()
    flip = r.random((n, m)) < 0.25
    P[flip] = 1 - P[flip]
    return Y, P


def random_multiclass(n=200, m=5, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, m, n)
    yhat = y.copy()
    idx = r.random(n) < 0.4
    yhat[idx] = r.integers(0, m, int(idx.sum()))
    return y, yhat


# ---------------------------------------------------------------------------
# Theorem 1: micro equals denominator-weighted macro, for decomposable measures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list_measures(decomposable=True))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_theorem1_holds_for_decomposable(name, seed):
    Y, P = random_multilabel(seed=seed)
    im = IndicatorMatrix(Y, P, task="multilabel")
    meas = get_measure(name)

    micro = im.compute(meas, "micro")
    pc = im.per_class()
    num = np.array([meas.numerator(tuple(r)) for r in pc])
    den = np.array([meas.denominator(tuple(r)) for r in pc])
    if den.sum() <= 0:
        pytest.skip("degenerate sample")

    # micro must equal the denominator-weighted average of the per-class values
    assert micro == pytest.approx(num.sum() / den.sum(), abs=1e-10)

    ok = den > 0
    weighted = np.average((num[ok] / den[ok]), weights=den[ok])
    assert micro == pytest.approx(weighted, abs=1e-10)


@pytest.mark.parametrize("name", ["balanced_accuracy", "gmean", "informedness",
                                  "mcc", "kappa"])
def test_theorem1_fails_for_non_decomposable(name):
    """No weighting reproduces the pooled value for these measures.

    Uses the confusion matrix from the paper's counterexample, for which the
    pooled value provably lies outside the per-class range.
    """
    CM = np.array([[110, 10, 10], [20, 40, 5], [25, 5, 50]])
    y, yhat = [], []
    for j in range(3):
        for k in range(3):
            y += [j] * CM[j, k]
            yhat += [k] * CM[j, k]
    im = IndicatorMatrix(np.array(y), np.array(yhat), task="multiclass")

    meas = get_measure(name)
    assert not meas.decomposable
    with pytest.raises(ValueError):
        meas.denominator((1, 1, 1, 1))

    if name in ("balanced_accuracy", "informedness", "mcc"):
        finding = check_hull(im, name)
        assert finding is not None, f"{name} should escape the per-class hull here"
        assert finding.code == "outside_hull"


def test_counterexample_matches_published_values():
    CM = np.array([[110, 10, 10], [20, 40, 5], [25, 5, 50]])
    y, yhat = [], []
    for j in range(3):
        for k in range(3):
            y += [j] * CM[j, k]
            yhat += [k] * CM[j, k]
    im = IndicatorMatrix(np.array(y), np.array(yhat), task="multiclass")

    assert im.n == 275
    assert im.compute("recall", "micro") == pytest.approx(0.7273, abs=5e-4)
    assert im.compute("specificity", "micro") == pytest.approx(0.8636, abs=5e-4)
    assert im.compute("balanced_accuracy", "macro") == pytest.approx(0.7713, abs=5e-4)
    assert im.compute("balanced_accuracy", "micro") == pytest.approx(0.7955, abs=5e-4)
    assert im.compute("gmean", "micro") == pytest.approx(0.7925, abs=5e-4)
    assert im.compute("mcc", "micro") == pytest.approx(0.5909, abs=5e-4)
    # the pooled value exceeds every per-class value
    assert im.compute("balanced_accuracy", "micro") > im.parts("balanced_accuracy").max()


# ---------------------------------------------------------------------------
# Theorem 3 (Section 7.3): the product t-norm is the unique marginally consistent one
# ---------------------------------------------------------------------------

def test_theorem3_product_is_uniquely_marginally_consistent():
    assert marginal_defect("product") == pytest.approx(0.0, abs=1e-12)
    for other in ("godel", "lukasiewicz"):
        assert marginal_defect(other) > 0.1


def test_corollary3_partition_follows():
    """Marginal consistency implies the four cells sum to one entrywise."""
    from aggmetrics.tnorms import soft_cells
    r = np.random.default_rng(3)
    Y, P = r.random((40, 5)), r.random((40, 5))
    TP, FP, FN, TN = soft_cells(Y, P, "product")
    assert np.allclose(TP + FP + FN + TN, 1.0)
    # and the marginals are recovered
    assert np.allclose(TP + FN, Y)
    assert np.allclose(TP + FP, P)


# ---------------------------------------------------------------------------
# Theorem 2 (Section 7.2): micro-collapse on single-label data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [2, 3, 5, 10])
def test_theorem2_micro_collapse(m):
    y, yhat = random_multiclass(n=300, m=m, seed=m)
    im = IndicatorMatrix(y, yhat, task="multiclass")
    acc = im.accuracy()
    for name in ("precision", "recall", "f1"):
        assert im.compute(name, "micro") == pytest.approx(acc, abs=1e-12)


def test_micro_collapse_does_not_hold_in_multilabel():
    Y, P = random_multilabel(seed=9)
    im = IndicatorMatrix(Y, P, task="multilabel")
    vals = {n: im.compute(n, "micro") for n in ("precision", "recall", "f1")}
    assert max(vals.values()) - min(vals.values()) > 1e-6


# ---------------------------------------------------------------------------
# Proposition 9 (Section 7.9): Bayes threshold and ERM-consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("label_accuracy", 0.5),
    ("error_rate", 0.5),
    ("recall", 0.0),
    ("specificity", 1.0),
])
def test_proposition9_thresholds(name, expected):
    assert get_measure(name).bayes_threshold() == pytest.approx(expected, abs=1e-12)


def test_proposition9_erm_consistency_flags():
    assert get_measure("label_accuracy").erm_consistent() is True
    assert get_measure("error_rate").erm_consistent() is True
    assert get_measure("recall").erm_consistent() is False
    assert get_measure("specificity").erm_consistent() is False
    # prediction-dependent denominators have no closed form
    assert get_measure("precision").bayes_threshold() is None
    assert get_measure("f1").bayes_threshold() is None


def test_proposition9_threshold_is_the_empirical_optimum():
    """The predicted threshold maximises the population objective."""
    r = np.random.default_rng(11)
    eta = r.random(4000)
    grid = np.linspace(0.0, 1.0, 1001)

    for name, (a, b, g, d) in [("label_accuracy", (1, 0, 0, 1)),
                               ("recall", (1, 0, 0, 0)),
                               ("specificity", (0, 0, 0, 1))]:
        vals = []
        for t in grid:
            dec = (eta > t).astype(float)
            tp = (dec * eta).sum(); fp = (dec * (1 - eta)).sum()
            fn = ((1 - dec) * eta).sum(); tn = ((1 - dec) * (1 - eta)).sum()
            vals.append(a * tp - b * fp - g * fn + d * tn)
        vals = np.asarray(vals)
        best = grid[vals >= vals.max() - 1e-9]
        t_star = get_measure(name).bayes_threshold()
        assert best.min() - 1e-9 <= t_star <= best.max() + 1e-9


# ---------------------------------------------------------------------------
# Proposition 5 (Section 7.4): attainable floors under one-hot encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [3, 4, 5, 10])
def test_proposition5_floors(m):
    n = 200
    y = np.arange(n) % m
    yhat = (y + 1) % m                      # every prediction wrong
    im = IndicatorMatrix(y, yhat, task="multiclass")
    assert im.counts[0] == 0                # no true positives at all

    assert im.compute("label_accuracy", "micro") == pytest.approx((m - 2) / m, abs=1e-12)
    assert im.compute("specificity", "micro") == pytest.approx((m - 2) / (m - 1), abs=1e-12)
    assert im.compute("balanced_accuracy", "micro") == pytest.approx(
        (m - 2) / (2 * (m - 1)), abs=1e-12)
    assert im.compute("mcc", "micro") == pytest.approx(-1 / (m - 1), abs=1e-12)
    assert im.compute("kappa", "micro") == pytest.approx(-1 / (m - 1), abs=1e-12)

    # measures whose value at tp = 0 does not involve tn do reach zero
    for name in ("recall", "precision", "f1", "jaccard", "gmean"):
        assert im.compute(name, "micro") == pytest.approx(0.0, abs=1e-12)
    assert im.accuracy() == pytest.approx(0.0, abs=1e-12)


def test_gmean_is_complete_yet_reaches_zero():
    """The G-mean uses all four cells but its tn-dependence vanishes at tp = 0.

    This is why 'depends on tn' is the wrong criterion and 'depends on tn at
    tp = 0' is the right one.
    """
    g = get_measure("gmean")
    assert g.tn_dependent_at_zero is False
    for m in (3, 5, 10):
        assert g.attainable_floor(m) == pytest.approx(0.0, abs=1e-12)
    ba = get_measure("balanced_accuracy")
    assert ba.tn_dependent_at_zero is True
    assert ba.attainable_floor(10) > 0.4


# ---------------------------------------------------------------------------
# Proposition 6 (Section 7.6): one-vs-rest recall is prior-free, specificity is not
# ---------------------------------------------------------------------------

def _ovr_from_rates(pi, R, n=2_000_000):
    m = len(pi)
    out = []
    for j in range(m):
        tp = n * pi[j] * R[j, j]
        fn = n * pi[j] * (1 - R[j, j])
        fp = n * sum(pi[k] * R[k, j] for k in range(m) if k != j)
        tn = n * sum(pi[k] * (1 - R[k, j]) for k in range(m) if k != j)
        out.append((tp, fp, fn, tn))
    return out


def test_proposition6_recall_prior_free_specificity_not():
    R = np.array([[0.80, 0.15, 0.05],
                  [0.30, 0.60, 0.10],
                  [0.05, 0.25, 0.70]])
    rec, spec = get_measure("recall"), get_measure("specificity")
    priors = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.8, 0.1, 0.1]),
              np.array([0.1, 0.1, 0.8])]
    macro_rec = [np.mean([rec(*c) for c in _ovr_from_rates(p, R)]) for p in priors]
    macro_spec = [np.mean([spec(*c) for c in _ovr_from_rates(p, R)]) for p in priors]

    assert max(macro_rec) - min(macro_rec) < 1e-12      # exactly invariant
    assert max(macro_spec) - min(macro_spec) > 1e-3     # demonstrably not


def test_proposition6_uniform_error_condition_restores_invariance():
    """Specificity is prior-free iff R[k,j] does not depend on k."""
    c = np.array([0.10, 0.15, 0.05])
    m = 3
    R = np.zeros((m, m))
    for k in range(m):
        for j in range(m):
            R[k, j] = c[j] if k != j else 1 - sum(c[i] for i in range(m) if i != k)
    assert np.allclose(R.sum(axis=1), 1.0)

    spec = get_measure("specificity")
    vals = []
    for p in (np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.05, 0.9, 0.05])):
        per = _ovr_from_rates(p, R)
        vals.append([spec(*cc) for cc in per])
    assert np.allclose(vals[0], vals[1])
    assert np.allclose(vals[0], 1 - c)


# ---------------------------------------------------------------------------
# Corollary 6 (Section 7.7): the micro-macro gap as a normalised covariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["recall", "precision", "f1", "jaccard",
                                  "label_accuracy"])
def test_corollary6_gap_is_normalised_covariance(name):
    Y, P = random_multilabel(n=200, m=9, seed=5)
    im = IndicatorMatrix(Y, P, task="multilabel")
    d = micro_macro_gap(im, name)
    assert d["exact"] is True
    assert d["gap"] == pytest.approx(d["predicted_gap"], abs=1e-9)


def test_corollary6_not_exact_for_non_decomposable():
    Y, P = random_multilabel(seed=6)
    im = IndicatorMatrix(Y, P, task="multilabel")
    d = micro_macro_gap(im, "balanced_accuracy")
    assert d["exact"] is False
    assert "note" in d


# ---------------------------------------------------------------------------
# Theorem 4 (Section 7.3): ordinal losses as cost-sensitive classification
# ---------------------------------------------------------------------------

def test_theorem4_mae_and_mse_as_cost_matrices():
    m, n = 5, 400
    r = np.random.default_rng(7)
    y = r.integers(0, m, n)
    yhat = np.clip(y + r.integers(-2, 3, n), 0, m - 1)
    im = IndicatorMatrix(y, yhat, task="multiclass")

    J, K = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    mae = im.cost(np.abs(J - K)) / n
    mse = im.cost((J - K) ** 2.0) / n
    assert mae == pytest.approx(np.abs(y - yhat).mean(), abs=1e-12)
    assert mse == pytest.approx(((y - yhat) ** 2).mean(), abs=1e-12)


def test_zero_one_cost_is_half_the_error_cells():
    """A 0/1 cost matrix counts misclassified examples, which is half of FP+FN."""
    m, n = 6, 300
    r = np.random.default_rng(8)
    y = r.integers(0, m, n)
    yhat = r.integers(0, m, n)
    im = IndicatorMatrix(y, yhat, task="multiclass")
    C = 1.0 - np.eye(m)
    _, fp, fn, _ = im.counts
    assert im.cost(C) == pytest.approx(0.5 * (fp + fn), abs=1e-9)
    assert im.cost(C) == pytest.approx(float((y != yhat).sum()), abs=1e-9)


# ---------------------------------------------------------------------------
# Proposition 6 (Section 7.6): uniform-error condition and OvR skew drift
# ---------------------------------------------------------------------------

def test_uniform_error_deviation_vanishes_for_uniform_errors():
    """When off-diagonal errors R_{kj} = c_j are uniform, deviation is 0."""
    from aggmetrics.diagnostics import check_ovr_skew_drift, uniform_error_deviation
    m = 3
    # Rates R: row 0 = [0.80, 0.15, 0.05], row 1 = [0.10, 0.85, 0.05], row 2 = [0.10, 0.15, 0.75]
    # Off-diagonal into col 0: (0.10, 0.10); col 1: (0.15, 0.15); col 2: (0.05, 0.05) -> all stds = 0
    C = np.array([
        [80, 15,  5],
        [10, 85,  5],
        [10, 15, 75],
    ])
    # Build actual (y) and predicted (yhat) vectors from confusion matrix C
    y_list, yhat_list = [], []
    for true_c in range(m):
        for pred_c in range(m):
            count = C[true_c, pred_c]
            y_list.extend([true_c] * count)
            yhat_list.extend([pred_c] * count)
    im = IndicatorMatrix(np.array(y_list), np.array(yhat_list), task="multiclass")

    dev = uniform_error_deviation(im)
    assert dev == pytest.approx(0.0, abs=1e-9)
    assert check_ovr_skew_drift(im) is None


def test_uniform_error_deviation_detects_asymmetric_errors():
    """Asymmetric off-diagonal errors produce positive uniform-error deviation."""
    from aggmetrics import diagnose
    from aggmetrics.diagnostics import check_ovr_skew_drift, uniform_error_deviation
    m = 3
    # Asymmetric off-diagonal errors (Example 2 from paper):
    # R = [[0.80, 0.15, 0.05], [0.30, 0.60, 0.10], [0.05, 0.25, 0.70]]
    C = np.array([
        [80, 15,  5],
        [30, 60, 10],
        [ 5, 25, 70],
    ])
    y_list, yhat_list = [], []
    for true_c in range(m):
        for pred_c in range(m):
            count = C[true_c, pred_c]
            y_list.extend([true_c] * count)
            yhat_list.extend([pred_c] * count)
    im = IndicatorMatrix(np.array(y_list), np.array(yhat_list), task="multiclass")

    dev = uniform_error_deviation(im)
    assert dev > 0.05
    f = check_ovr_skew_drift(im, threshold=0.01)
    assert f is not None
    assert f.code == "ovr_skew_drift"
    assert "Delta_BA = 0.5 * Delta_spec" in f.message

    # Verify that diagnose finds it
    findings = diagnose(im)
    codes = {f.code for f in findings}
    assert "ovr_skew_drift" in codes


# ---------------------------------------------------------------------------
# Proposition 3 and Remark 2 (Section 7.2): degrees of freedom, orthogonal basis, non-redundant suites
# ---------------------------------------------------------------------------

def test_degrees_of_freedom_and_canonical_coordinates():
    """Degrees of freedom equal 2 for binary and m(m-1) for multiclass."""
    from aggmetrics.diagnostics import check_redundancy

    # Binary classification
    y_bin = np.array([0, 1, 1, 0, 1, 0])
    p_bin = np.array([0, 1, 0, 0, 1, 1])
    im_bin = IndicatorMatrix(y_bin, p_bin, task="binary")
    assert im_bin.degrees_of_freedom == 2
    coords_bin = im_bin.canonical_coordinates()
    assert coords_bin.shape == (2,)
    assert coords_bin[0] == pytest.approx(im_bin.compute("recall", "micro"))
    assert coords_bin[1] == pytest.approx(im_bin.compute("specificity", "micro"))

    # Redundancy check in binary
    suite_non_redundant = ["recall", "specificity"]
    assert check_redundancy(im_bin, suite_non_redundant) is None

    suite_redundant = ["accuracy", "balanced_accuracy", "f1", "recall", "specificity"]
    f = check_redundancy(im_bin, suite_redundant)
    assert f is not None
    assert f.code == "redundant_suite"
    assert "exceeds the 2 independent degrees of freedom" in f.message

    # Multiclass classification (m = 4 -> d = 4 * 3 = 12)
    m = 4
    y_mc = np.arange(100) % m
    p_mc = (y_mc + 1) % m
    im_mc = IndicatorMatrix(y_mc, p_mc, task="multiclass")
    assert im_mc.degrees_of_freedom == 12
    coords_mc = im_mc.canonical_coordinates()
    assert coords_mc.shape == (12,)


# ---------------------------------------------------------------------------
# Ordinal cumulative matrix representation (Eq. 18 and Section 6.4)
# ---------------------------------------------------------------------------

def test_ordinal_cumulative_representation():
    """Ordinal task creates m-1 cumulative threshold binary columns."""
    # 4 classes: 0, 1, 2, 3 -> 3 threshold indicators (y > 0, y > 1, y > 2)
    y = np.array([0, 1, 2, 3])
    p = np.array([0, 2, 2, 2])
    im_ord = IndicatorMatrix(y, p, task="ordinal")
    assert im_ord.m == 3
    assert im_ord.n == 4
    # Expected Y:
    # 0 -> [0, 0, 0]
    # 1 -> [1, 0, 0]
    # 2 -> [1, 1, 0]
    # 3 -> [1, 1, 1]
    expected_Y = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ], float)
    assert np.allclose(im_ord.Y, expected_Y)

    # Expected P for [0, 2, 2, 2]:
    # 0 -> [0, 0, 0]
    # 2 -> [1, 1, 0]
    # 2 -> [1, 1, 0]
    # 2 -> [1, 1, 0]
    expected_P = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
    ], float)
    assert np.allclose(im_ord.P, expected_P)

    # Compute macro-recall and macro-precision across ordinal thresholds
    rec = im_ord.compute("recall", "macro")
    assert 0.0 <= rec <= 1.0


# ---------------------------------------------------------------------------
# Section 7.11 (resting on Proposition 3): universal Pareto dominance across monotonic measures
# ---------------------------------------------------------------------------

def test_pareto_dominance_binary_and_multiclass():
    """Dominance holds componentwise in canonical coordinates."""
    from aggmetrics.core import pareto_dominance

    # Binary: classifier A with fewer errors than B
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p_a = np.array([0, 0, 1, 1, 1, 0, 1, 0])  # perfect
    p_b = np.array([0, 1, 1, 0, 1, 0, 1, 0])  # 2 errors
    im_a = IndicatorMatrix(y, p_a, task="binary")
    im_b = IndicatorMatrix(y, p_b, task="binary")

    assert im_a.dominates(im_b) is True
    assert im_b.dominates(im_a) is False
    assert pareto_dominance(im_a, im_b) == "A_dominates_B"
    assert pareto_dominance(im_b, im_a) == "B_dominates_A"
    assert pareto_dominance(im_a, im_a) == "equivalent"

    # Incomparable classifiers (one has higher recall, other has higher specificity)
    p_c = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # predicts all 1 (rec=1, spec=0)
    p_d = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # predicts all 0 (rec=0, spec=1)
    im_c = IndicatorMatrix(y, p_c, task="binary")
    im_d = IndicatorMatrix(y, p_d, task="binary")
    assert pareto_dominance(im_c, im_d) == "incomparable"


# ---------------------------------------------------------------------------
# Fold-wise reduction operator (Sigma_k) and Cross-Validation Aggregation
# ---------------------------------------------------------------------------

def test_fold_aggregation_pooled_vs_averaged():
    """Fold pooling (Sigma_k) vs fold averaging across CV splits."""
    from aggmetrics.aggregation import fold_aggregate

    # Create 3 synthetic folds
    rng = np.random.default_rng(42)
    folds = []
    for _ in range(3):
        y = rng.integers(0, 3, 100)
        p = (y + rng.choice([0, 1], size=100, p=[0.7, 0.3])) % 3
        folds.append(IndicatorMatrix(y, p, task="multiclass"))

    # Pooled CV (Sigma_k before non-linear ratio)
    mcc_pooled = fold_aggregate(folds, "mcc", how="macro", pooling="pooled")
    assert -1.0 <= mcc_pooled <= 1.0

    # Averaged CV (ratio in each fold then weighted average)
    mcc_averaged = fold_aggregate(folds, "mcc", how="macro", pooling="averaged")
    assert -1.0 <= mcc_averaged <= 1.0

    # For a decomposable linear measure like label_accuracy under equal fold sizes,
    # pooled and averaged agree
    acc_pooled = fold_aggregate(folds, "label_accuracy", how="micro", pooling="pooled")
    acc_averaged = fold_aggregate(folds, "label_accuracy", how="micro", pooling="averaged")
    assert acc_pooled == pytest.approx(acc_averaged, abs=1e-12)



