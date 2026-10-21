# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""Behaviour that the theory tests do not reach: soft labels, input forms,
ordinal encodings, cost matrices, dominance, fold pooling, undefined-value
conventions, diagnostics edge cases and input validation.

Each test checks a value against an independent computation (plain NumPy), not
merely that a code path runs.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from aggmetrics import (BinaryMeasure, IndicatorMatrix, diagnose, fold_aggregate,
                        get_measure, get_tnorm, infer_properties, list_measures,
                        marginal_defect, pareto_dominance, soft_cells)
from aggmetrics.aggregation import _combine, aggregate, cells, sigma_k
from aggmetrics.diagnostics import (check_hull, check_ovr_skew_drift,
                                    check_redundancy, micro_macro_gap,
                                    uniform_error_deviation)
from aggmetrics.tnorms import hamacher


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else np.nan


# ---------------------------------------------------------------------------
# Soft labels through IndicatorMatrix
# ---------------------------------------------------------------------------

@pytest.fixture
def soft():
    rng = np.random.default_rng(0)
    return rng.random((30, 4)), rng.random((30, 4))


def _soft_cells(Y, P):
    return Y * P, (1 - Y) * P, Y * (1 - P), (1 - Y) * (1 - P)


def test_soft_micro_matches_manual_product_cells(soft):
    Y, P = soft
    TP, FP, _, _ = _soft_cells(Y, P)
    im = IndicatorMatrix(Y, P, task="multilabel", tnorm="product")
    assert im.compute("precision", "micro") == pytest.approx(TP.sum() / (TP.sum() + FP.sum()))


@pytest.mark.parametrize("how", ["macro", "weighted", "exemplar"])
def test_soft_per_part_operators_match_manual(soft, how):
    Y, P = soft
    TP, FP, FN, _ = _soft_cells(Y, P)
    axis = 1 if how == "exemplar" else 0
    tp, fp, fn = TP.sum(axis), FP.sum(axis), FN.sum(axis)
    parts = np.array([_f1(a, b, c) for a, b, c in zip(tp, fp, fn)])
    w = (tp + fn) if how == "weighted" else None
    expected = np.average(parts, weights=w)
    im = IndicatorMatrix(Y, P, task="multilabel", tnorm="product")
    assert im.compute("f1", how) == pytest.approx(expected)


def test_product_tnorm_on_crisp_input_reproduces_the_crisp_result():
    rng = np.random.default_rng(1)
    Y = (rng.random((40, 3)) < 0.4).astype(float)
    P = (rng.random((40, 3)) < 0.4).astype(float)
    crisp = IndicatorMatrix(Y, P, task="multilabel")
    fuzzy = IndicatorMatrix(Y, P, task="multilabel", tnorm="product")
    for how in ("micro", "macro", "weighted", "exemplar"):
        assert fuzzy.compute("f1", how) == pytest.approx(crisp.compute("f1", how))


def test_weighted_average_falls_back_to_uniform_when_there_are_no_positives():
    rng = np.random.default_rng(2)
    Y = np.zeros((20, 3))
    P = rng.random((20, 3))
    soft_im = IndicatorMatrix(Y, P, task="multilabel", tnorm="product")
    assert soft_im.compute("precision", "weighted") == pytest.approx(
        soft_im.compute("precision", "macro"))
    crisp = IndicatorMatrix(Y, (P > 0.5).astype(float), task="multilabel")
    assert crisp.compute("precision", "weighted") == pytest.approx(
        crisp.compute("precision", "macro"))


def test_soft_cells_respect_the_mask(soft):
    Y, P = soft
    mask = np.ones_like(Y)
    mask[:10] = 0
    TP, FP, _, _ = _soft_cells(Y[10:], P[10:])
    im = IndicatorMatrix(Y, P, task="multilabel", tnorm="product", mask=mask)
    assert im.compute("precision", "micro") == pytest.approx(TP.sum() / (TP.sum() + FP.sum()))


def test_per_example_counts_have_one_row_per_example(soft):
    Y, P = soft
    im = IndicatorMatrix(Y, P, task="multilabel", tnorm="product")
    pe = im.per_example()
    assert pe.shape == (30, 4)
    assert pe[:, 0] == pytest.approx((Y * P).sum(axis=1))


# ---------------------------------------------------------------------------
# t-norms
# ---------------------------------------------------------------------------

def test_hamacher_with_gamma_one_is_the_product_tnorm():
    a = np.linspace(0, 1, 7)
    A, B = np.meshgrid(a, a)
    assert hamacher(1.0)(A, B) == pytest.approx(A * B)
    assert marginal_defect(hamacher(1.0)) == pytest.approx(0.0, abs=1e-12)
    assert marginal_defect(hamacher(0.5)) > 0.01


def test_hamacher_is_zero_where_its_denominator_vanishes():
    assert hamacher(0.0)(0.0, 0.0) == 0.0


def test_unknown_tnorm_name_raises_and_callables_pass_through():
    with pytest.raises(KeyError, match="unknown t-norm"):
        get_tnorm("nonsense")
    f = hamacher(2.0)
    assert get_tnorm(f) is f
    assert get_tnorm(" Product ") is get_tnorm("prod")


def test_soft_cells_validate_their_input():
    with pytest.raises(ValueError, match="shape mismatch"):
        soft_cells(np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        soft_cells(np.array([[1.5]]), np.array([[0.5]]))


# ---------------------------------------------------------------------------
# Input forms and task inference
# ---------------------------------------------------------------------------

def test_task_is_inferred_from_the_shape_and_number_of_distinct_labels():
    assert IndicatorMatrix([0, 1, 1, 0], [0, 1, 0, 0]).task == "binary"
    assert IndicatorMatrix([0, 1, 2, 1], [0, 1, 1, 2]).task == "multiclass"
    assert IndicatorMatrix(np.eye(3), np.eye(3)).task == "multilabel"


Y_ORD, P_ORD = [0, 1, 2, 3], [0, 2, 2, 2]


@pytest.fixture
def ordinal_reference():
    return IndicatorMatrix(Y_ORD, P_ORD, task="ordinal")


def test_ordinal_encoding_is_cumulative(ordinal_reference):
    expected_y = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], float)
    expected_p = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], float)
    assert (ordinal_reference.Y == expected_y).all()
    assert (ordinal_reference.P == expected_p).all()


def test_ordinal_accepts_cumulative_truth_and_index_predictions(ordinal_reference):
    Y2 = (np.array(Y_ORD)[:, None] > np.arange(3)).astype(float)
    im = IndicatorMatrix(Y2, P_ORD, task="ordinal", n_classes=4)
    assert (im.Y == ordinal_reference.Y).all() and (im.P == ordinal_reference.P).all()


def test_ordinal_accepts_a_class_score_matrix(ordinal_reference):
    scores = np.eye(4)[P_ORD] * 0.7 + 0.05
    im = IndicatorMatrix(Y_ORD, scores, task="ordinal", n_classes=4)
    assert (im.P == ordinal_reference.P).all()


def test_ordinal_accepts_threshold_probabilities_and_crisp_thresholds(ordinal_reference):
    crisp = ordinal_reference.P
    probs = np.where(crisp == 1, 0.9, 0.1)
    for pred in (probs, crisp):
        im = IndicatorMatrix(Y_ORD, pred, task="ordinal", n_classes=4)
        assert (im.P == crisp).all()


def test_ordinal_rejects_predictions_of_the_wrong_width():
    with pytest.raises(ValueError, match="shape mismatch"):
        IndicatorMatrix(Y_ORD, np.zeros((4, 5)), task="ordinal", n_classes=4)


def test_multiclass_score_matrix_is_argmaxed():
    scores = np.array([[0.1, 0.8, 0.1], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    im = IndicatorMatrix([1, 0, 1], scores, task="multiclass")
    assert (im.P == np.eye(3)[[1, 0, 2]]).all()
    assert im.accuracy() == pytest.approx(2 / 3)


def test_shape_mismatch_between_truth_and_predictions_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        IndicatorMatrix(np.zeros((4, 2)), np.zeros((4, 3)), task="multilabel")


# ---------------------------------------------------------------------------
# IndicatorMatrix methods
# ---------------------------------------------------------------------------

def test_subset_accuracy_counts_exact_label_set_matches():
    Y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0]])
    P = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]])
    assert IndicatorMatrix(Y, P, task="multilabel").subset_accuracy() == pytest.approx(0.5)


def test_accuracy_outside_multiclass_is_label_accuracy():
    Y = np.array([[1, 0], [0, 1], [1, 1]])
    P = np.array([[1, 0], [1, 1], [1, 0]])
    im = IndicatorMatrix(Y, P, task="multilabel")
    assert im.accuracy() == pytest.approx((Y == P).mean())


def test_parts_returns_per_class_values_and_refuses_micro():
    im = IndicatorMatrix([0, 1, 2, 2], [0, 1, 1, 2], task="multiclass")
    assert im.parts("recall", "macro") == pytest.approx([1.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="single value"):
        im.parts("recall", "micro")


def test_report_falls_back_to_a_dict_without_pandas(monkeypatch):
    im = IndicatorMatrix([0, 1, 2, 2], [0, 1, 1, 2], task="multiclass")
    monkeypatch.setitem(sys.modules, "pandas", None)
    table = im.report(["recall"], ["micro", "macro"])
    assert isinstance(table, dict)
    assert table["recall"]["macro"] == pytest.approx(im.compute("recall", "macro"))
    assert im.report(["recall"], as_frame=False)["recall"]["micro"] == pytest.approx(
        im.compute("recall", "micro"))


def test_degrees_of_freedom_by_task():
    assert IndicatorMatrix([0, 1, 1], [0, 1, 0], task="binary").degrees_of_freedom == 2
    assert IndicatorMatrix(np.eye(4), np.eye(4), task="multilabel").degrees_of_freedom == 8
    assert IndicatorMatrix([0, 1, 2, 3], [0, 1, 2, 3],
                           task="multiclass").degrees_of_freedom == 12


# ---------------------------------------------------------------------------
# Cost matrices
# ---------------------------------------------------------------------------

@pytest.fixture
def cost_case():
    C = np.abs(np.subtract.outer(np.arange(3), np.arange(3)))
    return IndicatorMatrix([0, 1, 2, 2], [0, 2, 0, 2], task="multiclass"), C


def test_cost_under_each_aggregation(cost_case):
    im, C = cost_case
    assert im.cost(C, "micro") == pytest.approx(3.0)      # |1-2| + |2-0|
    assert im.cost(C, "exemplar") == pytest.approx(0.75)  # the mean absolute error
    assert im.cost(C, "macro") == pytest.approx(1.0)      # column totals (2, 0, 1)


def test_cost_respects_the_observation_mask(cost_case):
    _, C = cost_case
    mask = np.ones((4, 3))
    mask[2] = 0                                           # drop the |2-0| error
    im = IndicatorMatrix([0, 1, 2, 2], [0, 2, 0, 2], task="multiclass", mask=mask)
    assert im.cost(C, "micro") == pytest.approx(1.0)


def test_cost_validates_its_arguments(cost_case):
    im, C = cost_case
    with pytest.raises(ValueError, match="cost matrix must be"):
        im.cost(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="aggregation must be"):
        im.cost(C, "weighted")


# ---------------------------------------------------------------------------
# Dominance
# ---------------------------------------------------------------------------

Y3 = [0, 0, 1, 1, 2, 2]


def _im(pred):
    return IndicatorMatrix(Y3, pred, task="multiclass")


def test_multiclass_dominance_follows_the_confusion_rates():
    perfect = _im([0, 0, 1, 1, 2, 2])
    worse = _im([0, 1, 1, 1, 2, 2])
    assert perfect.dominates(worse) and perfect.dominates(worse, strict=True)
    assert not worse.dominates(perfect)
    assert pareto_dominance(perfect, worse) == "A_dominates_B"
    assert pareto_dominance(worse, perfect) == "B_dominates_A"


def test_identical_classifiers_dominate_weakly_but_not_strictly():
    a, b = _im([0, 1, 1, 1, 2, 2]), _im([0, 1, 1, 1, 2, 2])
    assert a.dominates(b) and not a.dominates(b, strict=True)
    assert pareto_dominance(a, b) == "equivalent"


def test_trade_offs_between_classes_are_incomparable():
    a = _im([0, 0, 1, 2, 2, 2])      # errs on class 1 only
    b = _im([0, 1, 1, 1, 2, 2])      # errs on class 0 only
    assert not a.dominates(b) and not b.dominates(a)
    assert pareto_dominance(a, b) == "incomparable"


def test_binary_dominance_is_strict_only_when_some_count_improves():
    a = IndicatorMatrix([0, 1, 0, 1], [0, 1, 0, 1], task="binary")
    b = IndicatorMatrix([0, 1, 0, 1], [0, 1, 0, 1], task="binary")
    assert a.dominates(b) and not a.dominates(b, strict=True)


def test_dominance_requires_comparable_problems():
    a = IndicatorMatrix([0, 1, 1], [0, 1, 1], task="binary")
    b = IndicatorMatrix([0, 1, 2], [0, 1, 2], task="multiclass")
    with pytest.raises(ValueError, match="same task"):
        a.dominates(b)


# ---------------------------------------------------------------------------
# Aggregation, conventions and fold pooling
# ---------------------------------------------------------------------------

Y_UNDEF = [0, 1, 2, 2]
P_UNDEF = [0, 1, 1, 1]          # class 2 is never predicted: its precision is 0/0


@pytest.mark.parametrize("convention,expected", [
    ("exclude", (1 + 1 / 3) / 2),
    ("zero", (1 + 1 / 3 + 0) / 3),
    ("one", (1 + 1 / 3 + 1) / 3),
])
def test_conventions_for_undefined_entries(convention, expected):
    im = IndicatorMatrix(Y_UNDEF, P_UNDEF, task="multiclass")
    assert im.compute("precision", "macro", convention=convention) == pytest.approx(expected)


def test_nan_convention_propagates_undefined_entries():
    im = IndicatorMatrix(Y_UNDEF, P_UNDEF, task="multiclass")
    assert np.isnan(im.compute("precision", "macro", convention="nan"))
    assert im.compute("recall", "macro", convention="nan") == pytest.approx(
        im.compute("recall", "macro"))


def test_excluding_when_nothing_is_defined_gives_nan():
    im = IndicatorMatrix(np.zeros((5, 2)), np.zeros((5, 2)), task="multilabel")
    assert np.isnan(im.compute("precision", "macro"))


def test_unknown_convention_and_operator_are_rejected():
    with pytest.raises(ValueError, match="convention must be"):
        _combine(np.array([1.0, np.nan]), "bogus")
    with pytest.raises(ValueError, match="how must be"):
        aggregate(np.eye(2), np.eye(2), "recall", "median")


def test_cells_validate_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        cells(np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="mask shape"):
        cells(np.zeros((3, 2)), np.zeros((3, 2)), mask=np.ones((2, 2)))


def test_sigma_k_requires_at_least_one_fold():
    with pytest.raises(ValueError, match="must not be empty"):
        sigma_k([])


@pytest.fixture
def folds():
    """Three folds of unequal size, as k-fold gives whenever n is not a multiple of k."""
    rng = np.random.default_rng(3)
    ys = [rng.integers(0, 3, n) for n in (8, 7, 9)]
    ps = [np.where(rng.random(len(y)) < 0.6, y, rng.integers(0, 3, len(y))) for y in ys]
    ims = [IndicatorMatrix(y, p, task="multiclass", n_classes=3) for y, p in zip(ys, ps)]
    return ims, IndicatorMatrix(np.concatenate(ys), np.concatenate(ps),
                                task="multiclass", n_classes=3)


@pytest.mark.parametrize("how", ["micro", "macro", "weighted", "exemplar"])
def test_pooling_folds_equals_evaluating_the_concatenated_data(folds, how):
    ims, whole = folds
    assert fold_aggregate(ims, "f1", how, pooling="pooled") == pytest.approx(
        whole.compute("f1", how))


def test_pooled_folds_must_share_the_label_space():
    a = IndicatorMatrix([0, 1, 1], [0, 1, 0], task="multiclass", n_classes=2)
    b = IndicatorMatrix([0, 1, 2], [0, 1, 2], task="multiclass", n_classes=3)
    with pytest.raises(ValueError, match="same number of classes"):
        fold_aggregate([a, b], "f1", "macro")


def test_averaged_pooling_is_the_sample_weighted_mean_of_fold_values(folds):
    ims, _ = folds
    values = [im.compute("f1", "macro") for im in ims]
    assert fold_aggregate(ims, "f1", "macro", pooling="averaged") == pytest.approx(
        np.average(values, weights=[im.n for im in ims]))


def test_fold_aggregate_validates_its_arguments(folds):
    ims, _ = folds
    with pytest.raises(ValueError, match="must not be empty"):
        fold_aggregate([], "f1")
    with pytest.raises(ValueError, match="how must be"):
        fold_aggregate(ims, "f1", "median")
    with pytest.raises(ValueError, match="pooling must be"):
        fold_aggregate(ims, "f1", "macro", pooling="stacked")


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def test_non_decomposable_measures_have_no_numerator_or_denominator():
    mcc = get_measure("mcc")
    with pytest.raises(ValueError, match="numerator"):
        mcc.numerator((1, 1, 1, 1))
    with pytest.raises(ValueError, match="denominator"):
        mcc.denominator((1, 1, 1, 1))


def test_bayes_threshold_is_undefined_when_the_coefficients_cancel():
    m = BinaryMeasure("ratio", lambda tp, fp, fn, tn: (tp + fp) / (tp + fn),
                      linear=((1, 1, 0, 0), (1, 0, 1, 0)))
    assert m.decomposable and m.bayes_threshold() is None
    assert m.erm_consistent() is None


def test_attainable_floor_needs_at_least_two_classes():
    with pytest.raises(ValueError, match="at least 2"):
        get_measure("mcc").attainable_floor(1)


@pytest.mark.parametrize("name,counts", [
    ("gmean", (0, 0, 0, 0)),             # recall and specificity both undefined
    ("kappa", (0, 0, 0, 0)),             # empty confusion matrix
    ("fowlkes_mallows", (0, 0, 0, 5)),   # precision undefined
])
def test_measures_return_nan_when_their_ingredients_are_undefined(name, counts):
    assert np.isnan(get_measure(name)(*counts))


def test_unknown_measure_lists_the_available_ones():
    with pytest.raises(KeyError, match="unknown measure"):
        get_measure("does_not_exist")


def test_list_measures_filters_by_property():
    dec = set(list_measures(decomposable=True))
    assert "mcc" not in dec and "recall" in dec
    assert set(list_measures(decomposable=False)).isdisjoint(dec)
    assert "recall" in list_measures(skew_invariant=True)


# ---------------------------------------------------------------------------
# Property inference on awkward formulas
# ---------------------------------------------------------------------------

def test_inference_survives_formulas_that_are_undefined_on_part_of_the_domain():
    props = infer_properties(lambda tp, fp, fn, tn: np.nan if tp < 30 else tp / (tp + fp))
    assert props.decomposable is False
    assert "numerator" not in props.summary()
    assert "decomposable" in props.summary()


def test_inference_ignores_scaled_points_where_the_formula_is_undefined():
    props = infer_properties(lambda tp, fp, fn, tn: np.nan if tp > 100 else tp / (tp + fp))
    assert props.decomposable is True
    assert props.skew_invariant is False
    assert "numerator coeffs" in props.summary()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

HULL_Y = [2, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 1]
HULL_P = [2, 0, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0]


def test_micro_mcc_can_fall_outside_the_range_of_the_per_class_values():
    im = IndicatorMatrix(HULL_Y, HULL_P, task="multiclass")
    f = check_hull(im, "mcc", "macro")
    assert f is not None and f.code == "outside_hull"
    assert f.detail["lo"] < f.detail["hi"] < f.detail["micro"]
    assert "outside_hull" in [x.code for x in diagnose(im, ["mcc"])]


def test_hull_check_is_silent_for_a_decomposable_measure_and_for_undefined_data():
    im = IndicatorMatrix(HULL_Y, HULL_P, task="multiclass")
    assert check_hull(im, "recall", "macro") is None
    empty = IndicatorMatrix(np.zeros((5, 2)), np.zeros((5, 2)), task="multilabel")
    assert check_hull(empty, "precision", "macro") is None


def test_micro_macro_gap_reports_when_the_identity_does_not_apply():
    im = IndicatorMatrix([0, 1, 2, 2], [0, 1, 1, 2], task="multiclass")
    assert micro_macro_gap(im, "mcc")["exact"] is False
    exact = micro_macro_gap(im, "recall")
    assert exact["exact"] is True
    assert exact["predicted_gap"] == pytest.approx(exact["gap"])
    empty = IndicatorMatrix(np.zeros((5, 2)), np.zeros((5, 2)), task="multilabel")
    assert micro_macro_gap(empty, "precision")["exact"] is False


def test_ovr_drift_needs_multiclass_with_asymmetric_errors():
    binary = IndicatorMatrix([0, 1, 1], [0, 1, 0], task="binary")
    assert uniform_error_deviation(binary) == 0.0 and check_ovr_skew_drift(binary) is None
    perfect = IndicatorMatrix(Y3, Y3, task="multiclass")
    assert check_ovr_skew_drift(perfect) is None
    skewed = IndicatorMatrix([0, 0, 0, 0, 1, 1, 2, 2], [1, 1, 1, 0, 1, 1, 2, 1],
                             task="multiclass")
    f = check_ovr_skew_drift(skewed)
    assert f is not None and f.code == "ovr_skew_drift"


def test_redundancy_is_reported_only_beyond_the_degrees_of_freedom():
    im = IndicatorMatrix([0, 1, 1, 0], [0, 1, 0, 0], task="binary")      # 2 degrees
    assert check_redundancy(im, None) is None
    assert check_redundancy(im, ["recall", "specificity"]) is None
    f = check_redundancy(im, ["recall", "specificity", "f1", "f1"])
    assert f is not None and f.detail["count"] == 3 and f.detail["dof"] == 2
    assert "redundant_suite" in [x.code for x in diagnose(im, ["recall", "specificity", "f1"])]


def test_diagnose_reports_undefined_entries_with_the_size_of_the_convention_effect():
    im = IndicatorMatrix(Y_UNDEF, P_UNDEF, task="multiclass")
    found = [f for f in diagnose(im, ["precision"]) if f.code == "undefined_entries"]
    assert len(found) == 1
    d = found[0].detail
    assert (d["undefined"], d["total"]) == (1, 3)
    assert d["exclude"] == pytest.approx(2 / 3) and d["zero"] == pytest.approx(4 / 9)


def test_micro_collapse_is_not_reported_when_predictions_are_not_single_label():
    Y = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    P = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1]], float)
    im = IndicatorMatrix(Y, P, task="multiclass")
    assert "micro_collapse" not in [f.code for f in diagnose(im)]


def test_drift_check_is_skipped_when_no_prior_sensitive_measure_is_requested():
    im = IndicatorMatrix([0, 0, 0, 0, 1, 1, 2, 2], [1, 1, 1, 0, 1, 1, 2, 1], task="multiclass")
    assert "ovr_skew_drift" not in [f.code for f in diagnose(im, ["precision"])]
    assert "ovr_skew_drift" in [f.code for f in diagnose(im, ["specificity"])]


def test_diagnose_skips_a_measure_that_cannot_be_evaluated_per_example():
    def fragile(tp, fp, fn, tn):
        if tp + fp + fn + tn < 3:
            raise RuntimeError("needs at least three cells")
        return tp / (tp + fp + 1)

    m = BinaryMeasure("fragile", fragile)
    im = IndicatorMatrix((np.arange(40).reshape(20, 2) % 2).astype(float),
                         (np.arange(40).reshape(20, 2) % 3 == 0).astype(float),
                         task="multilabel")
    assert all(f.code != "undefined_entries" for f in diagnose(im, [m]))


def test_report_returns_a_frame_with_one_row_per_measure():
    pd = pytest.importorskip("pandas")
    im = IndicatorMatrix([0, 1, 2, 2], [0, 1, 1, 2], task="multiclass")
    frame = im.report(["recall", "f1"], ["micro", "macro"])
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.index) == ["recall", "f1"] and list(frame.columns) == ["micro", "macro"]
    assert frame.loc["f1", "macro"] == pytest.approx(im.compute("f1", "macro"))


def test_drift_is_not_reported_when_errors_are_uniform():
    perfect = IndicatorMatrix(Y3, Y3, task="multiclass")
    assert "ovr_skew_drift" not in [f.code for f in diagnose(perfect, ["specificity"])]


def test_inference_rejects_a_formula_that_changes_between_the_fit_and_the_check():
    calls = {"n": 0}

    def drifting(tp, fp, fn, tn):
        calls["n"] += 1
        return tp / (tp + fp) if calls["n"] <= 40 else np.nan   # fit set, then check set

    assert infer_properties(drifting).decomposable is False


def test_sigma_k_sums_cell_matrices_elementwise():
    a = cells(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [1, 1]]))
    b = cells(np.array([[1, 1], [0, 0]]), np.array([[0, 1], [0, 0]]))
    total = sigma_k([a, b])
    for k in range(4):
        assert (total[k] == a[k] + b[k]).all()


def test_binary_problems_are_not_flagged_for_per_example_undefined_entries():
    y = [0, 1, 1, 0, 1, 0, 0, 1]
    p = [0, 1, 0, 0, 1, 1, 0, 1]
    im = IndicatorMatrix(y, p, task="binary")
    findings = diagnose(im, ["precision", "recall"])
    assert not [f for f in findings if f.code == "undefined_entries"]
    # a multilabel problem with several labels is still checked per example
    Y = np.array([[1, 0], [0, 0], [1, 1]])
    P = np.array([[1, 0], [0, 0], [0, 1]])
    ml = diagnose(IndicatorMatrix(Y, P, task="multilabel"), ["precision"])
    assert any(f.code == "undefined_entries" and f.detail["aggregation"] == "exemplar" for f in ml)
