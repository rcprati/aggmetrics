# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""API behaviour: construction, conventions, masking, sklearn interoperability."""
from __future__ import annotations

import numpy as np
import pytest

from aggmetrics import (IndicatorMatrix, aggregate, diagnose, get_measure,
                         list_measures, make_scorer, score, sklearn_average)
from aggmetrics.diagnostics import MeasureWarning


def test_construction_from_class_indices():
    im = IndicatorMatrix([0, 1, 2, 1], [0, 2, 2, 1], task="multiclass")
    assert (im.n, im.m) == (4, 3)
    assert im.accuracy() == pytest.approx(0.75)


def test_construction_from_indicator_matrices():
    Y = np.array([[1, 0, 1], [0, 1, 0]])
    P = np.array([[1, 1, 0], [0, 1, 0]])
    im = IndicatorMatrix(Y, P, task="multilabel")
    assert im.counts == (2.0, 1.0, 1.0, 2.0)


def test_probability_matrix_is_thresholded_for_multilabel():
    Y = np.array([[1, 0], [0, 1]])
    P = np.array([[0.9, 0.4], [0.2, 0.7]])
    im = IndicatorMatrix(Y, P, task="multilabel", threshold=0.5)
    assert np.array_equal(im.P, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert im.compute("f1", "micro") == pytest.approx(1.0)


def test_probability_matrix_uses_argmax_for_multiclass():
    Y = np.array([[1, 0, 0], [0, 0, 1]])
    P = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    im = IndicatorMatrix(Y, P, task="multiclass")
    assert im.P.sum(axis=1).tolist() == [1.0, 1.0]
    assert im.accuracy() == pytest.approx(1.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        IndicatorMatrix(np.zeros((3, 2)), np.zeros((3, 4)), task="multilabel")


def test_unknown_measure_lists_alternatives():
    with pytest.raises(KeyError, match="unknown measure"):
        get_measure("not_a_measure")


def test_aliases_resolve():
    assert get_measure("sensitivity").name == "recall"
    assert get_measure("IoU").name == "jaccard"
    assert get_measure("dice").name == "f1"
    assert get_measure("cohen-kappa").name == "kappa"


# ---------------------------------------------------------------------------
# Undefined-value conventions
# ---------------------------------------------------------------------------

def test_conventions_differ_when_entries_are_undefined():
    # column 1 is never predicted -> per-class precision undefined there
    Y = np.array([[1, 1], [1, 0], [0, 1]])
    P = np.array([[1, 0], [1, 0], [0, 0]])
    excl = aggregate(Y, P, "precision", "macro", convention="exclude")
    zero = aggregate(Y, P, "precision", "macro", convention="zero")
    one = aggregate(Y, P, "precision", "macro", convention="one")
    assert excl == pytest.approx(1.0)
    assert zero == pytest.approx(0.5)
    assert one == pytest.approx(1.0)
    assert np.isnan(aggregate(Y, P, "precision", "macro", convention="nan"))


def test_micro_is_defined_where_macro_is_not():
    """Pooling before dividing survives individual vanishing denominators."""
    Y = np.array([[1, 1], [1, 0]])
    P = np.array([[1, 0], [1, 0]])
    assert not np.isnan(aggregate(Y, P, "precision", "micro"))
    parts = IndicatorMatrix(Y, P, task="multilabel").parts("precision", "macro")
    assert np.isnan(parts).any()


def test_bad_convention_is_rejected():
    with pytest.raises(ValueError, match="convention must be"):
        aggregate(np.eye(3), np.eye(3), "f1", "macro", convention="bogus")


# ---------------------------------------------------------------------------
# Masking / partial labels
# ---------------------------------------------------------------------------

def test_mask_excludes_unobserved_entries():
    Y = np.array([[1, 0], [0, 1]])
    P = np.array([[1, 1], [0, 1]])
    full = IndicatorMatrix(Y, P, task="multilabel")
    masked = IndicatorMatrix(Y, P, task="multilabel",
                             mask=np.array([[1, 0], [1, 1]]))
    assert full.counts[1] == 1.0        # the false positive is counted
    assert masked.counts[1] == 0.0      # ... and masked away
    assert masked.compute("precision", "micro") == pytest.approx(1.0)


def test_abstention_contributes_true_negatives():
    """An all-zero predicted row still yields m - |Y_i| true negatives."""
    Y = np.array([[0, 0, 1, 0, 0]])
    P = np.zeros((1, 5))
    im = IndicatorMatrix(Y, P, task="multilabel")
    tp, fp, fn, tn = im.counts
    assert (tp, fp, fn) == (0.0, 0.0, 1.0)
    assert tn == 4.0


# ---------------------------------------------------------------------------
# Weighted averaging and the sklearn correspondence
# ---------------------------------------------------------------------------

def test_weighted_matches_prevalence_weighting():
    Y = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
    P = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    im = IndicatorMatrix(Y, P, task="multilabel")
    parts = im.parts("recall", "macro")
    prevalence = Y.sum(axis=0)
    expected = np.average(parts, weights=prevalence)
    assert im.compute("recall", "weighted") == pytest.approx(expected)


def test_sklearn_average_translation():
    assert sklearn_average("micro") == "micro"
    assert sklearn_average("samples") == "exemplar"
    assert sklearn_average(None) == "macro"
    with pytest.raises(ValueError):
        sklearn_average("nonsense")


def test_score_function_and_scorer_agree():
    y = np.array([0, 1, 2, 2, 1, 0])
    yhat = np.array([0, 2, 2, 1, 1, 0])
    direct = score(y, yhat, "f1", "macro", task="multiclass")

    class Dummy:
        def predict(self, X):
            return yhat

    scorer = make_scorer("f1", "macro", task="multiclass")
    assert scorer(Dummy(), None, y) == pytest.approx(direct)


def test_scorer_negates_lower_is_better_measures():
    y = np.array([0, 1, 1, 0])
    yhat = np.array([0, 0, 1, 0])

    class Dummy:
        def predict(self, X):
            return yhat

    s = make_scorer("error_rate", "micro", task="binary")(Dummy(), None, y)
    assert s <= 0.0     # greater is better after negation


@pytest.mark.skipif(pytest.importorskip("sklearn", reason="sklearn absent") is None,
                    reason="sklearn absent")
def test_scorer_works_inside_cross_val_score():
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=1000))])
    for name in ("f1", "mcc", "balanced_accuracy"):
        s = cross_val_score(clf, X, y, cv=3,
                            scoring=make_scorer(name, "macro", task="multiclass"))
        assert s.shape == (3,)
        assert np.isfinite(s).all()


# ---------------------------------------------------------------------------
# Reporting and diagnostics
# ---------------------------------------------------------------------------

def test_report_covers_requested_grid():
    y = np.array([0, 1, 2, 2, 1, 0])
    yhat = np.array([0, 2, 2, 1, 1, 0])
    im = IndicatorMatrix(y, yhat, task="multiclass")
    rep = im.report(measures=["f1", "mcc"], as_frame=False)
    assert set(rep) == {"f1", "mcc"}
    assert set(rep["f1"]) == {"micro", "macro", "exemplar"}


def test_diagnose_flags_inflated_floor_and_collapse():
    y = np.arange(200) % 8
    yhat = (y + 1) % 8
    im = IndicatorMatrix(y, yhat, task="multiclass")
    codes = {f.code for f in diagnose(im)}
    assert "inflated_floor" in codes
    assert "micro_collapse" in codes
    assert "degenerate_optimum" in codes


def test_diagnose_can_emit_warnings():
    y = np.arange(60) % 5
    yhat = (y + 1) % 5
    im = IndicatorMatrix(y, yhat, task="multiclass")
    with pytest.warns(MeasureWarning):
        diagnose(im, emit_warnings=True)


def test_list_measures_filters():
    dec = list_measures(decomposable=True)
    non = list_measures(decomposable=False)
    assert "f1" in dec and "recall" in dec
    assert "mcc" in non and "balanced_accuracy" in non
    assert not set(dec) & set(non)
    skew = list_measures(skew_invariant=True)
    assert "recall" in skew and "balanced_accuracy" in skew
    assert "precision" not in skew


def test_cost_matrix_shape_is_validated():
    im = IndicatorMatrix([0, 1], [1, 0], task="multiclass")
    with pytest.raises(ValueError, match="cost matrix must be"):
        im.cost(np.zeros((3, 3)))


def test_soft_labels_require_unit_interval():
    from aggmetrics.tnorms import soft_cells
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        soft_cells(np.array([[1.5]]), np.array([[0.5]]))
