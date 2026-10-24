# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Choosing a suite of measures (Remark 2): the Jacobian, its rank, and the selection.

The gradients are checked against derivatives written by hand, and the rank test
against the redundancies the paper proves (Theorem 2 and the degrees-of-freedom
bound).
"""
from __future__ import annotations

import numpy as np
import pytest

import aggmetrics.suites as suites
from aggmetrics import (
    IndicatorMatrix,
    list_measures,
    register_measure,
    select_suite,
    suite_jacobian,
    suite_rank,
)
from aggmetrics.measures import _REGISTRY


def binary(n=2000, prevalence=0.2, accuracy=0.8, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random(n) < prevalence).astype(int)
    p = np.where(r.random(n) < accuracy, y, 1 - y)
    return IndicatorMatrix(y, p, task="binary")


def multiclass(n=3000, m=3, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, m, n)
    p = np.where(r.random(n) < 0.7, y, r.integers(0, m, n))
    return IndicatorMatrix(y, p, task="multiclass")


def test_remark2_binary_gradients_match_derivatives_by_hand():
    im = binary()
    pi = float(im.Y.mean())
    rec, spec = im.canonical_coordinates()
    J, names = suite_jacobian(im, ["label_accuracy", "f1", "balanced_accuracy"], "micro")
    assert names == ["recall", "specificity"]
    np.testing.assert_allclose(J[0], [pi, 1 - pi], atol=1e-7)
    d = pi * (1 + rec) + (1 - pi) * (1 - spec)
    np.testing.assert_allclose(
        J[1],
        [2 * pi * (pi + (1 - pi) * (1 - spec)) / d**2,
         2 * pi * rec * (1 - pi) / d**2],
        atol=1e-7,
    )
    np.testing.assert_allclose(J[2], [0.5, 0.5], atol=1e-7)


def test_remark2_multiclass_gradients_match_derivatives_by_hand():
    im = multiclass()
    n_j = im.Y.sum(axis=0)
    J, names = suite_jacobian(im, [("recall", "macro"), ("precision", "micro")])
    assert len(names) == im.degrees_of_freedom == 6
    # macro-recall is the mean of 1 - sum_k R_jk, micro-precision is C/n
    np.testing.assert_allclose(J[0], -1 / im.m, atol=1e-7)
    expected = np.array([-n_j[int(nm[2])] / im.n for nm in names])
    np.testing.assert_allclose(J[1], expected, atol=1e-7)


def test_remark2_rank_of_binary_suites():
    im = binary()
    ok = suite_rank(im, ["recall", "specificity"], "micro")
    assert (ok.rank, ok.degrees_of_freedom, ok.redundant) == (2, 2, False)
    assert ok.volume == pytest.approx(1.0)

    dependent = suite_rank(im, ["recall", "fnr"], "micro")        # fnr = 1 - recall
    assert (dependent.rank, dependent.redundant) == (1, True)
    assert dependent.volume == pytest.approx(0.0, abs=1e-9)

    # more measures than degrees of freedom: Proposition 3
    three = suite_rank(im, ["label_accuracy", "f1", "balanced_accuracy"], "micro")
    assert three.rank == 2 and three.redundant
    assert len(three.singular_values) == 2          # min(measures, degrees of freedom)


def test_remark2_micro_collapse_gives_a_rank_one_suite():
    # Theorem 2: micro-precision, recall and F1 all equal accuracy on one-hot data
    r = suite_rank(multiclass(), ["precision", "recall", "f1"], "micro")
    assert r.rank == 1 and r.redundant


def test_remark2_pairing_row_and_column_measures_beats_two_row_measures():
    # the paper recommends macro-recall with macro-precision rather than two
    # row-conditioned measures; recall and fnr are the extreme case of the latter
    im = multiclass()
    pair = suite_rank(im, ["recall", "precision"], "macro")
    twin = suite_rank(im, ["recall", "fnr"], "macro")
    assert pair.rank == 2 and twin.rank == 1
    assert pair.volume > twin.volume


def test_select_suite_recovers_an_orthogonal_pair_in_the_binary_case():
    sel = select_suite(binary(), aggregation="micro")
    assert sel.method == "exhaustive" and sel.rank == 2
    assert sel.volume == pytest.approx(1.0)
    assert len(sel.items) == 2


def test_select_suite_default_size_is_capped_by_the_independent_candidates():
    im = binary()
    sel = select_suite(im, ["recall", "fnr", "specificity"], aggregation="micro")
    assert len(sel.items) == 2 and sel.rank == 2
    assert {n for n, _ in sel.items} & {"specificity"}


def test_select_suite_greedy_is_no_better_than_exhaustive(monkeypatch):
    im = multiclass()
    cand = [(m, o) for o in ("macro", "weighted") for m in list_measures()]
    exhaustive = select_suite(im, cand, size=3)
    assert exhaustive.method == "exhaustive"
    monkeypatch.setattr(suites, "_EXHAUSTIVE_LIMIT", 0)
    greedy = select_suite(im, cand, size=3)
    assert greedy.method == "greedy"
    assert greedy.volume <= exhaustive.volume + 1e-12
    assert greedy.rank == exhaustive.rank == 3


def test_select_suite_size_one_is_always_exhaustive(monkeypatch):
    monkeypatch.setattr(suites, "_EXHAUSTIVE_LIMIT", 0)
    sel = select_suite(binary(), size=1, aggregation="micro")
    assert sel.method == "exhaustive" and len(sel.items) == 1


def test_select_suite_over_many_candidates_uses_the_greedy_search():
    im = multiclass()
    cand = [(m, o) for o in ("micro", "macro", "weighted") for m in list_measures()]
    sel = select_suite(im, cand, size=5)
    assert sel.method == "greedy"
    assert len(set(sel.items)) == 5 and 3 <= sel.rank <= 5
    with pytest.raises(ValueError, match="size must be"):
        select_suite(im, cand, size=6)        # the candidates span only 5 directions


def test_operators_weighted_and_undefined_labels():
    # the last label has no positives, so its recall is undefined everywhere
    r = np.random.default_rng(3)
    Y = (r.random((400, 4)) < [0.4, 0.3, 0.5, 0.0]).astype(int)
    P = np.where(r.random(Y.shape) < 0.8, Y, 1 - Y)
    im = IndicatorMatrix(Y, P, task="multilabel")
    J, names = suite_jacobian(im, [("recall", "weighted"), ("recall", "macro"),
                                   ("precision", "micro")])
    assert J.shape == (3, 8) and names[0] == "recall[0]"
    assert np.isfinite(J).all()
    # the undefined label does not respond to the classifier's recall
    assert J[1, 6] == 0.0


def test_a_measure_that_does_not_respond_is_reported_flat_and_never_chosen():
    register_measure("flat_half", lambda tp, fp, fn, tn: 0.5 + 0.0 * tp, overwrite=True)
    try:
        im = binary()
        r = suite_rank(im, ["recall", "flat_half"], "micro")
        assert r.flat == (("flat_half", "micro"),) and r.rank == 1
        sel = select_suite(im, ["flat_half", "recall", "specificity"], aggregation="micro")
        assert all(name != "flat_half" for name, _ in sel.items)
        with pytest.raises(ValueError, match="size must be"):
            select_suite(im, ["flat_half"], size=1, aggregation="micro")
    finally:
        del _REGISTRY["flat_half"]


@pytest.mark.parametrize("bad", [
    dict(measures=[("recall", "exemplar")]),
    dict(measures=["recall", ("recall", "macro")]),
])
def test_suite_rank_rejects_unsupported_or_repeated_items(bad):
    with pytest.raises(ValueError):
        suite_rank(multiclass(), **bad)


def test_suite_functions_reject_unsupported_data():
    soft = IndicatorMatrix([[1.0, 0.0], [0.8, 0.2]], [[0.9, 0.1], [0.7, 0.3]],
                           task="multilabel", tnorm="product")
    with pytest.raises(ValueError, match="masks and t-norms"):
        suite_rank(soft, ["recall"])
    masked = IndicatorMatrix([[1, 0], [0, 1]], [[1, 0], [1, 1]],
                             task="multilabel", mask=[[1, 1], [1, 0]])
    with pytest.raises(ValueError, match="masks and t-norms"):
        suite_rank(masked, ["recall"])
    ordinal = IndicatorMatrix([0, 1, 2, 2, 1, 0], [0, 2, 2, 1, 1, 1], task="ordinal")
    with pytest.raises(ValueError, match="not supported"):
        suite_rank(ordinal, ["recall"])
    with pytest.raises(ValueError, match="size must be"):
        select_suite(binary(), size=9, aggregation="micro")
