# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Results of the paper turned into code: prior shift, several targets, balanced
denominators (Corollary 1) and taxonomy cost matrices.
"""
from __future__ import annotations

import numpy as np
import pytest

from aggmetrics import (
    IndicatorMatrix,
    denominator_balance,
    diagnose,
    get_measure,
    multioutput_aggregate,
    pu_correct,
    tree_cost_matrix,
    under_prior,
)


def binary(n=600, prevalence=0.3, accuracy=0.8, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random(n) < prevalence).astype(int)
    p = np.where(r.random(n) < accuracy, y, 1 - y)
    return IndicatorMatrix(y, p, task="binary")


def multiclass(sizes=(200, 100, 50), seed=0):
    r = np.random.default_rng(seed)
    y = np.repeat(np.arange(len(sizes)), sizes)
    p = np.where(r.random(len(y)) < 0.7, y, r.integers(0, len(sizes), len(y)))
    return IndicatorMatrix(y, p, task="multiclass")


def multilabel(n=500, seed=0):
    r = np.random.default_rng(seed)
    Y = (r.random((n, 5)) < [0.5, 0.3, 0.1, 0.02, 0.0]).astype(int)   # the last label is empty
    P = np.where(r.random(Y.shape) < 0.8, Y, 1 - Y)
    return IndicatorMatrix(Y, P, task="multilabel")


# -- prior shift ------------------------------------------------------------

@pytest.mark.parametrize("im, priors", [
    (binary(), None),
    (multiclass(), None),
    (multilabel(), None),
])
def test_under_prior_at_the_observed_prior_reproduces_compute(im, priors):
    if im.task == "binary":
        priors = float(im.Y.mean())
    elif im.task == "multiclass":
        priors = im.Y.mean(axis=0)
    else:
        priors = im.Y.mean(axis=0)
    shifted = under_prior(im, priors)
    for ms in ("precision", "recall", "f1", "mcc", "gmean"):
        for how in ("micro", "macro", "weighted"):
            for conv in ("exclude", "zero", "one"):
                a = im.compute(ms, how, convention=conv)
                b = shifted.compute(ms, how, convention=conv)
                assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-12)
    assert shifted.n == im.n
    np.testing.assert_allclose(shifted.per_class(), im.per_class())
    assert shifted.accuracy() == pytest.approx(im.accuracy())


def test_proposition6_skew_invariant_measures_do_not_move_and_precision_does():
    im = binary()
    rec, spec = im.canonical_coordinates()
    for pi in (0.02, 0.5, 0.9):
        s = under_prior(im, pi)
        assert s.compute("recall") == pytest.approx(rec)
        assert s.compute("specificity") == pytest.approx(spec)
        by_hand = pi * rec / (pi * rec + (1 - pi) * (1 - spec))
        assert s.compute("precision") == pytest.approx(by_hand)
    assert under_prior(im, 0.02).compute("precision") < im.compute("precision")


def test_multiclass_prior_shift_keeps_rates_and_moves_precision_and_accuracy():
    im = multiclass()
    rates = np.diag((im.Y.T @ im.P) / im.Y.sum(axis=0)[:, None])
    uniform = np.full(3, 1 / 3)
    s = under_prior(im, uniform)
    assert s.compute("recall", "macro") == pytest.approx(im.compute("recall", "macro"))
    assert s.accuracy() == pytest.approx(float((uniform * rates).sum()))
    assert s.compute("precision", "macro") != pytest.approx(im.compute("precision", "macro"))
    assert s.counts[0] == pytest.approx(float((uniform * rates).sum() * im.n))


def test_weighted_operator_without_any_positive_falls_back_like_compute():
    # every weight (positives per label) is zero, so the operator averages uniformly
    Y = np.zeros((8, 2), int)
    P = np.array([[0, 1], [0, 0], [1, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])
    im = IndicatorMatrix(Y, P, task="multilabel")
    shifted = under_prior(im, [0.0, 0.0])
    assert shifted.compute("specificity", "weighted") == pytest.approx(
        im.compute("specificity", "weighted"))


def test_multilabel_scalar_prior_applies_to_every_label():
    im = multilabel()
    s = under_prior(im.__class__(im.Y[:, :3], im.P[:, :3], task="multilabel"), 0.25)
    np.testing.assert_allclose(s.priors, 0.25)
    assert s.compute("f1", "micro") > 0


@pytest.mark.parametrize("im, priors, match", [
    (multiclass(), [0.5, 0.5], "must be 3"),
    (multiclass(), [0.5, 0.4, 0.2], "sum to 1"),
    (multiclass(), [-0.1, 0.6, 0.5], "non-negative"),
    (binary(), 1.5, "prevalence"),
    (binary(), [0.2, 0.3], "prevalence"),
])
def test_under_prior_rejects_invalid_priors(im, priors, match):
    with pytest.raises(ValueError, match=match):
        under_prior(im, priors)


def test_under_prior_rejects_impossible_priors_and_unsupported_data():
    absent = IndicatorMatrix([0, 0, 1, 1], [0, 1, 1, 1], task="multiclass", n_classes=3)
    with pytest.raises(ValueError, match="absent"):
        under_prior(absent, [0.3, 0.3, 0.4])
    assert under_prior(absent, [0.5, 0.5, 0.0]).compute("recall", "macro") == pytest.approx(0.75)
    no_positives = IndicatorMatrix([0, 0, 0, 0], [0, 1, 0, 0], task="binary")
    with pytest.raises(ValueError, match="without positives"):
        under_prior(no_positives, 0.3)
    ordinal = IndicatorMatrix([0, 1, 2, 2, 1, 0], [0, 2, 2, 1, 1, 1], task="ordinal")
    with pytest.raises(ValueError, match="not supported"):
        under_prior(ordinal, 0.5)
    with pytest.raises(ValueError, match="exemplar"):
        under_prior(binary(), 0.3).compute("recall", "exemplar")


# -- several targets ---------------------------------------------------------

def two_targets():
    r = np.random.default_rng(1)
    y1 = r.integers(0, 3, 300)
    y2 = r.integers(0, 2, 300)
    p1 = np.where(r.random(300) < 0.7, y1, r.integers(0, 3, 300))
    p2 = np.where(r.random(300) < 0.9, y2, 1 - y2)
    return [IndicatorMatrix(y1, p1, task="multiclass"),
            IndicatorMatrix(y2, p2, task="multiclass", n_classes=2)]


def test_theorem1_pooling_targets_is_the_denominator_weighted_target_average():
    ims = two_targets()
    for ms in ("recall", "precision", "f1", "jaccard"):
        meas = get_measure(ms)
        values = np.array([im.compute(ms, "micro") for im in ims])
        dens = np.array([meas.denominator(im.counts) for im in ims])
        pooled = multioutput_aggregate(ims, ms, "micro")
        assert pooled == pytest.approx(float((dens * values).sum() / dens.sum()))
        assert multioutput_aggregate(ims, ms, "target") == pytest.approx(values.mean())


def test_corollary1_equal_target_denominators_make_pooled_equal_the_target_average():
    # the same number of positives, hence the same recall denominator, in both targets
    y = np.array([1] * 40 + [0] * 60)
    ims = [IndicatorMatrix(y, np.where(np.arange(100) % 5 == 0, 1 - y, y), task="binary"),
           IndicatorMatrix(y, np.where(np.arange(100) % 3 == 0, 1 - y, y), task="binary")]
    assert multioutput_aggregate(ims, "recall", "micro") == pytest.approx(
        multioutput_aggregate(ims, "recall", "target"))
    assert multioutput_aggregate(ims, "recall", "micro") == pytest.approx(
        multioutput_aggregate(ims, "recall", "macro"))


def test_multioutput_macro_and_weighted_use_every_class_of_every_target():
    ims = two_targets()
    cols = np.concatenate([im.parts("recall", "macro") for im in ims])
    assert multioutput_aggregate(ims, "recall", "macro") == pytest.approx(cols.mean())
    prevalence = np.concatenate([im.Y.sum(axis=0) for im in ims])
    assert multioutput_aggregate(ims, "recall", "weighted") == pytest.approx(
        float((prevalence * cols).sum() / prevalence.sum()))
    within = multioutput_aggregate(ims, "recall", "target", within="macro")
    assert within == pytest.approx(np.mean([im.compute("recall", "macro") for im in ims]))


def test_multioutput_undefined_target_follows_the_convention():
    ok = binary()
    empty = IndicatorMatrix(np.zeros(ok.n, int), np.zeros(ok.n, int), task="binary")
    both = [ok, empty]                                   # recall is 0/0 in the second
    excl = multioutput_aggregate(both, "recall", "target")
    assert excl == pytest.approx(ok.compute("recall"))
    assert multioutput_aggregate(both, "recall", "target", convention="zero") == pytest.approx(
        ok.compute("recall") / 2)
    assert np.isnan(multioutput_aggregate(both, "recall", "target", convention="nan"))


def test_multioutput_rejects_bad_input():
    ims = two_targets()
    with pytest.raises(ValueError, match="at least one"):
        multioutput_aggregate([], "recall")
    with pytest.raises(ValueError, match="same examples"):
        multioutput_aggregate([ims[0], binary(n=50)], "recall")
    soft = IndicatorMatrix([[1.0, 0.0], [0.8, 0.2]], [[0.9, 0.1], [0.7, 0.3]],
                           task="multilabel", tnorm="product")
    with pytest.raises(ValueError, match="masks"):
        multioutput_aggregate([soft, soft], "recall")
    with pytest.raises(ValueError, match="how must be"):
        multioutput_aggregate(ims, "recall", "exemplar")


# -- Corollary 1: balanced denominators ---------------------------------------

def test_corollary1_balanced_class_sizes_make_micro_and_macro_recall_equal():
    im = multiclass(sizes=(100, 100, 100))
    info = denominator_balance(im, "recall")
    assert info["balanced"] and not info["structural"]
    np.testing.assert_allclose(info["denominators"], 100)
    assert info["micro"] == pytest.approx(info["macro"])
    unbalanced = denominator_balance(multiclass(), "recall")
    assert not unbalanced["balanced"]
    assert unbalanced["micro"] != pytest.approx(unbalanced["macro"])


def test_corollary1_precision_depends_on_the_predictions_not_the_class_sizes():
    im = multiclass(sizes=(100, 100, 100))
    predicted = im.P.sum(axis=0)
    assert denominator_balance(im, "precision")["balanced"] == bool(
        np.allclose(predicted, predicted[0]))


def test_denominator_balance_edge_cases():
    with pytest.raises(ValueError, match="not aggregation-decomposable"):
        denominator_balance(multiclass(), "mcc")
    assert denominator_balance(multiclass(), "label_accuracy")["structural"]
    assert not denominator_balance(binary(), "recall")["balanced"]          # one class
    empty = IndicatorMatrix(np.zeros((4, 2), int), np.zeros((4, 2), int), task="multilabel")
    assert not denominator_balance(empty, "recall")["balanced"]    # equal, but all zero


def test_diagnose_reports_balanced_denominators_only_when_not_structural():
    codes = lambda im, ms: [f.code for f in diagnose(im, ms)]        # noqa: E731
    assert "balanced_denominators" in codes(multiclass(sizes=(100, 100, 100)), ["recall"])
    assert "balanced_denominators" not in codes(multiclass(), ["recall"])
    # the denominator of label accuracy is n for every class: true on any data
    assert "balanced_denominators" not in codes(multiclass(sizes=(100, 100, 100)),
                                                ["label_accuracy"])
    assert "balanced_denominators" not in codes(multiclass(sizes=(100, 100, 100)), ["mcc"])
    finding = [f for f in diagnose(multiclass(sizes=(100, 100, 100)), ["recall"])
               if f.code == "balanced_denominators"][0]
    assert finding.severity == "note" and "Corollary 1" in finding.message


# -- taxonomies ---------------------------------------------------------------

TREE = {"root": None, "A": "root", "B": "root", "a1": "A", "a2": "A", "b1": "B"}


def test_tree_cost_matrix_counts_edges_between_leaves():
    C = tree_cost_matrix(TREE)
    assert C.astype(int).tolist() == [[0, 2, 4], [2, 0, 4], [4, 4, 0]]
    np.testing.assert_array_equal(C, C.T)


def test_tree_cost_matrix_accepts_inner_nodes_and_an_order():
    C = tree_cost_matrix(TREE, classes=["b1", "A", "a2"])
    assert C.astype(int).tolist() == [[0, 3, 4], [3, 0, 1], [4, 1, 0]]


def test_tree_cost_matrix_gives_the_hierarchical_loss_through_cost():
    C = tree_cost_matrix(TREE)                         # classes a1, a2, b1
    y = np.array([0, 0, 1, 2, 2, 1])
    p = np.array([0, 1, 1, 0, 2, 2])
    im = IndicatorMatrix(y, p, task="multiclass")
    by_hand = sum(C[a, b] for a, b in zip(y, p))       # 0 + 2 + 0 + 4 + 0 + 4
    assert im.cost(C) == pytest.approx(by_hand) == pytest.approx(10.0)


@pytest.mark.parametrize("tree, match", [
    ({"a": None, "b": None}, "exactly one root"),
    ({"r": None, "a": "b", "b": "a"}, "cycle"),
    ({"r": None, "a": "ghost"}, "not a node"),
])
def test_tree_cost_matrix_rejects_malformed_taxonomies(tree, match):
    with pytest.raises(ValueError, match=match):
        tree_cost_matrix(tree)


def test_tree_cost_matrix_rejects_unknown_classes():
    with pytest.raises(ValueError, match="not in the taxonomy"):
        tree_cost_matrix(TREE, classes=["a1", "zzz"])


# -- positive-unlabeled learning ----------------------------------------------

def pu_observed(tp=300, fp=200, fn=100, tn=1400, c=0.5):
    """Observed PU data whose cells are those the latent cells give in expectation."""
    obs = (c * tp, fp + (1 - c) * tp, c * fn, tn + (1 - c) * fn)
    tp_o, fp_o, fn_o, tn_o = (int(round(x)) for x in obs)
    s = np.array([1] * (tp_o + fn_o) + [0] * (fp_o + tn_o))
    p = np.array([1] * tp_o + [0] * fn_o + [1] * fp_o + [0] * tn_o)
    return IndicatorMatrix(s, p, task="binary")


def test_pu_correction_recovers_the_latent_cells_exactly_in_expectation():
    im = pu_observed()                                  # latent 300, 200, 100, 1400 at c = 0.5
    fixed = pu_correct(im, propensity=0.5)
    assert [round(float(x)) for x in fixed.counts] == [300, 200, 100, 1400]
    assert fixed.compute("precision") == pytest.approx(300 / 500)
    assert fixed.compute("specificity") == pytest.approx(1400 / 1600)
    assert fixed.accuracy() == pytest.approx(1700 / 2000)
    assert fixed.n == im.n and fixed.propensity == 0.5
    assert fixed.priors[0] == pytest.approx(400 / 2000)       # the latent prevalence


def test_pu_recall_is_the_same_for_every_propensity_and_precision_is_biased():
    # Section 6: the propensity cancels in recall, and fp_obs mixes in latent positives
    im = pu_observed()
    nominal_precision = im.compute("precision")
    for c in (0.5, 0.7, 1.0):
        fixed = pu_correct(im, propensity=c)
        assert fixed.compute("recall") == pytest.approx(im.compute("recall"))
    assert pu_correct(im, propensity=0.5).compute("precision") > nominal_precision
    # at c = 1 nothing is hidden: the observed labels are the latent ones
    one = pu_correct(im, propensity=1.0)
    assert one.compute("precision") == pytest.approx(nominal_precision)
    assert [float(x) for x in one.counts] == pytest.approx(list(im.counts))


def test_pu_prior_and_propensity_describe_the_same_correction():
    im = pu_observed()
    by_c = pu_correct(im, propensity=0.5)
    by_prior = pu_correct(im, prior=400 / 2000)
    assert by_prior.propensity == pytest.approx(0.5)
    assert by_prior.compute("f1") == pytest.approx(by_c.compute("f1"))


def test_pu_correction_on_a_simulated_scar_labelling():
    r = np.random.default_rng(7)
    n, c = 200_000, 0.4
    y = (r.random(n) < 0.2).astype(int)                        # latent labels
    p = np.where(r.random(n) < 0.85, y, 1 - y)                 # a classifier, 85% right
    s = ((y == 1) & (r.random(n) < c)).astype(int)             # observed labels
    truth = IndicatorMatrix(y, p, task="binary")
    observed = IndicatorMatrix(s, p, task="binary")
    fixed = pu_correct(observed, propensity=c)
    for ms in ("precision", "specificity", "f1"):
        assert fixed.compute(ms) == pytest.approx(truth.compute(ms), abs=0.01)
    assert abs(observed.compute("precision") - truth.compute("precision")) > 0.2


def test_pu_correction_rejects_bad_input():
    im = pu_observed()
    with pytest.raises(ValueError, match="exactly one"):
        pu_correct(im)
    with pytest.raises(ValueError, match="exactly one"):
        pu_correct(im, propensity=0.5, prior=0.2)
    for bad in (0.0, 1.2, -0.1):
        with pytest.raises(ValueError, match="propensity must be"):
            pu_correct(im, propensity=bad)
    with pytest.raises(ValueError, match="prior must be"):
        pu_correct(im, prior=0.0)
    with pytest.raises(ValueError, match="smaller than the share"):
        pu_correct(im, prior=0.05)                          # c would exceed 1
    with pytest.raises(ValueError, match="too small"):
        pu_correct(im, propensity=0.2)                      # needs negative counts
    with pytest.raises(ValueError, match="binary"):
        pu_correct(multiclass(), propensity=0.5)


def test_pu_correction_without_observed_positives_keeps_the_negatives():
    im = IndicatorMatrix(np.zeros(10, int), np.array([0] * 8 + [1] * 2), task="binary")
    fixed = pu_correct(im, propensity=0.9)
    assert [float(x) for x in fixed.counts] == pytest.approx([0.0, 2.0, 0.0, 8.0])
    assert fixed.compute("specificity") == pytest.approx(0.8)


def test_pu_correction_when_every_prediction_is_positive():
    im = IndicatorMatrix([1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], task="binary")
    fixed = pu_correct(im, propensity=0.5)       # tn_obs = 0 and fn_obs = 0: nothing to correct
    assert fixed.compute("recall") == pytest.approx(1.0)
    assert fixed.n == 6
