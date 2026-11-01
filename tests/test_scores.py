"""aggmetrics.scores: score matrices, deviation operators, conditional errors, Murphy."""
from __future__ import annotations

import warnings
from fractions import Fraction as F

import numpy as np
import pytest

with warnings.catch_warnings():  # scikit-learn may import pandas, which warns
    warnings.simplefilter("ignore", DeprecationWarning)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import ndcg_score, roc_auc_score

from aggmetrics import IndicatorMatrix
from aggmetrics.scores import (
    ConditionalErrors,
    DegenerateDecompositionWarning,
    Deviation,
    ScoreMatrix,
    auc,
    auc_cells,
    auc_parts,
    brier_under_prior,
    check_properness,
    class_weighted_brier,
    dcg,
    dcg_parts,
    geometric,
    get_deviation,
    list_deviations,
    log_weight,
    murphy,
    ndcg,
    ndcg_parts,
    pooled_auc,
    power,
    precision_at_k,
    row_ranks,
    top_k,
)

# The running example of the companion paper (four examples, labels A, B, C).
Y = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0]])
P = np.array([[.7, .6, .1], [.6, .1, .4], [.8, .2, .7], [.4, .3, .2]])


def approx(fracs):
    return pytest.approx([float(f) for f in fracs], abs=1e-12)


def random_scores(n=200, m=4, seed=0, binary_scores=False):
    r = np.random.default_rng(seed)
    y = (r.random((n, m)) < r.uniform(0.2, 0.6, m)).astype(float)
    p = np.clip(0.6 * y + 0.4 * r.random((n, m)), 0, 1)
    if binary_scores:
        p = (p > 0.5).astype(float)
    return y, p


# -- running example (golden values of Appendix A) ---------------------------------

def test_running_example_brier_collapse():
    sm = ScoreMatrix(Y, P)
    assert sm.task == "multilabel" and (sm.n, sm.m) == (4, 3)
    for agg in ("micro", "macro", "exemplar"):
        assert sm.brier(agg) == pytest.approx(float(F(13, 48)), abs=1e-12)
    assert sm.parts("squared", "class") == approx([F(21, 80), F(13, 40), F(9, 40)])
    assert sm.parts("squared", "example") == approx(
        [F(13, 150), F(53, 300), F(59, 100), F(23, 100)])
    assert sm.loss("squared", scale="sum") == pytest.approx(3 * 13 / 48)


def test_running_example_conditional_errors_and_mixture():
    ce = ScoreMatrix(Y, P).conditional_errors()
    assert ce.alpha == approx([F(1, 8), F(43, 100), F(9, 25)])
    assert ce.beta == approx([F(2, 5), F(1, 100), F(9, 50)])
    assert ce.prevalence == approx([F(1, 2), F(3, 4), F(1, 4)])
    assert ce.mixture() == approx([F(21, 80), F(13, 40), F(9, 40)])


def test_running_example_companions():
    assert ScoreMatrix(Y, P ** 2).brier() == pytest.approx(14039 / 40000)
    same = np.array([[.8, .7, .2], [.9, .4, .1], [.4, .2, .3], [.2, .1, .6]])
    assert ScoreMatrix(Y, same).brier() == pytest.approx(13 / 48)


def test_running_example_change_of_prior():
    sm = ScoreMatrix(Y, P)
    new = brier_under_prior(sm, [2 / 3, 3 / 4, 1 / 4])
    assert new == approx([F(13, 60), F(13, 40), F(9, 40)])
    ce = sm.conditional_errors()
    assert class_weighted_brier(sm, 1.0, 1.0) == pytest.approx(ce.alpha + ce.beta)


# -- construction ------------------------------------------------------------------

def test_binary_and_multiclass_inputs():
    sm = ScoreMatrix([1, 0, 1], [0.9, 0.2, 0.4])
    assert sm.task == "binary" and sm.Y.shape == (3, 1)
    assert sm.brier() == pytest.approx((0.01 + 0.04 + 0.36) / 3)
    sm2 = ScoreMatrix(np.array([[1], [0]]), np.array([[0.8], [0.3]]), task="binary")
    assert sm2.Y.shape == (2, 1)
    mc = ScoreMatrix([0, 2, 1], np.array([[.7, .2, .1], [.1, .1, .8], [.3, .6, .1]]))
    assert mc.task == "multiclass"
    assert mc.Y.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert mc.prevalence == pytest.approx([1 / 3] * 3)


@pytest.mark.parametrize("args,kw,msg", [
    (([0, 1], [0.1, 0.2]), {"task": "ordinal"}, "task must"),
    (([0, 1, 2], [0.1, 0.2, 0.3]), {"task": "multiclass"}, r"\(n, m\) matrix"),
    (([0, 3], np.full((2, 3), 0.3)), {}, "class indices"),
    (([0, 0.5], np.full((2, 3), 0.3)), {"task": "multiclass"}, "class indices"),
    (([0, 1, 2], [0.1, 0.2, 0.3]), {"task": "multiclass", "n_classes": 3},
     "shape mismatch"),
    ((np.ones((2, 2)), np.full((2, 3), 0.3)), {}, "shape mismatch"),
    ((np.full((2, 2), 2.0), np.full((2, 2), 0.3)), {}, "0/1"),
    ((np.ones((2, 2)), np.full((2, 2), 1.3)), {}, r"\[0, 1\]"),
    ((np.ones((2, 2)), np.full((2, 2), np.nan)), {}, "finite"),
])
def test_invalid_inputs(args, kw, msg):
    with pytest.raises(ValueError, match=msg):
        ScoreMatrix(*args, **kw)


def test_invalid_arguments():
    sm = ScoreMatrix(Y, P)
    with pytest.raises(ValueError, match="axis"):
        sm.parts(axis="row")
    with pytest.raises(ValueError, match="aggregation"):
        sm.loss(aggregation="weighted")
    with pytest.raises(ValueError, match="scale"):
        sm.loss(scale="total")
    with pytest.raises(ValueError, match="axis"):
        sm.conditional_errors(axis="row")
    ce = sm.conditional_errors()
    with pytest.raises(ValueError, match="shape"):
        ce.mixture([0.5, 0.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ce.mixture([0.5, 1.5, 0.2])


# -- deviation operators -----------------------------------------------------------

def test_registry_and_aliases():
    assert list_deviations() == ["absolute", "log", "squared"]
    assert get_deviation("Brier").name == "squared"
    assert get_deviation("mae").name == "absolute"
    assert get_deviation("logloss").name == "log"
    d = get_deviation("squared")
    assert get_deviation(d) is d
    with pytest.raises(ValueError, match="unknown"):
        get_deviation("hinge")
    with pytest.raises(TypeError):
        get_deviation(3)


def test_log_loss_is_extended_valued():
    sm = ScoreMatrix([1, 0, 1], [0.8, 0.0, 0.0])
    cells = sm.cells("log").ravel()
    assert cells[0] == pytest.approx(-np.log(0.8))
    assert cells[1] == 0.0 and np.isinf(cells[2])
    assert np.isinf(sm.loss("log"))
    ce = sm.conditional_errors("log")
    assert np.isinf(ce.alpha[0]) and ce.beta[0] == 0.0


def test_custom_phi_and_power():
    def quartic(t):
        return t ** 4
    d = get_deviation(quartic)
    assert d.name == "quartic" and not d.proper
    assert ScoreMatrix(Y, P).loss(quartic) == pytest.approx(np.mean((Y - P) ** 4))
    assert power(2).proper and not power(3).proper
    assert power(1)(np.array([-0.5])) == pytest.approx([0.5])
    with pytest.raises(ValueError):
        power(0)


@pytest.mark.parametrize("op,proper,strict", [
    ("squared", True, True),
    ("log", True, True),
    ("absolute", False, False),
    (power(4), False, False),
    (Deviation("poly", lambda t: 5 / 8 * t ** 2 - np.abs(t) ** 3 / 3 + t ** 4 / 4,
               False, False), True, True),
])
def test_check_properness(op, proper, strict):
    res = check_properness(op)
    assert res.proper is proper and res.strictly_proper is strict
    if proper:
        assert res.max_report_error < 1e-9
    else:
        assert res.max_gap > 1e-6


def test_check_properness_constant_phi_is_weakly_proper():
    res = check_properness(lambda t: np.zeros_like(t), grid=11)
    assert res.proper and not res.strictly_proper
    with pytest.raises(ValueError, match="grid"):
        check_properness("squared", grid=2)


# -- conditional errors ------------------------------------------------------------

def test_hard_scores_recover_one_minus_recall_and_specificity():
    y, p = random_scores(seed=3, binary_scores=True)
    ce = ScoreMatrix(y, p).conditional_errors()
    tp, fp, fn, tn = np.asarray(IndicatorMatrix(y, p, task="multilabel").per_class()).T
    assert ce.alpha == pytest.approx(1 - tp / (tp + fn))
    assert ce.beta == pytest.approx(1 - tn / (tn + fp))


def test_conditional_errors_without_support():
    y = np.array([[1, 0], [1, 0]])
    ce = ScoreMatrix(y, np.array([[.9, .2], [.7, .1]])).conditional_errors()
    assert np.isnan(ce.beta[0]) and np.isnan(ce.alpha[1])
    assert ce.mixture() == pytest.approx([(0.01 + 0.09) / 2, (0.04 + 0.01) / 2])


def test_example_axis_and_skew_invariance():
    y, p = random_scores(seed=4)
    sm = ScoreMatrix(y, p)
    rows = sm.conditional_errors(axis="example")
    assert rows.alpha.shape == (y.shape[0],)
    ok = (rows.prevalence > 0) & (rows.prevalence < 1)
    assert rows.mixture()[ok] == pytest.approx(sm.parts(axis="example")[ok])
    # replicating the positives of class 0 leaves alpha_0, beta_0 unchanged
    pos = y[:, 0] == 1
    rep = ScoreMatrix(np.vstack([y, y[pos]]), np.vstack([p, p[pos]]))
    a, b = sm.conditional_errors(), rep.conditional_errors()
    assert b.alpha[0] == pytest.approx(a.alpha[0])
    assert b.beta[0] == pytest.approx(a.beta[0])
    assert b.mixture()[0] == pytest.approx(rep.parts()[0])


# -- Murphy decompositions ---------------------------------------------------------

@pytest.mark.parametrize("axis", ["class", "example"])
@pytest.mark.parametrize("bins,strategy", [(None, "uniform"), (10, "uniform"),
                                           (5, "quantile")])
def test_murphy_identity(axis, bins, strategy):
    y, p = random_scores(seed=5)
    p = np.round(p, 1)  # repeated values give informative level sets
    md = murphy(ScoreMatrix(y, p), axis=axis, bins=bins, strategy=strategy)
    assert md.brier == pytest.approx(md.unc - md.res + md.rel, abs=1e-12)
    assert md.axis == axis and md.bins == bins
    if bins is None:
        assert md.binning_gap is None
        assert md.brier == pytest.approx(ScoreMatrix(y, p).parts(axis=axis))
    else:
        assert md.brier + md.binning_gap == pytest.approx(
            ScoreMatrix(y, p).parts(axis=axis))


def test_murphy_running_example_is_degenerate_without_bins():
    with pytest.warns(DegenerateDecompositionWarning):
        md = murphy(ScoreMatrix(Y, P))
    assert md.rel == pytest.approx(md.brier)
    assert md.res == pytest.approx(md.unc)


def test_murphy_paper_instance_rel_after_replication():
    # Corollary murphy_skew(c): calibrated at one prevalence, REL = 1/45 after
    # doubling the positives.
    y = np.array([1, 1, 0, 1, 0, 0], float).reshape(-1, 1)
    p = np.array([2 / 3] * 3 + [1 / 3] * 3).reshape(-1, 1)
    assert murphy(ScoreMatrix(y, p)).rel == pytest.approx([0.0], abs=1e-15)
    pos = y[:, 0] == 1
    rep = ScoreMatrix(np.vstack([y, y[pos]]), np.vstack([p, p[pos]]))
    assert murphy(rep).rel == pytest.approx([1 / 45])


def test_murphy_invalid_arguments():
    sm = ScoreMatrix(Y, P)
    with pytest.raises(ValueError, match="axis"):
        murphy(sm, axis="row")
    with pytest.raises(ValueError, match="bins"):
        murphy(sm, bins=0)
    with pytest.raises(ValueError, match="bins"):
        murphy(sm, bins=2.5)
    with pytest.raises(ValueError, match="strategy"):
        murphy(sm, bins=3, strategy="kmeans")


def test_conditional_errors_dataclass_roundtrip():
    ce = ConditionalErrors(alpha=np.array([0.2]), beta=np.array([0.1]),
                           prevalence=np.array([0.5]))
    assert ce.mixture([0.0]) == pytest.approx([0.1])
    assert ce.mixture([1.0]) == pytest.approx([0.2])


# -- isotonic (CORP) decomposition ---------------------------------------------------

@pytest.mark.parametrize("axis", ["class", "example"])
@pytest.mark.parametrize("rounded", [False, True])
def test_murphy_isotonic_matches_sklearn(axis, rounded):
    y, p = random_scores(seed=7)
    if rounded:
        p = np.round(p, 1)  # ties: the fit pools each level set first
    md = murphy(ScoreMatrix(y, p), axis=axis, method="isotonic")
    Yc, Pc = (y, p) if axis == "class" else (y.T, p.T)
    for k in range(Yc.shape[1]):
        f = IsotonicRegression().fit_transform(Pc[:, k], Yc[:, k])
        bs, bs_f = np.mean((Yc[:, k] - Pc[:, k]) ** 2), np.mean((Yc[:, k] - f) ** 2)
        ybar = Yc[:, k].mean()
        assert md.rel[k] == pytest.approx(bs - bs_f, abs=1e-12)
        assert md.res[k] == pytest.approx(ybar * (1 - ybar) - bs_f, abs=1e-12)
    assert md.brier == pytest.approx(md.unc - md.res + md.rel, abs=1e-12)
    assert md.method == "isotonic" and md.bins is None and md.binning_gap is None
    assert (md.rel >= -1e-12).all() and (md.res >= -1e-12).all()


def test_murphy_isotonic_running_example():
    md = murphy(ScoreMatrix(Y, P), method="isotonic")
    # label B is perfectly ordered: the recalibrated scores are the labels
    assert md.res[1] == pytest.approx(md.unc[1]) and md.rel[1] == pytest.approx(md.brier[1])
    assert md.brier == approx([F(21, 80), F(13, 40), F(9, 40)])
    assert md.miscalibration is md.rel and md.discrimination is md.res
    assert md.uncertainty is md.unc


def test_murphy_isotonic_on_calibrated_monotone_scores():
    # scores equal to the per-level frequencies and increasing: no miscalibration
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1], float).reshape(-1, 1)
    p = np.array([.25] * 4 + [.75] * 4).reshape(-1, 1)
    md = murphy(ScoreMatrix(y, p), method="isotonic")
    assert md.rel == pytest.approx([0.0], abs=1e-15)
    assert md.res == pytest.approx([0.0625])


def test_murphy_method_arguments():
    sm = ScoreMatrix(Y, P)
    with pytest.raises(ValueError, match="method"):
        murphy(sm, method="kde")
    with pytest.raises(ValueError, match="bins"):
        murphy(sm, method="isotonic", bins=5)
    assert murphy(sm, bins=2).method == "levels"


# -- ranking: DCG family -------------------------------------------------------------


W2 = 1 / np.log2(3)


def test_running_example_dcg_family():
    sm = ScoreMatrix(Y, P)
    raw = (1 + W2 + 1 + W2 + 0.5 + W2) / 12
    for agg in ("micro", "macro", "exemplar"):
        assert dcg(sm, agg) == pytest.approx(raw)
    assert dcg_parts(sm) == pytest.approx([2 / 4, (2 * W2 + 0.5) / 4, W2 / 4])
    assert ndcg(sm, "micro") == pytest.approx((2 + 0.5 + 3 * W2) / (2 * (1 + W2) + 2))
    assert ndcg_parts(sm) == pytest.approx([1.0, (2 * W2 + 0.5) / 3, W2])
    assert ndcg_parts(sm, "example") == pytest.approx([1.0, 1.0, 0.5, W2])
    assert ndcg(sm, "macro") == pytest.approx(0.739405, abs=1e-6)
    assert ndcg(sm, "exemplar") == pytest.approx(0.782732, abs=1e-6)
    assert row_ranks(sm).tolist() == [[1, 2, 3], [1, 3, 2], [1, 3, 2], [1, 2, 3]]
    assert precision_at_k(sm, 1) == pytest.approx([1, 1, 0, 0])  # top-1 is always A


def test_paper_two_by_three_witness():
    sm = ScoreMatrix(np.array([[0, 0, 1], [1, 1, 0]]),
                     np.array([[.9, .6, .2], [.9, .6, .2]]))
    assert ndcg(sm, "micro") == pytest.approx(0.809953, abs=1e-6)
    assert ndcg(sm, "macro") == pytest.approx(0.710310, abs=1e-6)
    assert ndcg(sm, "exemplar") == pytest.approx(0.75)


def test_exemplar_ndcg_matches_sklearn():
    y, p = random_scores(n=60, m=6, seed=8)
    y[y.sum(axis=1) == 0, 0] = 1  # every example has a relevant label
    sm = ScoreMatrix(y, p)
    assert ndcg(sm, "exemplar") == pytest.approx(ndcg_score(y, p))


def test_ranks_ties_and_weights():
    sm = ScoreMatrix(np.array([[1, 0, 1]]), np.array([[.5, .5, .2]]))
    assert row_ranks(sm, "min").tolist() == [[1, 1, 3]]
    assert row_ranks(sm, "average").tolist() == [[1.5, 1.5, 3]]
    assert row_ranks(sm, "ordinal").tolist() == [[1, 2, 3]]
    with pytest.raises(ValueError, match="ties"):
        row_ranks(sm, "dense")
    assert log_weight([1, 3]) == pytest.approx([1.0, 0.5])
    assert geometric(0.5)([1, 2, 3]) == pytest.approx([1, 0.5, 0.25])
    assert top_k(2)([1, 2, 3]).tolist() == [1, 1, 0]
    assert dcg(sm, weight=top_k(1)) == pytest.approx(1 / 3)
    for bad in (0, 1.0 + 1e-9):
        with pytest.raises(ValueError):
            geometric(bad)
    for bad in (0, 1.5):
        with pytest.raises(ValueError):
            top_k(bad)
    with pytest.raises(ValueError, match="weight"):
        dcg(sm, weight="exp")


def test_hard_top1_limit_proposition():
    # one-hot labels, hard top-1 predictions, min-rank: exemplar DCG is
    # (P@1 w(1) + (1 - P@1) w(2)) / m  (Proposition hard_ranking(a))
    r = np.random.default_rng(9)
    n, m = 50, 4
    c = r.integers(0, m, n)
    js = np.where(r.random(n) < 0.6, c, r.integers(0, m, n))
    y = np.eye(m)[c]
    sm = ScoreMatrix(y, np.eye(m)[js])
    p1 = np.mean(c == js)
    assert dcg(sm, "exemplar") == pytest.approx((p1 + (1 - p1) * log_weight(2)) / m)


def test_empty_rows_and_classes():
    y = np.array([[1, 0], [0, 0], [1, 0]])
    sm = ScoreMatrix(y, np.array([[.9, .1], [.4, .6], [.3, .2]]))
    assert ndcg_parts(sm, "example").tolist()[1] == 0.0
    assert np.isnan(ndcg_parts(sm, "example", empty="nan")[1])
    assert ndcg_parts(sm, "class")[1] == 0.0
    assert ndcg(sm, "macro", empty="nan") == pytest.approx(ndcg_parts(sm)[0])
    empty = ScoreMatrix(np.zeros((2, 2)), np.full((2, 2), 0.5))
    assert ndcg(empty, "micro") == 0.0
    with pytest.raises(ValueError, match="empty"):
        ndcg_parts(sm, empty="drop")


@pytest.mark.parametrize("fn,kw", [(dcg_parts, {"axis": "row"}),
                                   (ndcg_parts, {"axis": "row"}),
                                   (dcg, {"aggregation": "weighted"}),
                                   (ndcg, {"aggregation": "weighted"}),
                                   (auc_parts, {"axis": "row"}),
                                   (auc, {"aggregation": "weighted"})])
def test_ranking_invalid_arguments(fn, kw):
    with pytest.raises(ValueError):
        fn(ScoreMatrix(Y, P), **kw)


def test_precision_at_k_bounds():
    sm = ScoreMatrix(Y, P)
    assert precision_at_k(sm, 3) == pytest.approx(Y.sum(axis=1) / 3)
    for k in (0, 4, 1.5):
        with pytest.raises(ValueError):
            precision_at_k(sm, k)


# -- ranking: AUC family -------------------------------------------------------------

def test_running_example_auc_family():
    sm = ScoreMatrix(Y, P)
    assert auc_parts(sm) == approx([F(1, 2), F(1), F(2, 3)])
    assert auc(sm, "macro") == pytest.approx(13 / 18)
    assert auc(sm, "micro") == pytest.approx(7 / 9)
    assert auc_parts(sm, "example") == approx([F(3, 4), F(7, 12), F(1), F(1)])
    assert auc(sm, "exemplar") == pytest.approx(5 / 6)
    assert auc(sm, "raw") == pytest.approx(7 / 18)
    assert pooled_auc(sm) == pytest.approx(43 / 72)


def test_auc_matches_sklearn_with_ties():
    y, p = random_scores(n=80, m=5, seed=10)
    p = np.round(p, 1)  # ties: half credit, as in roc_auc_score
    sm = ScoreMatrix(y, p)
    expected = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]
    assert auc_parts(sm) == pytest.approx(expected)
    assert pooled_auc(sm) == pytest.approx(roc_auc_score(y.ravel(), p.ravel()))


def test_auc_raw_collapse_and_mean_rank_identity():
    y, p = random_scores(n=40, m=3, seed=11)
    sm = ScoreMatrix(y, p)
    pi = y.mean(axis=0)
    assert auc(sm, "raw") == pytest.approx((pi * auc_parts(sm)).mean())
    # Proposition auc_meanrank: AUC_j = (n - (n+ - 1)/2 - M_j) / n-, M_j the mean
    # (descending) rank of the positives, with distinct scores
    n = y.shape[0]
    for j in range(y.shape[1]):
        order = np.argsort(-p[:, j])
        rank = np.empty(n)
        rank[order] = np.arange(1, n + 1)
        npos = y[:, j].sum()
        M = rank[y[:, j] == 1].mean()
        assert auc_parts(sm)[j] == pytest.approx((n - (npos - 1) / 2 - M) / (n - npos))


def test_auc_undefined_classes_and_examples():
    y = np.array([[1, 1], [0, 1], [0, 1]])
    sm = ScoreMatrix(y, np.array([[.9, .5], [.2, .4], [.4, .7]]))
    cells = auc_cells(sm)
    assert np.isnan(cells[:, 1]).all() and cells[0, 0] == 1.0
    assert np.isnan(auc_parts(sm)[1])
    assert auc_parts(sm, empty="zero")[1] == 0.0
    assert auc(sm, "macro") == pytest.approx(1.0)
    assert auc(sm, "micro") == pytest.approx(1.0)
    assert np.isnan(auc_parts(sm, "example")).all()
    assert auc(sm, "exemplar", empty="zero") == 0.0
    with pytest.raises(ValueError, match="pooled"):
        pooled_auc(ScoreMatrix(np.ones((2, 2)), np.full((2, 2), 0.5)))


def test_duality_can_reverse_label_order():
    # running example: nDCG orders A > C > B, AUC orders B > C > A
    sm = ScoreMatrix(Y, P)
    assert np.argsort(-ndcg_parts(sm)).tolist() == [0, 2, 1]
    assert np.argsort(-auc_parts(sm)).tolist() == [1, 2, 0]
