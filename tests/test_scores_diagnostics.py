"""aggmetrics.scores.diagnostics: gaps, duality, skew, properness, the hard case."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from aggmetrics.diagnostics import MeasureWarning
from aggmetrics.scores import (
    ScoreMatrix,
    auc,
    auc_gaps,
    diagnose_scores,
    hard_bridge,
    ndcg,
    ndcg_gaps,
    power,
    properness_table,
    ranking_duality,
    skew_report,
)
from aggmetrics.scores.diagnostics import SKEW_INVARIANT

Y = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0]])
P = np.array([[.7, .6, .1], [.6, .1, .4], [.8, .2, .7], [.4, .3, .2]])


@pytest.fixture
def sm():
    return ScoreMatrix(Y, P)


def _codes(findings):
    return {f.code for f in findings}


# -- collapse and non-collapse ------------------------------------------------------

def test_ndcg_gaps_running_example(sm):
    g = ndcg_gaps(sm)
    assert g["micro"] == pytest.approx(ndcg(sm, "micro"))
    assert g["class_mean"] == pytest.approx(ndcg(sm, "macro"))
    assert g["exemplar"] == pytest.approx(ndcg(sm, "exemplar"))
    # Theorem dcg_collapse (b)-(d)
    assert g["micro_minus_exemplar"] == pytest.approx(g["covariance"] / g["mean_ideal"])
    assert np.nansum(g["class_contributions"]) == pytest.approx(g["micro_minus_class"])
    assert g["class_minus_exemplar"] == pytest.approx(g["class_mean"] - g["exemplar"])
    assert g["excluded_classes"].size == 0 and g["excluded_examples"] == 0


def test_ndcg_gaps_excludes_empty_rows_and_columns():
    y = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]])
    p = np.array([[.2, .5, .1], [.3, .3, .3], [.9, .4, .8], [.1, .6, .2]])
    s = ScoreMatrix(y, p)
    g = ndcg_gaps(s)
    assert g["excluded_classes"].tolist() == [2] and g["excluded_examples"] == 1
    assert np.isnan(g["class_contributions"][2])
    assert g["micro"] == pytest.approx(ndcg(s, "micro"))
    assert g["exemplar"] == pytest.approx(ndcg(s, "exemplar", empty="nan"))
    assert g["exemplar_zero"] == pytest.approx(ndcg(s, "exemplar"))
    assert g["micro_minus_exemplar"] == pytest.approx(g["covariance"] / g["mean_ideal"])
    assert np.nansum(g["class_contributions"]) == pytest.approx(g["micro_minus_class"])
    assert g["class_minus_exemplar"] == pytest.approx(g["class_mean"] - g["exemplar"])
    with pytest.raises(ValueError, match="relevant cell"):
        ndcg_gaps(ScoreMatrix(np.zeros((2, 2)), np.full((2, 2), .5)))


def test_auc_gaps_running_example(sm):
    g = auc_gaps(sm)
    assert (g["micro"], g["macro"], g["exemplar"]) == pytest.approx((7 / 9, 13 / 18,
                                                                       5 / 6))
    # Theorem auc_collapse (b)-(d)
    assert g["micro_minus_macro"] == pytest.approx(g["class_covariance"]
                                                   / g["mean_positives"])
    assert g["micro_minus_macro"] == pytest.approx(1 / 18)
    assert g["class_contributions"].sum() == pytest.approx(g["micro_minus_macro"])
    assert g["micro_minus_exemplar"] == pytest.approx(g["example_covariance"]
                                                      / g["mean_relevant"])
    assert g["macro_minus_exemplar"] == pytest.approx(13 / 18 - 5 / 6)


def test_auc_gaps_excludes_undefined_classes():
    y = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0]])   # class 1 has no negatives
    p = np.array([[.7, .6, .1], [.6, .1, .4], [.8, .2, .7], [.4, .3, .2]])
    s = ScoreMatrix(y, p)
    g = auc_gaps(s)
    assert g["excluded_classes"].tolist() == [1]
    assert g["macro"] == pytest.approx(auc(s, "macro"))
    assert g["micro"] == pytest.approx(auc(s, "micro"))
    assert g["micro_minus_exemplar"] == pytest.approx(g["example_covariance"]
                                                      / g["mean_relevant"])
    assert g["macro_minus_exemplar"] == pytest.approx(g["macro"] - g["exemplar"])
    with pytest.raises(ValueError, match="positives and negatives"):
        auc_gaps(ScoreMatrix(np.ones((3, 2)), np.full((3, 2), .5)))


# -- duality ---------------------------------------------------------------------------

def test_ranking_duality_running_example(sm):
    d = ranking_duality(sm)
    # Example running_duality: A, B and C are ordered oppositely by the two families
    assert d["discordant"] == [(0, 1), (0, 2), (1, 2)]
    assert d["concordant"] == 0 and d["tau"] == -1.0


def test_ranking_duality_concordant_ties_and_empty():
    # identical columns: both summaries tie, nothing is compared
    t = ScoreMatrix(np.array([[1, 1], [0, 0], [1, 1]]),
                    np.array([[.6, .6], [.3, .3], [.4, .4]]))
    dt = ranking_duality(t)
    assert dt["compared"] == 0 and np.isnan(dt["tau"])
    # a concordant pair
    c = ScoreMatrix(np.array([[1, 0], [0, 1], [1, 0], [0, 0]]),
                    np.array([[.9, .2], [.5, .3], [.8, .4], [.5, .6]]))
    dc = ranking_duality(c)
    assert dc["concordant"] == 1 and dc["tau"] == 1.0


# -- skew ------------------------------------------------------------------------------

def test_skew_report_invariants(sm):
    r = skew_report(sm, factor=3)
    for s in SKEW_INVARIANT:
        np.testing.assert_allclose(r["own"][s], 0.0, atol=1e-12)
    assert (r["own"]["brier"] > 0).all()          # the mixture moves with pi_j
    assert r["others"]["alpha"][0] > 0            # A shares examples with B and C
    assert r["globals"]["auc_micro"][1] > 0       # n_j^+-weighted
    assert r["globals"]["auc_macro"][1] == pytest.approx(0.0)


def test_skew_report_skips_unskewable_and_single_class():
    y = np.array([[1, 0], [1, 0], [0, 0]])
    p = np.array([[.9, .2], [.4, .6], [.3, .1]])
    r = skew_report(ScoreMatrix(y, p))
    assert np.isnan(r["own"]["alpha"][1]) and not np.isnan(r["own"]["alpha"][0])
    b = skew_report(ScoreMatrix(np.array([1, 0, 1, 0]), np.array([.8, .3, .4, .5])))
    assert b["others"]["alpha"][0] == 0.0
    assert b["own"]["auc"][0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="factor"):
        skew_report(ScoreMatrix(y, p), factor=1)


# -- properness ------------------------------------------------------------------------

def test_properness_table():
    rows = {r["name"]: r for r in properness_table([power(3)])}
    assert rows["squared"]["strictly_proper"] and rows["log"]["strictly_proper"]
    assert not rows["absolute"]["proper"]
    assert not rows["power(3)"]["proper"]
    assert rows["dcg"]["proper"] and not rows["dcg"]["strictly_proper"]
    assert rows["ndcg"]["proper"] is False and rows["auc"]["proper"] is None
    assert {r["family"] for r in rows.values()} == {"deviation", "ranking"}


# -- the hard case ---------------------------------------------------------------------

def test_hard_bridge(sm):
    h = hard_bridge(sm, threshold=.5)
    assert h["consistent"]
    np.testing.assert_allclose(h["alpha"], 1 - h["recall"])
    np.testing.assert_allclose(h["beta"], 1 - h["specificity"])
    y = np.array([[1, 0], [1, 0], [0, 0]])          # class 1 without positives
    h2 = hard_bridge(ScoreMatrix(y, np.array([[1., 0.], [0., 1.], [1., 0.]])))
    assert h2["consistent"] and np.isnan(h2["alpha"][1])
    with pytest.raises(ValueError, match="0/1"):
        hard_bridge(sm)


# -- the runner ------------------------------------------------------------------------

def test_diagnose_scores_running_example(sm):
    f = diagnose_scores(sm)
    assert _codes(f) == {"degenerate_murphy", "ndcg_no_collapse", "auc_no_collapse",
                         "pooled_auc_differs", "ranking_duality", "class_coupling"}
    with pytest.warns(MeasureWarning, match="degenerate_murphy"):
        diagnose_scores(sm, emit_warnings=True)


def test_diagnose_scores_quiet_cases():
    # one label with tied scores: everything collapses and nothing is reported
    s = ScoreMatrix(np.array([1, 0, 1, 0]), np.array([.8, .3, .8, .3]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert diagnose_scores(s, emit_warnings=True) == []
    # two identical labels: pooled AUC equals micro-AUC, every row carries both
    t = ScoreMatrix(np.array([[1, 1], [0, 0], [1, 1], [0, 0]]),
                    np.array([[.8, .8], [.3, .3], [.8, .8], [.4, .4]]))
    assert "pooled_auc_differs" not in _codes(diagnose_scores(t))
    # one-hot multiclass: no example carries two labels
    u = ScoreMatrix(np.array([0, 1, 2, 0]),
                    np.array([[.6, .3, .1], [.2, .5, .3], [.3, .3, .4], [.5, .4, .1]]))
    assert "class_coupling" not in _codes(diagnose_scores(u))


def test_diagnose_scores_undefined_and_empty():
    s = ScoreMatrix(np.array([[1, 0], [1, 0], [0, 0]]),
                    np.array([[.8, .1], [.8, .2], [.3, .1]]))
    assert "undefined_auc" in _codes(diagnose_scores(s))
    e = ScoreMatrix(np.zeros((3, 2)), np.array([[.1, .2], [.1, .2], [.1, .3]]))
    assert _codes(diagnose_scores(e)) == {"undefined_auc"}
