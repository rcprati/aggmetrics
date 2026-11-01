# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Diagnostics for scores: when the summaries of a :class:`ScoreMatrix` disagree, why.

The counterpart of :mod:`aggmetrics.diagnostics` for unthresholded scores. Each
function reports a result of the companion paper on the data at hand:

* :func:`ndcg_gaps` and :func:`auc_gaps` write the differences between the micro,
  class and exemplar summaries as the covariances and coefficient sums of
  Theorems ``dcg_collapse`` (b)--(d) and ``auc_collapse`` (b)--(d), with the
  contribution of each class;
* :func:`ranking_duality` lists the pairs of labels that per-class nDCG and
  per-class AUC order in opposite ways (Theorem ``duality``);
* :func:`skew_report` replicates the positive examples of each class and records
  which summaries move. Per-class ``alpha_j``, ``beta_j``, ``AUC_j`` and
  ``nDCG_j`` do not, while prevalence-weighted summaries do; because whole rows are
  replicated, the per-class summaries of the other classes can move as well;
* :func:`properness_table` states which operators are (strictly) proper;
* :func:`hard_bridge` checks that on 0/1 scores ``alpha_j = 1 - recall_j`` and
  ``beta_j = 1 - specificity_j``, as computed by :class:`aggmetrics.IndicatorMatrix`;
* :func:`diagnose_scores` runs the checks and returns
  :class:`aggmetrics.diagnostics.Finding` objects.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import numpy as np

from ..core import IndicatorMatrix
from ..diagnostics import Finding, MeasureWarning
from .calibration import DegenerateDecompositionWarning, murphy
from .deviation import check_properness, get_deviation, list_deviations
from .matrix import ScoreMatrix
from .ranking import (
    _ideal,
    _weight,
    auc,
    auc_cells,
    auc_parts,
    ndcg,
    ndcg_parts,
    pooled_auc,
    row_ranks,
)

__all__ = ["ndcg_gaps", "auc_gaps", "ranking_duality", "skew_report",
           "properness_table", "hard_bridge", "diagnose_scores"]


def _cov(a: np.ndarray, b: np.ndarray) -> float:
    """Uniform (population) covariance."""
    return float(np.mean((a - a.mean()) * (b - b.mean())))


# -- collapse and non-collapse ------------------------------------------------------

def ndcg_gaps(sm: ScoreMatrix, weight="log", ties: str = "min") -> dict:
    """Differences between micro-nDCG, the class mean and the exemplar mean.

    Classes and examples without a relevant cell are left out, as Theorem
    ``dcg_collapse`` assumes positive ideal denominators; with them excluded,

    * ``micro - exemplar = Cov(IDCG_i, nDCG_i) / mean(IDCG_i)`` (part b);
    * ``micro - class = sum_j T_j (1/B - 1/(m N_j))`` (part c), whose terms are
      ``class_contributions``;
    * ``class - exemplar = sum_ij D_ij (1/(m N_j) - 1/(n b_i))`` (part d).

    ``exemplar_zero`` is the exemplar mean with empty examples counted as ``0``,
    the default of :func:`~aggmetrics.scores.ndcg`.
    """
    w = _weight(weight)
    D = sm.Y * w(row_ranks(sm, ties))
    k = sm.Y.sum(axis=1)
    b = _ideal(k, w)
    N = sm.Y.sum(axis=0) * float(w(np.array([1.0]))[0])
    rows, cols = b > 0, N > 0
    if not rows.any():
        raise ValueError("nDCG gaps need at least one relevant cell")
    Dk = D[rows][:, cols]
    bk, Nk = b[rows], N[cols]
    m, n, B = int(cols.sum()), int(rows.sum()), float(bk.sum())
    r = Dk.sum(axis=1) / bk
    T = Dk.sum(axis=0)
    micro = float(T.sum() / B)
    class_mean = float(np.mean(T / Nk))
    exemplar = float(r.mean())
    contrib = np.full(sm.m, np.nan)
    contrib[cols] = T * (1.0 / B - 1.0 / (m * Nk))
    coef = 1.0 / (m * Nk)[None, :] - 1.0 / (n * bk)[:, None]
    return dict(
        micro=micro, class_mean=class_mean, exemplar=exemplar,
        exemplar_zero=float(r.sum() / sm.n),
        micro_minus_exemplar=micro - exemplar,
        covariance=_cov(bk, r), mean_ideal=float(bk.mean()),
        micro_minus_class=micro - class_mean, class_contributions=contrib,
        class_minus_exemplar=float((Dk * coef).sum()),
        excluded_classes=np.flatnonzero(~cols), excluded_examples=int((~rows).sum()),
    )


def auc_gaps(sm: ScoreMatrix) -> dict:
    """Differences between micro-AUC, macro-AUC and the exemplar mean.

    Classes whose AUC is undefined (no positives or no negatives) are left out;
    the exemplar mean runs over the examples with a positive among the remaining
    classes. Then (Theorem ``auc_collapse``)

    * ``micro - macro = Cov_j(n_j^+, AUC_j) / mean(n_j^+)`` (part b), whose terms
      ``n_j^+ (AUC_j - macro) / sum n^+`` are ``class_contributions``;
    * ``micro - exemplar = Cov_i(k_i, AUC_i) / mean(k_i)`` (part c);
    * ``macro - exemplar = sum_{Y_ij=1} A_ij (1/(m n_j^+) - 1/(n k_i))`` (part d).

    Examples
    --------
    >>> import numpy as np
    >>> from aggmetrics.scores import ScoreMatrix
    >>> Y = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0]])
    >>> P = np.array([[.7, .6, .1], [.6, .1, .4], [.8, .2, .7], [.4, .3, .2]])
    >>> g = auc_gaps(ScoreMatrix(Y, P))
    >>> round(g["micro_minus_macro"], 6)    # 7/9 - 13/18
    0.055556
    >>> round(g["class_covariance"] / g["mean_positives"], 6)
    0.055556
    """
    a = auc_parts(sm, "class")
    cols = ~np.isnan(a)
    if not cols.any():
        raise ValueError("AUC gaps need a class with positives and negatives")
    Y = sm.Y[:, cols]
    A = np.where(Y == 1, auc_cells(sm)[:, cols], 0.0)
    npos, aj = Y.sum(axis=0), a[cols]
    k = Y.sum(axis=1)
    rows = k > 0
    ai = A[rows].sum(axis=1) / k[rows]
    m, n = int(cols.sum()), int(rows.sum())
    micro = float((npos * aj).sum() / npos.sum())
    macro = float(aj.mean())
    exemplar = float(ai.mean())
    contrib = np.full(sm.m, np.nan)
    contrib[cols] = npos * (aj - macro) / npos.sum()
    coef = 1.0 / (m * npos)[None, :] - 1.0 / (n * k[rows])[:, None]
    return dict(
        micro=micro, macro=macro, exemplar=exemplar,
        micro_minus_macro=micro - macro,
        class_covariance=_cov(npos, aj), mean_positives=float(npos.mean()),
        class_contributions=contrib,
        micro_minus_exemplar=micro - exemplar,
        example_covariance=_cov(k[rows], ai), mean_relevant=float(k[rows].mean()),
        macro_minus_exemplar=float((A[rows] * coef).sum()),
        excluded_classes=np.flatnonzero(~cols),
    )


def ranking_duality(sm: ScoreMatrix, weight="log", ties: str = "min") -> dict:
    """Pairs of labels that per-class nDCG and per-class AUC order oppositely.

    Both are conditional means over the positives of a label (Theorem ``duality``),
    of a row-rank quality for nDCG and a column-rank quality for AUC, and they can
    disagree. ``tau`` is ``(concordant - discordant) / compared`` over the pairs
    where both summaries are defined and neither ties.
    """
    nd = ndcg_parts(sm, "class", weight, ties, empty="nan")
    au = auc_parts(sm, "class")
    ok = np.flatnonzero(~np.isnan(nd) & ~np.isnan(au))
    disc, conc = [], 0
    for x, j in enumerate(ok):
        for k in ok[x + 1:]:
            s = np.sign(nd[j] - nd[k]) * np.sign(au[j] - au[k])
            if s < 0:
                disc.append((int(j), int(k)))
            elif s > 0:
                conc += 1
    compared = conc + len(disc)
    return dict(ndcg=nd, auc=au, discordant=disc, concordant=conc,
                compared=compared,
                tau=(conc - len(disc)) / compared if compared else float("nan"))


# -- skew --------------------------------------------------------------------------

_OWN = ("alpha", "beta", "auc", "ndcg", "brier")
_GLOBAL = ("brier_micro", "auc_macro", "auc_micro", "ndcg_micro", "ndcg_macro",
           "ndcg_exemplar")
#: Per-class summaries that replicating the positives of their own class leaves
#: unchanged (Propositions ``ndcg_class_skew`` and ``auc_skew``; ``alpha_j`` and
#: ``beta_j`` are class-conditional by construction).
SKEW_INVARIANT = ("alpha", "beta", "auc", "ndcg")


def _summaries(sm: ScoreMatrix, weight, ties) -> dict:
    ce = sm.conditional_errors("squared")
    return dict(
        alpha=ce.alpha, beta=ce.beta, auc=auc_parts(sm, "class"),
        ndcg=ndcg_parts(sm, "class", weight, ties, empty="nan"),
        brier=sm.parts("squared"),
        brier_micro=sm.brier(), auc_macro=auc(sm, "macro"),
        auc_micro=auc(sm, "micro"),
        ndcg_micro=ndcg(sm, "micro", weight, ties),
        ndcg_macro=float(np.nanmean(ndcg_parts(sm, "class", weight, ties, "nan"))),
        ndcg_exemplar=float(ndcg_parts(sm, "example", weight, ties).mean()),
    )


def skew_report(sm: ScoreMatrix, factor: int = 2, weight="log",
                ties: str = "min") -> dict:
    """What moves when the positive examples of one class are replicated.

    For each class ``j`` with positives and negatives, the rows with ``Y_ij = 1``
    are repeated ``factor`` times and every summary is recomputed. The result maps

    * ``'own'``: for each per-class summary (``alpha``, ``beta``, ``auc``, ``ndcg``,
      ``brier``), the absolute change of the entry of class ``j``, one value per
      class (``nan`` for classes that cannot be skewed). The first four are
      invariant (``SKEW_INVARIANT``); the per-class Brier score is the mixture
      ``pi_j alpha_j + (1 - pi_j) beta_j`` and moves with ``pi_j``.
    * ``'others'``: the largest change of the same summary over the other classes.
      Replicating whole rows changes the positives or the negatives of the other
      classes too, so a nonzero value here is the coupling of the classes, not a
      failure of invariance; the paper's per-class convention holds the other
      classes fixed instead.
    * ``'global'``: the change of the aggregate summaries (micro Brier, macro and
      micro AUC, micro, macro and exemplar nDCG).
    """
    if int(factor) != factor or factor < 2:
        raise ValueError("factor must be an integer >= 2")
    base = _summaries(sm, weight, ties)
    pos = sm.Y.sum(axis=0)
    own = {s: np.full(sm.m, np.nan) for s in _OWN}
    others = {s: np.full(sm.m, np.nan) for s in _OWN}
    glob = {s: np.full(sm.m, np.nan) for s in _GLOBAL}
    for j in range(sm.m):
        if pos[j] == 0 or pos[j] == sm.n:
            continue
        reps = np.where(sm.Y[:, j] == 1, int(factor), 1)
        sk = ScoreMatrix(np.repeat(sm.Y, reps, axis=0), np.repeat(sm.P, reps, axis=0),
                         task="multilabel")
        new = _summaries(sk, weight, ties)
        rest = np.arange(sm.m) != j
        for s in _OWN:
            d = np.abs(new[s] - base[s])
            own[s][j] = d[j]
            others[s][j] = np.max(np.nan_to_num(d[rest]), initial=0.0)
        for s in _GLOBAL:
            glob[s][j] = abs(new[s] - base[s])
    return dict(factor=int(factor), own=own, others=others, globals=glob)


# -- properness --------------------------------------------------------------------

_RANKING_PROPERNESS = (
    dict(name="dcg", family="ranking", proper=True, strictly_proper=False,
         note=("proper among reports with distinct entries; any order-preserving "
               "transformation of the truth scores the same (Prop. disc_improper)")),
    dict(name="ndcg", family="ranking", proper=False, strictly_proper=False,
         note=("normalising by IDCG_i, which depends on the random k_i, breaks "
               "properness (Remark ndcg_not_proper)")),
    dict(name="precision_at_k", family="ranking", proper=True, strictly_proper=False,
         note="a positional operator with top-k weight; as for dcg"),
    dict(name="auc", family="ranking", proper=None, strictly_proper=False,
         note=("truthful marginals maximise expected AUC when the labels of "
               "different examples are independent, not in general; never strictly "
               "proper (Prop. auc_improper)")),
)


def properness_table(ops: Optional[Sequence] = None) -> List[dict]:
    """Properness of the deviation operators (checked numerically) and of the
    ranking operators (as stated in the companion paper).

    ``ops`` adds deviation operators (names, :class:`Deviation` objects or
    callables) to the registered ones. For ranking operators ``proper=None`` means
    that it depends on the dependence between the labels.
    """
    out = []
    for op in list(list_deviations()) + list(ops or ()):
        dev = get_deviation(op)
        chk = check_properness(dev)
        out.append(dict(name=dev.name, family="deviation", proper=chk.proper,
                        strictly_proper=chk.strictly_proper,
                        note=f"largest gap {chk.max_gap:.3g} on the grid"))
    return out + [dict(r) for r in _RANKING_PROPERNESS]


# -- the hard case -----------------------------------------------------------------

def hard_bridge(sm: ScoreMatrix, threshold: Optional[float] = None) -> dict:
    """Conditional errors of 0/1 scores against the rates of the indicator matrix.

    With scores in ``{0, 1}`` the squared and the absolute deviation of a cell are
    both its error indicator, so ``alpha_j = fn_j / (tp_j + fn_j) = 1 - recall_j``
    and ``beta_j = fp_j / (fp_j + tn_j) = 1 - specificity_j``. ``threshold``
    binarises the scores first (``P >= threshold``); without it the scores must
    already be 0/1. ``consistent`` is true when both identities hold.
    """
    P = sm.P if threshold is None else (sm.P >= threshold).astype(float)
    if not np.isin(P, (0.0, 1.0)).all():
        raise ValueError("scores must be 0/1, or pass threshold=")
    hard = ScoreMatrix(sm.Y, P, task="multilabel")
    ce = hard.conditional_errors("squared")
    tp, fp, fn, tn = np.asarray(
        IndicatorMatrix(sm.Y, P, task="multilabel").per_class(), float).T
    with np.errstate(invalid="ignore", divide="ignore"):
        miss = np.where(tp + fn > 0, fn / (tp + fn), np.nan)
        fpr = np.where(fp + tn > 0, fp / (fp + tn), np.nan)
    consistent = bool(np.allclose(ce.alpha, miss, equal_nan=True)
                      and np.allclose(ce.beta, fpr, equal_nan=True))
    return dict(alpha=ce.alpha, beta=ce.beta, recall=1 - miss, specificity=1 - fpr,
                consistent=consistent)


# -- the runner --------------------------------------------------------------------

def diagnose_scores(sm: ScoreMatrix, *, weight="log", ties: str = "min",
                    tol: float = 1e-3, emit_warnings: bool = False
                    ) -> List[Finding]:
    """Run the score diagnostics and return :class:`Finding` objects.

    Codes: ``'degenerate_murphy'``, ``'undefined_auc'``, ``'ndcg_no_collapse'``,
    ``'auc_no_collapse'``, ``'pooled_auc_differs'``, ``'ranking_duality'`` and
    ``'class_coupling'``. Summaries closer than ``tol`` are treated as equal. With
    ``emit_warnings=True`` the findings of severity ``'warning'`` are also raised as
    :class:`~aggmetrics.diagnostics.MeasureWarning`.
    """
    out: List[Finding] = []

    # 1. unbinned Murphy decomposition that only restates the Brier score
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        murphy(sm)
    if any(issubclass(c.category, DegenerateDecompositionWarning) for c in caught):
        out.append(Finding(
            code="degenerate_murphy", severity="warning",
            message=("every score is distinct within its column, so the unbinned "
                     "Murphy decomposition gives REL = BS and RES = UNC; use "
                     "murphy(..., bins=k) or method='isotonic'"),
            detail=dict(n=sm.n, m=sm.m)))

    # 2. classes whose AUC is undefined
    pos = sm.Y.sum(axis=0)
    bad = np.flatnonzero((pos == 0) | (pos == sm.n))
    if bad.size:
        out.append(Finding(
            code="undefined_auc", severity="warning",
            message=(f"{bad.size} of {sm.m} classes have no positives or no "
                     "negatives; their AUC is undefined and they are left out of "
                     "the macro, micro and exemplar AUC"),
            detail=dict(classes=bad.tolist())))

    # 3. normalised nDCG summaries that do not coincide (Theorem dcg_collapse)
    if sm.Y.any():
        g = ndcg_gaps(sm, weight, ties)
        spread = max(g["micro"], g["class_mean"], g["exemplar"]) - min(
            g["micro"], g["class_mean"], g["exemplar"])
        if spread > tol:
            lead = int(np.nanargmax(np.abs(g["class_contributions"])))
            out.append(Finding(
                code="ndcg_no_collapse", severity="note",
                message=(f"micro-nDCG {g['micro']:.4f}, class mean "
                         f"{g['class_mean']:.4f} and exemplar mean "
                         f"{g['exemplar']:.4f} differ; micro - exemplar is "
                         f"Cov(IDCG_i, nDCG_i)/mean IDCG = "
                         f"{g['micro_minus_exemplar']:+.4f}, and class {lead} "
                         "contributes most to micro - class"),
                detail=g))

    # 4. micro-AUC and macro-AUC that differ, and pooled AUC (Theorem auc_collapse)
    if bad.size < sm.m:
        g = auc_gaps(sm)
        vals = (g["micro"], g["macro"], g["exemplar"])
        if max(vals) - min(vals) > tol:
            out.append(Finding(
                code="auc_no_collapse", severity="note",
                message=(f"micro-AUC {g['micro']:.4f}, macro-AUC {g['macro']:.4f} "
                         f"and exemplar mean {g['exemplar']:.4f} differ; micro - "
                         f"macro is Cov(n_j^+, AUC_j)/mean n^+ = "
                         f"{g['micro_minus_macro']:+.4f}"),
                detail=g))
        if sm.m > 1:
            pooled = pooled_auc(sm)
            if abs(pooled - g["micro"]) > tol:
                out.append(Finding(
                    code="pooled_auc_differs", severity="note",
                    message=(f"pooled ROC-AUC ({pooled:.4f}, scikit-learn's "
                             f"'micro') differs from micro-AUC ({g['micro']:.4f}): "
                             "it compares positives of one class with negatives of "
                             "another, so it depends on how the classes' scores "
                             "are scaled"),
                    detail=dict(pooled=pooled, micro=g["micro"])))

    # 5. per-class nDCG and AUC ordering the labels oppositely (Theorem duality)
    d = ranking_duality(sm, weight, ties)
    if d["discordant"]:
        out.append(Finding(
            code="ranking_duality", severity="note",
            message=(f"per-class nDCG and AUC order {len(d['discordant'])} of "
                     f"{d['compared']} label pairs oppositely (tau {d['tau']:+.3f}); "
                     "nDCG asks where a label sits among the labels of its "
                     "positives, AUC whether its positives outscore its negatives"),
            detail=d))

    # 6. labels that share examples: replicating the positives of one label also
    #    replicates positives of the others (skew_report shows by how much)
    multi = sm.Y.sum(axis=1) > 1
    if sm.m > 1 and multi.any():
        out.append(Finding(
            code="class_coupling", severity="note",
            message=(f"{int(multi.sum())} examples carry several labels, so "
                     "replicating the positives of one label also replicates "
                     "positives of others and moves their alpha, AUC and nDCG; "
                     "brier_under_prior and the skew-invariance of alpha_j, "
                     "beta_j, AUC_j and nDCG_j hold class by class"),
            detail=dict(examples=int(multi.sum()))))

    if emit_warnings:
        for f in out:
            if f.severity == "warning":
                warnings.warn(f"{f.code}: {f.message}", MeasureWarning, stacklevel=2)
    return out
