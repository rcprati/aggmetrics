# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
aggmetrics.scores --- calibration and ranking of unthresholded scores (experimental).

A :class:`ScoreMatrix` holds actual labels and predicted scores in ``[0, 1]``.
Deviation operators (squared, absolute, log, power, or a user-defined ``phi``)
turn each cell into a loss; the Brier score, the class-conditional errors
``alpha_j`` and ``beta_j``, and the Murphy decompositions follow.

Ranking operators order the labels within each example (the DCG family:
:func:`dcg`, :func:`ndcg`, :func:`precision_at_k`) or the examples within each
label (the AUC family: :func:`auc`, :func:`pooled_auc` for contrast).

:func:`diagnose_scores` explains why these summaries disagree: the covariances
behind the micro, class and exemplar gaps, labels that nDCG and AUC order
oppositely, the summaries that move under a change of prior, and properness.
"""
from .calibration import (
    DegenerateDecompositionWarning,
    Murphy,
    brier_under_prior,
    class_weighted_brier,
    murphy,
)
from .deviation import (
    Deviation,
    Properness,
    check_properness,
    get_deviation,
    list_deviations,
    power,
)
from .diagnostics import (
    auc_gaps,
    diagnose_scores,
    hard_bridge,
    ndcg_gaps,
    properness_table,
    ranking_duality,
    skew_report,
)
from .matrix import ConditionalErrors, ScoreMatrix
from .ranking import (
    auc,
    auc_cells,
    auc_parts,
    dcg,
    dcg_parts,
    geometric,
    log_weight,
    ndcg,
    ndcg_parts,
    pooled_auc,
    precision_at_k,
    row_ranks,
    top_k,
)

__all__ = [
    "ConditionalErrors",
    "DegenerateDecompositionWarning",
    "Deviation",
    "Murphy",
    "Properness",
    "ScoreMatrix",
    "auc",
    "auc_cells",
    "auc_gaps",
    "auc_parts",
    "brier_under_prior",
    "check_properness",
    "class_weighted_brier",
    "dcg",
    "dcg_parts",
    "diagnose_scores",
    "geometric",
    "get_deviation",
    "hard_bridge",
    "list_deviations",
    "log_weight",
    "murphy",
    "ndcg",
    "ndcg_gaps",
    "ndcg_parts",
    "pooled_auc",
    "power",
    "precision_at_k",
    "properness_table",
    "ranking_duality",
    "row_ranks",
    "skew_report",
    "top_k",
]
