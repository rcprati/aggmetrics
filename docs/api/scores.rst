Scores (experimental)
=====================

Calibration and ranking of unthresholded scores. The API may change in a later 0.x release.

.. automodule:: aggmetrics.scores
   :members: ScoreMatrix, ConditionalErrors, Deviation, power, get_deviation, list_deviations, check_properness, Properness, murphy, Murphy, DegenerateDecompositionWarning, brier_under_prior, class_weighted_brier
   :show-inheritance:

Ranking
-------

.. autofunction:: aggmetrics.scores.row_ranks
.. autofunction:: aggmetrics.scores.log_weight
.. autofunction:: aggmetrics.scores.geometric
.. autofunction:: aggmetrics.scores.top_k
.. autofunction:: aggmetrics.scores.dcg
.. autofunction:: aggmetrics.scores.dcg_parts
.. autofunction:: aggmetrics.scores.ndcg
.. autofunction:: aggmetrics.scores.ndcg_parts
.. autofunction:: aggmetrics.scores.precision_at_k
.. autofunction:: aggmetrics.scores.auc_cells
.. autofunction:: aggmetrics.scores.auc_parts
.. autofunction:: aggmetrics.scores.auc
.. autofunction:: aggmetrics.scores.pooled_auc

Diagnostics
-----------

.. autofunction:: aggmetrics.scores.diagnose_scores
.. autofunction:: aggmetrics.scores.ndcg_gaps
.. autofunction:: aggmetrics.scores.auc_gaps
.. autofunction:: aggmetrics.scores.ranking_duality
.. autofunction:: aggmetrics.scores.skew_report
.. autofunction:: aggmetrics.scores.properness_table
.. autofunction:: aggmetrics.scores.hard_bridge
