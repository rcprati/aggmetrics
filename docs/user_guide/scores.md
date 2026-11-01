# Evaluating scores: calibration and ranking

The rest of the package scores decisions. `aggmetrics.scores` scores the unthresholded
predictions in `[0, 1]` that come before the decision: how close they are to the labels
(calibration) and how they order labels or examples (ranking). **The subpackage is
experimental**: names and signatures may change in a later 0.x release. It needs only NumPy.

The running example below has four examples and three labels.

```python
import numpy as np
from aggmetrics.scores import ScoreMatrix

Y = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 1, 0]])
P = np.array([[.7, .6, .1], [.6, .1, .4], [.8, .2, .7], [.4, .3, .2]])
sm = ScoreMatrix(Y, P)            # binary, multiclass (1-D labels) or multilabel
```

## Calibration

A deviation operator turns each cell into a loss `phi(Y - P)`. The Brier score is the squared
deviation, and **its micro, macro and exemplar averages are the same number**, because every
average divides by a fixed count.

```python
sm.brier()                                 # 0.270833 = 13/48, for any aggregation
ce = sm.conditional_errors()               # alpha_j: loss on positives, beta_j: on negatives
ce.alpha, ce.beta, ce.prevalence
```

The per-class Brier score is the mixture `pi_j alpha_j + (1 - pi_j) beta_j`. `alpha_j` and
`beta_j` do not change when the prevalence of the class does, so `brier_under_prior(sm, pi)`
gives the per-class score at another prevalence and `class_weighted_brier` a weighted version.
Operators are `"squared"`, `"absolute"`, `"log"` (infinite when a score is exactly wrong),
`power(s)`, or any function `phi`; `check_properness` tests numerically whether an operator is
(strictly) proper.

`murphy` splits the Brier score of each column (or row) as `BS = UNC - RES + REL`:

```python
from aggmetrics.scores import murphy

murphy(sm, bins=5)                         # exact for the binned predictor; binning_gap reported
murphy(sm, method="isotonic")              # CORP: isotonic recalibration, no bins
```

Without bins the decomposition uses the level sets of the scores. When every score in a
column is distinct, each level set is a single cell, so `REL = BS` and `RES = UNC`: the
decomposition restates the Brier score, and `murphy` warns (`DegenerateDecompositionWarning`).
Use `bins=` or the isotonic method instead. The isotonic method is the CORP approach of
Dimitriadis, Gneiting and Jordan (2021); if you need plots, quantile or expectile scores for a
single target, [model-diagnostics](https://lorentzenchr.github.io/model-diagnostics/) covers
them. This subpackage adds the multilabel aggregation, the class-conditional errors and the
ranking operators.

## Ranking

Within a row, the labels of one example are ordered by score (`row_ranks`, with `ties="min"`,
`"average"` or `"ordinal"`). The DCG family weighs each relevant label by `w(rank)`:

```python
from aggmetrics.scores import dcg, ndcg, ndcg_parts, precision_at_k, geometric, top_k

dcg(sm)                                    # raw: micro, macro and exemplar are equal
ndcg(sm, "micro"), ndcg(sm, "macro"), ndcg(sm, "exemplar")   # .8348, .7394, .7827: they differ
ndcg_parts(sm, "class")                    # [1, .5873, .6309]
ndcg(sm, weight=geometric(0.8))            # rank-biased precision weights
```

Within a column, the examples are ordered by their score for one label. The concordance cell
of a positive is the share of the column's negatives it outranks, with half credit for ties:

```python
from aggmetrics.scores import auc, auc_parts, pooled_auc

auc_parts(sm)                              # AUC_j = [1/2, 1, 2/3]
auc(sm, "macro"), auc(sm, "micro"), auc(sm, "exemplar")      # 13/18, 7/9, 5/6
pooled_auc(sm)                             # 43/72, scikit-learn's micro ROC-AUC
```

`auc(..., "micro")` weights each `AUC_j` by its number of positives. It is not the pooled
ROC-AUC, which compares a positive of one label with a negative of another and therefore
depends on how the labels' scores are scaled.

## Why the summaries disagree

`diagnose_scores` returns the same kind of `Finding` objects as `diagnose`:

```python
from aggmetrics.scores import diagnose_scores

for f in diagnose_scores(sm):
    print(f)
```

and the functions behind it give the numbers:

- `ndcg_gaps(sm)` and `auc_gaps(sm)` write the micro, class and exemplar differences as the
  covariances of the theorems (for example micro minus macro AUC is
  `Cov(n_j^+, AUC_j) / mean(n_j^+)`), with the contribution of each label.
- `ranking_duality(sm)` lists the pairs of labels that per-class nDCG and per-class AUC order in
  opposite ways. In the running example label A is ranked first in every row (`nDCG = 1`) but its
  positives do not outscore its negatives (`AUC = 1/2`); label B is the reverse.
- `skew_report(sm)` replicates the positives of each label and records what moves.
  `alpha_j`, `beta_j`, `AUC_j` and per-class nDCG do not; micro AUC, exemplar nDCG and the
  per-class Brier score do. Replicating whole rows also moves the other labels when examples carry
  several labels, and the report says by how much.
- `properness_table()` states which operators are proper: Brier and log strictly, absolute error
  not, raw DCG and AUC only weakly (they ignore order-preserving distortions), nDCG not in general.
- `hard_bridge(sm, threshold=.5)` checks that on 0/1 scores `alpha_j = 1 - recall_j` and
  `beta_j = 1 - specificity_j`, so the hard measures of the rest of the package are the special case.

## Limits

- The row ranks compare every pair of labels of an example, `O(n m^2)` time and memory, which suits
  up to a few hundred labels; the AUC cells cost `O(n log n)` per label.
- Scores must lie in `[0, 1]`. Regression, time series and multivariate scores are out of scope.
- For one-hot multiclass labels the usual multiclass Brier score divides by `n` only and is `m`
  times the value reported here; use `sm.loss("squared", scale="sum")` for that convention.
