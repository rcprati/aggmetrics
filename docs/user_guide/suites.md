# Choosing which measures to report

:::{note}
This module is experimental. Its signatures may change in a later 0.x release.
:::

A confusion matrix has few degrees of freedom: two for a binary problem (recall and
specificity, with the class sizes fixed), $m(m-1)$ for $m$ classes, $2m$ for $m$ labels.
Every other measure is a function of those, so a table with accuracy, balanced accuracy and
$F_1$ next to recall and specificity adds no information. Remark 2 of the companion paper
turns this into a test: a suite of $k \le d$ measures is locally non-redundant if and only
if the Jacobian of the measures with respect to the free parameters has full row rank $k$.

`suite_rank` computes that Jacobian at the data you have and its rank. `select_suite` picks
the measures that carry the most information.

## Is this suite redundant?

Binary case, 200 examples: 80 positives, 60 of them found, and 30 false alarms among the 120
negatives.

```python
import numpy as np
from aggmetrics import IndicatorMatrix, suite_rank

y = np.array([1] * 80 + [0] * 120)
p = np.array([1] * 60 + [0] * 20 + [1] * 30 + [0] * 90)
im = IndicatorMatrix(y, p, task="binary")

suite_rank(im, ["recall", "specificity"], "micro").rank    # 2: a basis
suite_rank(im, ["recall", "fnr"], "micro").rank            # 1: fnr = 1 - recall
r = suite_rank(im, ["label_accuracy", "f1", "balanced_accuracy"], "micro")
r.rank, r.redundant                                         # 2, True: three measures, two dimensions
```

A measure is a name or a `(name, operator)` pair, and the operators are `'micro'`, `'macro'` and
`'weighted'`. The exemplar operator depends on the examples and not on the confusion matrix,
so it has no Jacobian of this kind.

## Which measures carry the most information?

```python
from aggmetrics import select_suite

select_suite(im, aggregation="micro")
# SuiteSelection(items=(('recall', 'micro'), ('specificity', 'micro')), volume=1.0, rank=2, ...)
select_suite(im, ["f1", "precision", "recall", "jaccard", "mcc"], aggregation="micro")
# the pair (precision, recall), volume 0.949
```

On the full registry the choice is the canonical basis of the paper, recall and specificity.
Restricted to the measures a venue asks for, it is precision and recall, whose gradients are
close to orthogonal.

The *information volume* is a score added by the package, not a result of the paper. The rows of
the Jacobian are scaled to unit length and the volume is the volume of the parallelepiped they
span. It lies in $[0, 1]$, it is 0 exactly when the suite is redundant, and it is 1 when the
gradients are mutually orthogonal. The selection maximises it: exhaustively when there are at
most 200,000 subsets and greedily otherwise. Ties are broken by the order of the candidates.

## What to keep in mind

- The Jacobian is local. It is evaluated at the confusion matrix of the data you pass, with the
  class sizes fixed, so it says how the measures respond to the classifier and not to a change of
  prevalence.
- With more than two classes, measures aggregated over classes span few of the directions of the
  matrix. In notebook 12, 63 measure-and-operator pairs together span 11 of the 90 directions of a
  ten-class problem, and the volume of the best suite falls quickly as it grows.
- Micro-averaged precision, recall and $F_1$ are the same number on one-hot data (Theorem 2), and
  `suite_rank` reports them as rank 1.
- Masks, soft labels (t-norms) and ordinal data are not supported.
