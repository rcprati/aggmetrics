# Comparing models

Two models often disagree about which is better: one wins on $F_1$, the other on precision.
`pareto_dominance` decides whether one model beats the other on *every* measure that is
monotone in the confusion matrix, which settles the question whatever measure is chosen:

```python
from aggmetrics import pareto_dominance

pareto_dominance(im_a, im_b)
# 'A_dominates_B', 'B_dominates_A', 'equivalent' or 'incomparable'
```

## Dominance

Binary case, ten examples (four negatives, six positives). Model A makes one error. Model B
makes three, one of them a false positive. A is no worse in any of the four counts:

| | tp | fp | fn | tn |
|---|---|---|---|---|
| A | 5 | 0 | 1 | 4 |
| B | 4 | 1 | 2 | 3 |

```python
from aggmetrics import IndicatorMatrix, pareto_dominance

y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
im_a = IndicatorMatrix(y, [0, 0, 0, 0, 1, 1, 1, 1, 1, 0], task="binary")
im_b = IndicatorMatrix(y, [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], task="binary")
pareto_dominance(im_a, im_b)    # 'A_dominates_B'
```

So A is better on precision, recall, $F_1$, accuracy, MCC and every other measure that rewards
true positives and true negatives and penalises the errors. No choice of measure can reverse
that ranking.

## Incomparable models

Now compare C, which never raises a false alarm but misses half the positives, with D, which finds
every positive at the cost of two false alarms:

| | tp | fp | fn | tn | Precision | Recall | $F_1$ | Accuracy |
|---|---|---|---|---|---|---|---|---|
| C | 3 | 0 | 3 | 4 | 1.00 | 0.50 | 0.67 | 0.70 |
| D | 6 | 2 | 0 | 2 | 0.75 | 1.00 | 0.86 | 0.80 |

```python
im_c = IndicatorMatrix(y, [0, 0, 0, 0, 1, 1, 1, 0, 0, 0], task="binary")
im_d = IndicatorMatrix(y, [0, 0, 1, 1, 1, 1, 1, 1, 1, 1], task="binary")
pareto_dominance(im_c, im_d)    # 'incomparable'
```

C wins on precision and D wins on recall, $F_1$ and accuracy. Neither is better in every count,
so the ranking depends on how the two kinds of error are weighted, that is, on the measure you
choose. When the answer is `'incomparable'`, say which measure you optimise and why; do not
report the one that happens to favour your model.

`'equivalent'` means the two confusion profiles are identical.

## Multiclass

For multiclass problems the comparison is made on the confusion rates: A dominates B when every
per-class recall of A is at least that of B and every off-diagonal conditional error rate of A is
at most that of B. This is a stricter condition than a higher macro score, so incomparability is
common. In the agent-skill example, a random forest beats a logistic regression on accuracy and
macro-$F_1$ and still is `'incomparable'` with it.

`IndicatorMatrix.dominates(other, strict=False)` is the underlying check; the two models must have
the same task, number of classes and number of examples.

## Accounting for noise

Dominance on one test set is a statement about that test set. To compare models while accounting
for resampling, evaluate both on the *same* cross-validation folds, pool the predictions with
`fold_aggregate(..., pooling="pooled")`, and compare the gap between the models with the
standard deviation across folds, as in the
{doc}`multilabel notebook <../gallery/03_multilabel_ranking>`. `aggmetrics` does not implement
significance tests; per-fold values of a measure are available from each fold's
`IndicatorMatrix.compute` if you want to apply one.
