# Comparing models

Two models often disagree about which is better: one wins on $F_1$, the other on accuracy.
`pareto_dominance` decides whether one model beats the other on *every* measure that is
monotone in the confusion matrix:

```python
from aggmetrics import pareto_dominance

pareto_dominance(im_a, im_b)
# 'A_dominates_B', 'B_dominates_A', 'equivalent' or 'incomparable'
```

`'incomparable'` means the ranking depends on which measure you choose, so say which one.
For multiclass problems the comparison is made on the off-diagonal conditional error rates
and the per-class recalls.

For a comparison that accounts for resampling noise, evaluate each model on the same
cross-validation folds with `fold_aggregate(..., pooling="pooled")` and compare the gap with
the standard deviation across folds, as in the
{doc}`multilabel notebook <../gallery/03_multilabel_ranking>`.

:::{note}
**Sketch.** To write: a worked example of dominance and incomparability; paired comparisons
across folds (significance tests are not implemented).
:::
