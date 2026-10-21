# Aggregation: operators, conventions and folds

## Operators

```python
im.compute("f1", "micro")       # pool all entries first
im.compute("f1", "macro")       # per class, uniform average
im.compute("f1", "weighted")    # per class, weighted by prevalence
im.compute("f1", "exemplar")    # per example
```

Which one to report depends on the question:

- **micro**: the quality of the individual label decisions, dominated by the frequent classes.
  On single-label data, micro-precision, micro-recall and micro-$F_1$ all equal accuracy.
- **macro**: the quality on a typical class, each class counting once. This is the operator
  that exposes failure on rare classes.
- **weighted**: macro with prevalence weights, so it stays between macro and micro.
- **exemplar**: the quality on a typical example. Meaningful for multilabel problems; on
  single-label data each example is either right or wrong.

The {doc}`concepts` page works one example through all of them.

## Undefined entries

Per-class and per-example values divide before averaging, so an entry is undefined when its
denominator vanishes: precision of a class that is never predicted, $F_1$ of an example with
neither true nor predicted labels. The definitions in {doc}`measures` give the condition for
each measure. The treatment is an explicit argument.

| `convention` | Undefined entries are |
|---|---|
| `'exclude'` (default) | left out of the average |
| `'zero'` | counted as 0 (this is `zero_division=0` in `scikit-learn`) |
| `'one'` | counted as 1 |
| `'nan'` | propagated: the result is NaN |

```python
im.compute("precision", "macro", convention="zero")
```

The choice can matter. In the {doc}`imbalance notebook <../gallery/02_micro_macro_imbalance>`,
two of ten classes are never predicted, and macro-precision is 0.938 with `'exclude'`, 0.751 with
`'zero'` and 0.951 with `'one'`. `diagnose` reports the size of the effect
(`undefined_entries`).

## Observation masks

A mask restricts every operator to observed entries, which covers partial labels,
semi-supervised evaluation and abstention in one mechanism:

```python
IndicatorMatrix(Y, P, task="multilabel", mask=observed)
```

## Cross-validation folds

Averaging a non-linear measure over folds is not the same as scoring all the held-out
predictions together. `fold_aggregate` does either:

```python
from aggmetrics import fold_aggregate

fold_aggregate(folds, "f1", "macro", pooling="pooled")    # concatenate the folds' predictions
fold_aggregate(folds, "f1", "macro", pooling="averaged")  # sample-weighted mean of fold scores
```

`folds` is a list with one `IndicatorMatrix` per test fold. Folds may differ in size, which is
what $k$-fold gives whenever $n$ is not a multiple of $k$. They must share the same classes or
labels (pass `n_classes` when building them if a rare class could be missing from a fold).

### When do pooled and averaged agree?

For a decomposable measure, pooled evaluation is a weighted average of the fold values, with
weights equal to the fold *denominators*. The `averaged` option weights by fold size. The two are
therefore identical when the denominator is proportional to fold size, and close when folds are
large, similar in size and similar in class distribution. On `digits` with five stratified
folds of about 360 examples:

| Measure | Pooled | Averaged |
|---|---|---|
| micro-recall | 0.96939 | 0.96939 |
| macro-recall | 0.96938 | 0.96933 |
| macro-$F_1$ | 0.96943 | 0.96939 |
| macro-MCC (not decomposable) | 0.96613 | 0.96631 |
| micro-MCC (not decomposable) | 0.96599 | 0.96599 |

Micro-recall on single-label data is accuracy, whose denominator is the fold size, so the two
agree exactly. The others differ in the fourth or fifth decimal here. The difference grows with
small folds and rare classes, because a per-fold ratio is then computed from few examples and some
folds may not define it at all; the {doc}`multilabel notebook <../gallery/03_multilabel_ranking>`
shows the effect on a real benchmark. Pooling also avoids fold-level undefined values: a class missing from one fold is still present in
the pooled counts.

### Repeated cross-validation

Pass the folds of all repeats in one list. Every example then appears once per repeat, and the
pooled value is the measure on all the predictions together (for the micro operator this equals
scoring the concatenated predictions, which the tests check). The exemplar operator treats each
appearance of an example as a separate row.

### Uncertainty

`aggmetrics` does not compute confidence intervals across folds. The spread of the per-fold
values, as in the {doc}`multilabel notebook <../gallery/03_multilabel_ranking>`, is a useful
indication, but the folds share training data, so their standard deviation understates the
uncertainty of the estimate. Treat it as a lower bound, not as a confidence interval.
