# Aggregation: operators, conventions and folds

## Operators

```python
im.compute("f1", "micro")       # pool all entries first
im.compute("f1", "macro")       # per class, uniform average
im.compute("f1", "weighted")    # per class, weighted by prevalence
im.compute("f1", "exemplar")    # per example
```

## Undefined entries

Per-class and per-example values divide before averaging, so an entry is undefined when its
denominator vanishes: precision of a class that is never predicted, $F_1$ of an example with
neither true nor predicted labels. The treatment is an explicit argument.

| `convention` | Undefined entries are |
|---|---|
| `'exclude'` (default) | left out of the average |
| `'zero'` | counted as 0 (this is `zero_division=0` in `scikit-learn`) |
| `'one'` | counted as 1 |
| `'nan'` | propagated: the result is NaN |

```python
im.compute("precision", "macro", convention="zero")
```

`diagnose` reports how much the choice matters for the data at hand
(finding `undefined_entries`). The {doc}`imbalance notebook <../gallery/02_micro_macro_imbalance>`
works through an example.

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

Folds may differ in size. They must share the same set of classes or labels.

:::{note}
**Sketch.** To write: when pooled and averaged agree (decomposable measures on folds of equal
size and class distribution); repeated cross-validation; confidence intervals across folds
(not implemented).
:::
