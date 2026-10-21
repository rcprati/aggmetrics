# Concepts

**Indicator matrices.** For $n$ examples and $m$ classes or labels, the true and predicted
labels are matrices $Y, \hat Y \in \{0,1\}^{n \times m}$ (real-valued for soft labels). For
ordinal data with $m$ classes, $Y$ has $m-1$ columns of cumulative thresholds
$\mathbb{I}(y_i > j)$.

**Cell matrices.** Four matrices are formed entry by entry: true positives
$Y \wedge \hat Y$, false positives, false negatives and true negatives. Summing a cell
matrix over some axis gives the counts that a measure uses.

**Aggregation operators.** A binary measure $M(tp, fp, fn, tn)$ becomes a measure on the whole
problem once the cell matrices are reduced:

| Operator | Reduces | Name in the API |
|---|---|---|
| $\Sigma_1$ | all entries | `'micro'` |
| $\Sigma_m$ | down each column (per class or label) | `'macro'`, `'weighted'` |
| $\Sigma_n$ | along each row (per example) | `'exemplar'` |
| $\Sigma_k$ | across cross-validation folds | `fold_aggregate(..., pooling="pooled")` |

**Structural properties.** Each measure records whether it is *aggregation-decomposable*,
whether it is *skew-invariant*, and whether its value at $tp = 0$ depends on $tn$. These decide
which identities hold: for example, micro-averaging equals denominator-weighted macro-averaging
only for decomposable measures. See {doc}`measures`.

## A worked example

Four classes, ten examples. The classifier gets class 0 mostly right and confuses the others a
little:

```python
from aggmetrics import IndicatorMatrix

y_true = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]
y_pred = [0, 0, 0, 1, 1, 2, 2, 2, 3, 0]
im = IndicatorMatrix(y_true, y_pred, task="multiclass")
im.per_class()          # one row per class: tp, fp, fn, tn
```

| Class | tp | fp | fn | tn | Precision | Recall |
|---|---|---|---|---|---|---|
| 0 | 3 | 1 | 1 | 5 | 3/4 = 0.750 | 3/4 = 0.750 |
| 1 | 1 | 1 | 1 | 7 | 1/2 = 0.500 | 1/2 = 0.500 |
| 2 | 2 | 1 | 0 | 7 | 2/3 = 0.667 | 2/2 = 1.000 |
| 3 | 1 | 0 | 1 | 8 | 1/1 = 1.000 | 1/2 = 0.500 |
| **pooled** | 7 | 3 | 3 | 27 | 7/10 = 0.700 | 7/10 = 0.700 |

The three class-wise operators then differ only in how they combine the rows:

- **micro** applies the measure to the pooled counts: precision $7/10 = 0.7$.
- **macro** averages the per-class values with equal weights:
  $(0.750 + 0.500 + 0.667 + 1.000)/4 = 0.729$.
- **weighted** averages with weights equal to the number of true examples of each class,
  $(4, 2, 2, 2)$: $(0.750 \cdot 4 + 0.500 \cdot 2 + 0.667 \cdot 2 + 1.000 \cdot 2)/10 = 0.733$.

```python
im.compute("precision", "micro")      # 0.7
im.compute("precision", "macro")      # 0.7292
im.compute("precision", "weighted")   # 0.7333
im.compute("f1", "macro")             # 0.6792
```

The exemplar operator reduces along rows, so it needs several labels per example to be
interesting. Four examples with three labels:

```python
Y = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]
P = [[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 0]]
im = IndicatorMatrix(Y, P, task="multilabel")
im.parts("f1", "exemplar")            # [0.667, 0.667, 1.0, 0.0], one per example
im.compute("f1", "exemplar")          # 0.5833, the mean of those
im.compute("f1", "micro")             # 0.7273, over the same predictions
```

The same predictions give 0.73 or 0.58 depending on whether the unit of account is the label
decision or the example. Neither is wrong; they answer different questions.

## Relation to scikit-learn

The `average=` argument of `scikit-learn` corresponds exactly to these operators:

| `scikit-learn` `average` | `aggmetrics` operator |
|---|---|
| `'micro'` | `'micro'` ($\Sigma_1$) |
| `'macro'` | `'macro'` ($\Sigma_m$, uniform weights) |
| `'weighted'` | `'weighted'` ($\Sigma_m$, prevalence weights) |
| `'samples'` | `'exemplar'` ($\Sigma_n$) |

For $F_1$ and precision on the example above, `precision_score(y_true, y_pred, average=...)` and
`im.compute("precision", ...)` return the same values to the last digit. One difference to
keep in mind: when an entry is undefined, `scikit-learn` substitutes 0 (`zero_division=0`),
while `aggmetrics` leaves it out by default. Pass `convention="zero"` to match, as explained in
{doc}`aggregation`.
