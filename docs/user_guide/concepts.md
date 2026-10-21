# Concepts

**Indicator matrices.** For $n$ examples and $m$ classes or labels, the true and predicted
labels are matrices $Y, \hat Y \in \{0,1\}^{n \times m}$ (real-valued for soft
labels). For ordinal data with $m$ classes, $Y$ has $m-1$ columns of cumulative thresholds
$\mathbb{I}(y_i > j)$.

**Cell matrices.** Four matrices are formed entry by entry: true positives
$Y \wedge \hat Y$, false positives, false negatives and true negatives.

**Aggregation operators.** A binary measure $M(tp, fp, fn, tn)$ becomes a measure on the
whole problem once the cell matrices are reduced:

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

:::{note}
**Sketch.** To write: one worked example per operator on a four-class problem; the relation
between the operators and the `average=` argument of `scikit-learn`
(`'micro'`, `'macro'`, `'weighted'` and `'samples'` are $\Sigma_1$, $\Sigma_m$ with uniform
weights, $\Sigma_m$ with prevalence weights and $\Sigma_n$).
:::
