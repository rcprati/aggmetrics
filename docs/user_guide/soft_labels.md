# Soft labels and annotator proportions

When the reference is soft (the proportion of annotators who chose a label, or a calibrated
probability), the Boolean conjunction that forms the cell matrices is replaced by a t-norm:

```python
im = IndicatorMatrix(Y_soft, P_soft, task="multilabel", tnorm="product")
im.compute("f1", "exemplar")
```

## A worked example

Five simulated annotators label 400 examples, each correct 80% of the time. The soft label is
the proportion of annotators voting positive; the model produces a score in $[0,1]$.

```python
import numpy as np
from aggmetrics import IndicatorMatrix

rng = np.random.default_rng(0)
n = 400
truth = (rng.random(n) < 0.3).astype(int)
annotators = np.where(rng.random((n, 5)) < 0.8, truth[:, None], 1 - truth[:, None])
soft_y = annotators.mean(axis=1)                       # in {0, 0.2, ..., 1}
score = np.clip(0.55 * truth + 0.45 * rng.random(n), 0, 1)

im = IndicatorMatrix(soft_y, score, task="binary", tnorm="product")
im.compute("f1", "micro")                              # 0.5331
```

Thresholding both at 0.5 and scoring the hard labels gives $F_1 = 0.892$. The two numbers are
not comparable, and the soft one is lower for a reason: with a t-norm, uncertainty counts as
partial error. A score of 0.6 against a label of 0.8 contributes 0.48 to the true positives and
parts of the mass to false positives and false negatives. The soft cell matrices are the
expected counts if the label and the prediction were independent Bernoulli variables with these
probabilities. They are not counts of examples, and ratios of expected counts are not expected
values of the ratios.

## Why the product

Only the product t-norm preserves the marginals: the soft true positives and false negatives
add up to the soft count of positives, whatever the predictions are. On this example:

| T-norm | $tp + fn$ | `marginal_defect` |
|---|---|---|
| product | 137.60 (equals `soft_y.sum()`) | 0.0 |
| Gödel (minimum) | 187.62 | 0.5 |
| Łukasiewicz | 87.58 | 0.5 |

With the Gödel or Łukasiewicz t-norms the number of positives implied by the cells is not the
number of positives in the data, so measures such as recall are computed against a different
population. `marginal_defect` returns the largest violation over a grid on $[0,1]^2$ and is 0
only for the product.

The pointwise order of t-norms (Łukasiewicz $\le$ product $\le$ Gödel) applies to all four cell
matrices in the same direction, so it says nothing about the direction in which a ratio of them
moves. On this example the $F_1$ is 0.627, 0.533 and 0.561 under the three.

## Other t-norms

Any callable $T(a, b)$ can be passed as `tnorm`. `aggmetrics.tnorms.hamacher(gamma)` builds the
Hamacher family; `gamma = 1` is the product, and other values depart from marginal consistency:

```python
from aggmetrics.tnorms import hamacher
from aggmetrics import marginal_defect

marginal_defect(hamacher(1.0))     # 0.0
marginal_defect(hamacher(2.0))     # 0.101
IndicatorMatrix(soft_y, score, task="binary", tnorm=hamacher(2.0))
```

## Soft labels are not the same as predicted probabilities

A predicted probability matrix passed *without* `tnorm` is thresholded (multilabel, at
`threshold`, 0.5 by default) or argmaxed (multiclass) before the cells are formed. Pass `tnorm`
only when you want the soft calculation above. Soft *predictions* with a hard reference are also
handled by the t-norm path: a hard label is a soft label in $\{0,1\}$.
