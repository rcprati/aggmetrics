# What if the classes were more or less frequent?

A measure that is not skew-invariant depends on how often each class occurs, so the same
classifier scores differently where the prevalence differs (Theorem 5 and Proposition 6 of the
companion paper). `under_prior` answers "what would this measure be at another prior?" by
keeping the rates of the data and replacing the class sizes.

```python
import numpy as np
from aggmetrics import IndicatorMatrix, under_prior

y = np.array([1] * 80 + [0] * 120)                       # 40% positives
p = np.array([1] * 60 + [0] * 20 + [1] * 30 + [0] * 90)
im = IndicatorMatrix(y, p, task="binary")

im.compute("precision")                                  # 0.667
shifted = under_prior(im, 0.05)                          # 5% positives
shifted.compute("precision")                             # 0.136
shifted.compute("recall") == im.compute("recall")        # True: skew-invariant
```

The priors are, for a binary problem, the prevalence of the positive class; for a multiclass problem,
the frequency of each class (non-negative, summing to 1); for a multilabel problem, one prevalence
per label. The result offers `compute`, `accuracy`, `counts` and `per_class`. At the prior of the
data itself it reproduces the original values exactly, for every measure, operator and convention.

Recall and specificity do not move, and precision, $F_1$ and MCC do. Notebook 06 uses this to
simulate a long-tailed deployment from a balanced test set, and finds the imbalance at which two
models swap places.

## The assumption

The rates of each class are kept, and only its size changes. This is label shift. It holds when the
class-conditional distribution of the inputs is the same in the two settings, and it does not hold
when the change of prior also changes what the examples look like. A class absent from the data cannot
be given a positive prior, because its rates are unknown. The exemplar operator is not available: it
depends on the examples and not on the confusion matrix. Masks, soft labels and ordinal data are not
supported.
