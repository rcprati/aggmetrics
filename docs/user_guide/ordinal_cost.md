# Ordinal and cost-sensitive evaluation

When the classes are ordered (disease grades, ratings), an error of one grade and an error of
four grades are not the same event. Two tools treat them differently: the ordinal encoding and
cost matrices.

The examples use 300 simulated patients graded 0 to 4, with grade 0 the most common, and a
classifier that is wrong on about 40% of them, usually by one or two grades.

```python
import numpy as np
from aggmetrics import IndicatorMatrix

rng = np.random.default_rng(1)
n = 300
true = rng.choice(5, n, p=[0.45, 0.25, 0.15, 0.10, 0.05])
pred = np.clip(true + rng.choice([-2, -1, 0, 0, 0, 1, 2], n), 0, 4)

im = IndicatorMatrix(true, pred, task="multiclass", n_classes=5)
im.accuracy()                              # 0.617
```

## Cost matrices

`cost(C)` gives the cost under an $m \times m$ matrix $C$, where $C_{jk}$ is the cost of
predicting class $k$ when the truth is $j$. With $C_{jk} = |j-k|$ and $(j-k)^2$ it recovers the
mean absolute and mean squared error:

```python
import numpy as np

J, K = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
im.cost(np.abs(J - K)) / im.n       # 0.520, the mean absolute error
im.cost((J - K) ** 2.0) / im.n      # 0.793, the mean squared error
im.cost(1 - np.eye(5)) / im.n       # 0.383, which is 1 - accuracy
```

The last line shows that accuracy is the cost matrix with 0 on the diagonal and 1 elsewhere:
an error costs the same whatever its size.

**Asymmetric costs.** If under-grading a patient is three times worse than over-grading,
put that in the matrix (rows are the true grade $j$, columns the prediction $k$):

```python
C = np.where(K < J, 3.0 * (J - K), (K - J).astype(float))
im.cost(C) / im.n                   # 0.873
```

Here the classifier under-graded 44 patients and over-graded 71, and the asymmetric cost
(0.873) is well above the symmetric one (0.520), because the matrix prices the errors that
matter.

The `aggregation` argument sets how the cost is reduced:

| `aggregation` | Result |
|---|---|
| `'micro'` | the total cost over all examples |
| `'exemplar'` | the mean cost per example (the mean absolute error for $|j-k|$) |
| `'macro'` | the mean, over the predicted classes, of each class's total cost |

## The ordinal encoding

With `task="ordinal"`, $m$ classes become $m-1$ cumulative threshold indicators: column $j$
says "the grade is above $j$". Each threshold is a binary problem, and a measure under the
macro operator averages over the thresholds:

```python
im_ord = IndicatorMatrix(true, pred, task="ordinal", n_classes=5)
im_ord.parts("recall", "macro")     # [0.837, 0.839, 0.816, 0.667], one per threshold
im_ord.compute("recall", "macro")   # 0.790
```

For the same predictions the multiclass macro-recall is 0.539. A prediction one grade off is
wrong as a class but still right at most thresholds: predicting 2 for a true 3 is correct on
"above 0", "above 1" and "above 3", and wrong only on "above 2". The ordinal view gives partial
credit to near misses, as the cost matrix does.

When the true labels are already cumulative thresholds, pass `n_classes` so the number of
grades is not inferred from the largest label.

## Which to use

- Use **accuracy and macro measures** when every error is equally bad.
- Use **a cost matrix** when you can say what each kind of error costs. It is the most direct
  statement of the loss.
- Use **the ordinal encoding** when the order matters but you cannot price the errors; the
  per-threshold values also show where on the scale the classifier is weakest.
