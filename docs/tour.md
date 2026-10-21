# A five-minute tour

## Indicator matrices

Give `IndicatorMatrix` the true and predicted labels. It accepts class indices, binary
indicator matrices, probability matrices and ordinal classes, and converts them once.

```python
import numpy as np
from aggmetrics import IndicatorMatrix, diagnose

y_true = np.array([0, 1, 2, 2, 1, 0])
y_pred = np.array([0, 2, 2, 1, 1, 1])
im = IndicatorMatrix(y_true, y_pred, task="multiclass")
```

## Any measure under any operator

```python
im.compute("f1", "macro")      # 0.5222
im.compute("f1", "micro")      # 0.5      equals accuracy on single-label data
im.compute("mcc", "macro")     # 0.2942   macro-MCC: no standard implementation exists
im.accuracy()                  # 0.5
```

`report()` tabulates measures against operators (a pandas `DataFrame` when pandas is
installed):

```python
im.report()
#                    micro  macro  exemplar
# precision          0.500  0.611     0.500
# recall             0.500  0.500     0.500
# f1                 0.500  0.522     0.500
# ...
```

The operators and the 21 built-in measures are described in the {doc}`user_guide/aggregation`
and {doc}`user_guide/measures` pages.

## Ask the library whether a number can be trusted

```python
for finding in diagnose(im):
    print(finding)
```

`diagnose` returns structured `Finding` objects. The checks are listed in
{doc}`user_guide/diagnostics`. The notebooks in the {doc}`gallery <gallery/index>` show them on
real data.

## Properties are data

```python
from aggmetrics import get_measure

get_measure("mcc").decomposable             # False
get_measure("recall").bayes_threshold()     # 0.0
get_measure("mcc").attainable_floor(10)     # -0.111, not -1
```

## Your own measure

```python
from aggmetrics import register_measure

def prevalence_threshold(tp, fp, fn, tn):
    r, s = tp / (tp + fn), tn / (tn + fp)
    return (np.sqrt(r * (1 - s)) + s - 1) / (r + s - 1)

m = register_measure("prevalence_threshold", prevalence_threshold)
m.decomposable        # False, inferred from the formula
m.skew_invariant      # True
im.compute("prevalence_threshold", "exemplar")
```

## Inside scikit-learn

```python
from aggmetrics import make_scorer
from sklearn.model_selection import cross_val_score

cross_val_score(clf, X, y, cv=5, scoring=make_scorer("mcc", "macro", task="multiclass"))
```
