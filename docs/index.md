# aggmetrics

**Aggregating classification performance measures for machine learning.**

`aggmetrics` extends any binary performance measure of the four confusion-matrix counts
to multiclass, multilabel, ordinal, cost-sensitive, soft-label and cross-validation settings,
by choosing an *aggregation operator*: micro, macro, weighted, exemplar, or pooled over folds.
It also stores what is known about each measure, so it can tell you when a number is
misleading.

```python
import numpy as np
from aggmetrics import IndicatorMatrix, diagnose

y = np.arange(500) % 10
yhat = (y + 1) % 10                      # wrong on every example
im = IndicatorMatrix(y, yhat, task="multiclass")

im.accuracy()                            # 0.0
im.compute("label_accuracy", "micro")    # 0.8   (!)
diagnose(im)[0]                          # [warning] inflated_floor: ...
```

## Where to start

**New here?** Read the {doc}`tour`, then the notebooks in the {doc}`gallery/index`:
each starts from a mistake that is easy to make and shows what `aggmetrics` reports.

- **I want to know which number to report** → {doc}`user_guide/index`
- **I want the function signatures** → {doc}`api/index`
- **I use an AI coding assistant** → {doc}`agents`
- **I want to know why it is correct** → {doc}`theory`

```{toctree}
:hidden:
:maxdepth: 2

installation
tour
gallery/index
user_guide/index
api/index
agents
theory
contributing
changelog
```
