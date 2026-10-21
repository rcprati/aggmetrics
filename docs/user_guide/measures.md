# Measures and their properties

The registry holds 21 binary measures. Names resolve through aliases
(`iou` reaches `jaccard`, `dice` reaches `f1`). The table is generated from the registry
when the documentation is built.

```{include} ../_generated/measures_table.md
```

**Reading the table.**

- *Decomposable*: numerator and denominator are linear in the counts, so micro-averaging
  is a weighted average of the per-class values.
- *Skew-invariant*: a function of recall and specificity only.
- *Depends on tn at tp=0*: the value when there are no true positives still depends on the
  true negatives. This is what lifts the floor in multiclass problems (see the
  {doc}`floor-trap notebook <../gallery/01_floor_trap>`).
- *Bayes threshold*: the decision threshold on the conditional probability that maximises
  the measure, when a closed form exists.

```python
from aggmetrics import get_measure, list_measures

list_measures(decomposable=True)
get_measure("mcc").decomposable
```

:::{note}
**Sketch.** To write: the formulas in one table; which measures need which convention for
undefined values; a short guide to choosing among the skew-invariant measures.
:::
