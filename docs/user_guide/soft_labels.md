# Soft labels and annotator proportions

When the reference is soft (annotator proportions, crowd labels), the Boolean conjunction that
forms the cell matrices is replaced by a t-norm:

```python
im = IndicatorMatrix(Y_soft, P_soft, task="multilabel", tnorm="product")
im.compute("f1", "exemplar")
```

Only the product t-norm preserves the row and column marginals. `marginal_defect` measures
the departure: it is 0 for the product and 0.5 for the Gödel and Łukasiewicz t-norms.

```python
from aggmetrics import marginal_defect

marginal_defect("product")      # 0.0
marginal_defect("godel")        # 0.5
```

The pointwise order of t-norms applies to all four cell matrices in the same direction, so it
says nothing about the direction in which a ratio of them moves.

:::{note}
**Sketch.** To write: a worked example with simulated annotators; custom t-norms
(`aggmetrics.tnorms.hamacher`); the relation to probabilistic confusion matrices.
:::
