# Ordinal and cost-sensitive evaluation

**Ordinal classes.** With `task="ordinal"`, $m$ classes become $m-1$ cumulative threshold
indicators, so an error of one grade and an error of four grades are different events:

```python
im = IndicatorMatrix([0, 1, 2, 3], [0, 2, 2, 2], task="ordinal")
im.compute("recall", "macro")       # 0.6667
```

**Cost matrices.** For a multiclass problem, `cost(C)` gives the total or averaged cost under
an $m \times m$ matrix. With $C_{jk} = |j-k|$ and $(j-k)^2$ it recovers the mean absolute and
mean squared error:

```python
J, K = np.meshgrid(np.arange(im.m), np.arange(im.m), indexing="ij")
im.cost(np.abs(J - K)) / im.n       # mean absolute error
im.cost((J - K) ** 2.0) / im.n      # mean squared error
```

`aggregation` can be `'micro'`, `'macro'` or `'exemplar'`.

:::{note}
**Sketch.** To write: a clinical-grading example; asymmetric costs; how cost matrices relate to
the measures of the {doc}`measures` page.
:::
