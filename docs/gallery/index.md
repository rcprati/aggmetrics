# Pitfall gallery

Each notebook starts from a mistake that is easy to make, shows what the usual tools report,
and then what `aggmetrics` reports. They run in a few seconds, need only NumPy,
scikit-learn, pandas and matplotlib, and are executed in continuous integration.

| Notebook | What it shows |
|---|---|
| {doc}`01_floor_trap` | A classifier wrong on every example scores 0.998 on label accuracy with 1000 classes |
| {doc}`02_micro_macro_imbalance` | Micro-F1 equals accuracy; micro-MCC outside the per-class range; the hidden choice for 0/0 |
| {doc}`03_multilabel_ranking` | Three models on *yeast* ranked under 18 measure-operator combinations |
| {doc}`04_ordinal_and_cost` | Nine in ten errors are one grade off; a cost matrix changes which prediction is best |
| {doc}`05_model_selection_and_ci` | The scorer picks the model, pooled and averaged folds differ, and a test that fails on a flawed evaluation |
| {doc}`06_long_tail_images` | The same CIFAR-10-LT model scores 0.92 on a long-tailed set and 0.66 on a balanced one; what changes with the class prior |

```{toctree}
:hidden:

01_floor_trap
02_micro_macro_imbalance
03_multilabel_ranking
04_ordinal_and_cost
05_model_selection_and_ci
06_long_tail_images
```

:::{note}
Planned: thresholds and degenerate optima, soft labels from annotator proportions, pixel-level
segmentation measures, and custom measures.
:::
