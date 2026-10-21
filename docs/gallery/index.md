# Pitfall gallery

Each notebook starts from a mistake that is easy to make, shows what the usual tools report,
and then what `aggmetrics` reports. They run in a few seconds, need only NumPy,
scikit-learn, pandas and matplotlib, and are executed in continuous integration.

| Notebook | What it shows |
|---|---|
| {doc}`01_floor_trap` | A classifier wrong on every example scores 0.998 on label accuracy with 1000 classes |
| {doc}`02_micro_macro_imbalance` | Micro-F1 equals accuracy; micro-MCC outside the per-class range; the hidden choice for 0/0 |
| {doc}`03_multilabel_ranking` | Three models on *yeast* ranked under 18 measure-operator combinations |

```{toctree}
:hidden:

01_floor_trap
02_micro_macro_imbalance
03_multilabel_ranking
```

:::{note}
Planned: ordinal and cost-sensitive evaluation, soft labels from annotator proportions,
long-tailed image classification, and `scikit-learn` pipelines with `diagnose` in CI.
:::
