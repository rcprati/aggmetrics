# Diagnostics

`diagnose(im, measures=None, *, aggregation="macro", emit_warnings=False)` runs the checks
below and returns a list of `Finding` objects, each with a `code`, a `severity`
(`'warning'` or `'note'`), a `message` and the triggering values in `detail`.

| Code | Severity | Fires when |
|---|---|---|
| `outside_hull` | warning | a micro value lies outside the range of the per-class values, so it is not a weighted average of them |
| `inflated_floor` | warning | a measure that depends on true negatives is used on one-hot multiclass data, where a wrong classifier still scores high |
| `micro_collapse` | note | micro-precision, micro-recall and micro-F1 are all equal to accuracy (single-label data) |
| `undefined_entries` | warning | per-class or per-example values are undefined; the message gives the effect of the convention |
| `degenerate_optimum` | warning | a measure is maximised by predicting every instance positive or negative |
| `not_erm_consistent` | note | a measure is maximised at a threshold other than 0.5 |
| `ovr_skew_drift` | note | asymmetric off-diagonal errors make one-vs-rest specificity drift under prior shift |
| `redundant_suite` | note | more measures were requested than the confusion matrix has degrees of freedom |

## In tests and CI

With `emit_warnings=True`, each warning-level finding is also raised as a `MeasureWarning`,
so an evaluation script can fail on a flawed protocol:

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error", MeasureWarning)
    diagnose(im, emit_warnings=True)
```

## Individual checks

`check_hull`, `check_ovr_skew_drift`, `check_redundancy` and `micro_macro_gap` are available
on their own; see the {doc}`API reference <../api/diagnostics>`.

:::{note}
**Sketch.** To write: what to do about each finding (the "what to report" lists of the
notebooks, collected); the false-positive behaviour of each check.
:::
