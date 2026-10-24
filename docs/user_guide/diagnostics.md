# Diagnostics

`diagnose(im, measures=None, *, aggregation="macro", emit_warnings=False)` runs the checks below
and returns a list of `Finding` objects, each with a `code`, a `severity` (`'warning'` or
`'note'`), a `message` and the triggering values in `detail`. `measures` is the list of
measures you intend to report; by default it checks eight common ones.

| Code | Severity | Fires when | Depends on |
|---|---|---|---|
| `outside_hull` | warning | a micro value lies outside the range of the per-class values | the predictions |
| `inflated_floor` | warning | a measure that depends on true negatives is used on one-hot multiclass data | the measure and the problem type |
| `micro_collapse` | note | micro-precision, micro-recall and micro-F1 all equal accuracy | the problem type |
| `undefined_entries` | warning | per-class or per-example values are undefined | the predictions |
| `degenerate_optimum` | warning | a measure is maximised by predicting every instance positive or negative | the measure |
| `not_erm_consistent` | note | a measure is maximised at a threshold other than 0.5 | the measure |
| `ovr_skew_drift` | note | asymmetric off-diagonal errors make one-vs-rest specificity drift under prior shift | the predictions |
| `redundant_suite` | note | more measures were requested than the confusion matrix has degrees of freedom | the list you pass |

## Data-dependent and data-independent findings

Some findings describe the *data and model*, others describe the *measure or the problem type*.
The second kind fires even for a perfect classifier. On a 10-class problem, a classifier that is
right on every example still produces `inflated_floor`, `micro_collapse` and `degenerate_optimum`;
they are properties of the measures you asked about, not defects of the model. Read them as
reminders about how to interpret the numbers:

- `inflated_floor` fires for every one-hot multiclass problem with more than two classes whenever
  the list includes a measure that depends on true negatives. `micro_collapse` fires for every
  one-hot multiclass problem, whatever measures you request.
- `degenerate_optimum` fires only when the list includes `recall`, `fnr`, `specificity` or `fpr`,
  which the default list does.
- `not_erm_consistent` is triggered by no built-in measure. It exists for measures you register
  ({doc}`custom_measures`).
- `outside_hull`, `undefined_entries` and `ovr_skew_drift` depend on the predictions. They are
  the findings that tell you something about *this* model. `ovr_skew_drift` uses a threshold of
  0.01 on the mean uniform-error deviation; call `check_ovr_skew_drift(im, threshold=...)`
  directly to change it.

## What to do about each finding

- **`outside_hull`**: do not read the micro value as a weighted average of per-class values.
  Report the macro value, or both, and name the operator.
- **`inflated_floor`**: the measure cannot reach its nominal worst value on this problem. Report
  it next to its floor (`attainable_floor(m)`), or use a measure without the true-negative
  dependence (recall, precision, $F_1$, Jaccard); see the
  {doc}`floor-trap notebook <../gallery/01_floor_trap>`.
- **`micro_collapse`**: report accuracy once, and a macro measure for the class balance.
- **`undefined_entries`**: state the `convention`, or report the measure that is defined (macro
  $F_1$ rather than macro precision); the message gives both values.
- **`degenerate_optimum`**: never tune a threshold on this measure alone; use a balanced measure.
- **`not_erm_consistent`**: standard training does not target this measure; expect to tune the
  threshold.
- **`ovr_skew_drift`**: one-vs-rest specificity, and the measures built on it, will change if the
  class mix changes. Prefer recall per class for comparisons across priors.
- **`redundant_suite`**: some of the measures you list carry no extra information. Drop the
  collinear ones.

## In tests and CI

With `emit_warnings=True`, each warning-level finding is also raised as a `MeasureWarning`, so an
evaluation script can fail on a flawed protocol:

```python
import warnings
from aggmetrics.diagnostics import MeasureWarning

with warnings.catch_warnings():
    warnings.simplefilter("error", MeasureWarning)
    diagnose(im, emit_warnings=True)
```

Because some warnings fire for any one-hot multiclass problem, pass the list of measures you
actually report, and filter on the codes that matter to you, rather than failing on every warning.

## Individual checks

`check_hull`, `check_ovr_skew_drift`, `check_redundancy` and `micro_macro_gap` are available on
their own; see the {doc}`API reference <../api/diagnostics>`.

## When micro and macro are the same number

Corollary 1 of the paper says that, for a decomposable measure, micro equals unweighted macro for
every configuration of the per-class values if and only if all per-class denominators are equal.
For recall the denominator is the class size, so the two coincide when the classes have the same size.
`denominator_balance` shows the denominators, and `diagnose` reports the `balanced_denominators`
finding when they are equal:

```python
from aggmetrics import denominator_balance, diagnose

info = denominator_balance(im, "recall")     # im: three classes of 100 examples each
info["denominators"], info["balanced"]       # array([100., 100., 100.]), True
info["micro"], info["macro"]                 # 0.74, 0.74
```

Measures whose denominator is the total count (`error_rate`, `label_accuracy`) are balanced on any
data, so `diagnose` does not report them.
