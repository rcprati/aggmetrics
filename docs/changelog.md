# Changelog

## 0.1.0.dev0

First public pre-release.

- Indicator matrices for binary, multiclass, multilabel and ordinal problems, soft labels
  through t-norms, observation masks.
- 21 binary measures, four aggregation operators, explicit conventions for undefined values.
- Structural properties stored with each measure; property inference from a formula
  (`register_measure`, `infer_properties`, `verify_inference`).
- Diagnostics (`diagnose` and the individual checks).
- `scikit-learn` scorers.
- `fold_aggregate(pooling="pooled")` concatenates the folds' predictions, so folds may differ in
  size. (Earlier builds raised an error for folds of different size and summed unrelated rows
  for the exemplar operator.)
- `IndicatorMatrix` takes the number of classes from the width of a score matrix, so a class
  absent from `y_true` no longer raises.
- Test suite at 100% line and branch coverage; example notebooks; an agent skill.
