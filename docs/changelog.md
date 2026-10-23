# Changelog

## 0.1.1 (2026-10-22)

- Citation: the preprint describing the package ([arXiv:2610.30456](https://arxiv.org/abs/2610.30456)) is
  listed in `CITATION.cff` and the README.
- `diagnose` no longer reports undefined per-example entries (`undefined_entries` for the exemplar
  operator) on binary problems, where each example has a single label and every negative would be flagged.
- Example notebooks 04 to 10: ordinal grades and costs, model selection, long-tailed images,
  thresholds, soft labels (with the CIFAR-10H human labels), segmentation (with a CC0 subset of BBBC039), and
  custom measures.
- The package is clean under `ruff` (pycodestyle, pyflakes, import order, 88 columns), which CI now
  checks. `Sequence` was an undefined name in two annotations in `aggregation.py`; it is now imported.
  No behaviour changed apart from the `diagnose` item above.
- The notebooks and their data are in the repository, not in the pip package; the README and the
  installation page now say so.

## 0.1.0 (2026-10-21)

First stable release. A pre-release, 0.1.0.dev0, was uploaded earlier to reserve the name.
Archived on Zenodo: [10.5281/zenodo.23535052](https://doi.org/10.5281/zenodo.23535052) for this version,
[10.5281/zenodo.23535051](https://doi.org/10.5281/zenodo.23535051) for all versions.

**Library**

- Indicator matrices for binary, multiclass, multilabel and ordinal problems, soft labels
  through t-norms, observation masks.
- 21 binary measures, four aggregation operators, explicit conventions for undefined values.
- Structural properties stored with each measure; property inference from a formula
  (`register_measure`, `infer_properties`, `verify_inference`).
- Diagnostics (`diagnose` and the individual checks).
- `scikit-learn` scorers.

**Changes since 0.1.0.dev0**

- `fold_aggregate(pooling="pooled")` concatenates the folds' predictions, so folds may differ in
  size. (The pre-release raised an error for folds of different size and summed unrelated rows
  for the exemplar operator.)
- `IndicatorMatrix` takes the number of classes from the width of a score matrix, so a class
  absent from `y_true` no longer raises.

**Project**

- Test suite at 100% line and branch coverage, run on Linux, macOS and Windows with Python 3.9 to 3.12.
- Documentation at <https://aggmetrics.readthedocs.io>; three example notebooks that run offline
  (the *yeast* data they use is in `examples/data`); an agent skill with a worked session.
- Result numbering in the code, tests and skill follows the published article (Information Fusion
  138, 104767).
