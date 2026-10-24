# Changelog

## Unreleased

- `under_prior`: evaluate a measure at another class prior (label shift), from the skew results of the
  paper. Notebook 06 uses it in place of its hand-written reweighting (same numbers).
- `multioutput_aggregate`: the target-wise operator of the paper for several targets, and the usual
  operators over the block-structured matrix.
- `tree_cost_matrix`: a cost matrix from the edge distances of a taxonomy (hierarchical classification).
- `denominator_balance` and the `balanced_denominators` finding of `diagnose`: Corollary 1, micro equals
  macro when the per-class denominators are equal (not reported for measures whose denominator is the
  total count).
- `aggmetrics.suites` and the new modules share one internal description of the cells as functions of
  the rates (`aggmetrics._rates`).

## 0.2.0 (2026-10-24)

Archived on Zenodo: [10.5281/zenodo.23618910](https://doi.org/10.5281/zenodo.23618910) for this version,
[10.5281/zenodo.23535051](https://doi.org/10.5281/zenodo.23535051) for all versions. Published on PyPI on 2026-10-24.

- New module `aggmetrics.suites`: `suite_jacobian`, `suite_rank` and `select_suite` implement the rank
  criterion of Remark 2 (is a suite of measures locally non-redundant?) and add an information-volume score
  to pick the most informative suite. **Experimental**: the signatures may change in a later 0.x release.
  Notebook 12 and a user-guide page show it on data. 15 new tests.
- Example notebook 11: evaluating PyTorch and Keras models (logits, pooling batches instead of averaging a
  measure over them, accumulated counts, a Keras callback). It needs the new extra `deep` (torch, keras 3).

## 0.1.1 (2026-10-22)

Archived on Zenodo: [10.5281/zenodo.23585309](https://doi.org/10.5281/zenodo.23585309) for this version,
[10.5281/zenodo.23535051](https://doi.org/10.5281/zenodo.23535051) for all versions. Published on PyPI on 2026-10-23.

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
