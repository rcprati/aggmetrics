# aggmetrics

[![tests](https://github.com/rcprati/aggmetrics/actions/workflows/ci.yml/badge.svg)](https://github.com/rcprati/aggmetrics/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/aggmetrics/badge/?version=latest)](https://aggmetrics.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/aggmetrics)](https://pypi.org/project/aggmetrics/)
[![License](https://img.shields.io/badge/license-BSD--3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23535051.svg)](https://doi.org/10.5281/zenodo.23535051)

**Aggregating classification performance measures for machine learning: indicator matrices and aggregation operators.**

Any binary performance measure expressible in terms of the four confusion-matrix
counts is extended to multiclass, multilabel, ordinal, cost-sensitive and
soft-label settings by substituting one of a small set of aggregation operators.
Micro-, macro- and exemplar-averaging (and pooling across cross-validation folds)
become instances of a single construction rather than separate conventions.

The package also reports when a measure is being used outside the range where it
behaves as commonly assumed — which is the part that is hard to get from a
formula library.

```python
import numpy as np
from aggmetrics import IndicatorMatrix, diagnose

y_true = np.array([0, 1, 2, 2, 1, 0])
y_pred = np.array([0, 2, 2, 1, 1, 1])
im = IndicatorMatrix(y_true, y_pred, task='multiclass')

im.compute('f1', 'macro')          # 0.5222
im.compute('f1', 'micro')          # 0.5   -- equals accuracy on single-label data
im.compute('mcc', 'macro')         # macro-MCC: no standard implementation exists
im.compute('informedness', 'micro')
im.accuracy()                      # 0.5

im.report()                        # DataFrame: measures x operators
```

**Documentation:** <https://aggmetrics.readthedocs.io>

## Installation

```bash
pip install aggmetrics                # core, requires only numpy
pip install "aggmetrics[all]"         # adds scikit-learn and pandas support
```

## Why not just use `sklearn.metrics`?

`sklearn` exposes four `average` values for a fixed list of measures. The
correspondence is exact — `'micro'`, `'macro'`, `'weighted'` and `'samples'` are
Σ₁, Σₘ with uniform weights, Σₘ with prevalence weights and Σₙ respectively — but
the combinations available are a subset of those the construction generates, and
nothing in the library tells you when a combination is misleading.

Three things this package adds:

**Any measure under any operator.** Macro-MCC, exemplar-Jaccard, micro-informedness,
weighted G-mean. 21 binary measures × 4 operators, without special-casing.

**Structural properties as first-class data.** Each measure records whether it is
*aggregation-decomposable*, whether it is skew-invariant, and whether its value at
`tp = 0` depends on `tn`. These determine which identities apply.

```python
from aggmetrics import get_measure, list_measures

list_measures(decomposable=True)       # micro == denominator-weighted macro
get_measure('mcc').decomposable        # False
get_measure('recall').bayes_threshold()  # 0.0 -- maximised by predicting all positive
get_measure('mcc').attainable_floor(10)  # -0.111, not -1
```

**The generative claim, as an operation.** Supply a binary formula; the properties
that decide which identities apply are *derived*, not declared:

```python
from aggmetrics import register_measure, infer_properties

def prevalence_threshold(tp, fp, fn, tn):
    r, s = tp / (tp + fn), tn / (tn + fp)
    return (np.sqrt(r * (1 - s)) + s - 1) / (r + s - 1)

m = register_measure('prevalence_threshold', prevalence_threshold)
m.decomposable          # False  -- inferred by fitting <a,v>/<b,v> and validating
m.skew_invariant        # True   -- a function of recall and specificity
m.bayes_threshold()     # None   -- no closed form outside the decomposable class

im.compute('prevalence_threshold', 'exemplar')   # usable immediately
```

Inference is checked against the 21 measures whose properties are known
analytically — `verify_inference()` — and all 21 agree, with recovered coefficients
accurate to 4.2e-15. The check paid for itself: it found two errors in our own hand
annotations (error rate and FPR both retain a `tn` dependence at `tp = 0`, so both
*flatter* a wholly wrong classifier, reaching 2/m and 1/(m-1) respectively).

**Diagnostics.** `diagnose()` flags the traps:

```python
y = np.arange(500) % 10
yhat = (y + 1) % 10                    # every single prediction is wrong
im = IndicatorMatrix(y, yhat, task='multiclass')

im.compute('label_accuracy', 'micro')  # 0.800  (!)
im.compute('mcc', 'micro')             # -0.111 (!)
im.accuracy()                          # 0.000

for f in diagnose(im):
    print(f)
# [warning] inflated_floor: 'label_accuracy' cannot fall below 0.8000 on this
#           10-class one-hot problem even if every prediction is wrong ...
# [note]    micro_collapse: micro-precision, micro-recall and micro-F1 are all
#           equal to accuracy ...
# [warning] degenerate_optimum: 'recall' is maximised at threshold 0.000 ...
```

Checks currently implemented: pooled value outside the per-class range;
`tn`-dependent measures on one-hot multiclass data; micro-collapse on single-label
data; undefined per-class entries and the size of the convention effect; measures
whose optimal decision rule is not the Bayes rule for 0/1 loss.

## Features

- **21 binary measures**, resolvable by name or alias (`iou` → `jaccard`, `dice` → `f1`),
  plus `register_measure` for your own, with properties inferred.
- **Four operators**: `micro`, `macro`, `weighted`, `exemplar`.
- **Explicit undefined-value conventions**: `exclude` (default), `zero`
  (`sklearn`'s behaviour), `one`, `nan`. The choice is a parameter, not a default
  buried in the implementation.
- **Soft ground truth** via t-norms, with `marginal_defect()` showing that only the
  product t-norm preserves the marginals.
- **Cost matrices**, recovering MAE and MSE as special cases.
- **Observation masks** for partial labels, semi-supervised evaluation and abstention.
- **scikit-learn scorers** via `make_scorer`, usable in `cross_val_score` and
  `GridSearchCV`.
- **AI Agent Skills included** (`.agents/skills/aggmetrics-guide/`), providing machine-readable
  runbooks, axiomatic invariants, and diagnostic protocols for AI coding assistants.

## The test suite verifies the theory

Each result of the companion paper is an executable test, so a change that breaks
a theorem breaks a test:

```bash
python -m pytest tests/test_theory.py -v
```

| Test | Result verified |
|---|---|
| `test_theorem1_holds_for_decomposable` | micro = denominator-weighted macro, the 14 decomposable measures × 3 samples |
| `test_theorem1_fails_for_non_decomposable` | the identity has no extension beyond that class |
| `test_theorem3_product_is_uniquely_marginally_consistent` | only the product t-norm preserves marginals |
| `test_theorem2_micro_collapse` | micro-P = micro-R = micro-F₁ = accuracy, single-label |
| `test_proposition9_thresholds` | Bayes threshold `(β+δ)/(α+β+γ+δ)` |
| `test_proposition5_floors` | attainable floors under one-hot encoding |
| `test_proposition6_recall_prior_free_specificity_not` | one-vs-rest skew behaviour |
| `test_corollary6_gap_is_normalised_covariance` | the micro–macro gap identity |
| `test_theorem4_mae_and_mse_as_cost_matrices` | ordinal losses as cost matrices |
| `test_inference_reproduces_every_annotation` | inference recovers all 21 analytic annotations |

210 tests, no skips, 100% line and branch coverage.

## Examples

Notebooks, executed and checked in continuous integration. Each starts from a problem a
practitioner recognises, shows what the usual tools report, and then what `aggmetrics` reports.

| Notebook | What it shows | |
|---|---|---|
| [01 The floor trap](examples/notebooks/01_floor_trap.ipynb) | A classifier wrong on every example scores 0.998 on label accuracy with 1000 classes | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcprati/aggmetrics/blob/main/examples/notebooks/01_floor_trap.ipynb) |
| [02 Micro, macro and imbalance](examples/notebooks/02_micro_macro_imbalance.ipynb) | Micro-F1 equals accuracy; micro-MCC outside the per-class range; the hidden choice for 0/0 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcprati/aggmetrics/blob/main/examples/notebooks/02_micro_macro_imbalance.ipynb) |
| [03 Multilabel: the ranking depends on the operator](examples/notebooks/03_multilabel_ranking.ipynb) | Three models on *yeast* ranked under 18 measure-operator combinations | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcprati/aggmetrics/blob/main/examples/notebooks/03_multilabel_ranking.ipynb) |
| [04 Ordinal grades and costs](examples/notebooks/04_ordinal_and_cost.ipynb) | Nine in ten errors are one grade off; a cost matrix changes which prediction is best | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcprati/aggmetrics/blob/main/examples/notebooks/04_ordinal_and_cost.ipynb) |
| [05 Model selection](examples/notebooks/05_model_selection_and_ci.ipynb) | The scorer picks the model, pooled and averaged folds differ, and a test that fails on a flawed evaluation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcprati/aggmetrics/blob/main/examples/notebooks/05_model_selection_and_ci.ipynb) |
| [06 Long-tailed images](examples/notebooks/06_long_tail_images.ipynb) | The same CIFAR-10-LT model scores 0.92 on a long-tailed set and 0.66 on a balanced one; what changes with the class prior | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcprati/aggmetrics/blob/main/examples/notebooks/06_long_tail_images.ipynb) |
| [07 Choosing a threshold](examples/notebooks/07_thresholds_and_degenerate_optima.ipynb) | Recall, specificity and precision are maximised by trivial classifiers; a cost-weighted measure has a closed-form threshold | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcprati/aggmetrics/blob/main/examples/notebooks/07_thresholds_and_degenerate_optima.ipynb) |
| [08 Soft labels](examples/notebooks/08_soft_labels.ipynb) | Image classifiers against the distribution of 50 human annotators (CIFAR-10H): does the soft reference change the verdict, where do the models err, and a simulation of when it reverses the verdict | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rcprati/aggmetrics/blob/main/examples/notebooks/08_soft_labels.ipynb) |

To run them locally: `pip install "aggmetrics[examples]" jupyter`. A short script version of
the tour is in `examples/quickstart.py`:

```bash
python examples/quickstart.py
```

## Citing

Please cite both the software and the companion paper. The software is archived on Zenodo:
the DOI [10.5281/zenodo.23535051](https://doi.org/10.5281/zenodo.23535051) always resolves to the
latest version, and version 0.1.0 is [10.5281/zenodo.23535052](https://doi.org/10.5281/zenodo.23535052).
GitHub's "Cite this repository" button reads `CITATION.cff`. In BibTeX:

```bibtex
@software{prati2026aggmetrics,
  author  = {Prati, Ronaldo C.},
  title   = {aggmetrics: aggregating classification performance measures for machine learning},
  year    = {2026},
  doi     = {10.5281/zenodo.23535051},
  url     = {https://github.com/rcprati/aggmetrics}
}
```

The paper is:

```bibtex
@article{prati2027unified,
  author  = {Prati, Ronaldo C.},
  title   = {A unified algebraic framework for classification performance evaluation},
  journal = {Information Fusion},
  volume  = {138},
  pages   = {104767},
  year    = {2027},
  doi     = {10.1016/j.inffus.2026.104767}
}
```

## License

BSD 3-Clause.
