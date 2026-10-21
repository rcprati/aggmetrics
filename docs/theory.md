# Theory and tests

The package implements the construction of

> R. C. Prati, *A unified algebraic framework for classification performance evaluation*,
> Information Fusion 138 (2027), 104767.
> [doi:10.1016/j.inffus.2026.104767](https://doi.org/10.1016/j.inffus.2026.104767)

## From the paper to the code

| Idea in the paper | Where it is in the package |
|---|---|
| Indicator matrices and the four cell matrices | `IndicatorMatrix`, `cells` |
| The operators $\Sigma_1$, $\Sigma_m$, $\Sigma_n$ | `sigma_1`, `sigma_m`, `sigma_n`, `aggregate` |
| The fold operator $\Sigma_k$ and pooled versus averaged evaluation | `fold_aggregate`, `sigma_k` |
| Aggregation-decomposable measures; micro equals denominator-weighted macro | `BinaryMeasure.decomposable`, `numerator`, `denominator`; `list_measures(decomposable=True)` |
| The micro-macro gap as a normalised covariance | `micro_macro_gap` |
| Micro-precision, micro-recall and micro-$F_1$ collapse to accuracy | the `micro_collapse` finding |
| Pooled micro values outside the per-class range | `check_hull` |
| Bayes thresholds and degenerate optima | `BinaryMeasure.bayes_threshold`, `erm_consistent`; the `degenerate_optimum` finding |
| Attainable floors under one-hot encoding | `BinaryMeasure.attainable_floor`; the `inflated_floor` finding |
| Skew-invariance and one-vs-rest prior drift | `skew_invariant`; `uniform_error_deviation`, `check_ovr_skew_drift` |
| Soft ground truth and marginal consistency of the product t-norm | `tnorm=` argument, `soft_cells`, `marginal_defect` |
| Ordinal losses as cost matrices | `IndicatorMatrix.cost`, `task="ordinal"` |
| Degrees of freedom and canonical coordinates | `degrees_of_freedom`, `canonical_coordinates`, `check_redundancy` |
| Universal Pareto dominance | `pareto_dominance`, `IndicatorMatrix.dominates` |
| Recovery of structural properties from a formula | `infer_properties`, `register_measure`, `verify_inference` |

## Tests that verify the results

Each result of the paper is an executable test, so a change that breaks a theorem breaks a test:

| Test | Result verified |
|---|---|
| `test_theorem1_holds_for_decomposable` | micro equals denominator-weighted macro, the 14 decomposable measures × 3 samples |
| `test_theorem1_fails_for_non_decomposable` | the identity has no extension beyond that class |
| `test_theorem2_product_is_uniquely_marginally_consistent` | only the product t-norm preserves marginals |
| `test_theorem5_micro_collapse` | micro-P = micro-R = micro-F1 = accuracy, single-label |
| `test_proposition5_thresholds` | Bayes threshold $(\beta+\delta)/(\alpha+\beta+\gamma+\delta)$ |
| `test_proposition7_floors` | attainable floors under one-hot encoding |
| `test_proposition2_recall_prior_free_specificity_not` | one-vs-rest skew behaviour |
| `test_corollary5_gap_is_normalised_covariance` | the micro-macro gap identity |
| `test_theorem3_mae_and_mse_as_cost_matrices` | ordinal losses as cost matrices |
| `test_inference_reproduces_every_annotation` | inference recovers all analytic annotations |

```bash
python -m pytest tests/test_theory.py -v
```

The numbers in the test names are those of the results in the paper's draft, against which the
code was written. The behaviour they check does not depend on the numbering.

The same results are formalised in Lean 4 with Mathlib in a separate project; see the paper for
the correspondence.
