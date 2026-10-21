# Theory and tests

The package implements the construction of

> R. C. Prati, *A unified algebraic framework for classification performance evaluation*,
> Information Fusion 138 (2027), 104767.
> [doi:10.1016/j.inffus.2026.104767](https://doi.org/10.1016/j.inffus.2026.104767)

## From the paper to the code

Section and result numbers are those of the published article.

| Idea in the paper | Paper | Where it is in the package |
|---|---|---|
| Indicator matrices and the four cell matrices | Section 4 | `IndicatorMatrix`, `cells` |
| The operators $\Sigma_1$, $\Sigma_m$, $\Sigma_n$; extending a binary measure; correspondence with `scikit-learn` | Section 5 (5.2) | `sigma_1`, `sigma_m`, `sigma_n`, `aggregate`, `sklearn_average` |
| Soft classifier outputs | Section 6.1 | thresholding and argmax in `IndicatorMatrix` |
| Soft ground truth and t-norms | Section 6.2 | `tnorm=` argument, `soft_cells` |
| Cost-sensitive and ordinal evaluation | Sections 6.3, 6.4 | `IndicatorMatrix.cost`, `task="ordinal"` |
| Partial and missing labels via masking | Section 6.5 | `mask=` argument |
| The fold operator $\Sigma_k$; pooled versus averaged evaluation | Section 6.19 | `fold_aggregate`, `sigma_k` |
| Aggregation-decomposable measures; micro equals denominator-weighted macro | Definition 7, Proposition 1, Theorem 1 (7.1) | `BinaryMeasure.decomposable`, `numerator`, `denominator`; `list_measures(decomposable=True)` |
| Micro-averaging escaping the per-class range | Remark 1, Example 1 (7.1) | `check_hull`; the `outside_hull` finding |
| Micro-precision, micro-recall and micro-$F_1$ collapse to accuracy | Theorem 2 (7.2) | the `micro_collapse` finding |
| Degrees of freedom, the (recall, specificity) basis, redundant suites | Proposition 3, Remark 2 (7.2) | `degrees_of_freedom`, `canonical_coordinates`, `check_redundancy` |
| The product t-norm is the unique marginally consistent one | Definition 8, Theorem 3, Corollary 3 (7.3) | `marginal_defect` |
| Ordinal losses as cost matrices | Theorem 4, Corollary 4 (7.3) | `IndicatorMatrix.cost` |
| Lower boundary correctness fails under one-hot encoding (floors) | Proposition 5 (7.4) | `BinaryMeasure.attainable_floor`; the `inflated_floor` finding |
| Skew-invariant binary measures | Definition 11, Theorem 5 (7.5) | `BinaryMeasure.skew_invariant` |
| Prior dependence of one-vs-rest rates | Proposition 6 (7.6) | `uniform_error_deviation`, `check_ovr_skew_drift` |
| The micro-macro gap as a normalised covariance | Corollary 6 (7.7) | `micro_macro_gap` |
| Undefined denominators and the convention for them | Section 7.8 | the `convention` argument; the `undefined_entries` finding |
| Bayes-optimal thresholds, ERM-consistency, degenerate optima | Proposition 9 (7.9) | `BinaryMeasure.bayes_threshold`, `erm_consistent`; the `degenerate_optimum` finding |
| Dominance relations | Section 7.11 | `pareto_dominance`, `IndicatorMatrix.dominates` |
| Recovery of structural properties from a formula | the software, not the paper | `infer_properties`, `register_measure`, `verify_inference` |

## Tests that verify the results

Each result of the paper is an executable test, so a change that breaks a theorem breaks a test:

| Test | Result verified |
|---|---|
| `test_theorem1_holds_for_decomposable` | micro equals denominator-weighted macro, the 14 decomposable measures × 3 samples |
| `test_theorem1_fails_for_non_decomposable` | the identity has no extension beyond that class |
| `test_theorem3_product_is_uniquely_marginally_consistent` | only the product t-norm preserves marginals |
| `test_theorem2_micro_collapse` | micro-P = micro-R = micro-F1 = accuracy, single-label |
| `test_proposition9_thresholds` | Bayes threshold $(\beta+\delta)/(\alpha+\beta+\gamma+\delta)$ |
| `test_proposition5_floors` | attainable floors under one-hot encoding |
| `test_proposition6_recall_prior_free_specificity_not` | one-vs-rest skew behaviour |
| `test_corollary6_gap_is_normalised_covariance` | the micro-macro gap identity |
| `test_theorem4_mae_and_mse_as_cost_matrices` | ordinal losses as cost matrices |
| `test_inference_reproduces_every_annotation` | inference recovers all analytic annotations |

```bash
python -m pytest tests/test_theory.py -v
```

The numbers in the test names are those of the published article.

The same results are formalised in Lean 4 with Mathlib in a separate project; see the paper for
the correspondence.
