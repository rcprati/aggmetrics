# Theory and tests

The package implements the construction of

> R. C. Prati, *A unified algebraic framework for classification performance evaluation*,
> Information Fusion 138 (2027), 104767.
> [doi:10.1016/j.inffus.2026.104767](https://doi.org/10.1016/j.inffus.2026.104767)

Each result of the paper is an executable test, so a change that breaks a theorem breaks a
test:

| Test | Result verified |
|---|---|
| `test_theorem1_holds_for_decomposable` | micro equals denominator-weighted macro, 15 measures × 3 samples |
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

The same results are formalised in Lean 4 with Mathlib in a separate project; see the paper
for the correspondence.

:::{note}
**Sketch.** To write: a table mapping each function to the paper's definitions, with section
numbers.
:::
