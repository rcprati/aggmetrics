---
name: aggmetrics-guide
description: >-
  Use this skill whenever the user wants to evaluate classification models, compute evaluation
  measures via indicator matrices, analyze classifier diagnostics, work with soft/probabilistic
  confusion matrices (t-norms), define custom evaluation measures, choose or report an aggregation
  operator (micro, macro, weighted, exemplar, pooled folds), or evaluate segmentation, thresholds or
  cost-sensitive classifiers using the aggmetrics Python library.
---

# aggmetrics Evaluation Skill

A comprehensive guide for using `aggmetrics` to evaluate classification models (binary, multiclass, multilabel, and ordinal) using indicator matrices and aggregation operators.

## Core Concepts

In `aggmetrics`, classification evaluation is unified through:
1. **Indicator Matrices**: Ground truth $Y$ and predictions $P$ represented as matrices.
2. **Cell Matrices**: Four cell matrices $A, B, C, D$ corresponding to True Positives, False Positives, False Negatives, and True Negatives.
3. **Aggregation Operators**:
   - `'micro'` ($\Sigma_1$): Sums cell counts across all classes/samples first, then applies the measure formula.
   - `'macro'` ($\Sigma_m$): Computes the measure per class and averages uniformly.
   - `'weighted'` ($\Sigma_m$ weighted): Computes the measure per class and averages weighted by class prevalence.
   - `'exemplar'` / `'samples'` ($\Sigma_n$): Computes the measure per instance/sample (useful for multilabel).
4. **Diagnostics**: Detecting traps like *micro-collapse* (micro-P = micro-R = micro-F1 = accuracy), *inflated floors* (measures flattering wrong models due to TNs), and *degenerate optima*.

---

## What to do before reporting a number

An agent that reports an evaluation should do these, in this order. Each one is a failure that the example
notebooks show on real data.

1. **State the operator and the convention.** "F1" is not a number: micro, macro, weighted and exemplar
   give different values, and a macro value over undefined entries depends on `convention`
   (`exclude`, `zero`, `one`, `nan`). Report both (notebooks 02, 09).
2. **Run `diagnose(im)`** on the data and the suite of measures, and relay every `warning`. Typical
   findings: `inflated_floor` (a wrong model still scores high), `micro_collapse` (micro-F1 is accuracy),
   `undefined_entries` (the convention moves the value) (notebooks 01, 02).
3. **Compare models under more than one operator**, and say when the verdict changes (notebooks 03, 08, 10).
   A near-tie is a tie: do not rank on a gap that was not tested.
4. **Pool cross-validation folds** with `fold_aggregate(..., pooling="pooled")` for non-linear measures,
   and show the averaged value next to it (notebook 05).
5. **Do not maximise a measure that has a degenerate optimum** (`recall`, `specificity`, `precision`):
   predicting everything one way scores perfectly. Pick a cost-weighted measure or constrain the other
   error (notebook 07).
6. **Use a measure that rejects the trivial predictor.** Pixel or label accuracy gives high scores to
   "predict the majority"; compare with it (notebooks 06, 09).
7. **For soft labels, say what is soft.** A measure that is linear in the predicted probability rewards
   decisiveness, not calibration (notebook 08).
8. **A custom measure is only as good as its inferred properties.** Check them (notebook 10).

---

## Quickstart Recipes

### 1. Basic Classification (Binary & Multiclass)

```python
import numpy as np
from aggmetrics import IndicatorMatrix, diagnose

# Discrete predictions
y_true = np.array([0, 1, 2, 2, 1, 0])
y_pred = np.array([0, 2, 2, 1, 1, 1])

# Initialize matrix (task inferred or specified: 'binary', 'multiclass', 'multilabel', 'ordinal')
im = IndicatorMatrix(y_true, y_pred, task="multiclass")

# Compute standard and advanced aggregated measures
acc = im.accuracy()                  # 0.5
macro_f1 = im.compute("f1", "macro") # Macro-averaged F1
micro_f1 = im.compute("f1", "micro") # Micro-averaged F1 (equals accuracy on single-label)
macro_mcc = im.compute("mcc", "macro") # Macro-MCC

# Complete DataFrame report (measures x operators)
df_report = im.report()
```

### 2. Diagnostic Analysis

```python
# Run automatic checks against common evaluation pitfalls
findings = diagnose(im)
for f in findings:
    print(f"[{f.severity}] {f.code}: {f.message}")
```

### 3. Multilabel and Soft Labels (T-Norms)

```python
# Real-valued probabilities / soft labels
y_true_soft = np.array([[1.0, 0.0], [0.8, 0.2]])
y_pred_soft = np.array([[0.9, 0.1], [0.7, 0.3]])

# Product t-norm is uniquely marginally consistent
im_soft = IndicatorMatrix(y_true_soft, y_pred_soft, task="multilabel", tnorm="product")
exemplar_f1 = im_soft.compute("f1", "exemplar")
```

### 4. Custom Measures and Property Inference

```python
from aggmetrics import register_measure, infer_properties

# Define any binary measure function (tp, fp, fn, tn)
def custom_balanced_score(tp, fp, fn, tn):
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return 0.5 * (tpr + tnr)

# Register and automatically derive its mathematical properties
measure = register_measure("custom_balanced", custom_balanced_score)

print("Decomposable:", measure.decomposable)
print("Skew-invariant:", measure.skew_invariant)
print("Bayes decision threshold:", measure.bayes_threshold())

# Use immediately on any IndicatorMatrix
im.compute("custom_balanced", "macro")
```

The properties are inferred from sampled counts, so a piecewise formula (for example one that changes
below a count threshold) can be misclassified. Check `verify_inference`, and override a property by
keyword (`register_measure(..., linear=None, skew_invariant=False)`) when it is known. See
notebook 10.

### 5. Scikit-Learn Integration

```python
from aggmetrics.sklearn_compat import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Create an sklearn-compatible scorer for any aggmetrics metric + operator combination
macro_mcc_scorer = make_scorer("mcc", aggregation="macro")

clf = RandomForestClassifier()
# scores = cross_val_score(clf, X, y, scoring=macro_mcc_scorer, cv=5)
```

### 6. Orthogonality, Degrees of Freedom & Redundancy Analysis

```python
from aggmetrics.diagnostics import check_redundancy

# Check degrees of freedom (d = 2 for binary, d = m(m-1) for multiclass)
print("Degrees of Freedom:", im.degrees_of_freedom)

# Extract canonical orthogonal coordinates (row-decoupled conditional rates)
coords = im.canonical_coordinates()  # e.g., [recall, specificity] in binary

# Detect bloated / redundant evaluation metric suites
suite = ["accuracy", "balanced_accuracy", "f1", "recall", "specificity", "precision"]
finding = check_redundancy(im, suite)
if finding:
    print(finding.message)
```

### 7. Universal Pareto Dominance (Section 7.11)

```python
from aggmetrics import IndicatorMatrix, pareto_dominance

im_a = IndicatorMatrix(y_true, y_pred, task="multiclass")
im_b = IndicatorMatrix(y_true, [0, 1, 2, 2, 0, 0], task="multiclass")

# Compare two models componentwise in canonical coordinates
# If A dominates B, A is guaranteed to beat B on ALL monotonic performance measures
result = pareto_dominance(im_a, im_b)  # "A_dominates_B", "B_dominates_A", "incomparable", "equivalent"
if result == "incomparable":
    # Explain that any ranking between A and B depends arbitrarily on metric trade-off weights
    pass
```

### 8. Micro-Macro Gap Decomposition (Corollary 6)

```python
from aggmetrics import micro_macro_gap

# Exact algebraic decomposition of the gap as a normalized covariance
gap_info = micro_macro_gap(im, "f1")
print(f"Gap (Micro - Macro): {gap_info['gap']:.4f}")
print(f"Normalized Covariance: {gap_info['predicted_gap']:.4f}")
```

### 9. Ordinal Classification & Cost Matrices (Theorem 4)

```python
# MAE and MSE as cost matrices on indicator matrices
J, K = np.meshgrid(np.arange(im.m), np.arange(im.m), indexing="ij")
mae = im.cost(np.abs(J - K)) / im.n
mse = im.cost((J - K) ** 2.0) / im.n

# Cumulative threshold ordinal matrix (m-1 binary threshold indicators)
im_ord = IndicatorMatrix(y_true, y_pred, task="ordinal")
ord_macro_rec = im_ord.compute("recall", "macro")
```

### 10. Cross-Validation Aggregation ($\Sigma_k$ Operator)

```python
from aggmetrics.aggregation import fold_aggregate

# Pooled CV (Sigma_k) avoids Jensen's ratio-of-averages distortion for non-linear measures
mcc_pooled = fold_aggregate(folds, measure="mcc", how="macro", pooling="pooled")
mcc_averaged = fold_aggregate(folds, measure="mcc", how="macro", pooling="averaged")
```

---

## Detailed References

- [Measures and Properties Reference](./references/measures_and_properties.md): Full list of 21 built-in measures, aliases, algebraic properties, and micro-macro gap covariance.
- [Orthogonality & Redundancy Reference](./references/orthogonality_and_redundancy.md): Degrees of freedom ($d$), row-decoupling, Frobenius inner product, and canonical orthogonal bases.
- [Pareto Dominance Reference](./references/pareto_dominance_and_comparison.md): Universal model comparison, incomparability, and monotonic measure consistency.
- [Ordinal & Cost Matrices Reference](./references/ordinal_and_cost_matrices.md): Theorem 4 (MAE/MSE unification), custom cost matrices, and cumulative threshold encoding.
- [Cross-Validation Aggregation Reference](./references/cross_validation_aggregation.md): The inter-fold operator $\Sigma_k$, pooled vs averaged CV, and Jensen ratio bias.
- [Diagnostics Reference](./references/diagnostics.md): Detailed breakdown of diagnostic rules, OvR skew drift, and evaluation pitfalls.
- [T-Norms Reference](./references/tnorms_and_soft_labels.md): Soft label evaluation and marginal consistency.
- [Example notebooks](https://github.com/rcprati/aggmetrics/tree/main/examples/notebooks): ten executed notebooks, one per pitfall (floor trap, micro versus macro, multilabel ranking, ordinal and cost, model selection, long tail, thresholds, soft labels, segmentation, custom measures).
- [Code Examples](./examples/quickstart_recipes.py): Executable script patterns for all evaluation tasks.
- [Worked agent session](./examples/agent_session_example.md): A request, the recipes used, the real output, and the answer an agent gives (runnable: [`agent_session_example.py`](./examples/agent_session_example.py)).
