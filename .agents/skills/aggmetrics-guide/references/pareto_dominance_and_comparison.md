# Pareto Dominance & Universal Model Comparison

This reference details **Proposition 3** of the companion paper: how canonical orthogonal coordinates determine model dominance across **all monotonic performance measures simultaneously**.

---

## 1. The Core Theoretical Result (Proposition 3)

In classification benchmarks, ranking models using a single scalar metric (e.g. F1, Accuracy, or MCC) often produces contradictory orderings: Model A wins under F1, but Model B wins under Balanced Accuracy.

**Proposition 3 (Universal Pareto Dominance)** establishes that:
> If classifier $A$ componentwise dominates classifier $B$ in canonical orthogonal coordinates, then **$A$ achieves a strictly better (or equal) score than $B$ under EVERY performance measure that is monotonically increasing in true positives and decreasing in false positives / false negatives.**

### Canonical Conditions for Dominance:
- **Binary Classification**:
  $$A \succ B \iff \text{Recall}_A \ge \text{Recall}_B \quad \text{and} \quad \text{Specificity}_A \ge \text{Specificity}_B$$
  (with at least one strict inequality).
- **Multiclass Classification ($m$ classes)**:
  $$A \succ B \iff R_{jk}^A \le R_{jk}^B \quad \text{for all } j \neq k$$
  where $R_{jk} = P(\hat{Y}=k \mid Y=j)$ are the off-diagonal conditional misclassification rates.

---

## 2. Incomparability & Arbitrary Metric Weights

If neither classifier dominates the other in canonical coordinates:
$$\text{pareto\_dominance}(A, B) = \text{"incomparable"}$$
- **Key Insight**: When models are incomparable, **any scalar ranking between them is mathematically contingent upon the arbitrary trade-off weights embedded in the chosen metric**.
- For example, if Model A has higher recall but lower specificity than Model B, Model A will win under $F_2$, while Model B will win under $F_{0.5}$ or Specificity. Neither model is objectively superior without external cost specification.

---

## 3. Python API & Usage in `aggmetrics`

```python
import numpy as np
from aggmetrics import IndicatorMatrix, pareto_dominance

# Ground truth and two competing models
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
p_a    = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])  # Model A (perfect)
p_b    = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])  # Model B (2 errors)
p_c    = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Model C (predicts all 1: high recall, 0 specificity)

im_a = IndicatorMatrix(y_true, p_a, task="binary")
im_b = IndicatorMatrix(y_true, p_b, task="binary")
im_c = IndicatorMatrix(y_true, p_c, task="binary")

# Direct dominance query
print(im_a.dominates(im_b))  # True
print(im_b.dominates(im_a))  # False

# Comprehensive pair comparison
print(pareto_dominance(im_a, im_b))  # "A_dominates_B"
print(pareto_dominance(im_b, im_a))  # "B_dominates_A"
print(pareto_dominance(im_b, im_c))  # "incomparable"
print(pareto_dominance(im_a, im_a))  # "equivalent"
```

---

## 4. Best Practices for AI Assistants

1. **Before Claiming a "Winner"**: When comparing two candidate classification models on a test set, first check `pareto_dominance(im_a, im_b)`.
2. **If Incomparable**: Explicitly explain the trade-off to the user:
   > *"Model A achieves higher Recall (0.85 vs 0.70), but Model B achieves higher Specificity (0.92 vs 0.80). Because neither Pareto-dominates the other, Model A will rank higher on recall-focused metrics like $F_2$, while Model B will rank higher on precision/accuracy. A domain-specific cost matrix is needed to choose between them."*
