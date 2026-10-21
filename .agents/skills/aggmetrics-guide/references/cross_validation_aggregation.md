# Cross-Validation Aggregation & The Inter-Fold Operator ($\Sigma_k$)

This reference details the inter-fold reduction operator $\Sigma_k$ (`fold_aggregate`) for aggregating performance metrics across $k$-fold cross-validation splits.

---

## 1. The Ratio Bias Problem in Cross-Validation

In $k$-fold cross-validation, the standard convention used by most ML libraries (`scikit-learn`) is **fold-averaging**:
$$\text{Score}_{\text{avg}} = \frac{1}{K} \sum_{k=1}^K M(\text{Fold}_k)$$

### The Pitfall:
For non-linear measures (such as Matthews Correlation Coefficient, F1-score with imbalanced/sparse positive instances, G-mean, Informedness), calculating the ratio within each fold and averaging the resulting scores is statistically distorted by **Jensen's inequality** and undefined zero-denominator folds.

### The Algebraic Solution: Inter-Fold Pooling ($\Sigma_k$)
The framework introduces the fourth reduction operator $\Sigma_k$:
$$\text{Score}_{\text{pooled}} = M\left( \sum_{k=1}^K \text{Counts}(\text{Fold}_k) \right)$$
- Sums the raw confusion matrix counts across all $K$ test folds into a single global contingency matrix, and applies the metric formula once.
- Eliminates fold-level undefined denominators.
- Preserves asymptotic consistency.

---

## 2. When do Pooled and Averaged CV agree?

- For **linear decomposable measures** (such as Accuracy or Error Rate) evaluated on folds of equal size, pooled and averaged CV produce **strictly identical numerical values**.
- For **non-linear measures** (MCC, Cohen's Kappa, Macro-F1 on rare classes), pooled CV avoids the ratio-of-averages bias and is mathematically preferred.

---

## 3. Python API & Usage in `aggmetrics`

```python
import numpy as np
from aggmetrics import IndicatorMatrix
from aggmetrics.aggregation import fold_aggregate, sigma_k

# Assume we have 5 folds of cross-validation
# Each fold represented as an IndicatorMatrix
folds = []
for k in range(5):
    # y_true_k, y_pred_k for fold k
    im_k = IndicatorMatrix(y_true_k, y_pred_k, task="multiclass")
    folds.append(im_k)

# 1. Pooled CV (Sigma_k aggregation - recommended for non-linear metrics)
mcc_pooled = fold_aggregate(folds, measure="mcc", how="macro", pooling="pooled")

# 2. Averaged CV (standard mean across fold scores)
mcc_averaged = fold_aggregate(folds, measure="mcc", how="macro", pooling="averaged")

print(f"MCC (Pooled CV):   {mcc_pooled:.4f}")
print(f"MCC (Averaged CV): {mcc_averaged:.4f}")
```
