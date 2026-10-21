# Ordinal Classification & Cost-Sensitive Matrices Reference

This reference details **Theorem 4** (Section 7.3) and Section 6.4 of the companion paper: the unification of ordinal classification losses (such as MAE and MSE) with cost-sensitive classification via indicator matrices.

---

## 1. Theorem 4: Ordinal Losses as Cost Matrices

Traditional machine learning packages treat regression-like ordinal evaluation (MAE, MSE) as entirely separate paradigms from confusion matrix evaluation.

**Theorem 4** proves that for an $m$-class ordinal problem evaluated on $n$ instances:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| = \frac{1}{n} \operatorname{cost}(C_{\text{MAE}})$$
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n} \operatorname{cost}(C_{\text{MSE}})$$

where the cost matrices are given by:
$$C_{\text{MAE}}[j, k] = |j - k|, \qquad C_{\text{MSE}}[j, k] = (j - k)^2$$

### Standard 0/1 Error as Cost:
A standard 0/1 misclassification loss corresponds to the cost matrix $C_{0/1} = \mathbf{1} \mathbf{1}^T - I$, which satisfies:
$$\operatorname{im.cost}(C_{0/1}) = \frac{1}{2} (FP + FN) = \sum_{i=1}^n \mathbb{I}[y_i \neq \hat{y}_i]$$

---

## 2. Cumulative Ordinal Indicator Matrices (`task='ordinal'`)

In ordinal classification with $m$ classes $\{0, 1, \dots, m-1\}$, class labels carry a natural order.
Instead of one-hot encoding, `aggmetrics` can encode the labels into **$m-1$ cumulative threshold indicators**:
$$Y_{ik} = \mathbb{I}[y_i > k], \quad k \in \{0, 1, \dots, m-2\}$$

### Properties:
1. **Preserves monotonicity**: An instance of class $j$ is marked positive for all thresholds $k < j$.
2. **Threshold-wise Metrics**: Macro-averaging across the $m-1$ binary thresholds yields ordinal macro-recall, ordinal macro-precision, etc.

---

## 3. Python API & Usage in `aggmetrics`

### Evaluating Ordinal Costs (MAE, MSE, Custom Cost Matrix):
```python
import numpy as np
from aggmetrics import IndicatorMatrix

# 5-class ordinal predictions (e.g. ratings 0, 1, 2, 3, 4)
y_true = np.array([0, 1, 2, 3, 4, 2, 1, 3, 0, 4])
y_pred = np.array([0, 2, 2, 2, 4, 3, 1, 4, 0, 3])

im = IndicatorMatrix(y_true, y_pred, task="multiclass")
m = im.m
n = im.n

# Construct cost matrices
J, K = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
C_mae = np.abs(J - K)
C_mse = (J - K) ** 2.0

mae = im.cost(C_mae) / n
mse = im.cost(C_mse) / n

print(f"MAE via Cost Matrix: {mae:.4f}")  # Exact match to np.abs(y_true - y_pred).mean()
print(f"MSE via Cost Matrix: {mse:.4f}")  # Exact match to ((y_true - y_pred)**2).mean()
```

### Cumulative Ordinal Evaluation:
```python
# Task 'ordinal' creates m-1 cumulative threshold binary columns
im_ord = IndicatorMatrix(y_true, y_pred, task="ordinal")

print("Ordinal matrix shape (n x (m-1)):", im_ord.Y.shape)
print("Ordinal Macro-Recall:", im_ord.compute("recall", "macro"))
print("Ordinal Macro-Precision:", im_ord.compute("precision", "macro"))
```
