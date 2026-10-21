# T-Norms & Soft Labels Reference

When ground truth labels and/or predictions are real-valued probabilities in $[0, 1]$ (e.g., annotator agreement rates, crowd labels, probabilistic calibrated predictions), Boolean conjunctions are replaced by a continuous Triangular Norm ($t$-norm) $T(a, b)$ with complement $N(x) = 1 - x$:

$$\begin{aligned}
TP &= T(y, \hat{y}) \\
FP &= T(1-y, \hat{y}) \\
FN &= T(y, 1-\hat{y}) \\
TN &= T(1-y, 1-\hat{y})
\end{aligned}$$

---

## Available T-Norms

1. **`product`** ($T_P(a, b) = a \cdot b$):
   - **Crucial Mathematical Property**: The product $t$-norm is the **only** $t$-norm that is **marginally consistent**:
     $$TP + FN = y, \qquad TP + FP = \hat{y}$$
   - Preserves row and column marginal totals, behaving consistently like an empirical contingency table.
2. **`godel` / `min`** ($T_M(a, b) = \min(a, b)$):
   - The largest $t$-norm (Gödel / Zadeh minimum conjunction).
3. **`lukasiewicz`** ($T_L(a, b) = \max(0, a + b - 1)$):
   - Bounded difference $t$-norm; assigns non-zero overlap only when values sum to $> 1$.

---

## Checking Marginal Defect

You can measure the deviation of any $t$-norm from marginal consistency using `marginal_defect`:

```python
from aggmetrics.tnorms import marginal_defect

print(marginal_defect("product"))      # 0.0 (exact consistency)
print(marginal_defect("godel"))        # 0.5 (maximum marginal defect)
print(marginal_defect("lukasiewicz"))  # 0.5
```

---

## Evaluating Soft Labels with IndicatorMatrix

```python
import numpy as np
from aggmetrics import IndicatorMatrix

# Soft truth probabilities and soft predictions (n_samples, n_classes)
Y_soft = np.array([
    [0.8, 0.2],
    [0.1, 0.9],
    [0.6, 0.4]
])

P_soft = np.array([
    [0.7, 0.3],
    [0.2, 0.8],
    [0.5, 0.5]
])

# Evaluate using product t-norm
im = IndicatorMatrix(Y_soft, P_soft, task="multilabel", tnorm="product")

print("Micro F1:", im.compute("f1", "micro"))
print("Macro F1:", im.compute("f1", "macro"))
print("Exemplar F1:", im.compute("f1", "exemplar"))
```
