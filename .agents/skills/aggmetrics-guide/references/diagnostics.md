# aggmetrics Diagnostics Reference

The `diagnose(im)` function analyzes an `IndicatorMatrix` instance and returns structured `Finding` objects (eight codes, listed below) to alert users to statistical traps, redundant metrics, and misleading performance scores.

---

## Diagnostic Findings & Codes

### 1. `micro_collapse` (`severity: note`)
- **Condition**: On single-label multiclass or binary problems, $\text{micro-Precision} = \text{micro-Recall} = \text{micro-F1} = \text{Accuracy}$.
- **Insight**: Reporting multiple micro-averaged measures or micro-F1 alongside accuracy is redundant and adds no new information.

### 2. `inflated_floor` (`severity: warning`)
- **Condition**: Evaluated on $m$-class ($m > 2$) one-hot classification where a measure depends on $tn$ at $tp=0$ (e.g., `label_accuracy`, `specificity`, `npv`).
- **Insight**: Even when all predictions are completely wrong, misclassified examples contribute $m-2$ true negatives. The measure cannot reach its nominal minimum (e.g., `label_accuracy` floor is $\frac{m-2}{m}$), giving a falsely high sense of model competence.

### 3. `outside_hull` (`severity: warning`)
- **Condition**: A micro-averaged value falls outside the interval $[\min_j M_j, \max_j M_j]$ of the per-class macro values.
- **Insight**: Occurs only for **non-decomposable** measures (like `mcc`, `kappa`, `gmean`, `balanced_accuracy`). It proves that the micro-average cannot be interpreted as any weighted average of per-class performances.

### 4. `degenerate_optimum` (`severity: warning`)
- **Condition**: Optimizing decision thresholds for measures whose Bayes threshold is extreme (e.g., `recall` has $\theta^* = 0$, `specificity` has $\theta^* = 1$).
- **Insight**: Maximizing recall trivially leads to predicting positive for 100% of cases; optimizing such measures without constraint produces degenerate decision boundaries.

### 5. `ovr_skew_drift` (`severity: note`)
- **Condition**: In multiclass One-vs-Rest setups, off-diagonal error rates into classes are asymmetric.
- **Insight**: Macro-specificity and macro-balanced accuracy will drift under class prior distribution changes ($\Delta_{\text{BA}} = 0.5 \cdot \Delta_{\text{spec}}$).

### 6. `redundant_suite` (`severity: note`)
- **Condition**: The number of evaluated measures exceeds the intrinsic degrees of freedom $d = m(m-1)$ (or $d=2$ in binary).
- **Insight**: Extra scalar metrics beyond canonical bases have collinear gradients and yield redundant information.

### 7. `undefined_entries` (`severity: warning`)
- **Condition**: Some classes or instances have undefined counts (e.g. zero predictions or zero ground-truth instances resulting in $0/0$).
- **Insight**: Shows how much the chosen convention (`exclude`, `zero`, `one`) affects the final macro/exemplar score. The message gives the value under `exclude` and under `zero`.

### 8. `not_erm_consistent` (`severity: note`)
- **Condition**: A measure's closed-form Bayes threshold is not 0.5 (for example `recall`, `fnr`, `specificity`, `fpr`).
- **Insight**: Standard training, which targets the error rate at threshold 0.5, does not target such a measure; the decision threshold has to be tuned for it.

---

## Example Usage

```python
from aggmetrics import IndicatorMatrix, diagnose

im = IndicatorMatrix(y_true, y_pred, task="multiclass")

# Run diagnostic inspection
findings = diagnose(im, emit_warnings=False)

for finding in findings:
    print(f"[{finding.severity.upper()}] Code: {finding.code}")
    print(f"Message: {finding.message}")
    print(f"Details: {finding.detail}\n")
```
