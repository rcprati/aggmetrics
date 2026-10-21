# aggmetrics Measures & Axiomatic Properties Reference

`aggmetrics` includes 21 native binary performance measures. Each measure is characterized by structural/algebraic properties rather than just an empirical formula.

---

## The 21 Native Measures & Aliases

| Measure Name | Aliases | Decomposable ($\Sigma_1 = \Sigma_m^{\text{weighted}}$) | Skew-Invariant | Greater is Better | Notes |
|:---|:---|:---:|:---:|:---:|:---|
| `recall` | `sensitivity`, `tpr`, `hit_rate` | Yes | Yes | True | $\frac{tp}{tp+fn}$ |
| `specificity` | `tnr`, `selectivity` | Yes | Yes | True | $\frac{tn}{tn+fp}$, depends on $tn$ at $tp=0$ |
| `precision` | `ppv` | Yes | No | True | $\frac{tp}{tp+fp}$ |
| `npv` | - | Yes | No | True | $\frac{tn}{tn+fn}$, depends on $tn$ at $tp=0$ |
| `fpr` | `fallout` | Yes | Yes | False | $\frac{fp}{fp+tn}$, depends on $tn$ at $tp=0$ |
| `fnr` | `miss_rate` | Yes | Yes | False | $\frac{fn}{tp+fn}$ |
| `fdr` | - | Yes | No | False | $\frac{fp}{tp+fp}$ |
| `for` | `omission_rate` | Yes | No | False | $\frac{fn}{tn+fn}$, depends on $tn$ at $tp=0$ |
| `f1` | `dice`, `fscore`, `f1_score` | Yes | No | True | Harmonic mean of precision and recall |
| `f2` | - | Yes | No | True | $F_\beta$ with $\beta=2$ (recall weighted higher) |
| `f05` | `f0_5` | Yes | No | True | $F_\beta$ with $\beta=0.5$ (precision weighted higher) |
| `jaccard` | `iou`, `threat_score`, `critical_success_index`, `csi` | Yes | No | True | $\frac{tp}{tp+fp+fn}$ |
| `label_accuracy` | - | Yes | No | True | $\frac{tp+tn}{tp+fp+fn+tn}$, depends on $tn$ at $tp=0$ |
| `error_rate` | - | Yes | No | False | $\frac{fp+fn}{tp+fp+fn+tn}$, depends on $tn$ at $tp=0$ |
| `balanced_accuracy` | `bacc` | No | Yes | True | $\frac{1}{2}(TPR + TNR)$ |
| `informedness` | `youden`, `youden_j`, `bm`, `bookmaker_informedness` | No | Yes | True | $TPR + TNR - 1$ |
| `markedness` | `mk`, `delta_p` | No | No | True | $PPV + NPV - 1$ |
| `gmean` | `geometric_mean` | No | Yes | True | $\sqrt{TPR \cdot TNR}$ |
| `fowlkes_mallows` | `fm` | No | No | True | $\sqrt{PPV \cdot TPR}$ |
| `mcc` | `matthews_corrcoef` | No | No | True | Matthews correlation coefficient |
| `cohen_kappa` | `kappa` | No | No | True | Cohen's Kappa against chance agreement |

---

## Key Structural Properties

### 1. Aggregation Decomposability (`measure.decomposable`)
A measure is decomposable if it can be represented as a linear-fractional function:
$$M = \frac{\langle a, v \rangle}{\langle b, v \rangle}, \quad v = (tp, fp, fn, tn)$$
- **Theorem 1**: For decomposable measures, **micro-averaging ($\Sigma_1$) strictly equals denominator-weighted macro-averaging**.
- For non-decomposable measures (e.g., `mcc`, `kappa`, `gmean`, `balanced_accuracy`), micro-averaging does NOT decompose into a weighted sum, and the micro-average is not guaranteed to lie within the convex hull of per-class values.

### 2. Skew Invariance (`measure.skew_invariant`)
- A measure is skew-invariant if it depends only on $TPR$ and $TNR$ (or $FPR$ and $FNR$).
- It is unaffected by shifts in the class prior / base rate $p(y=1)$.
- In one-vs-rest multiclass problems, as the number of classes $m$ increases, non-skew-invariant measures experience severe skew drift.

### 3. Nominal Minimum & Attainable Floors (`measure.attainable_floor(m)`)
- On one-hot multiclass problems with $m$ classes, if a classifier misclassifies every instance ($tp=0$), each example still yields $m-2$ true negatives ($tn$).
- For measures that depend on $tn$ at $tp=0$ (e.g., `label_accuracy`, `specificity`), the score does **not** reach 0.
- For example, `label_accuracy` on a 10-class dataset where every prediction is wrong produces a score of **0.800**, misleading practitioners into thinking the model is performing well.

### 4. Bayes Decision Threshold (`measure.bayes_threshold()`)
- For linear-fractional measures whose denominator is prediction-independent, the threshold $\theta^*$ that maximizes expected utility under conditional probability $P(Y=1 \mid X)$ is given by:
$$\theta^* = \frac{\beta + \delta}{\alpha + \beta + \gamma + \delta}$$
- For example:
  - `accuracy`: $\theta^* = 0.5$ (ERM-consistent)
  - `recall`: $\theta^* = 0.0$ (predict all positive)
  - `specificity`: $\theta^* = 1.0$ (predict all negative)

### 5. The Micro-Macro Gap as Normalized Covariance (Corollary 5)
- For any **aggregation-decomposable measure**, the difference between micro-averaging and macro-averaging is mathematically identical to the sample covariance between per-class denominators $w_j$ and per-class values $v_j$, normalized by the mean denominator $\bar{w}$:
$$\text{Gap} = \text{Micro} - \text{Macro} = \frac{\operatorname{Cov}(w, v)}{\bar{w}}$$
- A positive gap indicates the model performs systematically better on high-volume/prevalent classes.
- For non-decomposable measures (like `mcc` or `balanced_accuracy`), this identity does not hold.
- Inspect via:
```python
from aggmetrics import micro_macro_gap
gap_info = micro_macro_gap(im, "f1")
print("Exact covariance gap:", gap_info["predicted_gap"])
```
