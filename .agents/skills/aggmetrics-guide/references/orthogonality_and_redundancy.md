# Orthogonality, Degrees of Freedom & Non-Redundant Benchmark Suites

This reference details the geometric and algebraic principles governing the degrees of freedom ($d$) and canonical orthogonal bases of confusion matrices in `aggmetrics`.

---

## 1. Intrinsic Degrees of Freedom ($d$)

A confusion matrix has a strict number of independent degrees of freedom:

| Classification Setting | Number of Classes / Labels | Degrees of Freedom ($d$) | Canonical Orthogonal Coordinates |
|:---|:---|:---:|:---|
| **Binary** | $m = 2$ | **2** | $(\text{recall}, \text{specificity})$ |
| **Multiclass** | $m \ge 2$ | **$m(m-1)$** | $m(m-1)$ off-diagonal conditional misclassification rates $R_{jk} = P(\hat{Y}=k \mid Y=j)$ for $k \neq j$ |
| **Multilabel** | $m$ labels | **$2m$** | $(\text{recall}_j, \text{specificity}_j)$ for each label $j \in \{1, \dots, m\}$ |

In `aggmetrics`, query this via:
```python
im.degrees_of_freedom
```

---

## 2. The Canonical Orthogonal Basis (Row-Decoupling)

### Why Row-Decoupling implies Orthogonality:
In multiclass and binary classification, each row $j$ of the normalized confusion matrix is governed by a separate conditional distribution:
$$P(\hat{Y} \mid Y = j)$$
Because rows are conditioned on mutually exclusive true classes ($Y=j$), the perturbations of one row are statistically independent of perturbations of another row.

Under the Frobenius inner product $\langle A, B \rangle_F = \text{Tr}(A^T B)$, the directional derivatives / gradients with respect to each off-diagonal conditional error rate $R_{jk}$ ($k \neq j$) are **mutually orthogonal**:
$$\langle \nabla R_{jk}, \nabla R_{j'k'} \rangle_F = 0 \quad \text{for } (j, k) \neq (j', k')$$

### Extracting Canonical Coordinates:
```python
coords = im.canonical_coordinates()
```
- In binary: returns `np.array([recall, specificity])`.
- In multiclass: returns a 1D array of shape `(m*(m-1),)` containing all off-diagonal entries of the row-normalized matrix $R = C / \text{row\_sums}(C)$.

---

## 3. Detecting Redundant Evaluation Suites

When a benchmark suite requests $k > d$ scalar measures (for example, reporting `accuracy`, `balanced_accuracy`, `f1`, `precision`, `recall`, and `specificity` in a binary problem where $d=2$), the measures inevitably suffer from:
1. **Collinear Gradients**: Multiple metrics measure the same underlying trade-off projected onto different non-linear curves.
2. **Prior Entanglement**: Mixing prior-dependent metrics (`precision`, `f1`, `accuracy`) with prior-free metrics (`recall`, `specificity`) obscures whether changes in score stem from classifier performance or test-set class balance.

### Running Redundancy Checks:
```python
from aggmetrics.diagnostics import check_redundancy

suite = ["accuracy", "balanced_accuracy", "f1", "recall", "specificity"]
finding = check_redundancy(im, suite)

if finding:
    print(f"[{finding.severity.upper()}] {finding.message}")
    # [NOTE] The requested suite of 5 measures exceeds the 2 independent degrees of freedom...
```

---

## 3b. The rank test of Remark 2 on data

`suite_rank(im, measures, aggregation)` computes the Jacobian of the measures with respect to the free
parameters of the confusion matrix (the class sizes held fixed) and its rank, which is the criterion of
Remark 2: a suite of $k \le d$ measures is locally non-redundant if and only if the rank is $k$.
Measures are names or `(name, operator)` pairs with operator `micro`, `macro` or `weighted`. The result
lists the rank, the singular values, the measures whose gradient vanishes, and an *information volume*.

`select_suite(im, candidates, size)` returns the suite of that size with the largest volume. The volume
(the rows of the Jacobian scaled to unit length, and the volume they span) is a score added by the package, not
part of the paper's remark. It is 1 for orthogonal gradients and 0 for a redundant suite. For a binary
problem the best pair among all measures is (recall, specificity). The Jacobian is local: it changes with the
model, and masks, soft labels and ordinal data are not supported.

---

## 4. Best Practices for AI Assistants & Practitioners

1. **Prefer the Orthogonal Basis**: When presenting a concise summary of classifier performance without redundancy:
   - For Binary: report $(\text{recall}, \text{specificity})$ or $(\text{tpr}, \text{fpr})$.
   - For Multiclass: examine the canonical error coordinate matrix $R_{jk}$.
2. **Warn against Over-parameterized Metrics**: If a user requests 5+ scalar binary metrics, alert them using `check_redundancy()` and clarify which metrics are linear combinations or projections of the canonical coordinates.
3. **Separate Prior from Discrimination**: Pair a prior-free metric (like `balanced_accuracy` or `informedness`) with class prevalence $p(y)$ rather than relying on prior-conflating metrics (`f1`, `precision`) without stating base rates.
