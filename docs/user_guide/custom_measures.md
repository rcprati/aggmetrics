# Your own measures

`register_measure(name, fn)` accepts any function of $(tp, fp, fn, tn)$ and infers its
structural properties from the formula. The measure is then usable under every operator, and
`diagnose` knows about it.

```python
import numpy as np
from aggmetrics import register_measure, infer_properties

def prevalence_threshold(tp, fp, fn, tn):
    r, s = tp / (tp + fn), tn / (tn + fp)
    return (np.sqrt(r * (1 - s)) + s - 1) / (r + s - 1)

props = infer_properties(prevalence_threshold)     # properties only, nothing registered
print(props.summary())
m = register_measure("prevalence_threshold", prevalence_threshold)
m.decomposable        # False
m.skew_invariant      # True
im.compute("prevalence_threshold", "exemplar")
```

`register_measure` also takes `aliases`, `note`, `overwrite=True` (to replace an existing name)
and `seed` (the inference samples count vectors).

## How the properties are inferred

| Property | How |
|---|---|
| decomposable | solve a homogeneous linear system for coefficients $(a, b)$ with $M = \langle a, v\rangle / \langle b, v\rangle$ over sampled count vectors $v$, then validate on fresh vectors (tolerance $10^{-8}$) |
| skew-invariant | check that $M(\lambda\,tp, \mu\,fp, \lambda\,fn, \mu\,tn) = M(tp, fp, fn, tn)$ for random $\lambda, \mu$ |
| depends on tn at tp=0 | check whether the value at $tp = 0$ changes when only $tn$ changes |
| greater is better | check whether *correcting* a prediction, which moves the counts by $(+1, -1, -1, +1)$, raises the value |

The properties of the 21 built-in measures are also known analytically, so
`verify_inference()` compares the two. They agree on all 21, with recovered coefficients within
$6\times 10^{-15}$, about 28 times the double-precision machine epsilon.

## What inference cannot determine

Inference is numerical. It samples counts between 1 and 60 and cannot see the whole domain:

- **A formula that is undefined on part of the sampled domain.** Count vectors where the formula
  returns NaN are skipped for the skew and tn tests, but the linear fit needs finite values on
  its samples. A formula that returns NaN for $tp < 30$, say, is reported as not decomposable
  even when it equals precision wherever it is defined.
- **Direction is a heuristic.** It looks at what happens when a single prediction is corrected.
  For a measure that is not monotone in that move, or that is constant along it, the result is
  not reliable.
- **A near-linear formula.** The linear fit is accepted when its relative residual on held-out
  vectors is below the tolerance. `infer_properties` takes a `tol` argument; `register_measure`
  uses the default, so override the property if the default decides wrongly.
- **Behaviour at the edges** (zero counts, very large counts) beyond the explicit $tp = 0$
  check on $tn$ dependence.

## Overriding a property

When you know a property analytically, pass it as a keyword. It replaces the inferred value:

```python
def precision_after_30(tp, fp, fn, tn):          # NaN below 30 true positives
    return np.nan if tp < 30 else tp / (tp + fp)

infer_properties(precision_after_30).decomposable        # False: inferred
m = register_measure("precision_after_30", precision_after_30,
                     linear=((1, 0, 0, 0), (1, 1, 0, 0)))
m.decomposable                                           # True: overridden
```

The accepted keywords are `linear` (a pair of coefficient vectors over $(tp, fp, fn, tn)$, or
`None`), `skew_invariant`, `tn_dependent_at_zero` and `greater_is_better`. An unknown keyword
raises `TypeError`.

An override is a claim you make, and the library does not check it against the formula. If it
is wrong, identities that depend on decomposability, such as the one that relates micro to macro
averaging, will give wrong results.
