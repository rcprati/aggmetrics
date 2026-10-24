# Positive-unlabeled data

In positive-unlabeled (PU) learning the ground truth is the observed label: 1 marks a confirmed
positive, and 0 an unlabeled example that may be a latent positive or a true negative. Under the
selected-completely-at-random assumption a positive is labelled with a constant propensity
$c = P(s = 1 \mid y = 1)$, and the cells computed from the observed labels are, in expectation,

$$
tp_\text{obs} = c\,tp, \quad fn_\text{obs} = c\,fn, \quad
fp_\text{obs} = fp + (1-c)\,tp, \quad tn_\text{obs} = tn + (1-c)\,fn,
$$

where $tp, fp, fn, tn$ are the cells with respect to the latent labels. Recall is the same in both,
because $c$ cancels. Precision, accuracy and every measure that uses $fp$ or $tn$ are biased.

`pu_correct` inverts the system, given $c$ or the prevalence $\pi = P(y = 1)$ of the latent positives
(which gives $c = P(s=1)/\pi$):

```python
import numpy as np
from aggmetrics import IndicatorMatrix, pu_correct

# observed labels s and a classifier's predictions
s = np.array([1] * 200 + [0] * 1800)
p = np.array([1] * 150 + [0] * 50 + [1] * 350 + [0] * 1450)
im = IndicatorMatrix(s, p, task="binary")

im.compute("precision"), im.compute("recall")            # 0.3, 0.75: nominal values
fixed = pu_correct(im, propensity=0.5)                   # or prior=0.2
fixed.compute("precision"), fixed.compute("recall")      # 0.6, 0.75
fixed.counts                                             # tp, fp, fn, tn = 300, 200, 100, 1400
```

The result offers `compute`, `accuracy`, `counts` and `per_class` on the corrected cells.

## What it needs and what it does not do

- The propensity (or the prevalence) cannot be recovered from the observed labels. It has to come from
  outside: a labelled sample, domain knowledge, or an estimator such as that of Elkan and Noto. The
  package does not estimate it.
- The correction is an expectation. On a finite sample it recovers the latent cells up to sampling error.
- A propensity too small for the data would need negative counts of false positives or true negatives
  and is rejected, with the smallest value that the data allow. In particular, the propensity must be at
  least the nominal precision.
- The assumption is that the labelled positives are a random sample of the positives. If the labelling
  depends on the example, the correction is wrong, and no result of the package says so.
- Only binary problems are supported.
