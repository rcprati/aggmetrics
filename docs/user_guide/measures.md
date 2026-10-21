# Measures and their properties

The registry holds 21 binary measures. Names resolve through aliases (`iou` reaches `jaccard`,
`dice` reaches `f1`). The table below is generated from the registry when the documentation is
built, so it always matches the code.

```{include} ../_generated/measures_table.md
```

**Reading the table.**

- *Decomposable*: numerator and denominator are linear in the counts, so micro-averaging
  is a weighted average of the per-class values.
- *Skew-invariant*: a function of recall and specificity only, so it does not change when the
  two classes are rescaled independently.
- *Depends on tn at tp=0*: the value when there are no true positives still depends on the
  true negatives. This is what lifts the floor in multiclass problems (see the
  {doc}`floor-trap notebook <../gallery/01_floor_trap>`).
- *Bayes threshold*: the decision threshold on the conditional probability that maximises
  the measure, when a closed form exists.

```python
from aggmetrics import get_measure, list_measures

list_measures(decomposable=True)
get_measure("mcc").decomposable            # False
get_measure("mcc").attainable_floor(10)    # -0.1111
```

## Definitions

All definitions are in terms of the four counts $tp, fp, fn, tn$ with $n = tp+fp+fn+tn$.
"Undefined when" is the condition under which the formula divides by zero; the value is then
NaN, and the treatment in an average is set by the `convention` argument
({doc}`aggregation`).

| Measure | Definition | Undefined when |
|---|---|---|
| `precision` | $tp / (tp+fp)$ | nothing is predicted positive: $tp+fp=0$ |
| `recall` | $tp / (tp+fn)$ | there are no positives: $tp+fn=0$ |
| `specificity` | $tn / (tn+fp)$ | there are no negatives: $tn+fp=0$ |
| `npv` | $tn / (tn+fn)$ | nothing is predicted negative: $tn+fn=0$ |
| `fnr` | $fn / (tp+fn)$ | $tp+fn=0$ |
| `fpr` | $fp / (fp+tn)$ | $tn+fp=0$ |
| `fdr` | $fp / (tp+fp)$ | $tp+fp=0$ |
| `for` | $fn / (fn+tn)$ | $tn+fn=0$ |
| `f1` (also `f0.5`, `f2`) | $F_\beta = (1+\beta^2)\,tp \,/\, ((1+\beta^2)\,tp + \beta^2\,fn + fp)$ | $tp=fp=fn=0$ |
| `jaccard` | $tp / (tp+fp+fn)$ | $tp=fp=fn=0$ |
| `balanced_accuracy` | $(\text{recall} + \text{specificity})/2$ | recall or specificity undefined |
| `gmean` | $\sqrt{\text{recall}\cdot\text{specificity}}$ | recall or specificity undefined |
| `informedness` | $\text{recall} + \text{specificity} - 1$ | recall or specificity undefined |
| `markedness` | $\text{precision} + \text{npv} - 1$ | precision or npv undefined |
| `fowlkes_mallows` | $\sqrt{\text{precision}\cdot\text{recall}}$ | precision or recall undefined |
| `mcc` | $(tp\,tn - fp\,fn) / \sqrt{(tp+fp)(tp+fn)(tn+fp)(tn+fn)}$ | any of the four sums in the root is 0 |
| `kappa` | $(p_o - p_e)/(1 - p_e)$, with $p_o = (tp+tn)/n$ and $p_e = ((tp+fn)(tp+fp) + (tn+fp)(tn+fn))/n^2$ | chance agreement is 1: all true and predicted labels coincide in one class |
| `label_accuracy` | $(tp+tn)/n$ | never (for $n>0$) |
| `error_rate` | $(fp+fn)/n$ | never (for $n>0$) |

Note that $F_1$ and Jaccard are defined whenever there is at least one positive example or
prediction: precision may be undefined for a class that is never predicted while its $F_1$ is
simply 0. This is why `macro-F1` can be computed in cases where `macro-precision` cannot.

## Choosing among the skew-invariant measures

Seven measures are skew-invariant: `recall`, `specificity`, `fnr`, `fpr`, `balanced_accuracy`,
`gmean` and `informedness`. Each is a function of recall and specificity alone, so the value
does not depend on the proportion of positives in the test set. The others (precision,
$F_1$, MCC, accuracy, ...) do.

- **Comparing across test sets with different class priors**, or reporting a number that
  should not change when the class mix does, favours a skew-invariant measure.
- **Predicting what will happen in deployment** at a known prevalence favours a measure that
  reflects it, such as precision or $F_1$, evaluated at that prevalence.
- Among the seven, `balanced_accuracy` and `informedness` are the same information
  ($\text{informedness} = 2\cdot\text{balanced accuracy} - 1$). `gmean` is zero as soon as
  either recall or specificity is zero, which makes it more severe than the arithmetic mean.
- Recall and specificity are *degenerate objectives*: each is maximised by predicting
  everything positive (respectively negative). `diagnose` flags this; do not use either alone to
  tune a threshold ({doc}`diagnostics`).
- On a many-class problem, most of these measures depend on true negatives and carry a floor
  (see the "Depends on tn" column); recall, precision, $F_1$ and Jaccard do not.
