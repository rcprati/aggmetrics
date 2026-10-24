# Evaluating a data stream

A stream is evaluated prequentially: each instance is predicted, and then, when its label is
known, used to update the evaluation. `aggmetrics.stream` keeps the state of that evaluation as a
confusion matrix that forgets, and computes any measure of the registry from it. **The subpackage is
experimental**: names and signatures may change in a later 0.x release. It needs only NumPy.

## A temporal state

```python
from aggmetrics.stream import TemporalState

state = TemporalState(2, alpha=0.98)          # exponential fading; or window=500
for y_true, y_pred in stream:                 # your prequential loop
    state.update(y_true, y_pred)

state.compute("f1")                           # any measure of the registry
state.compute("recall", "macro")              # multiclass: micro, macro or weighted
state.C                                       # the confusion matrix C[true, predicted]
```

With `alpha` the state is `C <- alpha C + y p^T`; with `window` it is the sum of the last
instances; with neither it is the cumulative count. A cumulative state gives exactly what
`IndicatorMatrix` gives on the same data, and a window state what it gives on the last `window`
instances. The state is a weighted count; divide by `state.mass` for the normalised state.
Every measure is a ratio of cells, so it does not depend on the normalisation.

A class with no support in the state (no instance in the window, or a mass below `tol` under
fading) has no value for the measures that divide by it. `compute` applies the convention you ask for
(`convention="exclude"` averages over the classes that have a value, `"zero"`, `"one"` and `"nan"`
are the others), and `return_audit=True` returns the audit: the value of each class, which classes
have one (`u`) and how many (`m_active`). A rare class that leaves a short window is therefore never
silently scored as zero.

## The value at another prior

Measures that are not skew-invariant (precision, F1, accuracy) move when the class prevalence
moves, even if the classifier does not. `compute(..., prior=pi)` returns the value the measure would
have with the rates of the state at the class prior `pi` (the prevalence of the positive class, or
the frequencies of the classes). Monitoring `state.compute("f1", prior=pi_ref)` gives a series that
a change of prevalence alone does not move. Recall and specificity are unchanged by construction.

## Rates that do not depend on the prior

Time-fading cells mix the prior and the rates: an accumulated recall is a prior-weighted average of
past recalls. `ClassTimeState` makes each class forget on its own arrivals, so its rates are free
of the prior, and a class that stops arriving keeps its last rates.

```python
from aggmetrics.stream import ClassTimeState

rates = ClassTimeState(2, alpha=0.98)
...
rates.canonical_coordinates()                 # (recall, specificity)
rates.compute("f1", prior=0.1)                # F1 with these rates at a 10% prevalence
```

## Labels that arrive late, or never

```python
from aggmetrics.stream import DelayedEvaluator

ev = DelayedEvaluator(2, alpha=0.98, eps=1e-3)
k = ev.arrive(proba, y_pred)                  # proba: the calibrated class probabilities
...
ev.label(k, y_true)                           # when the label arrives
ev.compute("recall")
```

Each instance first adds an interim term, the product t-norm of the stand-in distribution `proba`
and the prediction. If its label arrives while the instance is still in the queue, the interim
term is swapped for the verified one, with the weight it has decayed to. The product t-norm is the
only choice that keeps the prediction-side sums exact whatever the stand-ins are. With `eps`, queue
entries whose weight falls below `eps` are evicted: their term stays in the state and keeps decaying,
and the queue never holds more than `sizing.queue_bound(alpha, eps)` entries.

Because the label side of the state contains stand-ins, a class can have mass without a verified
instance. `ev.support()` returns the verified and the interim support, and `ev.coverage()` the share
of each class that is verified. `compute` uses the interim support for measures whose denominator is on
the prediction side (precision, accuracy), which is exact, and the verified support for the others
(recall, specificity, F1); `support="verified"` or `"interim"` forces one. A value for a class with
coverage between 0 and 1 is an estimate, and one with coverage 0 is not a measurement.

## Attributing a change

```python
from aggmetrics.stream import attribute

at = attribute("f1", reference.snapshot(), current.snapshot())
at.concept, at.prior, at.remainder
at.triage()                                    # 'concept' or 'prior'
```

The change of a measure between two snapshots is split into the part due to the rates (concept) and
the part due to the prior. When both change the split depends on the order; `method="counterfactual"`
applies the concept first and reports an interaction `remainder`, `"shapley"` averages the two orders,
`"jacobian"` integrates the gradient along the path (`steps`). `triage` calls a change concept-driven
if the concept part is at least a fraction `theta` of the total. The attribution classifies a change a
monitor already flagged; it does not detect concept change.

## Sizing before deployment

```python
from aggmetrics.stream import sizing

sizing.undefined_window_probability(0.02, 100, delay=25)   # P(class has no verified instance)
sizing.required_window(0.02, 0.05, delay=25)               # window for at most 5%
sizing.queue_bound(0.98, 1e-3)                             # queue cap under fading
sizing.effective_sample_size(0.98, kappa=8.8)              # what the estimate is worth
```

`undefined_window_probability` and `required_window` assume independent arrivals; the other functions
state their assumptions in their docstrings. `estimate_kappa` estimates the variance inflation factor
of dependent entries from a sequence; it includes any drift in the sequence.

## Monitoring (experimental)

`RateMonitor` is a chi-square on the class-time rates against a warm-up reference, and
`calibrate_threshold` calibrates its threshold by simulation to a run-level false alarm. The default is
not a 5% test: the statistic is evaluated at every step on an autocorrelated estimator, and the variances
assume independent arrivals, which native-order streams violate. Read the module docstring before using it.
