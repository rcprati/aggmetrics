# scikit-learn integration

`make_scorer(measure, aggregation="micro", *, task=None, convention="exclude", threshold=0.5)`
returns a scorer usable wherever `scikit-learn` takes a `scoring` argument. Measures for which
smaller is better are negated, so larger is always preferable: an `error_rate` scorer returns
$-0.006$ for an error rate of $0.006$.

```python
from aggmetrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score

scorer = make_scorer("mcc", "macro", task="multiclass")
cross_val_score(clf, X, y, cv=5, scoring=scorer)
GridSearchCV(clf, grid, scoring=scorer)
```

## Several measures at once

`scikit-learn` accepts a dictionary of scorers, so one cross-validation run reports a grid of
measures and operators:

```python
from sklearn.model_selection import cross_validate

scoring = {
    "macro_f1": make_scorer("f1", "macro"),
    "macro_mcc": make_scorer("mcc", "macro"),
    "error_rate": make_scorer("error_rate", "micro"),
}
result = cross_validate(clf, X, y, cv=5, scoring=scoring)
result["test_macro_f1"].mean()      # 0.9694 on digits
```

## One-shot scoring

`score(y_true, y_pred, measure, aggregation, *, task=None, convention="exclude")` is a function in
the style of `sklearn.metrics`:

```python
from aggmetrics import score

score(y, pred, "f1", "macro", task="multiclass")
```

`sklearn_average(average)` translates a `scikit-learn` `average=` value into an operator name
(`'samples'` becomes `'exemplar'`, `None` becomes `'macro'`).

## Pooled evaluation inside scikit-learn

A scorer is called once per fold, so the number `cross_val_score` reports is the *mean of
per-fold scores*. For a non-linear measure that differs slightly from scoring all the held-out
predictions together ({doc}`aggregation`). To pool, collect the out-of-fold predictions and score
them once:

```python
from sklearn.model_selection import cross_val_predict
from aggmetrics import IndicatorMatrix

pred = cross_val_predict(clf, X, y, cv=cv)
IndicatorMatrix(y, pred, task="multiclass", n_classes=10).compute("f1", "macro")
```

On `digits` with five stratified folds this gives macro-$F_1$ 0.96943, equal to
`fold_aggregate(folds, "f1", "macro")`, while the mean of the five scorer values is 0.9694
and the sample-weighted mean of fold scores is 0.96939.

## Things to know

- The scorer builds the `IndicatorMatrix` from `y` and the predictions of each fold and infers the
  number of classes from the largest label there. A class that is absent from both in some fold
  is either left out of that fold's macro average (if it is the highest-numbered class) or enters
  as an undefined entry (otherwise). With the default `'exclude'` convention neither changes the
  value; with `'zero'` or `'one'` they can. Pooling the out-of-fold predictions as above, with
  `n_classes`, avoids this.
- The scorer works with estimators that return labels from `predict`. For probabilistic
  outputs, build the `IndicatorMatrix` yourself and pass the probability matrix as `y_pred`
  (`task="multiclass"` argmaxes it, `"multilabel"` thresholds it).
- `diagnose` is not available through the scorer interface. Call it on the pooled predictions, in
  a test:

```python
def test_evaluation_integrity():
    im = IndicatorMatrix(y_test, y_pred, task="multiclass")
    findings = diagnose(im, ["f1", "recall"], emit_warnings=True)
    assert not any(f.code == "undefined_entries" for f in findings)
```
