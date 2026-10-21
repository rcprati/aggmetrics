# scikit-learn integration

`make_scorer(measure, aggregation="micro", *, task=None, convention="exclude", threshold=0.5)`
returns a scorer usable wherever `scikit-learn` takes a `scoring` argument. Measures for which
smaller is better are negated, so larger is always preferable.

```python
from aggmetrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score

scorer = make_scorer("mcc", "macro", task="multiclass")
cross_val_score(clf, X, y, cv=5, scoring=scorer)
GridSearchCV(clf, grid, scoring=scorer)
```

**A caution.** A scorer evaluates each fold separately, so it averages the per-fold scores.
For a non-linear measure, pooling the folds' predictions gives a different number; see
{doc}`aggregation`.

**Diagnostics in a test suite.**

```python
def test_evaluation_integrity():
    im = IndicatorMatrix(y_test, y_pred, task="multiclass")
    findings = diagnose(im, emit_warnings=True)
    assert not any(f.severity == "warning" for f in findings)
```

:::{note}
**Sketch.** To write: `score` and `sklearn_average`; pipelines that report several
measures at once; a pytest plugin idea.
:::
