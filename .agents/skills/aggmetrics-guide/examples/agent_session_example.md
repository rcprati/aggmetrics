# A coding agent using the `aggmetrics-guide` skill: a worked session

This page shows what an AI coding assistant does with the skill, on a realistic request. The
commands are real and the output is verbatim: it is produced by
[`agent_session_example.py`](./agent_session_example.py), which you can run yourself. The
narration is condensed.

## The request

> I trained a logistic regression and a random forest on an imbalanced 10-class problem
> (from 5 to 170 examples per class). The logistic regression has 0.89 accuracy. Which model
> should I deploy, and how should I report the result?

## Why the skill applies

The skill's `description` matches ("evaluate classification models", "classifier
diagnostics"), so the agent loads `SKILL.md`. Four of its recipes are relevant:

| Recipe | Used for |
|---|---|
| 1. Basic classification | indicator matrices, accuracy, macro measures |
| 2. Diagnostic analysis | what could make the headline number misleading |
| 7. Universal Pareto dominance | whether one model beats the other on *every* monotone measure |
| 10. Cross-validation aggregation | comparing the models with a pooled, fold-aware estimate |

## What the agent runs and sees

```text
1. Headline numbers (held-out half, 10 classes, 5 to 85 examples per class)
   logistic regression  accuracy=0.892  macro-F1=0.640  macro-recall=0.612
   random forest        accuracy=0.942  macro-F1=0.735  macro-recall=0.731

2. Diagnostics
   logistic regression: ['degenerate_optimum', 'inflated_floor', 'micro_collapse', 'undefined_entries']
     macro-precision: 2 of 10 classes never predicted; 0.938 with 'exclude' vs 0.751 with 'zero'
   random forest: ['degenerate_optimum', 'inflated_floor', 'micro_collapse', 'undefined_entries']
     macro-precision: 2 of 10 classes never predicted; 0.941 with 'exclude' vs 0.753 with 'zero'

3. Does one model beat the other on every monotone measure?
   pareto_dominance(logistic regression, random forest) = 'incomparable'
   -> the ranking depends on which measure is chosen; state the measure.

4. Five-fold cross-validation on all examples (pooled vs averaged macro-F1)
   logistic regression  pooled=0.696  averaged=0.691  sd across folds=0.028
   random forest        pooled=0.786  averaged=0.783  sd across folds=0.039
```

## What the skill changes in the agent's behaviour

The shortcut for this question is to compare the two accuracies and answer "the forest, 0.94
against 0.89". With the skill, the agent:

1. **Does not stop at accuracy.** The diagnostics flag `micro_collapse` (micro precision,
   recall and F1 all equal accuracy on single-label data), so the agent reports macro measures
   next to it, and names the operator with every number.
2. **Notices the never-predicted classes.** Each model leaves 2 of the 10 classes unpredicted
   on this split, so macro-precision is undefined for them. The agent reports the effect of the
   convention (0.938 against 0.751) instead of quoting one value, and does not use macro-precision
   as the headline.
3. **Checks for universal dominance before claiming a winner.** `pareto_dominance` returns
   `'incomparable'`: the forest is better on accuracy and macro-F1, but it does not dominate
   the logistic regression on every class, so the choice depends on the measure. The agent says so.
4. **Does not over-read one split.** With 2 to 5 test examples in the rarest classes, a single
   split is noisy. The agent pools five-fold cross-validation with `fold_aggregate`
   (`pooling="pooled"`, not the mean of per-fold scores), and compares the gap with the spread
   across folds.

## The answer the agent gives

> **Deploy the random forest, and report macro-F1 with the cross-validation spread.**
>
> - Pooled five-fold macro-F1 is **0.79** for the random forest against **0.70** for the
>   logistic regression. The gap (about 0.09) is larger than the standard deviation across folds
>   (0.03 to 0.04), so it is unlikely to be noise.
> - Accuracy (0.94 against 0.89) agrees but is not the right headline: on single-label data
>   micro precision, recall and F1 are all equal to it, and it is dominated by the frequent classes.
> - Neither model is better on *every* measure (`pareto_dominance` is `'incomparable'`), so state
>   which one you optimise. Here macro-F1, because the rare classes matter.
> - On the held-out half, each model leaves two of the ten classes unpredicted. Macro-precision
>   is therefore undefined for them and depends on the convention (0.94 excluding them, 0.75
>   counting them as zero). Report macro-recall or macro-F1, and state the `convention`.
> - Collect more examples of the rarest classes before relying on any per-class number.

## Reproduce

```bash
pip install "aggmetrics[examples]"
python .agents/skills/aggmetrics-guide/examples/agent_session_example.py
```
