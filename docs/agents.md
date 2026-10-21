# For AI coding assistants

The repository ships an [agent skill](https://github.com/rcprati/aggmetrics/tree/main/.agents/skills/aggmetrics-guide)
(`.agents/skills/aggmetrics-guide/`): a `SKILL.md` with recipes and a set of reference pages.
An assistant that loads it knows the API, the aggregation operators and the diagnostics,
and is told what to check before it reports a number.

- [`SKILL.md`](https://github.com/rcprati/aggmetrics/blob/main/.agents/skills/aggmetrics-guide/SKILL.md)
  lists ten recipes: classification, diagnostics, soft labels, custom measures,
  `scikit-learn`, redundancy, Pareto dominance, the micro-macro gap, ordinal and cost, and
  cross-validation.
- [A worked session](https://github.com/rcprati/aggmetrics/blob/main/.agents/skills/aggmetrics-guide/examples/agent_session_example.md)
  shows a real request answered by following the skill, with the output verbatim. The script
  behind it runs in continuous integration, so the example cannot go stale.

The recipes in `SKILL.md` are executed as part of the checks for this repository.
