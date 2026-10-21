# Contributing

```bash
git clone https://github.com/rcprati/aggmetrics
cd aggmetrics
pip install -e ".[test,docs,examples]"
```

**Tests.** `python -m pytest` runs the suite. Statement and branch coverage are held at
100%, and continuous integration fails below that:

```bash
python -m pytest --cov=aggmetrics --cov-report=term-missing
```

A test should check a value against an independent computation, not only that a line runs.
Code that cannot be reached (a defensive guard) is marked `# pragma: no cover` with the
reason.

**Notebooks.** The notebooks in `examples/notebooks` are committed with their outputs and
re-executed in continuous integration (`pytest --nbmake examples/notebooks`).

**Documentation.**

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

**Source files.** Each file starts with the copyright and `SPDX-License-Identifier: BSD-3-Clause`
header.

**Changes.** Open a pull request from a short-lived branch. Describe what changed and why; for
a fix, say what the old behaviour was.
