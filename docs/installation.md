# Installation

```bash
pip install aggmetrics                 # core: requires only NumPy
pip install "aggmetrics[all]"          # adds scikit-learn and pandas support
pip install "aggmetrics[examples]"     # the libraries the notebooks use (not the notebooks)
```

**The example notebooks** are in the repository, not in the pip package. Open them in Colab from
the badges in the README and the {doc}`gallery <gallery/index>`, or clone the repository and run
`pip install -e ".[examples]" jupyter`, then `jupyter lab examples/notebooks`.

**Requirements.** Python 3.9 or later and NumPy 1.20 or later. scikit-learn (1.0 or later)
is needed for the scorers and pandas (1.3 or later) for tabular reports; without pandas,
`IndicatorMatrix.report()` returns a dictionary.

**Development install.**

```bash
git clone https://github.com/rcprati/aggmetrics
cd aggmetrics
pip install -e ".[test,docs,examples]"
```

See {doc}`contributing` for the test, coverage and documentation commands.
