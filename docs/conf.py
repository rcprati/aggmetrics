# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""Sphinx configuration."""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import aggmetrics  # noqa: E402
import gen_measures  # noqa: E402

# -- generated content -------------------------------------------------------------
# the measures table comes from the registry, so it cannot drift from the code
gen_measures.write(HERE / "_generated" / "measures_table.md")
# the gallery is the example notebooks, copied so they have a single source
(HERE / "gallery").mkdir(exist_ok=True)
for nb in sorted((ROOT / "examples" / "notebooks").glob("*.ipynb")):
    shutil.copy2(nb, HERE / "gallery" / nb.name)

# -- project -----------------------------------------------------------------------
project = "aggmetrics"
author = "Ronaldo C. Prati"
copyright = "2026, Ronaldo C. Prati"
release = aggmetrics.__version__
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]
exclude_patterns = ["_build", "_generated", "Thumbs.db", ".DS_Store"]

# -- notebooks and markdown ----------------------------------------------------------
# The notebooks are executed in continuous integration; their stored outputs are shown.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- API -----------------------------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# -- theme ---------------------------------------------------------------------------
html_theme = "furo"
html_title = f"aggmetrics {release}"
html_theme_options = {
    "source_repository": "https://github.com/rcprati/aggmetrics/",
    "source_branch": "main",
    "source_directory": "docs/",
}
