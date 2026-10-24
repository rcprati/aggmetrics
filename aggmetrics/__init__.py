# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
aggmetrics --- classification evaluation via indicator matrices.

Any binary performance measure expressible in terms of the four confusion-matrix
counts is extended to multiclass, multilabel, ordinal, cost-sensitive and
soft-label settings by substituting one of three aggregation operators, so that
micro-, macro- and exemplar-averaging are three instances of one construction
rather than three separate conventions.

Quick start
-----------
>>> import numpy as np
>>> from aggmetrics import IndicatorMatrix
>>> y_true = np.array([0, 1, 2, 2, 1, 0])
>>> y_pred = np.array([0, 2, 2, 1, 1, 1])
>>> im = IndicatorMatrix(y_true, y_pred, task='multiclass')
>>> round(im.compute('f1', 'macro'), 4)
0.5222
>>> round(im.compute('f1', 'micro'), 4)
0.5
>>> round(im.accuracy(), 4)
0.5

Micro-F1 equals accuracy on single-label data, so reporting both adds nothing;
macro-F1 differs because it weights the three classes equally.

The package also reports when a measure is being used outside the range where it
behaves as commonly assumed:

>>> from aggmetrics import diagnose
>>> codes = sorted({f.code for f in diagnose(im)})
>>> 'micro_collapse' in codes
True
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "Ronaldo C. Prati"
__license__ = "BSD-3-Clause"

from .aggregation import (
    aggregate,
    cells,
    fold_aggregate,
    sigma_1,
    sigma_k,
    sigma_m,
    sigma_n,
)
from .core import IndicatorMatrix, pareto_dominance
from .diagnostics import (
    Finding,
    MeasureWarning,
    check_hull,
    check_ovr_skew_drift,
    check_redundancy,
    diagnose,
    micro_macro_gap,
    uniform_error_deviation,
)
from .inference import (
    InferredProperties,
    infer_properties,
    register_measure,
    verify_inference,
)
from .measures import MEASURES, BinaryMeasure, get_measure, list_measures
from .sklearn_compat import make_scorer, score, sklearn_average
from .suites import SuiteRank, SuiteSelection, select_suite, suite_jacobian, suite_rank
from .tnorms import TNORMS, get_tnorm, marginal_defect, soft_cells

__all__ = [
    "__version__",
    # core
    "IndicatorMatrix",
    "pareto_dominance",
    # measures
    "BinaryMeasure",
    "MEASURES",
    "get_measure",
    "list_measures",
    # operators
    "aggregate",
    "cells",
    "fold_aggregate",
    "sigma_1",
    "sigma_m",
    "sigma_n",
    "sigma_k",
    # soft labels
    "TNORMS",
    "get_tnorm",
    "marginal_defect",
    "soft_cells",
    # diagnostics
    "Finding",
    "MeasureWarning",
    "check_hull",
    "check_ovr_skew_drift",
    "check_redundancy",
    "diagnose",
    "micro_macro_gap",
    "uniform_error_deviation",
    # inference
    "register_measure",
    "infer_properties",
    "verify_inference",
    "InferredProperties",
    # choosing a suite of measures (Remark 2)
    "suite_jacobian",
    "suite_rank",
    "select_suite",
    "SuiteRank",
    "SuiteSelection",
    # scikit-learn
    "make_scorer",
    "score",
    "sklearn_average",
]
