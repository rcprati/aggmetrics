# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
aggmetrics.stream --- prequential evaluation of data streams (experimental).

The states keep a confusion matrix under a temporal operator (exponential fading
or a sliding window) and evaluate any measure of the registry on it.
"""
from . import sizing
from ._confusion import Audit
from .attribute import Attribution, attribute
from .delayed import DelayedEvaluator
from .monitor import RateMonitor, calibrate_threshold
from .state import ClassTimeState, Snapshot, TemporalState

__all__ = [
    "Attribution",
    "Audit",
    "ClassTimeState",
    "DelayedEvaluator",
    "RateMonitor",
    "Snapshot",
    "TemporalState",
    "attribute",
    "calibrate_threshold",
    "sizing",
]
