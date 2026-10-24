# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Positive-unlabeled (PU) learning: correcting the cells for the labelling propensity.

In PU data the ground truth is the observed label ``s``: ``s = 1`` marks a confirmed
positive, and ``s = 0`` is an unlabeled example that may be a latent positive
(``y = 1``) or a true negative. Under the selected-completely-at-random assumption a
positive is labelled with a constant propensity ``c = P(s = 1 | y = 1)``, and the
observed cells (Section 6 of the companion paper) are, in expectation,

    tp_obs = c tp      fn_obs = c fn
    fp_obs = fp + (1 - c) tp      tn_obs = tn + (1 - c) fn,

where ``tp, fp, fn, tn`` are the cells with respect to the latent ``y``. Recall is
therefore the same in both (the factor ``c`` cancels), while precision, accuracy and
every measure that uses ``fp`` or ``tn`` are biased. Given ``c``, or the prevalence
``pi = P(y = 1)``, which gives ``c = P(s = 1) / pi``, the system is inverted:

    tp = tp_obs / c     fn = fn_obs / c
    fp = fp_obs - (1 - c) tp_obs / c     tn = tn_obs - (1 - c) fn_obs / c.

``c`` and ``pi`` cannot be recovered from the observed labels alone: they have to
come from outside (a labelled sample, domain knowledge, or an estimator such as
that of Elkan and Noto). The correction is an expectation: with a finite sample it
recovers the latent cells up to sampling error, and a value of ``c`` too small for
the data gives negative cells, which is rejected.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from ._rates import Rates
from .shift import PriorShifted


class PUCorrected(PriorShifted):
    """The cells of PU data corrected for the labelling propensity.

    Built by :func:`pu_correct`. It offers ``compute``, ``accuracy``, ``counts`` and
    ``per_class`` on the corrected cells, as :class:`~aggmetrics.shift.PriorShifted`
    does. ``priors`` holds the estimated prevalence of the latent positives, and
    ``propensity`` the propensity used.
    """

    def __init__(self, model: Rates, propensity: float):
        super().__init__(model, model.pos / model.n, "binary")
        self.propensity = float(propensity)


def pu_correct(im, *, propensity: Optional[float] = None,
               prior: Optional[float] = None) -> PUCorrected:
    """Correct the cells of PU data for the labelling propensity.

    Parameters
    ----------
    im
        A binary :class:`~aggmetrics.core.IndicatorMatrix` whose ground truth is the
        observed label (1 for a confirmed positive, 0 for an unlabeled example).
    propensity
        ``c = P(s = 1 | y = 1)``, in (0, 1].
    prior
        Alternatively, the prevalence of the latent positives ``P(y = 1)``. Give one
        of the two.

    Returns
    -------
    PUCorrected
        An object with ``compute``, ``accuracy``, ``counts`` and ``per_class`` on the
        cells corrected to the latent labels.

    Raises
    ------
    ValueError
        If neither or both of ``propensity`` and ``prior`` are given, if the value is
        out of range, or if it is too small for the data (the corrected cells would
        be negative).

    Examples
    --------
    >>> import numpy as np
    >>> from aggmetrics import IndicatorMatrix, pu_correct
    >>> # latent cells tp=300, fp=200, fn=100, tn=1400 and propensity c=0.5, observed as
    >>> # tp=150, fp=350, fn=50, tn=1450
    >>> s = np.array([1] * 200 + [0] * 1800)
    >>> p = np.array([1] * 150 + [0] * 50 + [1] * 350 + [0] * 1450)
    >>> im = IndicatorMatrix(s, p, task="binary")
    >>> round(im.compute("precision"), 3), round(im.compute("recall"), 3)
    (0.3, 0.75)
    >>> fixed = pu_correct(im, propensity=0.5)
    >>> [round(float(x)) for x in fixed.counts]
    [300, 200, 100, 1400]
    >>> round(fixed.compute("precision"), 3), round(fixed.compute("recall"), 3)
    (0.6, 0.75)
    """
    if im.task != "binary":
        raise ValueError("pu_correct needs a binary IndicatorMatrix")
    if (propensity is None) == (prior is None):
        raise ValueError("give exactly one of propensity and prior")
    model = Rates(im)
    pos = float(model.pos[0])
    if prior is not None:
        if not 0 < prior <= 1:
            raise ValueError("prior must be in (0, 1]")
        propensity = pos / (model.n * prior)
    if not 0 < propensity <= 1:
        raise ValueError("propensity must be in (0, 1]" if prior is None else
                         "the prior is smaller than the share of labelled positives")
    c = float(propensity)
    tp, fp, fn, tn = (float(x[0]) for x in model.cells(model.theta))
    t_tp, t_fn = tp / c, fn / c
    t_fp, t_tn = fp - (1 - c) * tp / c, tn - (1 - c) * fn / c
    if min(t_fp, t_tn) < -1e-9:
        floor = max(tp / (tp + fp) if tp + fp else 0.0,
                    fn / (fn + tn) if fn + tn else 0.0)
        raise ValueError(f"propensity {c:.4g} is too small for these data: it would "
                         f"need negative counts of false positives or true negatives "
                         f"(the propensity must be at least {floor:.4g})")
    out = copy.copy(model)
    out.pos = np.array([t_tp + t_fn])
    out.neg = np.array([max(t_fp, 0.0) + max(t_tn, 0.0)])
    rec = t_tp / out.pos[0] if out.pos[0] > 0 else 0.0
    spec = max(t_tn, 0.0) / out.neg[0] if out.neg[0] > 0 else 0.0
    out.theta = np.array([rec, spec])
    return PUCorrected(out, c)
