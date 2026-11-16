# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Attributing the change of a measure to the concept and to the prior.

A measure that is not skew-invariant (precision, F1, accuracy) moves when the class
prior moves even if the classifier does not, so an alarm on it does not say what
changed. Between two snapshots, a reference and the current state, the change is
reparametrised in the canonical coordinates, the conditional rates ``R`` and the
prior ``pi``: the *concept* part is what the change of ``R`` does at a fixed prior,
and the *prior* part what the change of ``pi`` does at fixed rates.

When both change, the split depends on the order in which the two changes are
applied. ``method='counterfactual'`` applies the concept first, at the reference
prior, and reports the remainder as an interaction term; ``'shapley'`` averages the
two orders, which leaves no interaction; ``'jacobian'`` integrates the gradient of
the measure along the straight path between the snapshots in ``steps`` midpoint
steps, a first-order quadrature whose residual grows with the size of the change
and is reported.

The attribution does not detect concept change: it classifies a change that a
monitor on the measure has already flagged. A concept change that the measure does
not register is not seen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..measures import get_measure
from .state import Snapshot

__all__ = ["Attribution", "attribute"]


@dataclass(frozen=True)
class Attribution:
    """The change of a measure between two snapshots, split in two parts.

    ``total = value(current) - value(reference)``. ``concept`` and ``prior`` are the
    two parts; ``remainder`` is ``total - concept - prior`` (the interaction term for
    ``'counterfactual'``, the quadrature residual for ``'jacobian'``, zero for
    ``'shapley'``).
    """

    total: float
    concept: float
    prior: float
    remainder: float
    method: str

    def triage(self, theta: float = 0.5) -> str:
        """``'concept'`` if the concept part is at least a fraction ``theta`` of
        ``|total|``, ``'prior'`` otherwise, ``'none'`` if the measure did not move."""
        if not np.isfinite(self.total) or self.total == 0.0:
            return "none"
        return "concept" if abs(self.concept) >= theta * abs(self.total) else "prior"


def _value(model, theta, measure, how, cls, convention) -> float:
    if cls is None:
        return model.value(measure, how, theta, convention)
    tp, fp, fn, tn = model.cells(theta)
    return float(measure(tp[cls], fp[cls], fn[cls], tn[cls]))


def _at(snap: Snapshot, prior):
    return snap.model.with_priors(prior)


def _dprior(model_ref, theta, pri, dpri, measure, how, cls, convention) -> float:
    """Derivative of the measure along ``dpri`` at (theta, pri)."""
    pri, dpri = np.asarray(pri, float), np.asarray(dpri, float)
    nz = dpri != 0
    if not nz.any():
        return 0.0
    if model_ref.task == "multiclass":
        room = pri[nz] / np.abs(dpri[nz])
    else:
        room = np.minimum(pri, 1.0 - pri) / np.abs(dpri)
    h = float(min(1e-6, 0.25 * np.min(room)))
    hi = _value(model_ref.with_priors(pri + h * dpri), theta, measure, how, cls,
                convention)
    lo = _value(model_ref.with_priors(pri - h * dpri), theta, measure, how, cls,
                convention)
    return (hi - lo) / (2 * h)


def _dtheta(model, theta, measure, how, cls, convention, h=1e-6) -> np.ndarray:
    g = np.empty_like(theta)
    for i, t in enumerate(theta):
        a, b = theta.copy(), theta.copy()
        a[i], b[i] = max(t - h, 0.0), min(t + h, 1.0)
        g[i] = (_value(model, b, measure, how, cls, convention)
                - _value(model, a, measure, how, cls, convention)) / (b[i] - a[i])
    return g


def attribute(measure, reference: Snapshot, current: Snapshot, *,
              aggregation: str = "micro", cls=None, method: str = "counterfactual",
              steps: int = 1, convention: str = "exclude") -> Attribution:
    """Split the change of ``measure`` between two snapshots.

    Parameters
    ----------
    measure
        A measure of the registry.
    reference, current
        :class:`~aggmetrics.stream.state.Snapshot` objects of the same task and
        number of classes (``state.snapshot()``).
    aggregation
        ``'micro'``, ``'macro'`` or ``'weighted'``.
    cls
        Multiclass only: index of one class, to attribute its one-vs-rest measure
        instead of an aggregate.
    method
        ``'counterfactual'``, ``'shapley'`` or ``'jacobian'`` (see the module
        docstring).
    steps
        Jacobian only: number of midpoint steps along the path (1 is the midpoint
        rule; more is the path integral).
    """
    if (reference.task != current.task
            or reference.model.theta.shape != current.model.theta.shape):
        raise ValueError("the snapshots must have the same task and classes")
    if method not in ("counterfactual", "shapley", "jacobian"):
        raise ValueError("method must be 'counterfactual', 'shapley' or 'jacobian'")
    meas = get_measure(measure)
    if cls is not None and reference.task == "binary":
        raise ValueError("cls applies to multiclass snapshots")

    Rr, Rc = reference.model.theta, current.model.theta
    pr, pc = reference.prior, current.prior

    def M(theta, prior):
        return _value(_at(reference, prior), theta, meas, aggregation, cls, convention)

    m_ref, m_cur = M(Rr, pr), M(Rc, pc)
    total = m_cur - m_ref

    if method in ("counterfactual", "shapley"):
        concept_at_ref = M(Rc, pr) - m_ref     # concept first, at the reference prior
        prior_at_ref = M(Rr, pc) - m_ref       # prior first, at the reference rates
        if method == "counterfactual":
            concept, prior = concept_at_ref, prior_at_ref
        else:
            concept = 0.5 * (concept_at_ref + (m_cur - M(Rr, pc)))
            prior = 0.5 * (prior_at_ref + (m_cur - M(Rc, pr)))
        return Attribution(total, concept, prior, total - concept - prior, method)

    if steps < 1 or int(steps) != steps:
        raise ValueError("steps must be a positive integer")
    dR = Rc - Rr
    dpi = np.asarray(pc, float) - np.asarray(pr, float)
    concept = prior = 0.0
    for i in range(int(steps)):
        f = (i + 0.5) / steps
        theta_m = Rr + f * dR
        pri_m = np.asarray(pr, float) + f * dpi
        pri_m = float(pri_m) if pri_m.ndim == 0 else pri_m
        model_m = _at(reference, pri_m)
        grad = _dtheta(model_m, theta_m, meas, aggregation, cls, convention)
        concept += float(np.dot(grad, dR)) / steps
        prior += _dprior(reference.model, theta_m, pri_m, dpi, meas, aggregation,
                         cls, convention) / steps
    return Attribution(total, concept, prior, total - concept - prior, "jacobian")
