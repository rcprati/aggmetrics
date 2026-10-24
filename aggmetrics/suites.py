# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Choosing a suite of measures that carries as much information as possible.

Experimental: the signatures may change in a later 0.x release.

Remark 2 of the companion paper states a criterion for a *non-redundant* suite: a
set of ``k <= d`` measures is locally non-redundant if and only if the Jacobian
of the measures with respect to the free parameters of the confusion matrix has
full row rank ``k``, where ``d`` is the number of degrees of freedom
(``d = 2`` for a binary problem, ``m(m-1)`` for ``m`` classes, ``2m`` for ``m``
labels).

This module computes that Jacobian and its rank for a given data set, and adds a
numerical score that the paper does not state. The rows of the Jacobian are
scaled to unit length, and the *information volume* of a suite is the volume of
the parallelepiped they span, ``sqrt(det(J J^T))``. It lies in ``[0, 1]``, it is
0 exactly when the suite is redundant (the rank test fails), and it is 1 when the
gradients are mutually orthogonal, as for ``(recall, specificity)`` in the binary
case. ``select_suite`` picks the subset of candidates with the largest volume.

The Jacobian is local: it is evaluated at the confusion matrix of the data given,
with the class sizes held fixed, so it describes how the measures respond to
changes in the classifier and not to changes in the prevalence. Three operators
are functions of the confusion matrix alone and are supported: ``'micro'``,
``'macro'`` and ``'weighted'``. The exemplar operator depends on the examples and
is not.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._rates import Rates
from .measures import get_measure, list_measures

Item = Union[str, Tuple[str, str]]

_OPERATORS = ("micro", "macro", "weighted")
_EXHAUSTIVE_LIMIT = 200_000


@dataclass(frozen=True)
class SuiteRank:
    """Result of :func:`suite_rank`."""

    items: Tuple[Tuple[str, str], ...]
    rank: int
    degrees_of_freedom: int
    volume: float
    singular_values: np.ndarray
    flat: Tuple[Tuple[str, str], ...]

    @property
    def redundant(self) -> bool:
        """True if the suite does not have full row rank."""
        return self.rank < len(self.items)


@dataclass(frozen=True)
class SuiteSelection:
    """Result of :func:`select_suite`."""

    items: Tuple[Tuple[str, str], ...]
    volume: float
    rank: int
    method: str


def _items(measures: Sequence[Item], aggregation: str):
    out = []
    for it in measures:
        name, how = (it, aggregation) if isinstance(it, str) else it
        if how not in _OPERATORS:
            raise ValueError(
                f"operator {how!r} is not a function of the confusion matrix; "
                f"use one of {_OPERATORS}")
        out.append((get_measure(name).name, how))
    if len(set(out)) != len(out):
        raise ValueError("the suite repeats a (measure, operator) pair")
    return out


def _jacobian(model, items):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.vstack([model.gradient(m, h) for m, h in items])


def _volume(unit):
    gram = unit @ unit.T
    return float(np.sqrt(max(np.linalg.det(gram), 0.0)))


def _score(unit, rows):
    # rounded, so that ties between subsets are broken by the order of the candidates
    return round(_volume(unit[list(rows)]), 12)


def _unit_rows(J, tol):
    norms = np.linalg.norm(J, axis=1)
    flat = norms <= tol
    unit = J / np.where(flat, 1.0, norms)[:, None]
    unit[flat] = 0.0
    return unit, flat


def suite_jacobian(im, measures: Sequence[Item],
                   aggregation: str = "macro"):
    """Jacobian of a suite of measures with respect to the free parameters.

    Returns ``(J, names)``: ``J`` has one row per measure and one column per
    free parameter of the confusion matrix, evaluated at the data of ``im``, and
    ``names`` labels the columns. A measure is a name or a ``(name, operator)``
    pair; a bare name uses ``aggregation``.
    """
    model = Rates(im)
    return _jacobian(model, _items(measures, aggregation)), list(model.names)


def suite_rank(im, measures: Sequence[Item], aggregation: str = "macro",
               tol: float = 1e-6) -> SuiteRank:
    """Rank test of Remark 2: is the suite locally non-redundant?

    ``rank`` is the number of independent directions the suite spans. If it is
    smaller than the number of measures, some measure is a local function of
    the others. A measure whose gradient vanishes at the data (it does not
    respond to the classifier, for example a measure at its bound) is listed in
    ``flat``. ``volume`` is the information volume described in the module
    docstring.
    """
    model = Rates(im)
    items = _items(measures, aggregation)
    J = _jacobian(model, items)
    unit, flat = _unit_rows(J, tol)
    sv = np.linalg.svd(unit, compute_uv=False)
    return SuiteRank(
        items=tuple(items), rank=int((sv > tol).sum()),
        degrees_of_freedom=len(model.theta), volume=_volume(unit),
        singular_values=sv,
        flat=tuple(it for it, f in zip(items, flat) if f))


def select_suite(im, candidates: Optional[Sequence[Item]] = None,
                 size: Optional[int] = None, aggregation: str = "macro",
                 tol: float = 1e-6) -> SuiteSelection:
    """Pick the suite of ``size`` measures with the largest information volume.

    ``candidates`` defaults to every measure of the registry under
    ``aggregation``. ``size`` defaults to the degrees of freedom and cannot exceed the
    number of independent directions the candidates span (a larger suite is
    redundant by construction). The search is exhaustive when the number
    of subsets is at most 200,000 and greedy otherwise (the best pair first,
    then the candidate that most increases the volume at each step). Ties in the
    volume are broken by the order of ``candidates``. Candidates whose gradient
    vanishes at the data are never chosen.
    """
    model = Rates(im)
    if candidates is None:
        # measures where greater is better come first, so that they win ties
        candidates = sorted(list_measures(),
                            key=lambda n: not get_measure(n).greater_is_better)
    names = candidates
    items = _items(names, aggregation)
    J = _jacobian(model, items)
    unit, flat = _unit_rows(J, tol)
    keep = [i for i in range(len(items)) if not flat[i]]
    cap = int((np.linalg.svd(unit[keep], compute_uv=False) > tol).sum()) if keep else 0
    k = min(len(model.theta), cap) if size is None else size
    if k < 1 or k > cap:
        raise ValueError(f"size must be between 1 and {cap}, the number of independent "
                         f"directions the candidates span")
    if k == 1 or comb(len(keep), k) <= _EXHAUSTIVE_LIMIT:
        method = "exhaustive"
        best = max(combinations(keep, k), key=lambda s: _score(unit, s))
        chosen = list(best)
    else:
        method = "greedy"
        chosen = list(max(combinations(keep, 2), key=lambda s: _score(unit, s)))
        while len(chosen) < k:
            rest = [i for i in keep if i not in chosen]
            chosen.append(max(rest, key=lambda i: _score(unit, chosen + [i])))
    sv = np.linalg.svd(unit[chosen], compute_uv=False)
    return SuiteSelection(
        items=tuple(items[i] for i in chosen), volume=_volume(unit[chosen]),
        rank=int((sv > tol).sum()), method=method)
