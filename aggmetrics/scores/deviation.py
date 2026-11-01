# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Deviation operators: the cells of the Calibration Framework.

A deviation operator scores one cell by a function of the signed error,
``D(y, p) = phi(y - p)`` with ``phi`` even and ``phi(0) = 0``. The squared error
gives the Brier score, the absolute error the mean absolute error, and
``phi(t) = -log(1 - |t|)`` the binary log-loss, which is extended-valued: a
confident wrong prediction (``|t| = 1``) costs ``+inf``.

For an even, convex ``phi`` that is finite and continuous on ``[-1, 1]`` and
differentiable on ``(0, 1)``, the operator is a proper scoring rule exactly when
the weight ``phi'(t) / t`` is symmetric about ``1/2`` (Corollary ``savage_cases``(c)
of the companion paper). :func:`check_properness` tests properness numerically for
any operator, including user-defined ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Union

import numpy as np

__all__ = ["Deviation", "get_deviation", "list_deviations", "power",
           "check_properness", "Properness"]

Phi = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Deviation:
    """A deviation operator ``D(y, p) = phi(y - p)``."""

    name: str
    phi: Phi
    proper: bool
    strictly_proper: bool
    note: str = ""

    def __call__(self, t) -> np.ndarray:
        return self.phi(np.asarray(t, float))


def _squared(t):
    return t * t


def _absolute(t):
    return np.abs(t)


def _log(t):
    a = np.abs(t)
    with np.errstate(divide="ignore"):
        return np.where(a >= 1.0, np.inf, -np.log1p(-np.minimum(a, 1.0)))


def power(s: float) -> Deviation:
    """The power loss ``|t|**s``; proper only for ``s = 2`` among ``s >= 1``."""
    if s <= 0:
        raise ValueError("the exponent must be positive")
    s = float(s)
    return Deviation(f"power({s:g})", lambda t: np.abs(t) ** s,
                     proper=(s == 2.0), strictly_proper=(s == 2.0),
                     note="squared error is the only proper power loss")


_REGISTRY: Dict[str, Deviation] = {
    "squared": Deviation("squared", _squared, True, True,
                         "the Brier score; strictly proper"),
    "absolute": Deviation("absolute", _absolute, False, False,
                          "mean absolute error; elicits the median (0 or 1)"),
    "log": Deviation("log", _log, True, True,
                     "binary log-loss; extended-valued (+inf at |t| = 1)"),
}
_ALIASES = {"brier": "squared", "mse": "squared", "mae": "absolute",
            "logloss": "log", "log_loss": "log", "cross_entropy": "log"}


def list_deviations() -> list:
    """Names of the built-in deviation operators."""
    return sorted(_REGISTRY)


def get_deviation(op: Union[str, Deviation, Phi]) -> Deviation:
    """A deviation operator from a name, a :class:`Deviation` or a callable ``phi``.

    A plain callable is wrapped with ``proper`` unknown (recorded as ``False``);
    use :func:`check_properness` to test it.
    """
    if isinstance(op, Deviation):
        return op
    if isinstance(op, str):
        key = _ALIASES.get(op.lower(), op.lower())
        if key not in _REGISTRY:
            raise ValueError(f"unknown deviation operator {op!r}; "
                             f"available: {list_deviations()}")
        return _REGISTRY[key]
    if callable(op):
        name = getattr(op, "__name__", "custom")
        return Deviation(name, op, proper=False, strictly_proper=False,
                         note="user-defined; properness not checked")
    raise TypeError("op must be a name, a Deviation or a callable")


@dataclass(frozen=True)
class Properness:
    """Result of :func:`check_properness`."""

    proper: bool
    strictly_proper: bool
    max_gap: float
    """Largest excess of the truthful expected loss over the best report."""
    max_report_error: float
    """Largest distance between a truth ``p`` and its best report."""


def check_properness(op, grid: int = 201, tol: float = 1e-9) -> Properness:
    """Numerical properness test of a deviation operator for binary outcomes.

    For each truth ``p`` on a grid of ``(0, 1)`` the expected loss of a report ``q``
    is ``p * phi(1 - q) + (1 - p) * phi(q)``. The operator is proper if the truthful
    report attains the minimum over the grid, and strictly proper if it is moreover
    the only minimiser on the grid. A grid test is evidence, not a proof.
    """
    dev = get_deviation(op)
    if grid < 3:
        raise ValueError("grid must have at least 3 points")
    q = np.linspace(0.0, 1.0, grid)
    p = q[1:-1]
    with np.errstate(invalid="ignore"):
        loss = p[:, None] * dev(1.0 - q)[None, :] + (1 - p[:, None]) * dev(q)[None, :]
    truthful = np.diag(loss[:, 1:-1])
    best = loss.min(axis=1)
    gap = float(np.max(truthful - best))
    is_min = np.isclose(loss, best[:, None], rtol=0.0, atol=tol)
    unique = bool((is_min.sum(axis=1) == 1).all())
    report = q[np.argmin(loss, axis=1)]
    return Properness(proper=gap <= tol, strictly_proper=gap <= tol and unique,
                      max_gap=gap, max_report_error=float(np.max(np.abs(report - p))))
