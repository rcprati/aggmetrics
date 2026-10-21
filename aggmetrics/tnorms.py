# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Triangular norms for soft and uncertain ground truth.

When the label matrix is soft --- annotator proportions, crowd labels, or a
probabilistic reference standard --- the Boolean conjunction that defines the four
cells must be replaced by a t-norm :math:`T`, with the standard complement
:math:`N(x) = 1 - x`:

.. math::
    TP = T(y, \\hat{y}),\\quad FP = T(1-y, \\hat{y}),\\quad
    FN = T(y, 1-\\hat{y}),\\quad TN = T(1-y, 1-\\hat{y}).

The product t-norm is the only choice for which the resulting cells preserve the
row and column marginals,

.. math:: TP + FN = y, \\qquad TP + FP = \\hat{y},

and hence the only one under which the fuzzy confusion matrix behaves like a
contingency table. :func:`marginal_defect` measures departure from that property,
and returns zero only for ``product``.

Note that the pointwise ordering :math:`T_L \\le T_P \\le T_M` applies to *all four*
cell matrices in the same direction, since all four are produced by the same
t-norm. It therefore implies nothing about the direction in which a derived ratio
moves: numerator and denominator shrink together.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

__all__ = ["TNORMS", "get_tnorm", "marginal_defect", "godel", "product",
           "lukasiewicz"]


def godel(a, b):
    """Minimum (Gödel) t-norm: the largest t-norm."""
    return np.minimum(a, b)


def product(a, b):
    """Product t-norm: the unique marginally consistent choice."""
    return np.asarray(a) * np.asarray(b)


def lukasiewicz(a, b):
    """Łukasiewicz t-norm: credits only overlap above the unit threshold."""
    return np.maximum(0.0, np.asarray(a) + np.asarray(b) - 1.0)


def hamacher(gamma: float = 1.0) -> Callable:
    """Hamacher family; ``gamma == 1`` recovers the product t-norm."""
    def T(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        den = gamma + (1.0 - gamma) * (a + b - a * b)
        out = np.zeros_like(a)
        nz = np.abs(den) > 1e-12
        out[nz] = (a * b)[nz] / den[nz]
        return out
    return T


TNORMS: Dict[str, Callable] = {
    "godel": godel,
    "minimum": godel,
    "min": godel,
    "product": product,
    "prod": product,
    "lukasiewicz": lukasiewicz,
    "luk": lukasiewicz,
}


def get_tnorm(name: str | Callable) -> Callable:
    if callable(name):
        return name
    key = str(name).strip().lower()
    if key not in TNORMS:
        raise KeyError(f"unknown t-norm {name!r}. Available: "
                       f"{', '.join(sorted(set(TNORMS)))}")
    return TNORMS[key]


def marginal_defect(tnorm: str | Callable, grid: int = 21) -> float:
    """Maximum violation of marginal consistency over a grid of ``[0,1]^2``.

    Zero iff ``T(a,b) + T(a,1-b) == a`` and ``T(a,b) + T(1-a,b) == b`` throughout,
    which holds only for the product t-norm.
    """
    T = get_tnorm(tnorm)
    xs = np.linspace(0.0, 1.0, grid)
    A, B = np.meshgrid(xs, xs, indexing="ij")
    d1 = np.abs(T(A, B) + T(A, 1 - B) - A)
    d2 = np.abs(T(A, B) + T(1 - A, B) - B)
    return float(max(d1.max(), d2.max()))


def soft_cells(Y, P, tnorm="product"):
    """Fuzzy TP, FP, FN, TN matrices under the given t-norm."""
    T = get_tnorm(tnorm)
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    if Y.shape != P.shape:
        raise ValueError(f"shape mismatch: {Y.shape} vs {P.shape}")
    for name, arr in (("Y", Y), ("P", P)):
        if arr.min() < -1e-9 or arr.max() > 1 + 1e-9:
            raise ValueError(f"{name} must lie in [0, 1]")
    return T(Y, P), T(1 - Y, P), T(Y, 1 - P), T(1 - Y, 1 - P)
