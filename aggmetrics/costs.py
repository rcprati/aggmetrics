# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Cost matrices built from a class taxonomy (hierarchical classification).

Section 6 of the companion paper treats hierarchical classification as an instance
of the cost-sensitive construction: the content is the choice of the cost matrix
``C[j, k] = d(c_j, c_k)``, with ``d`` a distance in the taxonomy. The paper does not
say how a taxonomy becomes a distance. This module supplies the most common choice,
the number of edges on the path between two classes in the tree.
"""
from __future__ import annotations

from typing import Hashable, Mapping, Optional, Sequence

import numpy as np


def tree_cost_matrix(parents: Mapping[Hashable, Optional[Hashable]],
                     classes: Optional[Sequence[Hashable]] = None) -> np.ndarray:
    """Cost matrix whose entries are tree distances, in edges.

    Parameters
    ----------
    parents
        The taxonomy as a mapping from each node to its parent, with ``None`` for
        the root. Every node, including the root, is a key.
    classes
        The nodes that are classes, in the order of the columns of the indicator
        matrix. Defaults to the leaves, in the order of ``parents``.

    Returns
    -------
    numpy.ndarray
        The ``m x m`` matrix of distances, symmetric with a zero diagonal. Pass it
        to :meth:`aggmetrics.core.IndicatorMatrix.cost`.

    Examples
    --------
    >>> from aggmetrics import tree_cost_matrix
    >>> tree = {"root": None, "A": "root", "B": "root",
    ...         "a1": "A", "a2": "A", "b1": "B"}
    >>> tree_cost_matrix(tree).astype(int).tolist()   # leaves a1, a2, b1
    [[0, 2, 4], [2, 0, 4], [4, 4, 0]]
    """
    roots = [node for node, parent in parents.items() if parent is None]
    if len(roots) != 1:
        raise ValueError("the taxonomy must have exactly one root (parent None)")
    paths = {}
    for node in parents:
        path, seen, cur = [], set(), node
        while cur is not None:
            if cur in seen:
                raise ValueError(f"the taxonomy has a cycle at {cur!r}")
            if cur not in parents:
                raise ValueError(f"{cur!r} is a parent but not a node of the taxonomy")
            seen.add(cur)
            path.append(cur)
            cur = parents[cur]
        paths[node] = path[::-1]                      # from the root down to the node
    if classes is None:
        inner = {p for p in parents.values() if p is not None}
        classes = [node for node in parents if node not in inner]
    unknown = [c for c in classes if c not in parents]
    if unknown:
        raise ValueError(f"classes not in the taxonomy: {unknown}")
    m = len(classes)
    C = np.zeros((m, m))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            common = 0
            for x, y in zip(paths[a], paths[b]):
                if x != y:
                    break
                common += 1
            C[i, j] = len(paths[a]) + len(paths[b]) - 2 * common
    return C
