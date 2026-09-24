# =============================================================================
#
#      _                     ____                 _
#     / \   _ __ _   _  __ _ / ___|_ __ __ _ _ __ | |__
#    / _ \ | '__| | | |/ _` | |  _| '__/ _` | '_ \| '_ \
#   / ___ \| |  | |_| | (_| | |_| | | | (_| | |_) | | | |
#  /_/   \_\_|   \__, |\__,_|\____|_|  \__,_| .__/|_| |_|
#               |___/                     |_|
#
#          Graph & DAG visualization, analysis and simulation.
#
# -----------------------------------------------------------------------------
#  Copyright (c) 2026 Ali Mohammadi Ruzbahani
#  SPDX-License-Identifier: MIT
#
#  https://ruzbahani.com/aryagraph
# =============================================================================

"""Matrix representations and spectra.

Rows and columns follow graph iteration order unless *nodes* is given, which
is also the order :meth:`Graph.node_index` reports.
"""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np

from ..core.utils import WeightSpec, weight_fn

Node = Hashable


def adjacency_matrix(
    g: Any,
    weight: WeightSpec = "weight",
    *,
    nodes: list[Node] | None = None,
    dtype: Any = float,
) -> np.ndarray:
    """Dense adjacency matrix ``A[i, j]`` = weight of edge ``nodes[i] → nodes[j]``.

    Undirected graphs give a symmetric matrix. Edges without the *weight*
    attribute count as 1; ``weight=None`` gives a 0/1 matrix.
    """
    order = list(g._node) if nodes is None else list(nodes)
    index = {n: i for i, n in enumerate(order)}
    wf = weight_fn(weight)
    a = np.zeros((len(order), len(order)), dtype=dtype)
    for u, v, d in g.edges.data():
        i, j = index.get(u), index.get(v)
        if i is None or j is None:
            continue
        w = wf(u, v, d)
        a[i, j] = w
        if not g.directed:
            a[j, i] = w
    return a


def degree_vector(g: Any, weight: WeightSpec = None, *, nodes: list[Node] | None = None) -> np.ndarray:
    """Row sums of the (symmetrised, for directed graphs) adjacency matrix."""
    a = _symmetric_adjacency(g, weight, nodes)
    return a.sum(axis=1)


def _symmetric_adjacency(g: Any, weight: WeightSpec, nodes: list[Node] | None) -> np.ndarray:
    a = adjacency_matrix(g, weight, nodes=nodes)
    if g.directed:
        a = np.maximum(a, a.T)
    return a


def laplacian_matrix(
    g: Any,
    weight: WeightSpec = None,
    *,
    normalized: bool = False,
    nodes: list[Node] | None = None,
) -> np.ndarray:
    """Graph Laplacian ``L = D - A`` (or ``I - D^-1/2 A D^-1/2`` when *normalized*).

    Directed graphs are symmetrised first (an arc in either direction links the
    pair); self-loops are ignored, matching the usual combinatorial definition.
    """
    a = _symmetric_adjacency(g, weight, nodes)
    np.fill_diagonal(a, 0.0)
    deg = a.sum(axis=1)
    if not normalized:
        return np.diag(deg) - a
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    lap = -(inv_sqrt[:, None] * a * inv_sqrt[None, :])
    lap[np.diag_indices_from(lap)] = np.where(deg > 0, 1.0, 0.0)
    return lap


def incidence_matrix(g: Any, *, oriented: bool = False, nodes: list[Node] | None = None) -> np.ndarray:
    """``(n_nodes, n_edges)`` incidence matrix; *oriented* puts -1 at the tail, +1 at the head."""
    order = list(g._node) if nodes is None else list(nodes)
    index = {n: i for i, n in enumerate(order)}
    edges = list(g.edges)
    m = np.zeros((len(order), len(edges)))
    for k, (u, v) in enumerate(edges):
        if u == v:
            continue
        m[index[u], k] = -1.0 if oriented else 1.0
        m[index[v], k] = 1.0
    return m


def adjacency_spectrum(g: Any, weight: WeightSpec = "weight") -> np.ndarray:
    """Eigenvalues of the adjacency matrix (real & sorted for undirected graphs, complex otherwise)."""
    a = adjacency_matrix(g, weight)
    if g.directed:
        return np.linalg.eigvals(a)
    return np.linalg.eigvalsh(a)


def laplacian_spectrum(g: Any, weight: WeightSpec = None, *, normalized: bool = False) -> np.ndarray:
    """Ascending eigenvalues of the (symmetrised) Laplacian."""
    return np.linalg.eigvalsh(laplacian_matrix(g, weight, normalized=normalized))


def algebraic_connectivity(g: Any, weight: WeightSpec = None, *, normalized: bool = False) -> float:
    """Second-smallest Laplacian eigenvalue (Fiedler value); 0 iff the graph is disconnected."""
    if len(g) < 2:
        return 0.0
    vals = laplacian_spectrum(g, weight, normalized=normalized)
    return float(max(vals[1], 0.0))


__all__ = [
    "adjacency_matrix",
    "degree_vector",
    "laplacian_matrix",
    "incidence_matrix",
    "adjacency_spectrum",
    "laplacian_spectrum",
    "algebraic_connectivity",
]
