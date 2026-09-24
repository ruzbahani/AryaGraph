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

"""Spectral layout: nodes placed by low Laplacian eigenvectors.

The eigenvectors of the 2nd and 3rd smallest Laplacian eigenvalues minimise
``Σ_edges w_ij ‖x_i − x_j‖²`` under orthogonality constraints (Hall, 1970),
which reveals global structure (clusters, bottlenecks, grid-like geometry)
in one dense eigendecomposition. With ``normalized=True`` the
degree-normalised eigenvectors (Koren, 2005) are used, which keeps
low-degree nodes from collapsing onto hubs.
"""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np

from ..algorithms.matrix import laplacian_matrix
from ..core.utils import WeightSpec
from .base import Layout
from .geometric import _edge_arrays, _empty, _pack_by_components, circular

Node = Hashable


def spectral(g: Any, *, weight: WeightSpec = None, normalized: bool = True, seed: int = 0) -> Layout:
    """Layout from the Fiedler vector and the next Laplacian eigenvector.

    Parameters
    ----------
    weight:
        Edge weights for the Laplacian (``None`` = unweighted).
    normalized:
        Use the generalized problem ``L u = λ D u`` (degree-normalised
        eigenvectors) instead of the combinatorial Laplacian.

    Graphs with fewer than three nodes, no edges, or a degenerate spectrum
    (e.g. complete graphs, where the eigenvectors are arbitrary) fall back to
    :func:`circular`; disconnected graphs are laid out per component and
    packed. Coordinates are scaled to a median edge length of 1.
    Complexity: O(n³) (dense ``eigh``).
    """
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("spectral")
    packed = _pack_by_components(g, spectral, gap=1.0, weight=weight, normalized=normalized, seed=seed)
    if packed is not None:
        return packed
    index = {v: i for i, v in enumerate(nodes)}
    ei, ej, _ = _edge_arrays(g, index)

    def fallback(reason: str) -> Layout:
        lay = circular(g, seed=seed)
        return Layout(nodes, lay.xy, method="spectral", meta={"fallback": "circular", "reason": reason})

    if n < 3 or ei.size == 0:
        return fallback("too small")
    lap = laplacian_matrix(g, weight, nodes=nodes)
    deg = np.diag(lap).copy()
    if normalized and np.all(deg > 0):
        inv_sqrt = 1.0 / np.sqrt(deg)
        sym = inv_sqrt[:, None] * lap * inv_sqrt[None, :]
        vals, vecs = np.linalg.eigh(sym)
        vecs = inv_sqrt[:, None] * vecs
    else:
        vals, vecs = np.linalg.eigh(lap)
    scale = max(float(vals[-1]), 1e-12)
    # degenerate: the 2nd..last eigenvalues coincide (complete-graph-like spectrum)
    if (vals[-1] - vals[1]) <= 1e-9 * scale:
        return fallback("degenerate spectrum")
    xy = vecs[:, 1:3].copy()
    for k in range(2):
        col = xy[:, k]
        # deterministic sign: first clearly non-zero entry is negative (left / top)
        nz = np.flatnonzero(np.abs(col) > 1e-9 * (np.abs(col).max() or 1.0))
        if nz.size and col[nz[0]] > 0:
            xy[:, k] = -col
    if float(np.ptp(xy, axis=0).max()) <= 1e-12:
        return fallback("collapsed")
    lengths = np.hypot(*(xy[ei] - xy[ej]).T)
    lengths = lengths[lengths > 1e-12]
    if lengths.size:
        xy /= float(np.median(lengths))
    else:
        xy /= float(np.ptp(xy, axis=0).max())
    xy -= xy.mean(axis=0)
    return Layout(nodes, xy, method="spectral", meta={"eigenvalues": [float(vals[1]), float(vals[2])]})


__all__ = ["spectral"]
