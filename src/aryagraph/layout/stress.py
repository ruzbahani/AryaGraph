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

"""Stress majorization (SMACOF): the default layout for general graphs.

Stress measures how well geometric distances reproduce graph distances::

    stress(X) = Σ_{i<j} w_ij (‖x_i − x_j‖ − d_ij)²,   w_ij = d_ij⁻²

Gansner, Koren & North (2004) showed that majorizing this function gives a
monotone, fast-converging iteration that beats Kamada–Kawai's Newton–Raphson
both in quality and speed. Each step solves ``L_w X = L_Z X_old`` with the
weighted Laplacian ``L_w`` (factorised once), so an iteration is a handful of
dense ``(n, n)`` array operations.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Hashable, Mapping

import numpy as np

from ..algorithms._core import distance_matrix
from ..core.utils import WeightSpec, make_rng
from .base import Layout
from .geometric import _as_positions, _empty, _pack_by_components, _undirected_adjacency

Node = Hashable


def _hop_distances(adj: list[list[int]]) -> np.ndarray:
    """All-pairs hop distances by batched, vectorised BFS (``-1`` = unreachable).

    All sources advance one BFS level at a time: the frontier is a set of
    ``(source, node)`` pairs that numpy expands through a CSR adjacency, so the
    total work is O(n·m) element operations with no per-node Python loop.
    """
    n = len(adj)
    deg = np.fromiter((len(a) for a in adj), dtype=np.int64, count=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    indices = np.fromiter(itertools.chain.from_iterable(adj), dtype=np.int64, count=int(indptr[-1]))
    dist = np.full((n, n), -1, dtype=np.int32)
    chunk = max(1, min(n, 2_000_000 // max(1, indices.size)))
    for s0 in range(0, n, chunk):
        k = min(chunk, n - s0)
        sub = dist[s0 : s0 + k]
        rows = np.arange(k, dtype=np.int64)
        cur = np.arange(s0, s0 + k, dtype=np.int64)
        sub[rows, cur] = 0
        level = 0
        while rows.size:
            level += 1
            cnt = deg[cur]
            total = int(cnt.sum())
            if total == 0:
                break
            rep = np.repeat(rows, cnt)
            offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            nbr = indices[np.repeat(indptr[cur], cnt) + offs]
            fresh = sub[rep, nbr] < 0
            lin = np.unique(rep[fresh] * n + nbr[fresh])
            rows, cur = lin // n, lin % n
            sub[rows, cur] = level
    return dist


def graph_distances(g: Any, weight: WeightSpec = None, nodes: list[Node] | None = None) -> np.ndarray:
    """Symmetric shortest-path distance matrix used by the distance-based layouts.

    Directions are ignored; unreachable pairs are ``inf``. Unweighted graphs
    use the vectorised BFS, weighted ones Dijkstra.
    """
    order = list(g._node) if nodes is None else list(nodes)
    if weight is None:
        index = {v: i for i, v in enumerate(order)}
        adj = _undirected_adjacency(g, order, index)
        hops = _hop_distances(adj).astype(float)
        hops[hops < 0] = np.inf
        return hops
    d = distance_matrix(g, weight, nodes=order, undirected=True)
    return np.minimum(d, d.T)


def _classical_mds(d: np.ndarray, rng: np.random.Generator, pivots: int | None = None) -> np.ndarray:
    """Classical (Torgerson) MDS, or pivot MDS (Brandes & Pich) with *pivots* columns."""
    n = d.shape[0]
    if pivots is None or pivots >= n:
        d2 = d**2
        rm = d2.mean(axis=1)
        b = -0.5 * (d2 - rm[:, None] - rm[None, :] + rm.mean())
        vals, vecs = np.linalg.eigh(b)
        top = np.argsort(vals)[::-1][:2]
        return vecs[:, top] * np.sqrt(np.maximum(vals[top], 1e-12))
    # max-min pivot selection from a random start
    chosen = [int(rng.integers(n))]
    mind = d[chosen[0]].copy()
    for _ in range(pivots - 1):
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        mind = np.minimum(mind, d[nxt])
    c = d[:, chosen] ** 2
    c = -0.5 * (c - c.mean(axis=0)[None, :] - c.mean(axis=1)[:, None] + c.mean())
    vals, vecs = np.linalg.eigh(c.T @ c)
    top = np.argsort(vals)[::-1][:2]
    return c @ vecs[:, top]


def _pairwise(x: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix via the Gram identity (dtype of *x*, zero diagonal)."""
    sq = np.einsum("ij,ij->i", x, x)
    d2 = x @ x.T
    d2 *= -2.0
    d2 += sq[:, None]
    d2 += sq[None, :]
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2, out=d2)


def _pairwise_direct(x: np.ndarray) -> np.ndarray:
    """Distance matrix from coordinate differences (no cancellation, so float32-safe)."""
    dx = np.subtract.outer(x[:, 0], x[:, 0])
    dx *= dx
    dy = np.subtract.outer(x[:, 1], x[:, 1])
    dy *= dy
    dx += dy
    return np.sqrt(dx, out=dx)


def _smacof(d: np.ndarray, x: np.ndarray, iterations: int, tol: float) -> tuple[np.ndarray, float, int]:
    """Stress majorization from *x*; returns ``(X, stress, iterations_run)``.

    The elementwise ``(n, n)`` work arrays are float32 (half the memory
    traffic, ample precision for distances); the ill-conditioned Laplacian
    solve, the reductions and the iterate stay float64 so the majorization
    keeps its monotone descent.
    """
    n = d.shape[0]
    with np.errstate(divide="ignore"):
        dinv64 = 1.0 / d  # w_ij·d_ij with w = d⁻²
    np.fill_diagonal(dinv64, 0.0)
    lw = dinv64 * dinv64
    lw *= -1.0
    np.fill_diagonal(lw, -lw.sum(axis=1))
    # L_w is singular (constant null vector); L_w + J/n acts identically on centred vectors
    lw += 1.0 / n
    linv = np.linalg.inv(lw)
    del lw
    dinv = dinv64.astype(np.float32)
    del dinv64
    pairs = float(n * (n - 1))
    x = np.asarray(x, dtype=float) - np.mean(x, axis=0)
    # optimal uniform scale for the start configuration
    r = dinv * _pairwise_direct(x.astype(np.float32))
    num = float(np.sum(r, dtype=np.float64))
    den = float(np.sum(r * r, dtype=np.float64))
    if den > 0 and num > 0:
        x *= num / den
    history: list[float] = []
    best_s, best_x = math.inf, x
    it = 0
    for it in range(iterations + 1):
        x -= x.mean(axis=0)
        xf = x.astype(np.float32)
        dist = _pairwise_direct(xf)
        r = dinv * dist  # ‖x_i − x_j‖ / d_ij (0 on the diagonal)
        # Σ_{i≠j} (r − 1)² / 2 without materialising r − 1
        s = (float(np.sum(r * r, dtype=np.float64)) - 2.0 * float(np.sum(r, dtype=np.float64)) + pairs) / 2.0
        if s < best_s:
            best_s, best_x = s, x
        history.append(s)
        # stop when the relative decrease, averaged over the last 5 steps, is
        # below tol (single steps are too irregular to be a reliable signal)
        if len(history) > 5 and (history[-6] - s) <= 5 * tol * history[-6]:
            break
        if it == iterations:
            break
        np.maximum(dist, 1e-6, out=dist)  # coincident points: finite pull
        ratio = np.divide(dinv, dist, out=dist)  # diagonal stays 0 (dinv is 0 there)
        # both terms from the same (float32) coordinates, or their rounding
        # difference would swamp the late, small majorization steps
        bx = ratio.sum(axis=1, dtype=np.float64)[:, None] * xf - (ratio @ xf).astype(np.float64)
        x = linv @ bx
    return best_x, max(best_s, 0.0), it


def _principal_axes(x: np.ndarray) -> np.ndarray:
    """Centre *x* and rotate its principal axis onto the x axis (deterministic signs)."""
    x = x - x.mean(axis=0)
    if x.shape[0] < 2:
        return x
    cov = x.T @ x
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, np.argsort(vals)[::-1]]
    y = x @ vecs
    # fix the reflection: make the heavier tail point right / down
    for k in range(2):
        if np.sum(y[:, k] ** 3) < 0:
            y[:, k] = -y[:, k]
    return y


def stress(
    g: Any,
    *,
    weight: WeightSpec = None,
    seed: int = 0,
    iterations: int = 300,
    tol: float = 1e-5,
    init: str | Layout | Mapping[Node, Any] = "mds",
) -> Layout:
    """Stress-majorization layout (the modern replacement for Kamada–Kawai).

    Parameters
    ----------
    weight:
        ``None`` means every edge has length 1; otherwise edge lengths come
        from this attribute / callable (shortest-path distances via Dijkstra).
    iterations, tol:
        Stop after *iterations* majorization steps or when the relative stress
        decrease falls below *tol*.
    init:
        ``"mds"`` (classical MDS; pivot MDS with 50 pivots above 1000 nodes),
        ``"random"``, or a :class:`Layout` / ``{node: (x, y)}`` start.

    Returns
    -------
    Layout
        Abstract layout with edge lengths ≈ 1 (≈ weight), principal axis
        horizontal. ``meta["stress"]`` is the final normalised stress
        ``Σ (‖x_i − x_j‖/d_ij − 1)² / 2``. Disconnected graphs are laid out per
        component and packed.

    Notes
    -----
    Complexity: O(n·m) for distances, O(n³) once for the Laplacian solve and
    O(n²) per iteration (about a second for 1000 nodes).
    """
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("stress")
    packed = _pack_by_components(g, stress, gap=1.0, weight=weight, seed=seed, iterations=iterations, tol=tol, init=init)
    if packed is not None:
        return packed
    if n == 1:
        return Layout(nodes, np.zeros((1, 2)), method="stress", meta={"stress": 0.0})
    d = graph_distances(g, weight, nodes)
    pos_d = d[np.isfinite(d) & (d > 0)]
    tiny = 1e-3 * (float(np.median(pos_d)) if pos_d.size else 1.0)
    off = ~np.eye(n, dtype=bool)
    d[off & (d <= 0)] = tiny  # zero-length edges would give infinite weights
    if n == 2:
        half = float(d[0, 1]) / 2
        return Layout(nodes, np.array([[-half, 0.0], [half, 0.0]]), method="stress", meta={"stress": 0.0})
    rng = make_rng(seed)
    if isinstance(init, str):
        if init == "mds":
            x0 = _classical_mds(d, rng, pivots=50 if n > 1000 else None)
        elif init == "random":
            x0 = rng.random((n, 2)) * math.sqrt(n)
        else:
            raise ValueError(f"init must be 'mds', 'random' or positions, got {init!r}")
    else:
        given = _as_positions(init)
        x0 = rng.random((n, 2)) * math.sqrt(n)
        for i, v in enumerate(nodes):
            if v in given:
                x0[i] = np.asarray(given[v], dtype=float)[:2]
    # a tiny seeded jitter breaks exact symmetries (e.g. a collinear MDS start
    # would otherwise stay collinear under majorization)
    extent = float(np.ptp(x0, axis=0).max()) or 1.0
    x0 = x0 + rng.normal(scale=1e-3 * extent, size=x0.shape)
    x, s, its = _smacof(d, x0, int(iterations), float(tol))
    x = _principal_axes(x)
    return Layout(nodes, x, method="stress", meta={"stress": s, "iterations": its})


__all__ = ["stress", "graph_distances"]
