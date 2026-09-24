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

"""Shared primitives: components and single-source distances.

These are the few routines several subpackages build on (layouts need graph
distances, analysis needs components), kept dependency-free and fast.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import Any, Callable, Hashable, Iterable, Mapping

import numpy as np

from ..core.exceptions import NegativeWeightError, NodeNotFound
from ..core.utils import WeightSpec, weight_fn

Node = Hashable


def neighbor_map(g: Any, *, reverse: bool = False, undirected: bool = False) -> Callable[[Node], Iterable[tuple[Node, dict]]]:
    """``f(node) -> iterable of (neighbor, edge_attrs)`` for the chosen direction.

    *reverse* follows arcs backwards; *undirected* follows both directions.
    For undirected graphs both flags are irrelevant.
    """
    succ, pred = g._succ, g._pred
    if not g.directed:
        return lambda n: succ[n].items()
    if undirected:
        def both(n: Node) -> Iterable[tuple[Node, dict]]:
            yield from succ[n].items()
            for u, d in pred[n].items():
                if u not in succ[n]:
                    yield u, d
        return both
    adj = pred if reverse else succ
    return lambda n: adj[n].items()


def components(g: Any) -> list[list[Node]]:
    """Connected components (weakly connected for directed graphs).

    Components are listed by their first node in graph order, and the nodes of
    each component keep graph order, so results are stable across runs.
    """
    nbrs = neighbor_map(g, undirected=True)
    comp_of: dict[Node, int] = {}
    count = 0
    for s in g._node:
        if s in comp_of:
            continue
        comp_of[s] = count
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v, _ in nbrs(u):
                if v not in comp_of:
                    comp_of[v] = count
                    queue.append(v)
        count += 1
    groups: list[list[Node]] = [[] for _ in range(count)]
    for n in g._node:
        groups[comp_of[n]].append(n)
    return groups


def bfs_distances(
    g: Any,
    source: Node,
    *,
    reverse: bool = False,
    undirected: bool = False,
    cutoff: int | None = None,
) -> dict[Node, int]:
    """Hop distances from *source* to every reachable node (source included, at 0)."""
    if source not in g:
        raise NodeNotFound(source)
    nbrs = neighbor_map(g, reverse=reverse, undirected=undirected)
    dist = {source: 0}
    frontier = [source]
    level = 0
    while frontier and (cutoff is None or level < cutoff):
        level += 1
        nxt = []
        for u in frontier:
            for v, _ in nbrs(u):
                if v not in dist:
                    dist[v] = level
                    nxt.append(v)
        frontier = nxt
    return dist


def dijkstra_distances(
    g: Any,
    source: Node,
    weight: WeightSpec = "weight",
    *,
    reverse: bool = False,
    undirected: bool = False,
    cutoff: float | None = None,
) -> dict[Node, float]:
    """Weighted distances from *source* (Dijkstra; weights must be non-negative).

    Nodes are returned in order of increasing distance. A weight function
    returning ``None`` hides that edge.
    """
    if source not in g:
        raise NodeNotFound(source)
    wf = weight_fn(weight)
    nbrs = neighbor_map(g, reverse=reverse, undirected=undirected)
    dist: dict[Node, float] = {}
    seen = {source: 0.0}
    tie = itertools.count()
    heap: list[tuple[float, int, Node]] = [(0.0, next(tie), source)]
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in dist:
            continue
        dist[u] = d
        for v, attrs in nbrs(u):
            w = wf(u, v, attrs) if not reverse else wf(v, u, attrs)
            if w is None:
                continue
            if w < 0:
                raise NegativeWeightError(u, v, w, "use bellman_ford")
            nd = d + w
            if cutoff is not None and nd > cutoff:
                continue
            if v not in dist and (v not in seen or nd < seen[v]):
                seen[v] = nd
                heapq.heappush(heap, (nd, next(tie), v))
    return dist


def distance_matrix(
    g: Any,
    weight: WeightSpec = None,
    *,
    nodes: list[Node] | None = None,
    undirected: bool = False,
) -> np.ndarray:
    """All-pairs shortest-path distances as an ``(n, n)`` float array.

    Rows/columns follow *nodes* (graph order by default); unreachable pairs are
    ``inf``. Unweighted graphs use BFS, weighted ones Dijkstra.
    """
    order = list(g._node) if nodes is None else list(nodes)
    index = {n: i for i, n in enumerate(order)}
    n = len(order)
    out = np.full((n, n), np.inf)
    for i, s in enumerate(order):
        if weight is None:
            dist: Mapping[Node, float] = bfs_distances(g, s, undirected=undirected)
        else:
            dist = dijkstra_distances(g, s, weight, undirected=undirected)
        for t, d in dist.items():
            j = index.get(t)
            if j is not None:
                out[i, j] = d
    return out


__all__ = ["neighbor_map", "components", "bfs_distances", "dijkstra_distances", "distance_matrix"]
