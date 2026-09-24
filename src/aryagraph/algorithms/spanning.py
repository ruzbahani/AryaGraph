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

"""Minimum and maximum spanning trees (forests, when the graph is disconnected).

Weights follow the library convention: an edge-attribute name (missing ⇒ 1),
``None`` (every edge weighs 1), or ``f(u, v, attrs)`` (``None`` hides the
edge). Self-loops never belong to a spanning tree. Equal weights are broken by
edge order, so results are deterministic.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Any, Hashable, Iterable

from ..core.graph import Graph
from ..core.utils import WeightSpec, require_undirected, weight_fn

Node = Hashable

_ALGORITHMS = ("kruskal", "prim")


class _UnionFind:
    """Disjoint sets with path compression and union by rank (near-constant amortised time)."""

    __slots__ = ("parent", "rank")

    def __init__(self, items: Iterable[Node]) -> None:
        self.parent = {x: x for x in items}
        self.rank = dict.fromkeys(self.parent, 0)

    def find(self, x: Node) -> Node:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # compress the whole path
            parent[x], x = root, parent[x]
        return root

    def union(self, a: Node, b: Node) -> bool:
        """Merge the sets of *a* and *b*; False if they were already one set."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        rank = self.rank
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True


def _checked(w: Any, u: Node, v: Node) -> Any:
    if w != w:
        raise ValueError(f"edge ({u!r}, {v!r}) has a NaN weight")
    return w


def _spanning_edges(g: Graph, weight: WeightSpec, algorithm: str, maximum: bool) -> list[tuple[Node, Node, dict]]:
    require_undirected(g, "spanning trees")
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; use 'kruskal' or 'prim'")
    wf = weight_fn(weight)
    sign = -1 if maximum else 1
    if algorithm == "kruskal":
        ranked = []
        for k, (u, v, d) in enumerate(g._iter_edges()):
            if u == v:
                continue
            w = wf(u, v, d)
            if w is not None:
                ranked.append((sign * _checked(w, u, v), k, u, v, d))
        ranked.sort(key=lambda t: (t[0], t[1]))
        forest = _UnionFind(g._node)
        out = []
        target = len(g) - 1
        for _, _, u, v, d in ranked:
            if forest.union(u, v):
                out.append((u, v, d))
                if len(out) == target:
                    break
        return out
    # Prim, grown from the first unvisited node of every component (graph order).
    succ = g._succ
    tie = itertools.count()
    visited: set[Node] = set()
    out = []
    for root in g._node:
        if root in visited:
            continue
        visited.add(root)
        heap: list[tuple[Any, int, Node, Node, dict]] = []

        def push(u: Node) -> None:
            for v, d in succ[u].items():
                if v not in visited:
                    w = wf(u, v, d)
                    if w is not None:
                        heapq.heappush(heap, (sign * _checked(w, u, v), next(tie), u, v, d))

        push(root)
        while heap:
            _, _, u, v, d = heapq.heappop(heap)
            if v in visited:
                continue
            visited.add(v)
            out.append((u, v, d))
            push(v)
    return out


def minimum_spanning_edges(g: Graph, weight: WeightSpec = "weight", algorithm: str = "kruskal") -> list[tuple[Node, Node, dict]]:
    """Edges ``(u, v, attrs)`` of a minimum spanning forest.

    Kruskal lists them by increasing weight, Prim in growth order. *attrs* is
    the graph's own attribute dict (treat it as read-only).
    Kruskal: O(m log m); Prim: O(m log n).
    """
    return _spanning_edges(g, weight, algorithm, maximum=False)


def maximum_spanning_edges(g: Graph, weight: WeightSpec = "weight", algorithm: str = "kruskal") -> list[tuple[Node, Node, dict]]:
    """Edges ``(u, v, attrs)`` of a maximum spanning forest (see :func:`minimum_spanning_edges`)."""
    return _spanning_edges(g, weight, algorithm, maximum=True)


def _as_graph(g: Graph, edges: list[tuple[Node, Node, dict]]) -> Graph:
    tree = Graph()
    tree.attrs = dict(g.attrs)
    tree.add_nodes((n, dict(d)) for n, d in g._node.items())
    tree.add_edges((u, v, dict(d)) for u, v, d in edges)
    return tree


def minimum_spanning_tree(g: Graph, weight: WeightSpec = "weight", algorithm: str = "kruskal") -> Graph:
    """Minimum spanning tree of an undirected graph, as a new :class:`Graph`.

    A disconnected graph gives a spanning forest (one tree per component).
    Every node is kept; node, edge and graph attributes are copied.

    Parameters
    ----------
    algorithm:
        ``"kruskal"`` (sort + union-find) or ``"prim"`` (heap).
    """
    return _as_graph(g, minimum_spanning_edges(g, weight, algorithm))


def maximum_spanning_tree(g: Graph, weight: WeightSpec = "weight", algorithm: str = "kruskal") -> Graph:
    """Maximum spanning tree (forest) of an undirected graph; see :func:`minimum_spanning_tree`."""
    return _as_graph(g, maximum_spanning_edges(g, weight, algorithm))


__all__ = [
    "minimum_spanning_tree",
    "maximum_spanning_tree",
    "minimum_spanning_edges",
    "maximum_spanning_edges",
]
