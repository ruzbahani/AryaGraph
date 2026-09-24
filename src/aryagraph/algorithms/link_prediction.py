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

"""Link prediction: neighbourhood-based scores for pairs of unlinked nodes.

Each scorer returns ``[(u, v, score), ...]`` like networkx. With
``pairs=None`` every non-adjacent pair of distinct nodes is scored, in graph
order (``u`` before ``v``); networkx enumerates them in set order instead.
All scorers need an undirected graph; neighbour sets ignore the pair's own
nodes where networkx does (see each function).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any, Callable, Hashable

from ..core.exceptions import NodeNotFound
from ..core.utils import require_undirected

Node = Hashable
Scored = list[tuple[Node, Node, float]]


def _non_edges(g: Any) -> Iterator[tuple[Node, Node]]:
    nodes = list(g._node)
    adj = g._succ
    for i, u in enumerate(nodes):
        nbrs = adj[u]
        for v in nodes[i + 1 :]:
            if v not in nbrs:
                yield u, v


def _pairs(g: Any, pairs: Iterable[tuple[Node, Node]] | None) -> Iterator[tuple[Node, Node]]:
    if pairs is None:
        yield from _non_edges(g)
        return
    for pair in pairs:
        u, v = pair
        for x in (u, v):
            if x not in g._node:
                raise NodeNotFound(x)
        if u == v:
            raise ValueError(f"link prediction scores pairs of distinct nodes, got ({u!r}, {v!r})")
        yield u, v


def _common(adj: dict, u: Node, v: Node) -> list[Node]:
    """Common neighbours of *u* and *v* (excluding u and v), in u's neighbour order."""
    nv = adj[v]
    return [w for w in adj[u] if w in nv and w != u and w != v]


def _apply(g: Any, what: str, pairs: Iterable[tuple[Node, Node]] | None, score: Callable[[Node, Node], float]) -> Scored:
    require_undirected(g, what)
    return [(u, v, score(u, v)) for u, v in _pairs(g, pairs)]


def common_neighbors(g: Any, u: Node, v: Node) -> list[Node]:
    """Nodes adjacent to both *u* and *v* (other than u and v themselves), in u's neighbour order."""
    require_undirected(g, "common_neighbors")
    for x in (u, v):
        if x not in g._node:
            raise NodeNotFound(x)
    return _common(g._succ, u, v)


def jaccard_coefficient(g: Any, pairs: Iterable[tuple[Node, Node]] | None = None) -> Scored:
    """``|N(u) ∩ N(v)| / |N(u) ∪ N(v)|`` (0 when both neighbourhoods are empty).

    As in networkx, the union keeps u and v when they are neighbours (of each
    other or through a self-loop) while the intersection excludes them.
    """
    adj = g._succ

    def score(u: Node, v: Node) -> float:
        union = len(adj[u].keys() | adj[v].keys())
        return len(_common(adj, u, v)) / union if union else 0.0

    return _apply(g, "jaccard_coefficient", pairs, score)


def adamic_adar_index(g: Any, pairs: Iterable[tuple[Node, Node]] | None = None) -> Scored:
    """``Σ 1 / log deg(w)`` over the common neighbours *w* (Adamic & Adar 2003)."""
    adj = g._succ
    deg = g.degree()
    return _apply(g, "adamic_adar_index", pairs, lambda u, v: sum(1 / math.log(deg[w]) for w in _common(adj, u, v)))


def resource_allocation_index(g: Any, pairs: Iterable[tuple[Node, Node]] | None = None) -> Scored:
    """``Σ 1 / deg(w)`` over the common neighbours *w* (Zhou, Lü & Zhang 2009)."""
    adj = g._succ
    deg = g.degree()
    return _apply(g, "resource_allocation_index", pairs, lambda u, v: sum(1 / deg[w] for w in _common(adj, u, v)))


def preferential_attachment(g: Any, pairs: Iterable[tuple[Node, Node]] | None = None) -> Scored:
    """``deg(u) · deg(v)`` (self-loops count twice, as in :meth:`Graph.degree`)."""
    deg = g.degree()
    return _apply(g, "preferential_attachment", pairs, lambda u, v: deg[u] * deg[v])


def _common_count(g: Any, pairs: Iterable[tuple[Node, Node]] | None = None) -> Scored:
    adj = g._succ
    return _apply(g, "common_neighbors", pairs, lambda u, v: len(_common(adj, u, v)))


_METHODS: dict[str, Callable[..., Scored]] = {
    "common_neighbors": _common_count,
    "jaccard": jaccard_coefficient,
    "adamic_adar": adamic_adar_index,
    "resource_allocation": resource_allocation_index,
    "preferential_attachment": preferential_attachment,
}


def _two_hop_pairs(g: Any) -> list[tuple[Node, Node]]:
    """Unlinked pairs with at least one common neighbour, in graph order."""
    adj = g._succ
    index = {v: i for i, v in enumerate(g._node)}
    found: set[tuple[int, int]] = set()
    for w, nbrs in adj.items():
        ns = [x for x in nbrs if x != w]
        for a in range(len(ns)):
            for b in range(a + 1, len(ns)):
                x, y = ns[a], ns[b]
                if y not in adj[x]:
                    i, j = index[x], index[y]
                    found.add((i, j) if i < j else (j, i))
    nodes = list(g._node)
    return [(nodes[i], nodes[j]) for i, j in sorted(found)]


def predict_links(g: Any, method: str = "adamic_adar", k: int = 10) -> Scored:
    """The *k* most likely missing links as ``(u, v, score)``, best first.

    *method* is ``"adamic_adar"``, ``"resource_allocation"``, ``"jaccard"``,
    ``"common_neighbors"`` or ``"preferential_attachment"``. Ties keep graph
    order. Neighbourhood methods only score pairs two hops apart (all others
    score 0), and pad with zero-score pairs when fewer than *k* exist.

    >>> predict_links(g, "jaccard", k=3)
    [(u1, v1, 0.8), (u2, v2, 0.75), (u3, v3, 0.5)]
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_METHODS)}")
    require_undirected(g, "predict_links")
    if k <= 0:
        return []
    scorer = _METHODS[method]
    if method == "preferential_attachment":
        scored = scorer(g)
    else:
        candidates = _two_hop_pairs(g)
        scored = scorer(g, candidates)
        if len(scored) < k:
            taken = set(candidates)
            extra = (p for p in _non_edges(g) if p not in taken)
            scored += [(u, v, 0.0) for u, v in islice(extra, k - len(scored))]
    return heapq.nlargest(k, scored, key=lambda t: t[2])



__all__ = [
    "common_neighbors",
    "jaccard_coefficient",
    "adamic_adar_index",
    "resource_allocation_index",
    "preferential_attachment",
    "predict_links",
]
