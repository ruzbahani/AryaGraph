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

"""Vertex coloring and bipartiteness.

Colorings ignore arc directions (two nodes linked either way must differ) and
self-loops (a node cannot differ from itself). Colors are integers from 0.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any, Callable, Hashable, Iterable

from ..core.exceptions import GraphTypeError
from ..core.graph import Graph
from ..core.results import NodeMap
from ..core.utils import make_rng

Node = Hashable

_ALIASES = {
    "largest_first": "largest_first",
    "smallest_last": "smallest_last",
    "dsatur": "dsatur",
    "DSATUR": "dsatur",
    "saturation_largest_first": "dsatur",
    "independent_set": "independent_set",
    "random_sequential": "random_sequential",
    "connected_sequential_bfs": "connected_sequential_bfs",
    "connected_sequential": "connected_sequential_bfs",
    "connected_sequential_dfs": "connected_sequential_dfs",
    "insertion": "insertion",
}


def _neighbors(g: Graph) -> dict[Node, list[Node]]:
    """Distinct neighbors of every node, both arc directions, self excluded, graph order."""
    if not g.directed:
        return {n: [v for v in nbrs if v != n] for n, nbrs in g._succ.items()}
    succ, pred = g._succ, g._pred
    return {n: [v for v in dict.fromkeys([*succ[n], *pred[n]]) if v != n] for n in g._node}


def greedy_color(
    g: Graph,
    strategy: str | Callable[[Graph], Iterable[Node]] = "largest_first",
    *,
    seed: Any = None,
) -> NodeMap:
    """A proper vertex coloring by the greedy method: each node takes the smallest free color.

    Parameters
    ----------
    strategy:
        The order nodes are colored in:

        * ``"largest_first"``: decreasing degree (graph order breaks ties);
        * ``"smallest_last"``: repeatedly strip a minimum-degree node, color
          in reverse (uses at most degeneracy + 1 colors);
        * ``"dsatur"`` (alias ``"saturation_largest_first"``): next is the node
          seeing the most distinct colors, then the highest degree;
        * ``"independent_set"``: peel off maximal independent sets (min-degree
          first), one color each;
        * ``"random_sequential"``: a random order drawn from *seed*;
        * ``"connected_sequential_bfs"`` / ``"connected_sequential_dfs"``:
          BFS / DFS order within each component;
        * ``"insertion"``: graph order;
        * a callable ``f(g)`` returning every node once, in the order to color.

    Returns
    -------
    :class:`NodeMap` ``{node: color}`` in graph order. Never uses more than
    max degree + 1 colors.
    """
    nbrs = _neighbors(g)
    if callable(strategy):
        order = list(strategy(g))
        if len(order) != len(g) or set(order) != set(g._node):
            raise ValueError("a strategy callable must return every node of the graph exactly once")
        colors = _greedy(order, nbrs)
    else:
        name = _ALIASES.get(strategy)
        if name is None:
            raise ValueError(f"unknown coloring strategy {strategy!r}; choose from {sorted(set(_ALIASES.values()))}")
        if name == "dsatur":
            colors = _dsatur(g, nbrs)
        else:
            colors = _greedy(_ORDERS[name](g, nbrs, seed), nbrs)
    return NodeMap(((n, colors[n]) for n in g._node), name="color")


def _greedy(order: Iterable[Node], nbrs: dict[Node, list[Node]]) -> dict[Node, int]:
    colors: dict[Node, int] = {}
    for u in order:
        used = {colors[v] for v in nbrs[u] if v in colors}
        c = 0
        while c in used:
            c += 1
        colors[u] = c
    return colors


def _order_largest_first(g: Graph, nbrs: dict, seed: Any) -> list[Node]:
    return sorted(g._node, key=lambda n: len(nbrs[n]), reverse=True)  # stable: ties keep graph order


def _order_smallest_last(g: Graph, nbrs: dict, seed: Any) -> list[Node]:
    """Degeneracy order via bucket queue, O(n + m); ties take the longest-waiting node."""
    deg = {n: len(nbrs[n]) for n in g._node}
    buckets: list[dict[Node, None]] = [{} for _ in range(max(deg.values(), default=0) + 1)]
    for n, d in deg.items():
        buckets[d][n] = None
    removed: set[Node] = set()
    order: list[Node] = []
    d = 0
    for _ in range(len(deg)):
        while not buckets[d]:
            d += 1
        u = next(iter(buckets[d]))
        del buckets[d][u]
        removed.add(u)
        order.append(u)
        for v in nbrs[u]:
            if v not in removed:
                dv = deg[v]
                del buckets[dv][v]
                deg[v] = dv - 1
                buckets[dv - 1][v] = None
        d = max(d - 1, 0)  # a removal lowers other degrees by at most one
    order.reverse()
    return order


def _order_independent_set(g: Graph, nbrs: dict, seed: Any) -> list[Node]:
    """Concatenated maximal independent sets, each grown min-degree-first (as networkx does)."""
    index = g.node_index()
    remaining = dict.fromkeys(g._node)
    order: list[Node] = []
    while remaining:
        cand = dict.fromkeys(remaining)
        deg = {v: sum(1 for w in nbrs[v] if w in cand) for v in cand}
        heap = [(deg[v], index[v], v) for v in cand]
        heapq.heapify(heap)
        while heap:
            d, _, v = heapq.heappop(heap)
            if v not in cand or d != deg[v]:
                continue  # stale entry
            order.append(v)
            del remaining[v]
            dropped = [v, *(w for w in nbrs[v] if w in cand)]
            for x in dropped:
                del cand[x]
            for x in dropped:
                for y in nbrs[x]:
                    if y in cand:
                        deg[y] -= 1
                        heapq.heappush(heap, (deg[y], index[y], y))
    return order


def _order_random(g: Graph, nbrs: dict, seed: Any) -> list[Node]:
    nodes = list(g._node)
    return [nodes[i] for i in make_rng(seed).permutation(len(nodes))]


def _order_connected(g: Graph, nbrs: dict, breadth_first: bool) -> list[Node]:
    seen: set[Node] = set()
    order: list[Node] = []
    for root in g._node:
        if root in seen:
            continue
        seen.add(root)
        if breadth_first:
            queue = deque([root])
            while queue:
                u = queue.popleft()
                order.append(u)
                for v in nbrs[u]:
                    if v not in seen:
                        seen.add(v)
                        queue.append(v)
        else:
            order.append(root)
            stack = [iter(nbrs[root])]
            while stack:
                for v in stack[-1]:
                    if v not in seen:
                        seen.add(v)
                        order.append(v)
                        stack.append(iter(nbrs[v]))
                        break
                else:
                    stack.pop()
    return order


_ORDERS: dict[str, Callable[[Graph, dict, Any], list[Node]]] = {
    "largest_first": _order_largest_first,
    "smallest_last": _order_smallest_last,
    "independent_set": _order_independent_set,
    "random_sequential": _order_random,
    "connected_sequential_bfs": lambda g, nbrs, seed: _order_connected(g, nbrs, True),
    "connected_sequential_dfs": lambda g, nbrs, seed: _order_connected(g, nbrs, False),
    "insertion": lambda g, nbrs, seed: list(g._node),
}


def _dsatur(g: Graph, nbrs: dict[Node, list[Node]]) -> dict[Node, int]:
    """Brélaz's DSatur with a lazy max-heap on (saturation, degree, graph order)."""
    index = g.node_index()
    deg = {n: len(nbrs[n]) for n in g._node}
    seen_colors: dict[Node, set[int]] = {n: set() for n in g._node}
    heap = [(0, -deg[n], index[n], n) for n in g._node]
    heapq.heapify(heap)
    colors: dict[Node, int] = {}
    while heap:
        neg_sat, _, _, u = heapq.heappop(heap)
        if u in colors or -neg_sat != len(seen_colors[u]):
            continue  # stale entry
        used = seen_colors[u]
        c = 0
        while c in used:
            c += 1
        colors[u] = c
        for v in nbrs[u]:
            if v not in colors and c not in seen_colors[v]:
                seen_colors[v].add(c)
                heapq.heappush(heap, (-len(seen_colors[v]), -deg[v], index[v], v))
    return colors


# ---------------------------------------------------------------------- #
# bipartiteness
# ---------------------------------------------------------------------- #
def _two_coloring(g: Graph) -> tuple[dict[Node, int], list[Node] | None]:
    """BFS 2-coloring; returns ``(colors, None)`` or ``(partial, odd_cycle)``.

    The first node of every component (graph order) gets color 0. Arc
    directions are ignored; a self-loop is an odd cycle ``[a, a]``.
    """
    succ, pred = g._succ, g._pred
    color: dict[Node, int] = {}
    parent: dict[Node, Node | None] = {}
    for root in g._node:
        if root in color:
            continue
        color[root] = 0
        parent[root] = None
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for adj in (succ[u], pred[u]) if g.directed else (succ[u],):
                for v in adj:
                    if v not in color:
                        color[v] = 1 - color[u]
                        parent[v] = u
                        queue.append(v)
                    elif color[v] == color[u]:
                        return color, _odd_cycle(u, v, parent)
    return color, None


def _odd_cycle(u: Node, v: Node, parent: dict[Node, Node | None]) -> list[Node]:
    """Close the BFS-tree paths of two equally colored neighbors into an odd cycle."""
    if u == v:
        return [u, u]
    up = [u]
    while parent[up[-1]] is not None:
        up.append(parent[up[-1]])  # type: ignore[arg-type]
    pos = {x: i for i, x in enumerate(up)}
    vp = [v]
    while vp[-1] not in pos:
        vp.append(parent[vp[-1]])  # type: ignore[arg-type]
    lca = vp[-1]
    return [*up[: pos[lca] + 1], *reversed(vp[:-1]), u]


def is_bipartite(g: Graph) -> bool:
    """True when the nodes split into two sides with every edge crossing (directions ignored)."""
    return _two_coloring(g)[1] is None


def bipartite_sets(g: Graph) -> tuple[set[Node], set[Node]]:
    """The two sides ``(top, bottom)`` of a bipartite graph.

    Every component is 2-colored from its first node (in graph order), which
    goes to *top*, so disconnected graphs get a deterministic split where
    networkx refuses. Raises :class:`GraphTypeError`, naming an odd cycle,
    when the graph is not bipartite.
    """
    color, odd = _two_coloring(g)
    if odd is not None:
        raise GraphTypeError("graph is not bipartite: odd cycle " + " - ".join(map(repr, odd)))
    top = {n for n, c in color.items() if c == 0}
    return top, set(g._node) - top


__all__ = ["greedy_color", "is_bipartite", "bipartite_sets"]
