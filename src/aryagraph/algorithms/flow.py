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

"""Maximum flow and minimum cut (Dinic's algorithm).

Capacities come from an edge attribute (missing ⇒ unlimited, as in networkx)
or a callable ``f(u, v, attrs)``. Arcs of a directed graph carry flow one way;
an undirected edge can carry flow either way, up to its capacity. Self-loops
are ignored.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Hashable

from ..core.exceptions import UnboundedFlowError
from ..core.graph import Graph
from ..core.utils import require_node

Node = Hashable


@dataclass(frozen=True, repr=False)
class FlowResult:
    """A maximum flow: its ``value`` and ``flow[u][v]`` on every edge (0 where unused).

    For an undirected edge both ``flow[u][v]`` and ``flow[v][u]`` exist and at
    most one is positive. Opposite arcs ``u → v`` / ``v → u`` of a directed
    graph never both carry flow (it is netted out).
    """

    value: Any
    flow: dict

    def __repr__(self) -> str:
        busy = sum(1 for row in self.flow.values() for f in row.values() if f > 0)
        return f"FlowResult(value={self.value!r}, {busy} arcs carrying flow)"


@dataclass(frozen=True, repr=False)
class CutResult:
    """A minimum ``s``–``t`` cut: its capacity, both node sides and the edges crossing it.

    ``cut_edges`` are oriented from the source side to the sink side.
    """

    value: Any
    source_side: set
    sink_side: set
    cut_edges: list

    def __repr__(self) -> str:
        return (
            f"CutResult(value={self.value!r}, |source_side|={len(self.source_side)}, "
            f"|sink_side|={len(self.sink_side)}, {len(self.cut_edges)} cut edges)"
        )


class _Network:
    """Residual network in arc arrays: arc ``e`` and its partner ``e ^ 1`` point opposite ways."""

    __slots__ = ("nodes", "index", "head", "res", "flow", "adj", "edges")

    def __init__(self, g: Graph, capacity: str | Callable[[Node, Node, dict], Any]) -> None:
        self.nodes = list(g._node)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.head: list[int] = []
        self.res: list[Any] = []  # residual capacity; saturation is exact (res -= res)
        self.flow: list[Any] = []  # net flow, tracked separately so infinite arcs report it
        self.adj: list[list[int]] = [[] for _ in self.nodes]
        self.edges: list[tuple[Node, Node, int]] = []
        cap_of = capacity if callable(capacity) else (lambda u, v, d: d.get(capacity))
        for u, v, d in g._iter_edges():
            if u == v:
                continue
            c = cap_of(u, v, d)
            if c is None:
                c = math.inf
            elif c != c or c < 0:
                raise ValueError(f"capacity of edge ({u!r}, {v!r}) must be a non-negative number, got {c!r}")
            i, j = self.index[u], self.index[v]
            e = len(self.head)
            self.head += (j, i)
            self.res += (c, c if not g.directed else 0)
            self.flow += (0, 0)
            self.adj[i].append(e)
            self.adj[j].append(e + 1)
            self.edges.append((u, v, e))

    def infinite_path(self, s: int, t: int) -> list[int] | None:
        """An s→t path made only of infinite-capacity arcs (the flow would be unbounded)."""
        parent = {s: -1}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for e in self.adj[u]:
                v = self.head[e]
                if v not in parent and self.res[e] == math.inf:
                    parent[v] = u
                    if v == t:
                        path = [t]
                        while parent[path[-1]] != -1:
                            path.append(parent[path[-1]])
                        return path[::-1]
                    queue.append(v)
        return None

    def dinic(self, s: int, t: int) -> Any:
        """Push a maximum flow from s to t; returns its value. O(n²·m)."""
        head, res, flow, adj = self.head, self.res, self.flow, self.adj
        n = len(adj)
        total: Any = 0
        while True:
            level = [-1] * n
            level[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for e in adj[u]:
                    v = head[e]
                    if level[v] < 0 and res[e] > 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[t] < 0:
                return total
            it = [0] * n  # current-arc pointers
            while True:
                path: list[int] = []
                u = s
                while u != t:
                    arcs = adj[u]
                    i = it[u]
                    while i < len(arcs):
                        e = arcs[i]
                        if res[e] > 0 and level[head[e]] == level[u] + 1:
                            break
                        i += 1
                    it[u] = i
                    if i < len(arcs):
                        path.append(arcs[i])
                        u = head[arcs[i]]
                    elif u == s:
                        break
                    else:  # dead end: never enter u again in this phase
                        level[u] = -1
                        u = head[path.pop() ^ 1]
                        it[u] += 1
                if u != t:
                    break  # blocking flow reached
                f = min(res[e] for e in path)
                for e in path:
                    res[e] -= f
                    res[e ^ 1] += f
                    flow[e] += f
                    flow[e ^ 1] -= f
                total += f


def _solve(g: Graph, s: Node, t: Node, capacity: Any) -> tuple[_Network, Any]:
    require_node(g, s)
    require_node(g, t)
    if s == t:
        raise ValueError("source and sink must be different nodes")
    net = _Network(g, capacity)
    si, ti = net.index[s], net.index[t]
    inf_path = net.infinite_path(si, ti)
    if inf_path is not None:
        shown = " → ".join(repr(net.nodes[i]) for i in inf_path)
        raise UnboundedFlowError(f"maximum flow is unbounded: every arc of {shown} has infinite capacity")
    # Without an all-infinite path every augmenting path has a finite bottleneck,
    # so infinite capacities can stay infinite (no big-M substitute needed).
    return net, net.dinic(si, ti)


def maximum_flow(g: Graph, s: Node, t: Node, capacity: str | Callable[[Node, Node, dict], Any] = "capacity") -> FlowResult:
    """Maximum ``s``→``t`` flow by Dinic's algorithm (blocking flows on BFS level graphs).

    Parameters
    ----------
    capacity:
        Edge-attribute name (edges without it have unlimited capacity) or
        ``f(u, v, attrs)`` (``None`` = unlimited).

    Returns
    -------
    FlowResult
        The flow value and ``flow[u][v]`` for every edge.

    Raises
    ------
    NodeNotFound
        If *s* or *t* is not in the graph.
    ValueError
        If ``s == t`` or a capacity is negative/NaN.
    UnboundedFlowError
        If an all-infinite path makes the flow unbounded (a subclass of
        ``ValueError``).

    Notes
    -----
    O(n²·m) in general, O(m·√n) on unit-capacity networks.
    """
    net, value = _solve(g, s, t, capacity)
    raw: dict[Node, dict[Node, Any]] = {u: {v: 0 for v in nbrs} for u, nbrs in g._succ.items()}
    for u, v, e in net.edges:
        f = net.flow[e]
        if g.directed:
            raw[u][v] = f
        elif f > 0:
            raw[u][v] = f
        elif f < 0:
            raw[v][u] = -f
    if g.directed:  # net out flow running both ways between the same two nodes
        for u, v, _ in net.edges:
            back = raw[v].get(u)
            if back and raw[u][v]:
                m = min(raw[u][v], back)
                raw[u][v] -= m
                raw[v][u] -= m
    return FlowResult(value=value, flow=raw)


def minimum_cut(g: Graph, s: Node, t: Node, capacity: str | Callable[[Node, Node, dict], Any] = "capacity") -> CutResult:
    """Minimum ``s``–``t`` cut (max-flow min-cut theorem); same arguments as :func:`maximum_flow`.

    The source side is every node still reachable from *s* in the residual
    network of a maximum flow (the cut closest to *s*). ``cut_edges`` lists,
    in edge order, every edge from the source side to the sink side; their
    capacities add up to ``value``.
    """
    net, value = _solve(g, s, t, capacity)
    si = net.index[s]
    seen = [False] * len(net.nodes)
    seen[si] = True
    stack = [si]
    while stack:
        u = stack.pop()
        for e in net.adj[u]:
            v = net.head[e]
            if not seen[v] and net.res[e] > 0:
                seen[v] = True
                stack.append(v)
    source_side = {n for i, n in enumerate(net.nodes) if seen[i]}
    cut_edges: list[tuple[Node, Node]] = []
    for u, v in g.edges:
        if u == v:
            continue
        if u in source_side and v not in source_side:
            cut_edges.append((u, v))
        elif not g.directed and v in source_side and u not in source_side:
            cut_edges.append((v, u))
    return CutResult(
        value=value,
        source_side=source_side,
        sink_side=set(net.nodes) - source_side,
        cut_edges=cut_edges,
    )


__all__ = ["FlowResult", "CutResult", "maximum_flow", "minimum_cut"]
