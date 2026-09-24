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

"""DAG: a directed graph that stays acyclic.

Every mutation that could close a cycle is checked, and a failing mutation
leaves the graph exactly as it was. The error names the cycle it would have
created, so the cause is visible instead of implied.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Hashable, Iterable

from .exceptions import CycleError, NodeNotFound
from .graph import DiGraph, Graph, _unpack_edge
from .results import NodeMap

Node = Hashable


def _kahn(g: Graph) -> tuple[list[Node], list[Node] | None]:
    """Kahn's algorithm in insertion order.

    Returns ``(order, None)`` for an acyclic graph, else ``(partial_order, cycle)``
    where *cycle* is a closed node list ``[a, b, …, a]``.
    """
    pred = g._pred
    succ = g._succ
    indeg = {n: len(pred[n]) for n in g._node}
    queue = deque(n for n, k in indeg.items() if k == 0)
    order: list[Node] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) == len(indeg):
        return order, None
    remaining = dict.fromkeys(n for n, k in indeg.items() if k > 0)  # graph order, O(1) lookups
    return order, _cycle_in(pred, remaining)


def _cycle_in(pred: dict, remaining: dict) -> list[Node]:
    """A directed cycle inside *remaining*, where every node keeps a predecessor in it.

    The walk starts at the first remaining node in graph order and follows
    predecessors in insertion order, so the reported cycle is the same in
    every process.
    """
    cur = next(iter(remaining))
    path: list[Node] = []
    index: dict[Node, int] = {}
    while cur not in index:
        index[cur] = len(path)
        path.append(cur)
        cur = next(p for p in pred[cur] if p in remaining)
    k = index[cur]
    return [cur, *reversed(path[k + 1 :]), cur]


class DAG(DiGraph):
    """Directed acyclic graph.

    Adding an edge that would create a cycle raises :class:`CycleError` whose
    ``cycle`` attribute lists the offending loop; the graph is left unchanged.

    >>> dag = DAG([("extract", "clean"), ("clean", "train"), ("clean", "report")])
    >>> dag.topological_order()
    ['extract', 'clean', 'train', 'report']
    >>> dag.add_edge("report", "extract")
    Traceback (most recent call last):
    CycleError: edge 'report' → 'extract' would close the cycle 'report' → 'extract' → 'clean' → 'report'
    """

    _topo_cache: tuple[int, list[Node]] | None = None

    # ------------------------------------------------------------------ #
    # guarded mutation
    # ------------------------------------------------------------------ #
    def add_edge(self, u: Node, v: Node, **attrs: Any) -> None:
        """Add arc ``u → v``; raises :class:`CycleError` if it would close a cycle."""
        if u == v:
            raise CycleError(f"self-loop on {u!r} is not allowed in a DAG", [u, u])
        if u in self._node and v in self._node and v not in self._succ[u]:
            path = self._path(v, u)
            if path is not None:
                cycle = [u, *path]
                raise CycleError(f"edge {u!r} → {v!r} would close the cycle " + " → ".join(map(repr, cycle)), cycle)
        super().add_edge(u, v, **attrs)

    def add_edges(self, edges: Iterable[Any], **common: Any) -> None:
        """Add many arcs atomically: all of them, or none if any cycle would appear.

        Validation runs once over the whole graph (linear time) instead of once
        per arc, so bulk loading stays fast.
        """
        batch = [_unpack_edge(e) for e in edges]
        if len(batch) == 1:
            u, v, attrs = batch[0]
            self.add_edge(u, v, **{**common, **attrs})
            return
        new_nodes: list[Node] = []
        new_edges: list[tuple[Node, Node]] = []
        restored: list[tuple[dict, dict]] = []
        for u, v, attrs in batch:
            if u == v:
                self._rollback(new_edges, new_nodes, restored)
                raise CycleError(f"self-loop on {u!r} is not allowed in a DAG", [u, u])
            for n in (u, v):
                if n not in self._node:
                    new_nodes.append(n)
            existing = self.get_edge_data(u, v)
            if existing is None:
                new_edges.append((u, v))
            else:
                restored.append((existing, dict(existing)))
            DiGraph.add_edge(self, u, v, **({**common, **attrs} if common else attrs))
        _, cycle = _kahn(self)
        if cycle is not None:
            self._rollback(new_edges, new_nodes, restored)
            raise CycleError("these edges would create the cycle " + " → ".join(map(repr, cycle)), cycle)

    def _rollback(self, edges: list, nodes: list, restored: list) -> None:
        for u, v in edges:
            if self.has_edge(u, v):
                DiGraph.remove_edge(self, u, v)
        for n in nodes:
            if n in self._node:
                DiGraph.remove_node(self, n)
        for d, old in restored:
            d.clear()
            d.update(old)

    def _path(self, source: Node, target: Node) -> list[Node] | None:
        """A directed path source → … → target, or None (iterative DFS)."""
        if source == target:
            return [source]
        parent: dict[Node, Node] = {source: source}
        stack = [source]
        succ = self._succ
        while stack:
            x = stack.pop()
            for y in succ[x]:
                if y not in parent:
                    parent[y] = x
                    if y == target:
                        path = [y]
                        while path[-1] != source:
                            path.append(parent[path[-1]])
                        return path[::-1]
                    stack.append(y)
        return None

    # ------------------------------------------------------------------ #
    # order and structure
    # ------------------------------------------------------------------ #
    def is_dag(self) -> bool:
        return True

    def to_dag(self) -> "DAG":
        return self.copy()

    def topological_order(self) -> list[Node]:
        """Nodes so that every arc points forward (Kahn's algorithm, insertion-order tie-break)."""
        cache = self._topo_cache
        if cache is None or cache[0] != self._version:
            order, cycle = _kahn(self)
            if cycle is not None:  # only reachable if internals were bypassed
                raise CycleError("graph contains a cycle", cycle)
            cache = (self._version, order)
            self._topo_cache = cache
        return list(cache[1])

    def sources(self) -> list[Node]:
        """Nodes with no incoming arcs."""
        return [n for n in self._node if not self._pred[n]]

    def sinks(self) -> list[Node]:
        """Nodes with no outgoing arcs."""
        return [n for n in self._node if not self._succ[n]]

    def ancestors(self, node: Node) -> set[Node]:
        """Every node with a path to *node* (excluding *node*)."""
        return self._reach(node, self._pred)

    def descendants(self, node: Node) -> set[Node]:
        """Every node reachable from *node* (excluding *node*)."""
        return self._reach(node, self._succ)

    def _reach(self, node: Node, adj: dict) -> set[Node]:
        if node not in self._node:
            raise NodeNotFound(node)
        seen: set[Node] = set()
        stack = [node]
        while stack:
            for y in adj[stack.pop()]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return seen

    def levels(self) -> NodeMap:
        """Longest-path level of each node: sources are 0, every arc goes to a higher level."""
        level: dict[Node, int] = {}
        for v in self.topological_order():
            preds = self._pred[v]
            level[v] = 1 + max(level[u] for u in preds) if preds else 0
        return NodeMap(level, name="level")

    def generations(self) -> list[list[Node]]:
        """Nodes grouped by :meth:`levels` into the stages a pipeline can run in parallel."""
        groups: list[list[Node]] = []
        for n, k in self.levels().items():
            while len(groups) <= k:
                groups.append([])
            groups[k].append(n)
        return groups

    def depth(self) -> int:
        """Number of nodes on the longest path (0 for an empty DAG)."""
        return len(self.generations())

    # ------------------------------------------------------------------ #
    # delegates to aryagraph.algorithms.dag
    # ------------------------------------------------------------------ #
    def longest_path(self, weight: str | None = None) -> list[Node]:
        """Heaviest source-to-sink path (by node count when *weight* is None)."""
        from ..algorithms.dag import dag_longest_path

        return dag_longest_path(self, weight=weight)

    def critical_path(self, duration: str = "duration", default: float = 1.0):
        """Critical-path analysis (CPM); see :func:`aryagraph.algorithms.critical_path`."""
        from ..algorithms.dag import critical_path

        return critical_path(self, duration=duration, default=default)

    def transitive_reduction(self) -> "DAG":
        """Smallest DAG with the same reachability."""
        from ..algorithms.dag import transitive_reduction

        return transitive_reduction(self)

    def transitive_closure(self) -> "DAG":
        """DAG with an arc u → v for every path u ⇝ v."""
        from ..algorithms.dag import transitive_closure

        return transitive_closure(self)


__all__ = ["DAG"]
