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

"""Shortest paths, path enumeration and reachability.

Weights follow the library convention (see :func:`aryagraph.core.utils.weight_fn`):
``None`` means unweighted, a string names an edge attribute (missing ⇒ 1), a
callable ``f(u, v, attrs)`` is used as is and may return ``None`` to hide an
edge. Directed graphs follow arcs forwards; undirected edges work both ways
(so on an undirected graph a single negative edge is already a negative cycle).
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import Any, Callable, Hashable, Iterable, Iterator

import numpy as np

from ..core.exceptions import NegativeCycleError, NegativeWeightError, NoPath
from ..core.graph import Graph
from ..core.results import NodeMap
from ..core.utils import WeightFn, WeightSpec, require_node, weight_fn

Node = Hashable
Pred = dict[Node, list[Node]]

_METHODS = ("auto", "bfs", "dijkstra", "bellman-ford")


# ---------------------------------------------------------------------- #
# engines (shared by the public functions below)
# ---------------------------------------------------------------------- #
def _bfs_pred(g: Graph, source: Node, target: Node | None = None) -> tuple[dict[Node, int], Pred]:
    """Hop distances and *all* shortest-path predecessors from *source*.

    With *target*, stops once target's layer is complete (its predecessor
    list is then final).
    """
    succ = g._succ
    dist: dict[Node, int] = {source: 0}
    pred: Pred = {source: []}
    frontier = [source]
    level = 0
    while frontier:
        level += 1
        nxt: list[Node] = []
        for u in frontier:
            for v in succ[u]:
                dv = dist.get(v)
                if dv is None:
                    dist[v] = level
                    pred[v] = [u]
                    nxt.append(v)
                elif dv == level:
                    pred[v].append(u)
        if target is not None and target in dist:
            break
        frontier = nxt
    return dist, pred


def _dijkstra(
    g: Graph,
    sources: Iterable[Node],
    wf: WeightFn,
    target: Node | None = None,
    cutoff: float | None = None,
) -> tuple[dict[Node, Any], Pred]:
    """Multi-source Dijkstra returning settled distances and all tight predecessors.

    Nodes appear in *dist* in settling order. With *target*, the search stops
    after every node at target's distance is settled, so ``pred[target]`` is
    complete even across zero-weight edges.
    """
    succ = g._succ
    dist: dict[Node, Any] = {}
    seen: dict[Node, Any] = {}
    pred: Pred = {}
    origin = set()
    tie = itertools.count()
    heap: list[tuple[Any, int, Node]] = []
    for s in sources:
        if s not in origin:
            origin.add(s)
            seen[s] = 0
            pred[s] = []
            heap.append((0, next(tie), s))
    stop_at = None
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in dist:
            continue
        if stop_at is not None and d > stop_at:
            break
        dist[u] = d
        if u == target:
            stop_at = d
        for v, attrs in succ[u].items():
            w = wf(u, v, attrs)
            if w is None:
                continue
            if w < 0:
                raise NegativeWeightError(u, v, w, "use bellman_ford")
            if v == u:  # a non-negative self-loop never shortens a path
                continue
            nd = d + w
            if cutoff is not None and nd > cutoff:
                continue
            if v in dist:  # already settled: only a zero-weight tie can reach it
                if nd == dist[v] and v not in origin:
                    pred[v].append(u)
                continue
            old = seen.get(v)
            if old is None or nd < old:
                seen[v] = nd
                pred[v] = [u]
                heapq.heappush(heap, (nd, next(tie), v))
            elif nd == old and v not in origin:
                pred[v].append(u)
    return dist, {v: pred[v] for v in dist}


def _bellman_ford(g: Graph, sources: Iterable[Node], wf: WeightFn) -> tuple[dict[Node, Any], Pred]:
    """Queue-based Bellman–Ford (FIFO passes) with exact negative-cycle extraction.

    ``pred[v][0]`` is always the parent through which ``dist[v]`` last
    strictly improved. A node improved during pass ``k`` has a parent chain
    of at least ``k`` arcs unless that chain is cyclic, so an improvement in
    pass ``n`` proves the chain from it closes a cycle, and cycles in the
    parent graph always have negative weight. Worst case O(n·m).
    """
    succ = g._succ
    n = len(g)
    origin = list(dict.fromkeys(sources))
    origin_set = set(origin)
    dist: dict[Node, Any] = {s: 0 for s in origin}
    pred: Pred = {s: [] for s in origin}
    queue = deque(origin)
    queued = set(origin)
    passes = 0
    while queue:
        passes += 1
        for _ in range(len(queue)):
            u = queue.popleft()
            queued.discard(u)
            du = dist[u]
            for v, attrs in succ[u].items():
                w = wf(u, v, attrs)
                if w is None:
                    continue
                if v == u:
                    if w < 0:
                        _raise_negative_cycle(g, wf, [u, u])
                    continue
                nd = du + w
                dv = dist.get(v)
                if dv is None or nd < dv:
                    dist[v] = nd
                    pred[v] = [u]
                    if passes >= n:
                        _raise_negative_cycle(g, wf, _parent_cycle(pred, v))
                    if v not in queued:
                        queued.add(v)
                        queue.append(v)
                elif nd == dv and v not in origin_set and u not in pred[v]:
                    pred[v].append(u)
    return dist, pred


def _parent_cycle(pred: Pred, start: Node) -> list[Node]:
    """Follow first-parent links from *start* until a node repeats; return that cycle, closed."""
    pos: dict[Node, int] = {}
    walk: list[Node] = []
    x = start
    while x not in pos:
        pos[x] = len(walk)
        walk.append(x)
        x = pred[x][0]
    # walk = [..., x, parent(x), parent(parent(x)), ..., y] with parent(y) = x
    return [x, *reversed(walk[pos[x] + 1 :]), x]


def _raise_negative_cycle(g: Graph, wf: WeightFn, cycle: list[Node]) -> None:
    total = sum(wf(u, v, g._succ[u][v]) for u, v in zip(cycle, cycle[1:]))
    shown = " → ".join(map(repr, cycle))
    raise NegativeCycleError(f"negative cycle {shown} (total weight {total:g})", cycle)


def _has_negative_weight(g: Graph, wf: WeightFn) -> bool:
    for u, v, d in g._iter_edges():
        w = wf(u, v, d)
        if w is not None and w < 0:
            return True
        if not g.directed:
            w = wf(v, u, d)
            if w is not None and w < 0:
                return True
    return False


def _resolve_method(g: Graph, weight: WeightSpec, method: str) -> tuple[str, WeightFn]:
    if method == "bellman_ford":
        method = "bellman-ford"
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {', '.join(_METHODS)}")
    if method == "bfs":
        if weight is not None:
            raise ValueError("method='bfs' is unweighted; pass weight=None or choose 'dijkstra'")
        return "bfs", weight_fn(None)
    wf = weight_fn(weight)
    if method == "auto":
        if weight is None:
            return "bfs", wf
        return ("bellman-ford" if _has_negative_weight(g, wf) else "dijkstra"), wf
    return method, wf


def _single_source(
    g: Graph, source: Node, weight: WeightSpec, method: str, target: Node | None = None
) -> tuple[dict[Node, Any], Pred]:
    require_node(g, source)
    if target is not None:
        require_node(g, target)
    how, wf = _resolve_method(g, weight, method)
    if how == "bfs":
        return _bfs_pred(g, source, target)
    if how == "dijkstra":
        return _dijkstra(g, (source,), wf, target=target)
    return _bellman_ford(g, (source,), wf)


def _path_to(pred: Pred, source: Node, target: Node) -> list[Node]:
    path = [target]
    while path[-1] != source:
        path.append(pred[path[-1]][0])
    path.reverse()
    return path


def _all_paths(pred: Pred, dist: dict[Node, Any], source: Node) -> dict[Node, list[Node]]:
    """One shortest path per reached node, sharing the first-parent tree."""
    paths: dict[Node, list[Node]] = {source: [source]}
    for v in dist:
        if v in paths:
            continue
        chain = [v]
        while chain[-1] not in paths:
            chain.append(pred[chain[-1]][0])
        base = paths[chain.pop()]
        while chain:
            x = chain.pop()
            base = base + [x]
            paths[x] = base
    return {v: paths[v] for v in dist}


# ---------------------------------------------------------------------- #
# public API
# ---------------------------------------------------------------------- #
def shortest_path(
    g: Graph,
    source: Node,
    target: Node | None = None,
    weight: WeightSpec = None,
    method: str = "auto",
) -> list[Node] | dict[Node, list[Node]]:
    """A shortest path from *source* to *target*, or to every reachable node.

    Parameters
    ----------
    weight:
        ``None`` for hop count, an edge-attribute name, or ``f(u, v, attrs)``.
    method:
        ``"auto"`` (BFS when unweighted, Dijkstra when every weight is
        non-negative, Bellman–Ford otherwise), ``"bfs"``, ``"dijkstra"`` or
        ``"bellman-ford"``.

    Returns
    -------
    The node list ``[source, …, target]``; with ``target=None`` a dict
    ``{node: path}`` over every node reachable from *source*.

    Raises
    ------
    NodeNotFound, NoPath, NegativeCycleError (Bellman–Ford only)

    >>> shortest_path(Graph([("a", "b"), ("b", "c"), ("a", "c", 5)]), "a", "c", weight="weight")
    ['a', 'b', 'c']
    """
    dist, pred = _single_source(g, source, weight, method, target)
    if target is None:
        return _all_paths(pred, dist, source)
    if target not in dist:
        raise NoPath(source, target)
    return _path_to(pred, source, target)


def shortest_path_length(
    g: Graph,
    source: Node,
    target: Node | None = None,
    weight: WeightSpec = None,
    method: str = "auto",
) -> Any:
    """Length of a shortest path (hops, or total weight); same arguments as :func:`shortest_path`.

    With ``target=None`` returns a :class:`NodeMap` ``{node: distance}`` over
    every node reachable from *source* (in order of discovery).
    """
    dist, _ = _single_source(g, source, weight, method, target)
    if target is None:
        return NodeMap(dist, name="distance")
    if target not in dist:
        raise NoPath(source, target)
    return dist[target]


def dijkstra(
    g: Graph,
    source: Node,
    weight: WeightSpec = "weight",
    *,
    target: Node | None = None,
    cutoff: float | None = None,
) -> tuple[NodeMap, dict[Node, list[Node]]]:
    """Dijkstra's algorithm: distances and shortest-path predecessors from *source*.

    Returns ``(dist, pred)`` where ``pred[v]`` lists *every* predecessor of
    ``v`` on some shortest path (``pred[source] == []``), which is enough to
    rebuild all shortest paths. With *target*, the search stops once target
    (and any node tied with it) is settled; with *cutoff*, nodes farther than
    it are left out. Weights must be non-negative (``ValueError`` otherwise).
    O((n + m) log n).
    """
    require_node(g, source)
    if target is not None:
        require_node(g, target)
    dist, pred = _dijkstra(g, (source,), weight_fn(weight), target=target, cutoff=cutoff)
    return NodeMap(dist, name="distance"), pred


def bellman_ford(g: Graph, source: Node, weight: WeightSpec = "weight") -> tuple[NodeMap, dict[Node, list[Node]]]:
    """Bellman–Ford shortest paths from *source*; negative weights allowed.

    Returns ``(dist, pred)`` like :func:`dijkstra`. If a negative cycle is
    reachable from *source*, raises :class:`NegativeCycleError` whose
    ``cycle`` is that cycle as a closed node list ``[a, b, …, a]``.
    Worst case O(n·m).
    """
    require_node(g, source)
    dist, pred = _bellman_ford(g, (source,), weight_fn(weight))
    return NodeMap(dist, name="distance"), pred


def astar_path(
    g: Graph,
    source: Node,
    target: Node,
    heuristic: Callable[[Node, Node], float] | None = None,
    weight: WeightSpec = "weight",
) -> list[Node]:
    """A* search for a shortest path from *source* to *target*.

    ``heuristic(u, target)`` must never overestimate the remaining distance
    (admissible); it need not be consistent, since nodes are re-expanded
    when a cheaper route turns up. ``None`` means 0 everywhere (plain Dijkstra).
    Weights must be non-negative.
    """
    require_node(g, source)
    require_node(g, target)
    wf = weight_fn(weight)
    h_cache: dict[Node, float] = {}

    def h(n: Node) -> float:
        if heuristic is None:
            return 0
        val = h_cache.get(n)
        if val is None:
            val = h_cache[n] = heuristic(n, target)
        return val

    succ = g._succ
    tie = itertools.count()
    best: dict[Node, Any] = {source: 0}
    parent: dict[Node, Node | None] = {source: None}
    heap: list[tuple[Any, int, Any, Node]] = [(h(source), next(tie), 0, source)]
    while heap:
        _, _, d, u = heapq.heappop(heap)
        if u == target:
            path = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])  # type: ignore[arg-type]
            path.reverse()
            return path
        if d > best[u]:  # stale entry
            continue
        for v, attrs in succ[u].items():
            w = wf(u, v, attrs)
            if w is None or v == u:
                continue
            if w < 0:
                raise NegativeWeightError(u, v, w, "A* needs non-negative weights")
            nd = d + w
            old = best.get(v)
            if old is None or nd < old:
                best[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd + h(v), next(tie), nd, v))
    raise NoPath(source, target)


def all_shortest_paths(
    g: Graph,
    source: Node,
    target: Node,
    weight: WeightSpec = None,
    method: str = "auto",
) -> Iterator[list[Node]]:
    """Every shortest path from *source* to *target* (a lazy iterator).

    Missing nodes and unreachable targets raise immediately, before any path
    is produced. The number of shortest paths can grow exponentially, so the
    paths are generated one at a time from the predecessor DAG. With float
    weights, paths count as tied only when their lengths are exactly equal.
    """
    dist, pred = _single_source(g, source, weight, method, target)
    if target not in dist:
        raise NoPath(source, target)
    return _paths_from_pred(pred, source, target)


def _paths_from_pred(pred: Pred, source: Node, target: Node) -> Iterator[list[Node]]:
    """Enumerate the simple source→target paths of the predecessor graph (iteratively)."""
    stack: list[list[Any]] = [[target, 0]]
    on_path = {target}
    while stack:
        frame = stack[-1]
        node, i = frame
        if node == source:
            yield [n for n, _ in reversed(stack)]
            stack.pop()
            on_path.discard(node)
            continue
        preds = pred[node]
        if i < len(preds):
            frame[1] = i + 1
            p = preds[i]
            if p not in on_path:  # zero-weight cycles can loop the predecessor graph
                on_path.add(p)
                stack.append([p, 0])
        else:
            stack.pop()
            on_path.discard(node)


def all_pairs_shortest_path_length(g: Graph, weight: WeightSpec = None) -> dict[Node, NodeMap]:
    """Distances between every pair of mutually reachable nodes, as ``{u: NodeMap{v: d}}``.

    BFS when unweighted, Dijkstra for non-negative weights, and Johnson's
    reweighting (one Bellman–Ford + n Dijkstra runs) when some weight is
    negative. Unreachable pairs are absent. Raises
    :class:`NegativeCycleError` if the graph has a negative cycle.
    """
    out: dict[Node, NodeMap] = {}
    if weight is None:
        for s in g._node:
            out[s] = NodeMap(_bfs_pred(g, s)[0], name="distance")
        return out
    wf = weight_fn(weight)
    if not _has_negative_weight(g, wf):
        for s in g._node:
            out[s] = NodeMap(_dijkstra(g, (s,), wf)[0], name="distance")
        return out
    # Johnson: potentials from a virtual source joined to every node by 0-arcs.
    h, _ = _bellman_ford(g, g._node, wf)

    def reweighted(u: Node, v: Node, d: dict) -> Any:
        w = wf(u, v, d)
        if w is None:
            return None
        return max(w + h[u] - h[v], 0)  # clamp float round-off; exact for integers

    for s in g._node:
        dist, _ = _dijkstra(g, (s,), reweighted)
        out[s] = NodeMap({v: d - h[s] + h[v] for v, d in dist.items()}, name="distance")
    return out


def floyd_warshall(g: Graph, weight: WeightSpec = "weight") -> tuple[np.ndarray, list[Node]]:
    """All-pairs distances by Floyd–Warshall, vectorised over numpy.

    Returns ``(D, nodes)``: ``D[i, j]`` is the distance from ``nodes[i]`` to
    ``nodes[j]`` (``inf`` if unreachable, 0 on the diagonal); *nodes* is graph
    order. Negative weights are fine; a negative cycle raises
    :class:`NegativeCycleError` naming the cycle. O(n³) time, O(n²) memory.
    """
    nodes = list(g._node)
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    wf = weight_fn(weight)
    dmat = np.full((n, n), np.inf)
    np.fill_diagonal(dmat, 0.0)
    for u, v, d in g._iter_edges():
        pairs = ((u, v),) if g.directed else ((u, v), (v, u))
        for a, b in pairs:
            w = wf(a, b, d)
            if w is None:
                continue
            i, j = index[a], index[b]
            if i == j:
                if w < 0:
                    _raise_negative_cycle(g, wf, [a, a])
                continue
            if w < dmat[i, j]:
                dmat[i, j] = w
    diagonal = np.diagonal(dmat)  # a live view
    for k in range(n):
        np.minimum(dmat, dmat[:, k, None] + dmat[None, k, :], out=dmat)
        if diagonal.min() < 0:  # stop before negative cycles blow values up to -inf
            break
    neg = np.flatnonzero(diagonal < 0)
    if neg.size:
        # A negative closed walk through this node exists; Bellman–Ford names a cycle.
        _bellman_ford(g, (nodes[int(neg[0])],), wf)
        raise NegativeCycleError("graph contains a negative cycle")  # pragma: no cover - defensive
    return dmat, nodes


def has_path(g: Graph, source: Node, target: Node) -> bool:
    """True when *target* is reachable from *source* (every node reaches itself)."""
    require_node(g, source)
    require_node(g, target)
    if source == target:
        return True
    succ = g._succ
    seen = {source}
    stack = [source]
    while stack:
        for v in succ[stack.pop()]:
            if v == target:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def all_simple_paths(g: Graph, source: Node, target: Node, cutoff: int | None = None) -> Iterator[list[Node]]:
    """Every simple path (no repeated node) from *source* to *target*, lazily.

    *cutoff* bounds the number of edges. ``source == target`` yields the
    single trivial path ``[source]``, as networkx does. Paths come out in
    depth-first order over graph-ordered neighbors. Missing nodes raise
    immediately.
    """
    require_node(g, source)
    require_node(g, target)
    limit = len(g) - 1 if cutoff is None else cutoff
    return _simple_paths(g._succ, source, target, limit)


def _simple_paths(succ: dict, source: Node, target: Node, limit: int) -> Iterator[list[Node]]:
    if source == target:
        if limit >= 0:
            yield [source]
        return
    if limit < 1:
        return
    path = [source]
    on_path = {source}
    stack = [iter(succ[source])]
    while stack:
        for v in stack[-1]:
            if v in on_path:
                continue
            if v == target:
                yield [*path, v]
            elif len(path) < limit:  # room for v plus at least one more edge
                path.append(v)
                on_path.add(v)
                stack.append(iter(succ[v]))
                break
        else:
            stack.pop()
            on_path.discard(path.pop())


def k_shortest_paths(g: Graph, source: Node, target: Node, k: int, weight: WeightSpec = "weight") -> list[list[Node]]:
    """The *k* shortest loopless paths from *source* to *target* (Yen's algorithm).

    Paths are returned shortest first (fewer than *k* if fewer exist); ties
    keep discovery order. Weights must be non-negative. Raises
    :class:`NoPath` if *target* is unreachable. O(k·n·(m + n log n)).
    """
    require_node(g, source)
    require_node(g, target)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    wf = weight_fn(weight)
    if source == target:
        return [[source]]
    first = _restricted_dijkstra(g, source, target, wf, frozenset(), frozenset())
    if first is None:
        raise NoPath(source, target)
    succ = g._succ

    def cost(path: list[Node]) -> list[Any]:
        acc = [0]
        for u, v in zip(path, path[1:]):
            acc.append(acc[-1] + wf(u, v, succ[u][v]))
        return acc

    accepted: list[list[Node]] = [first[1]]
    known = {tuple(first[1])}
    candidates: list[tuple[Any, int, list[Node]]] = []
    tie = itertools.count()
    while len(accepted) < k:
        prev = accepted[-1]
        prefix_cost = cost(prev)
        for i in range(len(prev) - 1):
            root = prev[: i + 1]
            banned_edges = set()
            for p in accepted:
                if p[: i + 1] == root:
                    banned_edges.add((p[i], p[i + 1]))
                    if not g.directed:
                        banned_edges.add((p[i + 1], p[i]))
            spur = _restricted_dijkstra(g, prev[i], target, wf, frozenset(root[:-1]), banned_edges)
            if spur is None:
                continue
            total = root[:-1] + spur[1]
            key = tuple(total)
            if key not in known:
                known.add(key)
                heapq.heappush(candidates, (prefix_cost[i] + spur[0], next(tie), total))
        if not candidates:
            break
        accepted.append(heapq.heappop(candidates)[2])
    return accepted


def _restricted_dijkstra(
    g: Graph,
    source: Node,
    target: Node,
    wf: WeightFn,
    banned_nodes: frozenset | set,
    banned_edges: frozenset | set,
) -> tuple[Any, list[Node]] | None:
    """Shortest source→target path avoiding some nodes and arcs, or None."""
    succ = g._succ
    dist: dict[Node, Any] = {}
    seen: dict[Node, Any] = {source: 0}
    parent: dict[Node, Node] = {}
    tie = itertools.count()
    heap = [(0, next(tie), source)]
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in dist:
            continue
        dist[u] = d
        if u == target:
            path = [u]
            while path[-1] != source:
                path.append(parent[path[-1]])
            path.reverse()
            return d, path
        for v, attrs in succ[u].items():
            if v in dist or v in banned_nodes or (u, v) in banned_edges:
                continue
            w = wf(u, v, attrs)
            if w is None:
                continue
            if w < 0:
                raise NegativeWeightError(u, v, w, "k_shortest_paths needs non-negative weights")
            nd = d + w
            old = seen.get(v)
            if old is None or nd < old:
                seen[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, next(tie), v))
    return None


__all__ = [
    "shortest_path",
    "shortest_path_length",
    "dijkstra",
    "bellman_ford",
    "astar_path",
    "all_shortest_paths",
    "all_pairs_shortest_path_length",
    "floyd_warshall",
    "has_path",
    "all_simple_paths",
    "k_shortest_paths",
]
