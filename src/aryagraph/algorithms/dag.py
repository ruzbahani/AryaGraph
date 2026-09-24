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

"""Algorithms for directed acyclic graphs, plus cycle finding for any graph.

Functions that need acyclicity accept a :class:`~aryagraph.DAG` or any acyclic
:class:`~aryagraph.DiGraph`; a cyclic input raises :class:`CycleError` whose
``cycle`` names the offending loop, and an undirected graph raises
:class:`GraphTypeError`. Cycles are always reported as closed node lists
``[a, b, …, a]`` (the form :class:`CycleError` uses).
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Iterator

from ..core.dag import DAG, _kahn
from ..core.exceptions import CycleError, GraphTypeError, NodeNotFound
from ..core.graph import DiGraph, Graph
from ..core.results import NodeMap
from ..core.utils import WeightSpec, require_directed, require_node
from .connectivity import _scc
from .matching import _hopcroft_karp_core

Node = Hashable


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _topological_order(g: Graph, what: str) -> list[Node]:
    """Topological order of *g* (Kahn, graph-order ties), or CycleError / GraphTypeError."""
    if not g.directed:
        raise GraphTypeError(f"{what} requires a directed graph")
    if isinstance(g, DAG):
        return g.topological_order()
    order, cycle = _kahn(g)
    if cycle is not None:
        raise CycleError(f"{what} requires an acyclic graph; found the cycle " + " → ".join(map(repr, cycle)), cycle)
    return order


def _edge_weight(weight: WeightSpec, default: float) -> Callable[[Node, Node, dict], Any]:
    """Weight function for DAG path lengths: ``None`` counts edges, a name reads that attribute."""
    if weight is None:
        return lambda u, v, d: 1
    if callable(weight):
        return weight
    if isinstance(weight, str):
        return lambda u, v, d: d.get(weight, default)
    raise TypeError(f"weight must be None, an attribute name or a callable, got {weight!r}")


def _iter_bits(x: int) -> Iterator[int]:
    """Positions of the set bits of *x*, ascending."""
    s = bin(x)[:1:-1]  # little-endian digits without the '0b' prefix
    i = s.find("1")
    while i >= 0:
        yield i
        i = s.find("1", i + 1)


def _descendant_bits(g: Graph, order: list[Node], index: dict[Node, int]) -> dict[Node, int]:
    """For an acyclic *g*: ``{u: bitset of the nodes reachable from u}`` (u excluded)."""
    succ = g._succ
    reach: dict[Node, int] = {}
    for u in reversed(order):
        r = 0
        for v in succ[u]:
            r |= (1 << index[v]) | reach[v]
        reach[u] = r
    return reach


# ---------------------------------------------------------------------- #
# orders
# ---------------------------------------------------------------------- #
def is_dag(g: Graph) -> bool:
    """True for a directed graph without directed cycles (undirected graphs: False)."""
    if not g.directed:
        return False
    return isinstance(g, DAG) or _kahn(g)[1] is None


def topological_sort(g: Graph, key: Callable[[Node], Any] | None = None) -> list[Node]:
    """Nodes ordered so that every arc points forward.

    Without *key*, Kahn's algorithm with graph-order tie-breaking (the order
    :meth:`DAG.topological_order` gives). With *key*, the lexicographically
    smallest order under ``key(node)`` (a heap replaces the queue; equal keys
    fall back to graph order). O((n + m) log n) with a key.

    Raises :class:`CycleError` (with the cycle) if *g* has a cycle.
    """
    if key is None:
        return _topological_order(g, "topological_sort")
    require_directed(g, "topological_sort")
    pred, succ = g._pred, g._succ
    indeg = {n: len(pred[n]) for n in g._node}
    heap = [(key(n), i, n) for i, n in enumerate(g._node) if indeg[n] == 0]
    heapq.heapify(heap)
    index = g.node_index()
    order: list[Node] = []
    while heap:
        _, _, u = heapq.heappop(heap)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (key(v), index[v], v))
    if len(order) != len(g):
        _topological_order(g, "topological_sort")  # raises with the cycle
    return order


def all_topological_sorts(g: Graph) -> Iterator[list[Node]]:
    """Every topological order of *g*, generated lazily by backtracking.

    A cyclic graph raises :class:`CycleError` immediately. There can be up to
    n! orders; each is produced in O(n) amortised time beyond the output.
    """
    _topological_order(g, "all_topological_sorts")
    return _all_topological_sorts(g)


def _all_topological_sorts(g: Graph) -> Iterator[list[Node]]:
    succ = g._succ
    indeg = {n: len(g._pred[n]) for n in g._node}
    n = len(indeg)
    current: list[Node] = []
    stack: list[list[Any]] = [[[v for v in g._node if indeg[v] == 0], 0]]
    while stack:
        frame = stack[-1]
        available, i = frame
        if len(current) == n:
            yield list(current)
        elif i < len(available):
            frame[1] = i + 1
            v = available[i]
            current.append(v)
            nxt = available[:i] + available[i + 1 :]
            for w in succ[v]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    nxt.append(w)
            stack.append([nxt, 0])
            continue
        stack.pop()
        if current:  # undo the choice that opened this frame
            for w in succ[current.pop()]:
                indeg[w] += 1


# ---------------------------------------------------------------------- #
# cycles
# ---------------------------------------------------------------------- #
def find_cycle(g: Graph, source: Node | Iterable[Node] | None = None) -> list[Node] | None:
    """Some cycle of *g* as a closed node list ``[a, b, …, a]``, or ``None``.

    Works for directed graphs (a directed cycle) and undirected ones (an
    edge walked back and forth is not a cycle; a self-loop is ``[a, a]``).
    With *source* (a node or an iterable of nodes) only the part of the
    graph reachable from it is searched. Iterative DFS, O(n + m).
    """
    if source is None:
        roots: Iterable[Node] = g._node
    elif source in g:
        roots = (source,)
    else:
        try:
            roots = list(source)  # type: ignore[arg-type]
        except TypeError:
            raise NodeNotFound(source) from None
        for r in roots:
            require_node(g, r)
    if g.directed and isinstance(g, DAG):
        return None
    succ = g._succ
    directed = g.directed
    state: dict[Node, int] = {}  # 1 = on the DFS path, 2 = finished
    for root in roots:
        if root in state:
            continue
        state[root] = 1
        path = [root]
        pos = {root: 0}
        stack = [iter(succ[root])]
        while stack:
            u = path[-1]
            parent = path[-2] if len(path) > 1 else None
            for v in stack[-1]:
                s = state.get(v)
                if s is None:
                    state[v] = 1
                    pos[v] = len(path)
                    path.append(v)
                    stack.append(iter(succ[v]))
                    break
                if s == 1 and (directed or v != parent or v == u):
                    return [*path[pos[v] :], v]
            else:
                stack.pop()
                done = path.pop()
                del pos[done]
                state[done] = 2
    return None


def simple_cycles(g: Graph) -> Iterator[list[Node]]:
    """Every simple cycle of *g*, each once, as a closed node list ``[a, …, a]``.

    Johnson's algorithm on directed graphs, O((n + m)(c + 1)) for c cycles.
    Each cycle starts at its earliest node in graph order. Self-loops come
    first, as ``[a, a]``. Undirected graphs are supported too: every cycle of
    three or more nodes is reported once (in one of its two orientations).

    networkx lists cycles open (``[a, b, c]``); AryaGraph closes them to match
    :func:`find_cycle` and :attr:`CycleError.cycle`.
    """
    index = g.node_index()
    directed = g.directed
    for n in g._node:
        if n in g._succ[n]:
            yield [n, n]
    adj = {n: [v for v in g._succ[n] if v != n] for n in g._node}

    def by_first(comp: set[Node]) -> int:
        return min(index[v] for v in comp)

    pending = sorted((set(c) for c in _scc(adj, adj) if len(c) > 1), key=by_first, reverse=True)
    while pending:
        comp = pending.pop()
        start = min(comp, key=index.__getitem__)
        sub = {v: [w for w in adj[v] if w in comp] for v in comp}
        for cycle in _circuits(start, sub):
            if directed or (len(cycle) > 3 and index[cycle[1]] < index[cycle[-2]]):
                yield cycle
        comp.discard(start)
        rest = {v: [w for w in sub[v] if w != start] for v in comp}
        found = [set(c) for c in _scc(rest, rest) if len(c) > 1]
        pending.extend(sorted(found, key=by_first, reverse=True))


def _circuits(start: Node, adj: dict[Node, list[Node]]) -> Iterator[list[Node]]:
    """Johnson's CIRCUIT: all elementary cycles through *start* in *adj* (iterative)."""
    path = [start]
    blocked = {start}
    blocked_by: dict[Node, set[Node]] = defaultdict(set)
    closed = [False]
    stack = [iter(adj[start])]
    while stack:
        for w in stack[-1]:
            if w == start:
                yield [*path, start]
                closed[-1] = True
            elif w not in blocked:
                path.append(w)
                closed.append(False)
                blocked.add(w)
                stack.append(iter(adj[w]))
                break
        else:
            stack.pop()
            v = path.pop()
            if closed.pop():
                if closed:
                    closed[-1] = True
                todo = [v]
                while todo:  # unblock v and, transitively, everything waiting on it
                    u = todo.pop()
                    if u in blocked:
                        blocked.discard(u)
                        todo.extend(blocked_by[u])
                        blocked_by[u].clear()
            else:
                for w in adj[v]:
                    blocked_by[w].add(v)


# ---------------------------------------------------------------------- #
# longest paths and scheduling
# ---------------------------------------------------------------------- #
def _longest(g: Graph, weight: WeightSpec, default: float) -> tuple[list[Node], dict[Node, Any], dict[Node, Node | None]]:
    order = _topological_order(g, "dag_longest_path")
    wf = _edge_weight(weight, default)
    pred = g._pred
    best: dict[Node, Any] = {}
    parent: dict[Node, Node | None] = {}
    for v in order:
        bv: Any = None
        pv = None
        for u, d in pred[v].items():
            cand = best[u] + wf(u, v, d)
            if bv is None or cand > bv:
                bv, pv = cand, u
        if bv is None or bv < 0:  # starting afresh at v beats every way in
            bv, pv = 0, None
        best[v] = bv
        parent[v] = pv
    return order, best, parent


def dag_longest_path(g: Graph, weight: WeightSpec = None, default: float = 1.0) -> list[Node]:
    """A heaviest path of a DAG, as a node list.

    ``weight=None`` counts edges (so the path with the most nodes wins); an
    attribute name sums that **edge** attribute, *default* standing in where
    it is missing; a callable ``f(u, v, attrs)`` is used as is. With negative
    weights the best path may start anywhere, as in networkx. Ties go to the
    path ending earliest in topological order. O(n + m).
    """
    order, best, parent = _longest(g, weight, default)
    if not order:
        return []
    end = max(order, key=best.__getitem__)
    path = [end]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])  # type: ignore[arg-type]
    path.reverse()
    return path


def dag_longest_path_length(g: Graph, weight: WeightSpec = None, default: float = 1.0) -> Any:
    """Length of :func:`dag_longest_path` (number of edges when *weight* is None; 0 if empty)."""
    _, best, _ = _longest(g, weight, default)
    return max(best.values(), default=0)


def dag_levels(g: Graph) -> NodeMap:
    """Longest-path level of every node: sources are 0 and every arc climbs at least one level."""
    order = _topological_order(g, "dag_levels")
    pred = g._pred
    level: dict[Node, int] = {}
    for v in order:
        level[v] = 1 + max(level[u] for u in pred[v]) if pred[v] else 0
    return NodeMap({n: level[n] for n in g._node}, name="level")


@dataclass(frozen=True, repr=False)
class CriticalPath:
    """Result of :func:`critical_path` (activity-on-node CPM).

    Attributes
    ----------
    path:
        A critical chain from a source to a sink whose durations add up to
        :attr:`length`.
    length:
        The makespan: the earliest time every activity can be finished.
    earliest_start:
        Per-node schedule value (:class:`NodeMap`): the earliest time each
        activity can start, the latest earliest finish among its
        predecessors (0 for a source).
    earliest_finish:
        Per-node schedule value (:class:`NodeMap`): the earliest time each
        activity can finish, its earliest start plus its duration.
    latest_start:
        Per-node schedule value (:class:`NodeMap`): the latest time each
        activity can start without delaying the project, its latest finish
        minus its duration.
    latest_finish:
        Per-node schedule value (:class:`NodeMap`): the latest time each
        activity can finish without delaying the project, the earliest
        latest start among its successors (the makespan for a sink).
    slack:
        Per-node schedule value (:class:`NodeMap`): latest start minus
        earliest start. Slack is how long an activity can slip without
        delaying the project.
    critical:
        Every activity with zero slack (any of them slipping delays the end).
    """

    path: list
    length: float
    earliest_start: NodeMap
    earliest_finish: NodeMap
    latest_start: NodeMap
    latest_finish: NodeMap
    slack: NodeMap
    critical: set

    @property
    def makespan(self) -> float:
        """Alias of :attr:`length`."""
        return self.length

    def __repr__(self) -> str:
        shown = " → ".join(map(repr, self.path[:8])) + (" → …" if len(self.path) > 8 else "")
        return (
            f"CriticalPath(length={self.length:g}, path=[{shown}], "
            f"critical={len(self.critical)} of {len(self.slack)} activities)"
        )


def critical_path(
    g: Graph,
    duration: str | Callable[[Node], float] = "duration",
    default: float = 1.0,
) -> CriticalPath:
    """Critical-path method (CPM) with activities on the nodes.

    Each node is an activity lasting ``g.nodes[n][duration]`` (or
    ``duration(n)`` when a callable is given; *default* when the attribute is
    missing); an arc ``u → v`` means v cannot start before u finishes. A
    forward pass gives earliest start/finish, a backward pass from the
    makespan gives latest start/finish, and slack is their difference.
    Activities whose absolute slack is ≤ 1e-9·max(1, makespan) count as critical (their
    slack is reported as exactly 0). O(n + m).

    >>> dag = DAG([("design", "build"), ("design", "docs"), ("build", "ship"), ("docs", "ship")])
    >>> for n, d in {"design": 2, "build": 5, "docs": 1, "ship": 1}.items():
    ...     dag.nodes[n]["duration"] = d
    >>> critical_path(dag).path
    ['design', 'build', 'ship']
    """
    order = _topological_order(g, "critical_path")
    nodes = g._node
    if callable(duration):
        raw = {n: duration(n) for n in order}
    else:
        raw = {n: nodes[n].get(duration) for n in order}
    dur: dict[Node, float] = {}
    for n, d in raw.items():
        d = default if d is None else d
        try:
            d = float(d)
        except (TypeError, ValueError):
            raise ValueError(f"duration of {n!r} must be a number, got {d!r}") from None
        if not math.isfinite(d) or d < 0:
            raise ValueError(f"duration of {n!r} must be finite and non-negative, got {d!r}")
        dur[n] = d
    pred, succ = g._pred, g._succ
    es: dict[Node, float] = {}
    ef: dict[Node, float] = {}
    for v in order:
        es[v] = max((ef[u] for u in pred[v]), default=0.0)
        ef[v] = es[v] + dur[v]
    makespan = max(ef.values(), default=0.0)
    ls: dict[Node, float] = {}
    lf: dict[Node, float] = {}
    for v in reversed(order):
        lf[v] = min((ls[w] for w in succ[v]), default=makespan)
        ls[v] = lf[v] - dur[v]
    tol = 1e-9 * max(1.0, abs(makespan))
    slack: dict[Node, float] = {}
    for v in order:
        s = ls[v] - es[v]
        slack[v] = 0.0 if abs(s) <= tol else s
    path: list[Node] = []
    if order:
        # EF never decreases along an arc, so some sink finishes at the makespan;
        # walk back through predecessors that finish exactly when v starts.
        end = next(v for v in order if not succ[v] and ef[v] == makespan)
        path.append(end)
        while pred[path[-1]]:
            v = path[-1]
            path.append(next(u for u in pred[v] if ef[u] == es[v]))
        path.reverse()

    def ordered(values: dict[Node, float], name: str) -> NodeMap:
        return NodeMap({n: values[n] for n in nodes}, name=name)

    return CriticalPath(
        path=path,
        length=makespan,
        earliest_start=ordered(es, "earliest_start"),
        earliest_finish=ordered(ef, "earliest_finish"),
        latest_start=ordered(ls, "latest_start"),
        latest_finish=ordered(lf, "latest_finish"),
        slack=ordered(slack, "slack"),
        critical={n for n in nodes if slack[n] == 0.0},
    )


# ---------------------------------------------------------------------- #
# closure and reduction
# ---------------------------------------------------------------------- #
def transitive_closure(g: Graph, reflexive: bool | None = False) -> DiGraph:
    """Graph with an arc ``u → v`` for every directed path ``u ⇝ v``.

    A :class:`DAG` gives a DAG, any other directed graph a graph of its own
    type. Node, edge and graph attributes are kept; new arcs have none.

    Parameters
    ----------
    reflexive:
        As in networkx: ``False`` adds a self-loop to every node lying on a
        cycle, ``True`` to every node (the result is then a :class:`DiGraph`
        even for a DAG), ``None`` adds none.

    Notes
    -----
    Strongly connected components are contracted first and reachability is
    propagated as bitsets, so the cost is O(n·m / w) plus the output size.
    """
    require_directed(g, "transitive_closure")
    if reflexive not in (None, True, False):
        raise ValueError(f"reflexive must be None, True or False, got {reflexive!r}")
    nodes = list(g._node)
    index = {n: i for i, n in enumerate(nodes)}
    succ = g._succ
    comps = _scc(g._node, succ)  # reverse topological order: successors come first
    comp_of: dict[Node, int] = {}
    for i, c in enumerate(comps):
        for n in c:
            comp_of[n] = i
    member_bits: list[int] = []
    reach: list[int] = []
    for i, c in enumerate(comps):
        members = 0
        for n in c:
            members |= 1 << index[n]
        member_bits.append(members)
        cyclic = len(c) > 1 or c[0] in succ[c[0]]
        r = members if cyclic else 0
        for n in c:
            for v in succ[n]:
                j = comp_of[v]
                if j != i:  # j < i: Tarjan emitted that component already
                    r |= member_bits[j] | reach[j]
        reach.append(r)
    out = DiGraph(g) if (isinstance(g, DAG) and reflexive is True) else g.copy()
    new_edges = []
    for u in nodes:
        bits = reach[comp_of[u]]
        me = 1 << index[u]
        if reflexive is True:
            bits |= me
        elif reflexive is None:
            bits &= ~me
        have = out._succ[u]
        new_edges.extend((u, nodes[i]) for i in _iter_bits(bits) if nodes[i] not in have)
    out.add_edges(new_edges)
    return out


def transitive_reduction(g: Graph) -> DAG:
    """The smallest DAG with the same reachability as the acyclic graph *g*.

    Keeps every node (with its attributes) and exactly those arcs ``u → v``
    for which no other path ``u ⇝ v`` exists; kept arcs keep their
    attributes. Raises :class:`CycleError` on a cyclic input (the reduction
    of a cyclic graph is not unique). O(n·m / w) with bitsets.
    """
    order = _topological_order(g, "transitive_reduction")
    index = g.node_index()
    reach = _descendant_bits(g, order, index)
    out = DAG()
    out.attrs = dict(g.attrs)
    out.add_nodes((n, dict(d)) for n, d in g._node.items())
    kept = []
    for u, nbrs in g._succ.items():
        covered = 0
        for v in nbrs:
            covered |= reach[v]
        kept.extend((u, v, dict(d)) for v, d in nbrs.items() if not (covered >> index[v]) & 1)
    out.add_edges(kept)
    return out


# ---------------------------------------------------------------------- #
# ancestry and width
# ---------------------------------------------------------------------- #
def lowest_common_ancestors(g: Graph, a: Node, b: Node) -> set[Node]:
    """All lowest common ancestors of *a* and *b* in a DAG.

    A common ancestor (every node counts as its own ancestor) is *lowest*
    when none of its descendants is also a common ancestor. Trees have one;
    general DAGs may have several, or none (empty set). networkx's
    ``lowest_common_ancestor`` returns just one of these. O(n + m).
    """
    require_node(g, a)
    require_node(g, b)
    _topological_order(g, "lowest_common_ancestors")
    common = _ancestors_inclusive(g, a) & _ancestors_inclusive(g, b)
    succ = g._succ
    return {x for x in common if not any(y in common for y in succ[x])}


def _ancestors_inclusive(g: Graph, node: Node) -> set[Node]:
    pred = g._pred
    seen = {node}
    stack = [node]
    while stack:
        for u in pred[stack.pop()]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return seen


def _dilworth(g: Graph, what: str) -> tuple[list[Node], list[list[int]], list[int], list[int]]:
    """Maximum matching of the comparability bipartite graph (u → v whenever u ⇝ v)."""
    order = _topological_order(g, what)
    nodes = list(g._node)
    index = {n: i for i, n in enumerate(nodes)}
    reach = _descendant_bits(g, order, index)
    adj = [list(_iter_bits(reach[n])) for n in nodes]
    mate_l, mate_r = _hopcroft_karp_core(adj, len(nodes))
    return nodes, adj, mate_l, mate_r


def dag_width(g: Graph) -> int:
    """Size of a largest antichain (set of pairwise unreachable nodes) of a DAG.

    By Dilworth's theorem this equals the minimum number of chains covering
    the nodes: n minus a maximum matching (Hopcroft–Karp) in the bipartite
    graph linking u to every node reachable from u. O(n·c·√n) for c
    comparable pairs.
    """
    nodes, _, mate_l, _ = _dilworth(g, "dag_width")
    return len(nodes) - sum(1 for m in mate_l if m >= 0)


def maximum_antichain(g: Graph) -> set[Node]:
    """A largest set of pairwise incomparable nodes of a DAG (its size is :func:`dag_width`).

    Built from the maximum matching via König's theorem: the nodes whose
    left copy is reachable by an alternating path from an unmatched left
    copy while their right copy is not.
    """
    nodes, adj, mate_l, mate_r = _dilworth(g, "maximum_antichain")
    left_seen = [False] * len(nodes)
    right_seen = [False] * len(nodes)
    stack = [u for u in range(len(nodes)) if mate_l[u] < 0]
    for u in stack:
        left_seen[u] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not right_seen[v]:
                right_seen[v] = True
                w = mate_r[v]
                if w >= 0 and not left_seen[w]:
                    left_seen[w] = True
                    stack.append(w)
    return {nodes[i] for i in range(len(nodes)) if left_seen[i] and not right_seen[i]}


def minimum_chain_partition(g: Graph) -> list[list[Node]]:
    """Fewest chains covering every node of a DAG exactly once (Dilworth decomposition).

    Each chain lists nodes in topological order; consecutive nodes are
    comparable (a path joins them) but need not be adjacent. There are
    :func:`dag_width` chains, ordered by their first node's graph position.
    """
    nodes, _, mate_l, mate_r = _dilworth(g, "minimum_chain_partition")
    chains = []
    for i in range(len(nodes)):
        if mate_r[i] >= 0:
            continue  # not the head of a chain
        chain = [nodes[i]]
        j = mate_l[i]
        while j >= 0:
            chain.append(nodes[j])
            j = mate_l[j]
        chains.append(chain)
    return chains


__all__ = [
    "is_dag",
    "topological_sort",
    "all_topological_sorts",
    "find_cycle",
    "simple_cycles",
    "dag_longest_path",
    "dag_longest_path_length",
    "dag_levels",
    "CriticalPath",
    "critical_path",
    "transitive_closure",
    "transitive_reduction",
    "lowest_common_ancestors",
    "dag_width",
    "maximum_antichain",
    "minimum_chain_partition",
]
