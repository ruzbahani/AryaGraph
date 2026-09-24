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

"""Connected, strongly connected and biconnected structure.

Component lists are ordered largest first; equal sizes keep the order in
which their first node appears in the graph. All searches are iterative.
"""

from __future__ import annotations

import heapq
from typing import Hashable, Iterable, Mapping

from ..core.dag import DAG
from ..core.graph import Graph
from ..core.utils import require_directed, require_node, require_undirected
from ._core import components

Node = Hashable


def _largest_first(groups: Iterable[Iterable[Node]], index: dict[Node, int]) -> list[set[Node]]:
    keyed = [(len(s), min(index[n] for n in s), s) for s in map(set, groups)]
    keyed.sort(key=lambda t: (-t[0], t[1]))
    return [s for _, _, s in keyed]


# ---------------------------------------------------------------------- #
# undirected components
# ---------------------------------------------------------------------- #
def connected_components(g: Graph) -> list[set[Node]]:
    """Connected components of an undirected graph, largest first.

    Raises :class:`GraphTypeError` for directed graphs; use
    :func:`strongly_connected_components` or :func:`weakly_connected_components`.
    """
    require_undirected(g, "connected_components")
    return _largest_first(components(g), g.node_index())


def number_connected_components(g: Graph) -> int:
    """How many connected components an undirected graph has (0 for the empty graph)."""
    require_undirected(g, "number_connected_components")
    return len(components(g))


def is_connected(g: Graph) -> bool:
    """True when an undirected graph has exactly one connected component.

    The empty graph is reported as *not* connected (networkx raises instead).
    """
    require_undirected(g, "is_connected")
    return len(g) > 0 and len(_reach_undirected(g, next(iter(g._node)))) == len(g)


def node_connected_component(g: Graph, node: Node) -> set[Node]:
    """The set of nodes in *node*'s connected component (including *node*)."""
    require_undirected(g, "node_connected_component")
    require_node(g, node)
    return _reach_undirected(g, node)


def _reach_undirected(g: Graph, node: Node) -> set[Node]:
    """Nodes linked to *node* ignoring arc directions."""
    succ, pred = g._succ, g._pred
    seen = {node}
    stack = [node]
    while stack:
        u = stack.pop()
        for adj in (succ[u], pred[u]) if g.directed else (succ[u],):
            for v in adj:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    return seen


# ---------------------------------------------------------------------- #
# directed components
# ---------------------------------------------------------------------- #
def _tarjan(g: Graph) -> list[list[Node]]:
    """Strongly connected components of *g* (see :func:`_scc`)."""
    return _scc(g._node, g._succ)


def _scc(nodes: Iterable[Node], succ: Mapping[Node, Iterable[Node]]) -> list[list[Node]]:
    """Strongly connected components by iterative Tarjan over any adjacency mapping.

    ``succ[v]`` may only mention nodes of *nodes*. Components come out in
    reverse topological order of the condensation: every component is
    emitted after all components it can reach.
    """
    index: dict[Node, int] = {}
    low: dict[Node, int] = {}
    on_stack: set[Node] = set()
    stack: list[Node] = []
    out: list[list[Node]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(succ[root]))]
        while work:
            v, nbrs = work[-1]
            for w in nbrs:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(succ[w])))
                    break
                if w in on_stack and index[w] < low[v]:
                    low[v] = index[w]
            else:
                work.pop()
                if work:
                    u = work[-1][0]
                    if low[v] < low[u]:
                        low[u] = low[v]
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    out.append(comp)
    return out


def strongly_connected_components(g: Graph) -> list[set[Node]]:
    """Strongly connected components of a directed graph, largest first.

    Iterative Tarjan, O(n + m). Every node belongs to exactly one component
    (a node on no cycle is a component of its own).
    """
    require_directed(g, "strongly_connected_components")
    return _largest_first(_tarjan(g), g.node_index())


def weakly_connected_components(g: Graph) -> list[set[Node]]:
    """Components of a directed graph when arc directions are ignored, largest first."""
    require_directed(g, "weakly_connected_components")
    return _largest_first(components(g), g.node_index())


def number_strongly_connected_components(g: Graph) -> int:
    """How many strongly connected components a directed graph has."""
    require_directed(g, "number_strongly_connected_components")
    return len(_tarjan(g))


def number_weakly_connected_components(g: Graph) -> int:
    """How many weakly connected components a directed graph has."""
    require_directed(g, "number_weakly_connected_components")
    return len(components(g))


def is_strongly_connected(g: Graph) -> bool:
    """True when every node reaches every other node (False for the empty graph)."""
    require_directed(g, "is_strongly_connected")
    if len(g) == 0:
        return False
    start = next(iter(g._node))
    return len(_reach(g._succ, start)) == len(g) and len(_reach(g._pred, start)) == len(g)


def is_weakly_connected(g: Graph) -> bool:
    """True when the graph is connected once arc directions are ignored (False if empty)."""
    require_directed(g, "is_weakly_connected")
    return len(g) > 0 and len(_reach_undirected(g, next(iter(g._node)))) == len(g)


def _reach(adj: dict, start: Node) -> set[Node]:
    seen = {start}
    stack = [start]
    while stack:
        for v in adj[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def condensation(g: Graph, scc: Iterable[Iterable[Node]] | None = None) -> DAG:
    """The DAG obtained by contracting each strongly connected component to one node.

    Component ids are ``0 … k-1`` numbered in topological order (ties broken
    by the graph position of each component's first node), so every arc of
    the result goes from a lower to a higher id. Each node carries the
    attribute ``members`` (the set of original nodes); the graph attribute
    ``mapping`` maps every original node to its component id.

    Parameters
    ----------
    scc:
        Precomputed strongly connected components (any order), to skip Tarjan.
    """
    require_directed(g, "condensation")
    index = g.node_index()
    groups = [set(c) for c in (_tarjan(g) if scc is None else scc)]
    first = [min(index[n] for n in c) for c in groups]
    raw_of: dict[Node, int] = {}
    for i, c in enumerate(groups):
        for n in c:
            raw_of[n] = i
    if len(raw_of) != len(g):
        raise ValueError("scc must partition the nodes of the graph")
    succ_c: list[dict[int, None]] = [{} for _ in groups]
    indeg = [0] * len(groups)
    for u, nbrs in g._succ.items():
        cu = raw_of[u]
        for v in nbrs:
            cv = raw_of[v]
            if cu != cv and cv not in succ_c[cu]:
                succ_c[cu][cv] = None
                indeg[cv] += 1
    # Kahn with a heap keyed by first appearance: a canonical topological numbering.
    heap = [(first[i], i) for i in range(len(groups)) if indeg[i] == 0]
    heapq.heapify(heap)
    new_id = [-1] * len(groups)
    k = 0
    while heap:
        _, i = heapq.heappop(heap)
        new_id[i] = k
        k += 1
        for j in succ_c[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, (first[j], j))
    if k != len(groups):
        raise ValueError("scc is not a partition into strongly connected components")
    order = sorted(range(len(groups)), key=new_id.__getitem__)
    dag = DAG()
    dag.add_nodes((new_id[i], {"members": groups[i]}) for i in order)
    dag.add_edges((new_id[i], new_id[j]) for i in order for j in succ_c[i])
    dag.attrs["mapping"] = {n: new_id[raw_of[n]] for n in g._node}
    return dag


# ---------------------------------------------------------------------- #
# biconnectivity
# ---------------------------------------------------------------------- #
def _biconnected(g: Graph, want_components: bool) -> tuple[set[Node], set[frozenset], list[set[Node]]]:
    """Iterative Hopcroft–Tarjan: articulation points, bridges and biconnected components.

    Self-loops are ignored. Components are node sets; isolated nodes belong
    to none, a bridge forms a two-node component.
    """
    succ = g._succ
    disc: dict[Node, int] = {}
    low: dict[Node, int] = {}
    cut: set[Node] = set()
    bridges: set[frozenset] = set()
    comps: list[set[Node]] = []
    counter = 0
    for root in g._node:
        if root in disc:
            continue
        disc[root] = low[root] = counter
        counter += 1
        root_children = 0
        edge_stack: list[tuple[Node, Node]] = []
        work = [(root, None, iter(succ[root]))]
        while work:
            u, parent, nbrs = work[-1]
            for v in nbrs:
                if v == u or v == parent:  # self-loop, or the tree edge back up
                    continue
                if v not in disc:
                    disc[v] = low[v] = counter
                    counter += 1
                    if want_components:
                        edge_stack.append((u, v))
                    work.append((v, u, iter(succ[v])))
                    break
                if disc[v] < disc[u]:  # back edge to an ancestor
                    if disc[v] < low[u]:
                        low[u] = disc[v]
                    if want_components:
                        edge_stack.append((u, v))
            else:
                work.pop()
                if parent is None:
                    continue
                if low[u] < low[parent]:
                    low[parent] = low[u]
                if low[u] >= disc[parent]:
                    if parent == root:
                        root_children += 1
                    else:
                        cut.add(parent)
                    if want_components:
                        comp: set[Node] = set()
                        while True:
                            a, b = edge_stack.pop()
                            comp.add(a)
                            comp.add(b)
                            if a == parent and b == u:
                                break
                        comps.append(comp)
                if low[u] > disc[parent]:
                    bridges.add(frozenset((parent, u)))
        if root_children > 1:
            cut.add(root)
    return cut, bridges, comps


def articulation_points(g: Graph) -> list[Node]:
    """Cut vertices of an undirected graph (removal disconnects their component), in graph order."""
    require_undirected(g, "articulation_points")
    cut = _biconnected(g, False)[0]
    return [n for n in g._node if n in cut]


def bridges(g: Graph) -> list[tuple[Node, Node]]:
    """Bridges of an undirected graph (edges whose removal disconnects), in edge order."""
    require_undirected(g, "bridges")
    found = _biconnected(g, False)[1]
    return [(u, v) for u, v in g.edges if u != v and frozenset((u, v)) in found]


def biconnected_components(g: Graph) -> list[set[Node]]:
    """Maximal biconnected subgraphs of an undirected graph, as node sets, largest first.

    Articulation points belong to several components; isolated nodes to none.
    """
    require_undirected(g, "biconnected_components")
    return _largest_first(_biconnected(g, True)[2], g.node_index())


def is_biconnected(g: Graph) -> bool:
    """True when the graph is connected and has no articulation point.

    Follows networkx: a single edge (two nodes) is biconnected; the empty
    graph and a single node are not.
    """
    require_undirected(g, "is_biconnected")
    comps = _biconnected(g, True)[2]
    return len(comps) == 1 and len(comps[0]) == len(g)


__all__ = [
    "connected_components",
    "number_connected_components",
    "is_connected",
    "node_connected_component",
    "strongly_connected_components",
    "weakly_connected_components",
    "number_strongly_connected_components",
    "number_weakly_connected_components",
    "is_strongly_connected",
    "is_weakly_connected",
    "condensation",
    "articulation_points",
    "bridges",
    "biconnected_components",
    "is_biconnected",
]
