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

"""Breadth-first and depth-first traversal.

Every routine is iterative (no recursion limit), visits neighbors in graph
order, and follows successors on directed graphs, so the same graph always
yields the same order. Orders and edge lists match networkx's
``bfs_edges`` / ``dfs_preorder_nodes`` / ``dfs_edges`` … on graphs built in
the same insertion order.
"""

from __future__ import annotations

from typing import Hashable, Iterable

from ..core.exceptions import NodeNotFound
from ..core.graph import DiGraph, Graph
from ..core.utils import require_node

Node = Hashable


# ---------------------------------------------------------------------- #
# breadth-first search
# ---------------------------------------------------------------------- #
def _bfs(g: Graph, source: Node, reverse: bool, depth_limit: int | None) -> list[tuple[Node, Node]]:
    """Tree edges ``(parent, child)`` of a BFS from *source*, in discovery order."""
    require_node(g, source)
    adj = g._pred if (reverse and g.directed) else g._succ
    seen = {source}
    edges: list[tuple[Node, Node]] = []
    frontier = [source]
    depth = 0
    while frontier and (depth_limit is None or depth < depth_limit):
        depth += 1
        nxt: list[Node] = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    edges.append((u, v))
                    nxt.append(v)
        frontier = nxt
    return edges


def bfs_edges(
    g: Graph, source: Node, *, reverse: bool = False, depth_limit: int | None = None
) -> list[tuple[Node, Node]]:
    """Edges of the breadth-first search tree rooted at *source*, in discovery order.

    Parameters
    ----------
    reverse:
        On a directed graph, follow arcs backwards (predecessors). Tree edges
        are still reported ``(parent, child)``, i.e. against the arc direction.
    depth_limit:
        Only explore nodes at most this many hops from *source*.

    Examples
    --------
    >>> bfs_edges(Graph([(0, 1), (0, 2), (1, 3)]), 0)
    [(0, 1), (0, 2), (1, 3)]
    """
    return _bfs(g, source, reverse, depth_limit)


def bfs_order(
    g: Graph, source: Node, *, reverse: bool = False, depth_limit: int | None = None
) -> list[Node]:
    """Nodes reachable from *source* in breadth-first order (*source* first).

    See :func:`bfs_edges` for *reverse* and *depth_limit*.
    """
    return [source, *(v for _, v in _bfs(g, source, reverse, depth_limit))]


def bfs_layers(g: Graph, sources: Node | Iterable[Node]) -> list[list[Node]]:
    """Nodes grouped by hop distance from *sources*.

    *sources* is a single node or an iterable of nodes (a value that is itself
    a node of *g* is always treated as a single node). Layer 0 is the sources;
    layer ``k`` holds the nodes first reached after ``k`` hops.

    >>> bfs_layers(Graph([(0, 1), (1, 2), (0, 3)]), 0)
    [[0], [1, 3], [2]]
    """
    starts = [sources] if sources in g else list(dict.fromkeys(sources))  # type: ignore[arg-type]
    for s in starts:
        require_node(g, s)
    succ = g._succ
    seen = set(starts)
    layers: list[list[Node]] = []
    layer = starts
    while layer:
        layers.append(layer)
        nxt: list[Node] = []
        for u in layer:
            for v in succ[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        layer = nxt
    return layers


def bfs_tree(g: Graph, source: Node, *, reverse: bool = False, depth_limit: int | None = None) -> DiGraph:
    """Breadth-first search tree rooted at *source*, as a :class:`DiGraph`.

    Arcs point away from the root. Node and edge attributes are copied from
    *g* (networkx's ``bfs_tree`` drops them).
    """
    edges = _bfs(g, source, reverse, depth_limit)
    return _tree(g, [source, *(v for _, v in edges)], edges)


# ---------------------------------------------------------------------- #
# depth-first search
# ---------------------------------------------------------------------- #
def _dfs_events(
    g: Graph, source: Node | None, depth_limit: int | None
) -> tuple[list[Node], list[Node], list[tuple[Node, Node]]]:
    """One iterative DFS returning ``(preorder, postorder, tree_edges)``.

    With ``source=None`` every node is used as a root in graph order, so the
    result covers the whole graph (a DFS forest).
    """
    if source is None:
        roots: Iterable[Node] = g._node
    else:
        require_node(g, source)
        roots = (source,)
    succ = g._succ
    limit = len(g) if depth_limit is None else depth_limit
    seen: set[Node] = set()
    pre: list[Node] = []
    post: list[Node] = []
    edges: list[tuple[Node, Node]] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        pre.append(root)
        stack = [(root, iter(succ[root]))]
        while stack:
            u, nbrs = stack[-1]
            for v in nbrs:
                if v not in seen:
                    seen.add(v)
                    pre.append(v)
                    edges.append((u, v))
                    if len(stack) < limit:
                        stack.append((v, iter(succ[v])))
                    else:  # at the depth limit: v is a leaf of the search
                        post.append(v)
                    break
            else:
                stack.pop()
                post.append(u)
    return pre, post, edges


def dfs_preorder(g: Graph, source: Node | None = None, *, depth_limit: int | None = None) -> list[Node]:
    """Nodes in depth-first preorder (a node before its descendants).

    With ``source=None`` the search restarts from every unvisited node in
    graph order, covering the whole graph.
    """
    return _dfs_events(g, source, depth_limit)[0]


def dfs_postorder(g: Graph, source: Node | None = None, *, depth_limit: int | None = None) -> list[Node]:
    """Nodes in depth-first postorder (a node after all its descendants).

    On a DAG, the reversed postorder is a topological order. With
    *depth_limit*, nodes cut off at the limit are still listed (networkx's
    ``dfs_postorder_nodes`` drops them, although its preorder keeps them).
    """
    return _dfs_events(g, source, depth_limit)[1]


def dfs_edges(
    g: Graph, source: Node | None = None, *, depth_limit: int | None = None
) -> list[tuple[Node, Node]]:
    """Tree edges ``(parent, child)`` of the depth-first search, in discovery order."""
    return _dfs_events(g, source, depth_limit)[2]


def dfs_tree(g: Graph, source: Node | None = None, *, depth_limit: int | None = None) -> DiGraph:
    """Depth-first search tree (forest when ``source=None``) as a :class:`DiGraph`.

    Node and edge attributes are copied from *g*.
    """
    pre, _, edges = _dfs_events(g, source, depth_limit)
    return _tree(g, pre, edges)


# ---------------------------------------------------------------------- #
# reachability
# ---------------------------------------------------------------------- #
def descendants(g: Graph, node: Node) -> set[Node]:
    """Every node reachable from *node*, excluding *node* itself.

    For an undirected graph this is *node*'s connected component minus *node*.
    """
    return _reach(g, node, reverse=False)


def ancestors(g: Graph, node: Node) -> set[Node]:
    """Every node from which *node* is reachable, excluding *node* itself.

    For an undirected graph this equals :func:`descendants`.
    """
    return _reach(g, node, reverse=True)


def _reach(g: Graph, node: Node, reverse: bool) -> set[Node]:
    if node not in g:
        raise NodeNotFound(node)
    adj = g._pred if reverse else g._succ
    seen = {node}
    stack = [node]
    while stack:
        for v in adj[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    seen.discard(node)
    return seen


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _tree(g: Graph, nodes: list[Node], edges: list[tuple[Node, Node]]) -> DiGraph:
    """A DiGraph with *nodes* (in that order) and the tree *edges*, attributes copied."""
    tree = DiGraph()
    tree.attrs = dict(g.attrs)
    tree.add_nodes((n, dict(g._node[n])) for n in nodes)
    succ = g._succ
    for u, v in edges:
        attrs = succ[u].get(v)
        if attrs is None:  # reverse traversal: the underlying arc is v -> u
            attrs = succ[v][u]
        tree.add_edge(u, v, **attrs)
    return tree


__all__ = [
    "bfs_order",
    "bfs_edges",
    "bfs_layers",
    "bfs_tree",
    "dfs_preorder",
    "dfs_postorder",
    "dfs_edges",
    "dfs_tree",
    "descendants",
    "ancestors",
]
