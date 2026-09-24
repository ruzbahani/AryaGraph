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

"""Graph and DiGraph: the core data structures.

Storage is the classic adjacency-dict design (fast neighbor iteration, O(1)
edge lookup) with a few firm guarantees:

* **Deterministic order.** Nodes iterate in insertion order; so do each node's
  neighbors. Every algorithm, layout and renderer in AryaGraph inherits this, so the
  same input always produces the same output.
* **Shared edge attributes.** An undirected edge has exactly one attribute dict,
  reachable from both endpoints.
* **O(1) sizes.** Node and edge counts are maintained, not recomputed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterator

from .exceptions import EdgeNotFound, GraphTypeError, NodeNotFound
from .results import NodeMap
from .views import AdjacencyView, EdgeView, NodeView

if TYPE_CHECKING:
    from .dag import DAG

Node = Hashable


class Graph:
    """Undirected simple graph (self-loops allowed, no parallel edges).

    Parameters
    ----------
    data:
        Optional initial content: another graph (copied), an adjacency mapping
        ``{u: [v, ...]}`` / ``{u: {v: attrs}}``, or an iterable of edges where
        each edge is ``(u, v)``, ``(u, v, attrs_dict)`` or ``(u, v, weight)``.
    nodes:
        Extra nodes to add, each ``n`` or ``(n, attrs_dict)``. They are added
        *before* the content of *data*, so they lead the graph order; pass
        them to fix the order nodes are iterated, laid out and listed in.
    name:
        Optional display name (stored as ``g.attrs['name']``).
    **attrs:
        Graph-level attributes.

    Examples
    --------
    >>> g = Graph([("a", "b"), ("b", "c", 2.5)], name="demo")
    >>> g.add_node("d", color="red")
    >>> g.edges["b", "c"]
    {'weight': 2.5}
    """

    directed: bool = False

    def __init__(
        self,
        data: Any = None,
        *,
        nodes: Iterable[Any] | None = None,
        name: str | None = None,
        **attrs: Any,
    ) -> None:
        self._node: dict[Node, dict] = {}
        self._succ: dict[Node, dict[Node, dict]] = {}
        # Undirected graphs keep a single adjacency dict under both names.
        self._pred: dict[Node, dict[Node, dict]] = {} if self.directed else self._succ
        self._n_edges = 0
        self._version = 0
        self.attrs: dict[str, Any] = {}
        if isinstance(data, Graph):
            self.attrs.update(data.attrs)
        self.attrs.update(attrs)
        if name is not None:
            self.attrs["name"] = name
        if nodes is not None:
            self.add_nodes(nodes)
        if data is not None:
            self._load(data)

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _load(self, data: Any) -> None:
        if isinstance(data, Graph):
            self.add_nodes((n, dict(d)) for n, d in data._node.items())
            self.add_edges((u, v, dict(d)) for u, v, d in data._iter_edges())
        elif _is_networkx_graph(data):
            if data.is_multigraph():
                raise TypeError("multigraphs are not supported; collapse parallel edges first (e.g. nx.Graph(G))")
            self.attrs.update(getattr(data, "graph", {}) or {})
            self.add_nodes((n, dict(d)) for n, d in data.nodes(data=True))
            self.add_edges((u, v, dict(d)) for u, v, d in data.edges(data=True))
        elif isinstance(data, Mapping) and not isinstance(data, AbstractSet):
            # adjacency: {u: [v, ...]} or {u: {v: attrs}}
            for u, nbrs in data.items():
                self.add_node(u)
                if isinstance(nbrs, Mapping):
                    for v, d in nbrs.items():
                        if not isinstance(d, Mapping):
                            raise TypeError(
                                "mapping input must be an adjacency dict {u: [v, ...]} or {u: {v: attrs}}; "
                                f"got {{{u!r}: {{{v!r}: {d!r}}}}}"
                            )
                        self.add_edge(u, v, **d)
                else:
                    for v in nbrs:
                        self.add_edge(u, v)
        elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            # edge collections (including set-like views such as networkx's G.edges)
            self.add_edges(data)
        else:
            raise TypeError(f"cannot build a graph from {type(data).__name__}")

    @classmethod
    def from_edges(cls, edges: Iterable[Any], **kwargs: Any):
        """Build a graph from an edge iterable (see the class docstring for edge forms)."""
        g = cls(**kwargs)
        g.add_edges(edges)
        return g

    # ------------------------------------------------------------------ #
    # basic protocol
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self.attrs.get("name", "")

    @name.setter
    def name(self, value: str) -> None:
        self.attrs["name"] = value

    def is_directed(self) -> bool:
        return self.directed

    def __len__(self) -> int:
        return len(self._node)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._node)

    def __contains__(self, node: object) -> bool:
        try:
            return node in self._node
        except TypeError:
            return False

    def __getitem__(self, node: Node) -> Mapping[Node, dict]:
        """Neighbor mapping ``{nbr: edge_attrs}`` of *node* (successors if directed)."""
        try:
            return MappingProxyType(self._succ[node])
        except (KeyError, TypeError):
            raise NodeNotFound(node) from None

    def __bool__(self) -> bool:  # an empty graph is still a graph object
        return True

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"<{type(self).__name__}{label}: {self.num_nodes} nodes, {self.num_edges} edges>"

    @property
    def num_nodes(self) -> int:
        return len(self._node)

    @property
    def num_edges(self) -> int:
        return self._n_edges

    # ------------------------------------------------------------------ #
    # views
    # ------------------------------------------------------------------ #
    @property
    def nodes(self) -> NodeView:
        """Live view of the nodes; ``g.nodes[n]`` is n's attribute dict."""
        return NodeView(self)

    @property
    def edges(self) -> EdgeView:
        """Live view of the edges; ``g.edges[u, v]`` is the edge's attribute dict."""
        return EdgeView(self)

    @property
    def adj(self) -> AdjacencyView:
        """Read-only ``{node: {neighbor: attrs}}`` (successors for directed graphs)."""
        return AdjacencyView(self._succ)

    @property
    def succ(self) -> AdjacencyView:
        """Read-only successor adjacency (same as :attr:`adj` for undirected graphs)."""
        return AdjacencyView(self._succ)

    @property
    def pred(self) -> AdjacencyView:
        """Read-only predecessor adjacency (same as :attr:`adj` for undirected graphs)."""
        return AdjacencyView(self._pred)

    # ------------------------------------------------------------------ #
    # nodes
    # ------------------------------------------------------------------ #
    def add_node(self, node: Node, **attrs: Any) -> None:
        """Add *node* (no-op if present) and update its attributes with *attrs*."""
        if node is None:
            raise ValueError("None cannot be a node")
        if node not in self._node:
            self._node[node] = {}
            self._succ[node] = {}
            if self._pred is not self._succ:
                self._pred[node] = {}
            self._version += 1
        if attrs:
            self._node[node].update(attrs)

    def add_nodes(self, nodes: Iterable[Any], **common: Any) -> None:
        """Add many nodes; each item is ``n`` or ``(n, attrs_dict)``. *common* attrs apply to all."""
        for item in nodes:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Mapping):
                n, d = item
                self.add_node(n, **{**common, **d})
            else:
                self.add_node(item, **common)

    def remove_node(self, node: Node) -> None:
        """Remove *node* and all its incident edges."""
        if node not in self._node:
            raise NodeNotFound(node)
        for v in list(self._succ[node]):
            self._drop_edge(node, v)
        if self._pred is not self._succ:
            for u in list(self._pred[node]):
                self._drop_edge(u, node)
            del self._pred[node]
        del self._succ[node]
        del self._node[node]
        self._version += 1

    def remove_nodes(self, nodes: Iterable[Node]) -> None:
        """Remove every node in *nodes* (missing ones raise :class:`NodeNotFound`)."""
        for n in list(nodes):
            self.remove_node(n)

    def has_node(self, node: Node) -> bool:
        return node in self

    # ------------------------------------------------------------------ #
    # edges
    # ------------------------------------------------------------------ #
    def add_edge(self, u: Node, v: Node, **attrs: Any) -> None:
        """Add edge ``(u, v)`` (creating missing endpoints) and update its attributes."""
        self.add_node(u)
        self.add_node(v)
        d = self._succ[u].get(v)
        if d is None:
            d = {}
            self._succ[u][v] = d
            self._pred[v][u] = d
            self._n_edges += 1
            self._version += 1
        if attrs:
            d.update(attrs)

    def add_edges(self, edges: Iterable[Any], **common: Any) -> None:
        """Add many edges: ``(u, v)``, ``(u, v, attrs_dict)`` or ``(u, v, weight)``.

        *common* attributes are applied to every edge (per-edge values win).
        """
        for e in edges:
            u, v, attrs = _unpack_edge(e)
            if common:
                attrs = {**common, **attrs}
            self.add_edge(u, v, **attrs)

    def remove_edge(self, u: Node, v: Node) -> None:
        """Remove edge ``(u, v)``."""
        if not self.has_edge(u, v):
            raise EdgeNotFound(u, v)
        self._drop_edge(u, v)
        self._version += 1

    def remove_edges(self, edges: Iterable[tuple[Node, Node]]) -> None:
        for e in list(edges):
            self.remove_edge(e[0], e[1])

    def _drop_edge(self, u: Node, v: Node) -> None:
        del self._succ[u][v]
        if u != v or self._pred is not self._succ:
            del self._pred[v][u]
        self._n_edges -= 1

    def has_edge(self, u: Node, v: Node) -> bool:
        try:
            return v in self._succ[u]
        except (KeyError, TypeError):
            return False

    def edge_attrs(self, u: Node, v: Node) -> dict:
        """The (mutable) attribute dict of edge ``(u, v)``."""
        try:
            return self._succ[u][v]
        except (KeyError, TypeError):
            raise EdgeNotFound(u, v) from None

    def get_edge_data(self, u: Node, v: Node, default: Any = None) -> Any:
        """Edge attribute dict, or *default* when the edge does not exist."""
        try:
            return self._succ[u][v]
        except (KeyError, TypeError):
            return default

    def _iter_edges(self) -> Iterator[tuple[Node, Node, dict]]:
        seen: set = set()
        for u, nbrs in self._succ.items():
            for v, d in nbrs.items():
                if v not in seen:
                    yield u, v, d
            seen.add(u)

    def selfloops(self) -> list[tuple[Node, Node]]:
        """Edges whose two endpoints coincide."""
        return [(u, u) for u, nbrs in self._succ.items() if u in nbrs]

    # ------------------------------------------------------------------ #
    # neighborhoods and degrees
    # ------------------------------------------------------------------ #
    def neighbors(self, node: Node) -> Iterator[Node]:
        """Iterate over the neighbors of *node* (successors for directed graphs)."""
        try:
            return iter(self._succ[node])
        except (KeyError, TypeError):
            raise NodeNotFound(node) from None

    def degree(self, node: Node | None = None, weight: str | None = None) -> Any:
        """Degree of *node*, or a :class:`NodeMap` of all degrees.

        With *weight*, sums that edge attribute (missing values count as 1).
        A self-loop contributes 2 to its node's degree.
        """
        if node is not None:
            if node not in self._node:
                raise NodeNotFound(node)
            return self._degree_of(node, weight)
        return NodeMap(((n, self._degree_of(n, weight)) for n in self._node), name="degree")

    def _degree_of(self, n: Node, weight: str | None) -> Any:
        nbrs = self._succ[n]
        if weight is None:
            return len(nbrs) + (1 if n in nbrs else 0)
        total = sum(d.get(weight, 1) for d in nbrs.values())
        if n in nbrs:
            total += nbrs[n].get(weight, 1)
        return total

    # ------------------------------------------------------------------ #
    # derived graphs
    # ------------------------------------------------------------------ #
    def copy(self):
        """Deep-enough copy: new structure and new attribute dicts (values are shared)."""
        g = self._empty_like()
        g._load(self)
        return g

    def _empty_like(self, cls: type | None = None):
        g = (cls or type(self))()
        g.attrs = dict(self.attrs)
        return g

    def subgraph(self, nodes: Iterable[Node]):
        """Induced subgraph on *nodes* (a new graph; attributes are copied)."""
        keep = [n for n in dict.fromkeys(nodes) if n in self]
        keep_set = set(keep)
        g = self._empty_like()
        g.add_nodes((n, dict(self._node[n])) for n in keep)
        g.add_edges((u, v, dict(d)) for u, v, d in self._iter_edges() if u in keep_set and v in keep_set)
        return g

    def edge_subgraph(self, edges: Iterable[tuple[Node, Node]]):
        """Subgraph made of *edges* and their endpoints."""
        g = self._empty_like()
        picked = []
        for u, v in edges:
            if not self.has_edge(u, v):
                raise EdgeNotFound(u, v)
            g.add_node(u, **self._node[u])
            g.add_node(v, **self._node[v])
            picked.append((u, v, dict(self._succ[u][v])))
        g.add_edges(picked)
        return g

    def to_undirected(self) -> "Graph":
        """Undirected copy (for directed graphs, ``u→v`` and ``v→u`` merge; later attrs win)."""
        g = self._empty_like(Graph)
        g.add_nodes((n, dict(d)) for n, d in self._node.items())
        g.add_edges((u, v, dict(d)) for u, v, d in self._iter_edges())
        return g

    def to_directed(self) -> "DiGraph":
        """Directed copy; every undirected edge becomes a pair of opposite arcs."""
        g = self._empty_like(DiGraph)
        g.add_nodes((n, dict(d)) for n, d in self._node.items())
        for u, v, d in self._iter_edges():
            g.add_edge(u, v, **d)
            if not self.directed:
                g.add_edge(v, u, **d)
        return g

    def relabel(self, mapping: Mapping[Node, Node] | Callable[[Node], Node]):
        """Copy with nodes renamed by *mapping* (dict or function); unmapped nodes keep their name."""
        f = mapping if callable(mapping) else (lambda n: mapping.get(n, n))  # type: ignore[union-attr]
        new_names = [f(n) for n in self._node]
        if len(set(new_names)) != len(new_names):
            raise ValueError("relabel mapping is not one-to-one")
        g = self._empty_like()
        g.add_nodes((f(n), dict(d)) for n, d in self._node.items())
        g.add_edges((f(u), f(v), dict(d)) for u, v, d in self._iter_edges())
        return g

    def compose(self, other: "Graph"):
        """Union of nodes and edges of both graphs (*other*'s attributes win on overlap)."""
        if other.directed != self.directed:
            raise GraphTypeError("cannot compose a directed graph with an undirected one")
        g = self.copy()
        for n, d in other._node.items():
            g.add_node(n, **d)
        for u, v, d in other._iter_edges():
            g.add_edge(u, v, **d)
        return g

    def clear(self) -> None:
        """Remove every node and edge (graph attributes are kept)."""
        self._node.clear()
        self._succ.clear()
        self._pred.clear()
        self._n_edges = 0
        self._version += 1

    # ------------------------------------------------------------------ #
    # numeric bridge
    # ------------------------------------------------------------------ #
    def node_index(self) -> dict[Node, int]:
        """``{node: position}`` in iteration order, which is the row order of every matrix AryaGraph builds."""
        return {n: i for i, n in enumerate(self._node)}

    def nodelist(self) -> list[Node]:
        return list(self._node)

    # ------------------------------------------------------------------ #
    # fluent shortcuts into the other subpackages
    # ------------------------------------------------------------------ #
    def draw(self, **kwargs: Any):
        """Render this graph; see :func:`aryagraph.draw` for every option."""
        from ..render import draw

        return draw(self, **kwargs)

    def layout(self, method: str = "auto", **kwargs: Any):
        """Compute a layout; see :func:`aryagraph.layout.compute`."""
        from ..layout import compute

        return compute(self, method, **kwargs)

    def analyze(self, **kwargs: Any):
        """Full analytical report; see :func:`aryagraph.analyze`."""
        from ..analysis import analyze

        return analyze(self, **kwargs)


class DiGraph(Graph):
    """Directed simple graph (self-loops allowed, at most one arc per ordered pair)."""

    directed = True

    def _iter_edges(self) -> Iterator[tuple[Node, Node, dict]]:
        for u, nbrs in self._succ.items():
            for v, d in nbrs.items():
                yield u, v, d

    def successors(self, node: Node) -> Iterator[Node]:
        """Iterate over the heads of arcs leaving *node*."""
        try:
            return iter(self._succ[node])
        except (KeyError, TypeError):
            raise NodeNotFound(node) from None

    def predecessors(self, node: Node) -> Iterator[Node]:
        """Iterate over the tails of arcs entering *node*."""
        try:
            return iter(self._pred[node])
        except (KeyError, TypeError):
            raise NodeNotFound(node) from None

    def neighbors(self, node: Node) -> Iterator[Node]:
        """Successors of *node* (use :meth:`all_neighbors` for both directions)."""
        return self.successors(node)

    def all_neighbors(self, node: Node) -> Iterator[Node]:
        """Predecessors and successors of *node*, each listed once."""
        if node not in self._node:
            raise NodeNotFound(node)
        return iter(dict.fromkeys([*self._pred[node], *self._succ[node]]))

    def in_edges(self, node: Node) -> list[tuple[Node, Node]]:
        return [(u, node) for u in self.predecessors(node)]

    def out_edges(self, node: Node) -> list[tuple[Node, Node]]:
        return [(node, v) for v in self.successors(node)]

    def _degree_of(self, n: Node, weight: str | None) -> Any:
        return self._in_degree_of(n, weight) + self._out_degree_of(n, weight)

    def _in_degree_of(self, n: Node, weight: str | None) -> Any:
        if weight is None:
            return len(self._pred[n])
        return sum(d.get(weight, 1) for d in self._pred[n].values())

    def _out_degree_of(self, n: Node, weight: str | None) -> Any:
        if weight is None:
            return len(self._succ[n])
        return sum(d.get(weight, 1) for d in self._succ[n].values())

    def in_degree(self, node: Node | None = None, weight: str | None = None) -> Any:
        """In-degree of *node*, or a :class:`NodeMap` of all in-degrees."""
        if node is not None:
            if node not in self._node:
                raise NodeNotFound(node)
            return self._in_degree_of(node, weight)
        return NodeMap(((n, self._in_degree_of(n, weight)) for n in self._node), name="in_degree")

    def out_degree(self, node: Node | None = None, weight: str | None = None) -> Any:
        """Out-degree of *node*, or a :class:`NodeMap` of all out-degrees."""
        if node is not None:
            if node not in self._node:
                raise NodeNotFound(node)
            return self._out_degree_of(node, weight)
        return NodeMap(((n, self._out_degree_of(n, weight)) for n in self._node), name="out_degree")

    def reverse(self) -> "DiGraph":
        """Copy with every arc reversed."""
        g = self._empty_like()
        g.add_nodes((n, dict(d)) for n, d in self._node.items())
        g.add_edges((v, u, dict(d)) for u, v, d in self._iter_edges())
        return g

    def to_undirected(self, reciprocal: bool = False) -> Graph:
        """Undirected copy; with *reciprocal*, keep only pairs linked in both directions."""
        if not reciprocal:
            return super().to_undirected()
        g = self._empty_like(Graph)
        g.add_nodes((n, dict(d)) for n, d in self._node.items())
        g.add_edges((u, v, dict(d)) for u, v, d in self._iter_edges() if u in self._succ[v])
        return g

    def is_dag(self) -> bool:
        """True when the graph has no directed cycle."""
        from .dag import _kahn

        return _kahn(self)[1] is None

    def to_dag(self) -> "DAG":
        """Copy as a :class:`~aryagraph.DAG` (raises :class:`CycleError` if a cycle exists)."""
        from .dag import DAG

        return DAG(self)


def _is_networkx_graph(obj: Any) -> bool:
    """Duck-typed check so ``Graph(nx_graph)`` works without importing networkx."""
    return (
        not isinstance(obj, Graph)
        and all(hasattr(obj, a) for a in ("adj", "nodes", "edges", "is_directed", "is_multigraph"))
        and callable(getattr(obj, "is_multigraph", None))
    )


def _unpack_edge(e: Any) -> tuple[Node, Node, dict]:
    try:
        n = len(e)
    except TypeError:
        raise TypeError(f"edge must be a tuple like (u, v), got {e!r}") from None
    if n == 2:
        return e[0], e[1], {}
    if n == 3:
        extra = e[2]
        if isinstance(extra, Mapping):
            return e[0], e[1], dict(extra)
        if isinstance(extra, Real) and not isinstance(extra, bool):
            return e[0], e[1], {"weight": extra}
        raise TypeError(f"third edge element must be an attribute dict or a number, got {extra!r}")
    raise TypeError(f"edge must have 2 or 3 elements, got {n}: {e!r}")


__all__ = ["Graph", "DiGraph", "Node"]
