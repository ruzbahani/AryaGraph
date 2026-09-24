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

"""Read-only views over a graph's nodes and edges.

Views are live: they reflect later mutations of the graph. Attribute dicts they
hand out are the graph's own, so ``g.nodes['a']['color'] = 'red'`` edits the
graph in place (the structure itself can only change through graph methods).
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import TYPE_CHECKING, Any, Hashable, Iterator

from .exceptions import EdgeNotFound, NodeNotFound

if TYPE_CHECKING:
    from .graph import Graph

_MISSING = object()


def _preview(items: list, limit: int = 8) -> str:
    shown = ", ".join(repr(x) for x in items[:limit])
    return shown + (", …" if len(items) > limit else "")


class NodeView(Mapping):
    """``g.nodes``: iterate nodes, look up attributes, test membership.

    >>> g.nodes['a']                 # attribute dict of node 'a'
    >>> list(g.nodes.data('color'))  # [(node, color-or-None), ...]
    """

    __slots__ = ("_graph",)

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def __getitem__(self, node: Hashable) -> dict:
        try:
            return self._graph._node[node]
        except KeyError:
            raise NodeNotFound(node) from None
        except TypeError:  # unhashable key
            raise NodeNotFound(node) from None

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._graph._node)

    def __len__(self) -> int:
        return len(self._graph._node)

    def __contains__(self, node: object) -> bool:
        try:
            return node in self._graph._node
        except TypeError:
            return False

    def data(self, key: str | None = None, default: Any = None) -> Iterator[tuple[Hashable, Any]]:
        """Yield ``(node, attrs)``, or ``(node, attrs.get(key, default))`` when *key* is given."""
        if key is None:
            yield from self._graph._node.items()
        else:
            for n, d in self._graph._node.items():
                yield n, d.get(key, default)

    def __call__(self, data: bool | str = False, default: Any = None):
        """networkx-style call: ``g.nodes(data=True)`` / ``g.nodes(data='color')``."""
        if data is False:
            return list(self)
        if data is True:
            return list(self.data())
        return list(self.data(data, default))

    def __repr__(self) -> str:
        return f"NodeView([{_preview(list(self))}])"


class EdgeView(Set):
    """``g.edges``: iterate edges as ``(u, v)``, look up their attributes.

    For undirected graphs every edge is reported once and ``(u, v) in g.edges``
    is true in both orientations.
    """

    __slots__ = ("_graph",)

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def __iter__(self) -> Iterator[tuple[Hashable, Hashable]]:
        for u, v, _ in self._graph._iter_edges():
            yield u, v

    def __len__(self) -> int:
        return self._graph._n_edges

    def __contains__(self, edge: object) -> bool:
        try:
            u, v = edge  # type: ignore[misc]
        except (TypeError, ValueError):
            return False
        return self._graph.has_edge(u, v)

    def __getitem__(self, edge: tuple[Hashable, Hashable]) -> dict:
        u, v = edge
        try:
            return self._graph._succ[u][v]
        except (KeyError, TypeError):
            raise EdgeNotFound(u, v) from None

    def data(self, key: str | None = None, default: Any = None) -> Iterator[tuple]:
        """Yield ``(u, v, attrs)``, or ``(u, v, attrs.get(key, default))`` when *key* is given."""
        if key is None:
            yield from self._graph._iter_edges()
        else:
            for u, v, d in self._graph._iter_edges():
                yield u, v, d.get(key, default)

    def __call__(self, data: bool | str = False, default: Any = None):
        """networkx-style call: ``g.edges(data=True)`` / ``g.edges(data='weight')``."""
        if data is False:
            return list(self)
        if data is True:
            return list(self.data())
        return list(self.data(data, default))

    @classmethod
    def _from_iterable(cls, it):  # Set mixin operations (&, |, -) return plain sets
        return set(it)

    def __repr__(self) -> str:
        return f"EdgeView([{_preview(list(self))}])"


class AdjacencyView(Mapping):
    """Read-only ``node -> {neighbor: attrs}`` mapping (``g.adj``, ``g.succ``, ``g.pred``).

    The inner neighbor dicts are the graph's own storage and must be treated as
    read-only; this keeps neighbor iteration in algorithms allocation-free.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, node: Hashable) -> Mapping[Hashable, dict]:
        try:
            return self._data[node]
        except (KeyError, TypeError):
            raise NodeNotFound(node) from None

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, node: object) -> bool:
        try:
            return node in self._data
        except TypeError:
            return False

    def __repr__(self) -> str:
        return f"AdjacencyView({len(self._data)} nodes)"


__all__ = ["NodeView", "EdgeView", "AdjacencyView"]
