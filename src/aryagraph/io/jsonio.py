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

"""JSON node-link format.

AryaGraph writes::

    {"directed": false, "dag": false, "multigraph": false,
     "graph": {...graph attributes...},
     "nodes": [{"id": 0, ...node attributes...}, ...],
     "edges": [{"source": 0, "target": 1, ...edge attributes...}, ...]}

which networkx (``node_link_graph``) and d3 read directly. On input the
networkx and d3 variants are accepted too: ``links`` instead of ``edges``,
``multigraph: true`` (parallel edges are merged, later attributes win, and
the ``key`` field is dropped), and d3's older index form where nodes have no
``id`` and edge endpoints are positions in the ``nodes`` list.

Values JSON cannot hold are converted on output:

* tuples → arrays; node ids that are tuples come back as tuples, attribute
  values come back as lists;
* sets → sorted arrays; numpy scalars → Python numbers; numpy arrays → nested lists;
* dates and times → ISO-8601 strings; enums → their value; other numbers →
  ``int``/``float``;
* mapping keys → strings; anything else → ``str(value)``.

NaN and ±infinity are written as ``NaN``/``Infinity`` (Python's JSON dialect,
which networkx also uses) so they survive a round trip; strict JSON parsers
reject them.
"""

from __future__ import annotations

import datetime
import json
import numbers
import os
from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np

from ..core.dag import DAG
from ..core.graph import DiGraph, Graph

PathLike = str | os.PathLike


# ---------------------------------------------------------------------- #
# value conversion
# ---------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    """JSON-compatible copy of an attribute value (see the module docstring)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {k if isinstance(k, str) else str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(v) for v in value]
        try:
            return sorted(items)
        except TypeError:  # mixed types: still deterministic
            return sorted(items, key=lambda x: (type(x).__name__, repr(x)))
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return str(value)


def _json_id(node: Any) -> Any:
    """JSON form of a node id; only scalars and (nested) tuples of them round-trip."""
    if isinstance(node, (str, bool, int, float)):
        return node
    if isinstance(node, np.generic):
        return node.item()
    if isinstance(node, tuple):
        return [_json_id(x) for x in node]
    raise TypeError(f"node {node!r} of type {type(node).__name__} cannot be written as a JSON id")


def _hashable(x: Any) -> Any:
    """Inverse of :func:`_json_id`: JSON arrays become tuples."""
    if isinstance(x, list):
        return tuple(_hashable(v) for v in x)
    if isinstance(x, dict):
        raise TypeError(f"a JSON object cannot be a node id: {x!r}")
    return x


# ---------------------------------------------------------------------- #
# dict form
# ---------------------------------------------------------------------- #
def to_dict(g: Graph, *, node_key: str = "id", source: str = "source", target: str = "target") -> dict[str, Any]:
    """Node-link dictionary of *g* (see the module docstring for the layout).

    *node_key*, *source* and *target* rename the reserved fields; a node or
    edge attribute with the same name raises ``ValueError``.

    >>> to_dict(Graph([("a", "b")]))["edges"]
    [{'source': 'a', 'target': 'b'}]
    """
    nodes = []
    for n, attrs in g._node.items():
        if node_key in attrs:
            raise ValueError(f"node {n!r} has an attribute named {node_key!r}, which is the id field; pass node_key=...")
        nodes.append({node_key: _json_id(n), **_jsonable(attrs)})
    edges = []
    for u, v, attrs in g._iter_edges():
        for key in (source, target):
            if key in attrs:
                raise ValueError(f"edge ({u!r}, {v!r}) has an attribute named {key!r}; pass source=/target=")
        edges.append({source: _json_id(u), target: _json_id(v), **_jsonable(attrs)})
    return {
        "directed": g.directed,
        "dag": isinstance(g, DAG),
        "multigraph": False,
        "graph": _jsonable(g.attrs),
        "nodes": nodes,
        "edges": edges,
    }


def from_dict(
    data: Mapping[str, Any],
    *,
    node_key: str = "id",
    source: str = "source",
    target: str = "target",
) -> Graph:
    """Graph from a node-link dictionary (AryaGraph, networkx or d3 flavour).

    Returns a :class:`DAG` when ``data["dag"]`` is true (raising
    :class:`CycleError` if the edges contain a cycle), a :class:`DiGraph` when
    ``data["directed"]`` is true, else a :class:`Graph`.
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"node-link data must be a mapping, got {type(data).__name__}")
    dag = bool(data.get("dag", False))
    directed = dag or bool(data.get("directed", False))
    multigraph = bool(data.get("multigraph", False))
    graph_attrs = data.get("graph", {}) or {}
    if isinstance(graph_attrs, list):  # networkx 1.x stored [[key, value], ...]
        graph_attrs = dict(graph_attrs)
    cls = DAG if dag else DiGraph if directed else Graph
    g = cls()
    g.attrs.update((str(k), v) for k, v in graph_attrs.items())

    raw_nodes = data.get("nodes", []) or []
    by_index = bool(raw_nodes) and not any(isinstance(d, Mapping) and node_key in d for d in raw_nodes)
    index_ids: list[Any] = []
    for i, d in enumerate(raw_nodes):
        if not isinstance(d, Mapping):
            raise ValueError(f"nodes[{i}] must be an object, got {d!r}")
        attrs = dict(d)
        if by_index:
            node = i
        elif node_key in attrs:
            node = _hashable(attrs.pop(node_key))
        else:
            raise ValueError(f"nodes[{i}] has no {node_key!r} field")
        g.add_node(node, **attrs)
        index_ids.append(node)

    edges_field = "edges" if "edges" in data else "links"
    batch = []
    for i, e in enumerate(data.get(edges_field, []) or []):
        if not isinstance(e, Mapping):
            raise ValueError(f"{edges_field}[{i}] must be an object, got {e!r}")
        attrs = dict(e)
        try:
            ends = [attrs.pop(source), attrs.pop(target)]
        except KeyError as exc:
            raise ValueError(f"{edges_field}[{i}] has no {exc.args[0]!r} field") from None
        for j, end in enumerate(ends):
            if isinstance(end, Mapping):  # d3 replaces endpoints by node objects after a simulation
                end = end.get(node_key, end.get("index"))
            if by_index:
                if not isinstance(end, int) or not 0 <= end < len(index_ids):
                    raise ValueError(f"{edges_field}[{i}]: endpoint {end!r} is not a valid node index")
                end = index_ids[end]
            ends[j] = _hashable(end)
        if multigraph:
            attrs.pop("key", None)
        batch.append((ends[0], ends[1], attrs))
    g.add_edges(batch)
    return g


# ---------------------------------------------------------------------- #
# text and files
# ---------------------------------------------------------------------- #
def to_json(g: Graph, *, indent: int | None = 2) -> str:
    """Node-link JSON text of *g* (UTF-8 friendly: non-ASCII characters are kept as is)."""
    return json.dumps(to_dict(g), ensure_ascii=False, indent=indent)


def from_json(text: str) -> Graph:
    """Graph from node-link JSON text; see :func:`from_dict`."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from None
    return from_dict(data)


def write_json(g: Graph, path: PathLike, indent: int | None = 2) -> None:
    """Write *g* as node-link JSON (UTF-8)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(to_json(g, indent=indent))
        fh.write("\n")


def read_json(path: PathLike) -> Graph:
    """Read a node-link JSON file written by AryaGraph, networkx or d3."""
    with open(path, encoding="utf-8-sig") as fh:
        return from_json(fh.read())


__all__ = ["to_dict", "from_dict", "to_json", "from_json", "write_json", "read_json"]
