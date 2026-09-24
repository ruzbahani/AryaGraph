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

"""GraphML (``.graphml``), the XML format of yEd, Gephi, Cytoscape and networkx.

Attributes are declared with typed ``<key>`` elements. The writer picks the
narrowest GraphML type holding every value of an attribute: ``boolean``,
``int`` (``long`` beyond 32 bits), ``double`` (ints and floats mixed), else
``string``. In a ``string`` attribute non-string values are written as text
(containers such as tuples, lists, dicts and sets as JSON) and are read back
as strings.

GraphML ids are strings. So that other node types survive a round trip, the
writer marks each non-string id with ``aryagraph:type="int"`` (or ``float``,
``bool``, ``tuple``) in the ``urn:aryagraph`` XML namespace, and a DAG with
``aryagraph:dag="true"`` on ``<graph>``; other tools ignore these attributes,
and files without them read exactly as networkx reads them (string ids).

The reader accepts any GraphML: key defaults are applied, nested graphs are
flattened, yFiles graphics keys (no ``attr.name``) are skipped, parallel
edges are merged (later attributes win), and hyperedges are rejected.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from ..core.dag import DAG
from ..core.graph import DiGraph, Graph
from .jsonio import _hashable, _json_id, _jsonable

PathLike = str | os.PathLike

GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ARYAGRAPH_NS = "urn:aryagraph"
_SCHEMA = f"{GRAPHML_NS} http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
_INVALID_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿\ud800-\udfff]")
_INT32 = (-(2**31), 2**31 - 1)


# ---------------------------------------------------------------------- #
# helpers shared with gexf.py
# ---------------------------------------------------------------------- #
def _local(tag: Any) -> str:
    """Tag or attribute name without its ``{namespace}``."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _xml_text(s: str, what: str) -> str:
    if _INVALID_XML.search(s):
        raise ValueError(f"{what} {s!r} contains characters that XML 1.0 cannot represent")
    return s


def _encode_id(node: Any) -> tuple[str, str | None]:
    """``(text, type_hint)`` for a node id; the hint is ``None`` for strings."""
    if isinstance(node, np.generic):
        node = node.item()
    if isinstance(node, str):
        return _xml_text(node, "node"), None
    if isinstance(node, bool):
        return ("true" if node else "false"), "bool"
    if isinstance(node, int):
        return str(node), "int"
    if isinstance(node, float):
        return repr(node), "float"
    if isinstance(node, tuple):
        return _xml_text(json.dumps(_json_id(node), ensure_ascii=False), "node"), "tuple"
    raise TypeError(f"node {node!r} of type {type(node).__name__} cannot be written as an XML id")


def _decode_id(text: str, hint: str | None) -> Any:
    if not hint:
        return text
    try:
        if hint == "int":
            return int(text)
        if hint == "float":
            return float(text)
        if hint == "bool":
            return text == "true"
        if hint == "tuple":
            return _hashable(json.loads(text))
    except ValueError:
        raise ValueError(f"node id {text!r} does not match its aryagraph:type {hint!r}") from None
    raise ValueError(f"unknown aryagraph:type {hint!r} on node {text!r}")


def _id_table(g: Graph) -> dict[Any, tuple[str, str | None]]:
    """Encoded ids of every node, checking that distinct nodes get distinct texts."""
    table: dict[Any, tuple[str, str | None]] = {}
    owner: dict[str, Any] = {}
    for n in g._node:
        text, hint = _encode_id(n)
        if text in owner:
            raise ValueError(f"nodes {owner[text]!r} and {n!r} would both be written with the id {text!r}")
        owner[text] = n
        table[n] = (text, hint)
    return table


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _infer_type(values: Iterable[Any], *, int32: str = "int") -> str:
    """Narrowest XML schema type for *values*: boolean, int/long, double or string."""
    kinds: set[str] = set()
    big = False
    for v in values:
        v = _scalar(v)
        if isinstance(v, bool):
            kinds.add("boolean")
        elif isinstance(v, int):
            kinds.add("int")
            big = big or not _INT32[0] <= v <= _INT32[1]
        elif isinstance(v, float):
            kinds.add("double")
        else:
            kinds.add("string")
    if kinds == {"boolean"}:
        return "boolean"
    if kinds == {"int"}:
        return "long" if big else int32
    if kinds <= {"int", "double"} and kinds:
        return "double"
    return "string"


def _value_text(value: Any, xml_type: str, what: str) -> str:
    value = _scalar(value)
    if xml_type == "boolean":
        return "true" if value else "false"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (int, str)):
        return _xml_text(str(value), what)
    return _xml_text(json.dumps(_jsonable(value), ensure_ascii=False), what)


_BOOL_TEXT = {"true": True, "false": False, "1": True, "0": False}


def _parse_value(text: str, xml_type: str, what: str) -> Any:
    t = xml_type.lower()
    try:
        if t == "boolean":
            return _BOOL_TEXT[text.strip().lower()]
        if t in ("int", "long", "integer", "short", "byte"):
            return int(text.strip())
        if t in ("float", "double"):
            return float(text.strip())
    except (KeyError, ValueError):
        raise ValueError(f"{what}: {text!r} is not a valid {xml_type}") from None
    return text


def _collect(items: Iterable[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """``{attribute: [values…]}`` over *items*, in first-seen order, skipping ``None``."""
    out: dict[str, list[Any]] = {}
    for attrs in items:
        for k, v in attrs.items():
            if v is not None:
                out.setdefault(k, []).append(v)
    return out


# ---------------------------------------------------------------------- #
# writing
# ---------------------------------------------------------------------- #
def _graphml_element(g: Graph) -> ET.Element:
    root = ET.Element(
        "graphml",
        {"xmlns": GRAPHML_NS, "xmlns:xsi": XSI_NS, "xmlns:aryagraph": ARYAGRAPH_NS, "xsi:schemaLocation": _SCHEMA},
    )
    domains = {
        "graph": _collect([g.attrs]),
        "node": _collect(g._node.values()),
        "edge": _collect(d for _, _, d in g._iter_edges()),
    }
    keys: dict[tuple[str, str], tuple[str, str]] = {}
    for domain, attrs in domains.items():
        for name, values in attrs.items():
            key_id = f"d{len(keys)}"
            xml_type = _infer_type(values)
            keys[domain, name] = (key_id, xml_type)
            ET.SubElement(
                root,
                "key",
                {"id": key_id, "for": domain, "attr.name": _xml_text(str(name), "attribute name"), "attr.type": xml_type},
            )

    def add_data(parent: ET.Element, domain: str, attrs: Mapping[str, Any]) -> None:
        for name, value in attrs.items():
            if value is None:
                continue
            key_id, xml_type = keys[domain, name]
            ET.SubElement(parent, "data", {"key": key_id}).text = _value_text(value, xml_type, f"attribute {name!r}")

    graph = ET.SubElement(root, "graph", {"id": "G", "edgedefault": "directed" if g.directed else "undirected"})
    if isinstance(g, DAG):
        graph.set("aryagraph:dag", "true")
    add_data(graph, "graph", g.attrs)
    ids = _id_table(g)
    for n, attrs in g._node.items():
        text, hint = ids[n]
        el = ET.SubElement(graph, "node", {"id": text})
        if hint:
            el.set("aryagraph:type", hint)
        add_data(el, "node", attrs)
    for u, v, attrs in g._iter_edges():
        el = ET.SubElement(graph, "edge", {"source": ids[u][0], "target": ids[v][0]})
        add_data(el, "edge", attrs)
    ET.indent(root, space="  ")
    return root


def to_graphml(g: Graph) -> str:
    """GraphML document for *g* as a string (see the module docstring)."""
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(_graphml_element(g), encoding="unicode") + "\n"


def write_graphml(g: Graph, path: PathLike) -> None:
    """Write *g* as a UTF-8 GraphML file."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(to_graphml(g))


# ---------------------------------------------------------------------- #
# reading
# ---------------------------------------------------------------------- #
def from_graphml(text: str, *, node_type: Callable[[str], Any] | None = None, dag: bool | None = None) -> Graph:
    """Graph from a GraphML document string; see :func:`read_graphml`."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"invalid GraphML XML: {exc}") from None
    return _read_root(root, node_type, dag)


def read_graphml(path: PathLike, *, node_type: Callable[[str], Any] | None = None, dag: bool | None = None) -> Graph:
    """Read the first graph of a GraphML file.

    Parameters
    ----------
    node_type:
        Convert every node id with this callable (e.g. ``int`` for files
        written by other tools); by default ids are strings unless the file
        carries AryaGraph's ``aryagraph:type`` hints.
    dag:
        ``None`` follows the file (a DAG if AryaGraph wrote one), ``True`` forces
        a :class:`DAG` (``CycleError`` on a cycle, ``ValueError`` for an
        undirected file), ``False`` a plain graph.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid GraphML XML in {os.fspath(path)!r}: {exc}") from None
    return _read_root(root, node_type, dag)


def _read_root(root: ET.Element, node_type: Callable[[str], Any] | None, dag: bool | None) -> Graph:
    if _local(root.tag) != "graphml":
        raise ValueError(f"not a GraphML document: the root element is <{_local(root.tag)}>")
    keys: dict[str, tuple[str, str, str, Any]] = {}  # id -> (domain, name, type, default)
    skipped: set[str | None] = set()
    for el in root:
        if _local(el.tag) != "key":
            continue
        key_id = el.get("id")
        name = el.get("attr.name")
        if key_id is None or name is None:  # yFiles graphics and similar extensions
            skipped.add(key_id)
            continue
        xml_type = el.get("attr.type", "string")
        domain = el.get("for", "all")
        default = None
        for child in el:
            if _local(child.tag) == "default":
                default = _parse_value(child.text or "", xml_type, f"default of key {key_id!r}")
        keys[key_id] = (domain, name, xml_type, default)
    graph_el = next((el for el in root if _local(el.tag) == "graph"), None)
    if graph_el is None:
        raise ValueError("GraphML document contains no <graph>")
    directed = graph_el.get("edgedefault", "undirected") == "directed"
    make_dag = dag if dag is not None else graph_el.get(f"{{{ARYAGRAPH_NS}}}dag") == "true"
    if make_dag and not directed:
        raise ValueError("dag=True needs a directed graph (edgedefault='directed')")
    g: Graph = DiGraph() if directed else Graph()

    def data_of(el: ET.Element, domain: str) -> dict[str, Any]:
        out = {}
        for d in el:
            if _local(d.tag) != "data":
                continue
            key_id = d.get("key")
            if key_id not in keys:
                if key_id in skipped:
                    continue
                raise ValueError(f"<data> refers to the undeclared key {key_id!r}")
            _, name, xml_type, _ = keys[key_id]
            out[name] = _parse_value(d.text or "", xml_type, f"key {name!r}")
        for _, (dom, name, _, default) in keys.items():
            if default is not None and dom in (domain, "all") and name not in out:
                out[name] = default
        return out

    g.attrs.update(data_of(graph_el, "graph"))
    ids: dict[str, Any] = {}
    edges: list[tuple[Any, Any, dict]] = []

    def node_of(text: str, hint: str | None = None) -> Any:
        if text not in ids:
            ids[text] = node_type(text) if node_type is not None else _decode_id(text, hint)
        return ids[text]

    def walk(graph: ET.Element) -> None:
        for el in graph:
            tag = _local(el.tag)
            if tag == "node":
                text = el.get("id")
                if text is None:
                    raise ValueError("<node> without an id")
                g.add_node(node_of(text, el.get(f"{{{ARYAGRAPH_NS}}}type")), **data_of(el, "node"))
                for sub in el:
                    if _local(sub.tag) == "graph":
                        walk(sub)
            elif tag == "edge":
                s, t = el.get("source"), el.get("target")
                if s is None or t is None:
                    raise ValueError("<edge> needs both a source and a target")
                edges.append((node_of(s), node_of(t), data_of(el, "edge")))
                for sub in el:
                    if _local(sub.tag) == "graph":
                        walk(sub)
            elif tag == "hyperedge":
                raise ValueError("GraphML hyperedges are not supported")

    walk(graph_el)
    for u, v, attrs in edges:
        g.add_edge(u, v, **attrs)
    return DAG(g) if make_dag else g


__all__ = ["write_graphml", "read_graphml", "to_graphml", "from_graphml"]
