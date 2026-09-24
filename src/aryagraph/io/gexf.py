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

"""GEXF 1.3 (``.gexf``), Gephi's native format.

The writer declares every node and edge attribute (``integer``/``long``,
``double``, ``boolean`` or ``string``, chosen as in :mod:`aryagraph.io.graphml`)
and maps a few attributes onto GEXF's own fields:

* node ``label`` → the node's ``label`` (defaults to the id text);
* node ``pos`` → ``<viz:position>``, with y negated because Gephi's y axis
  points up while AryaGraph's screen coordinates point down;
* node ``size`` (a number) → ``<viz:size>``; node ``color`` (``#rrggbb`` or
  ``#rrggbbaa``) → ``<viz:color>``;
* edge ``weight`` (a number) and ``label`` → the edge's XML attributes.

The graph name is stored as the meta ``<description>``; other graph
attributes have no place in GEXF and are not written. Non-string node ids and
DAGs are marked with ``aryagraph:`` attributes exactly as in GraphML.

The reader handles GEXF 1.1–1.3 static graphs (dynamic spells are ignored,
nested nodes flattened, parallel edges merged). ``weight`` is read as an
``int`` when written without a decimal point, else as a ``float``.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any

from ..core.dag import DAG
from ..core.graph import DiGraph, Graph
from .graphml import (
    ARYAGRAPH_NS,
    XSI_NS,
    _collect,
    _decode_id,
    _id_table,
    _infer_type,
    _local,
    _parse_value,
    _scalar,
    _value_text,
    _xml_text,
)

PathLike = str | os.PathLike

GEXF_NS = "http://gexf.net/1.3"
VIZ_NS = "http://gexf.net/1.3/viz"
_SCHEMA = "http://gexf.net/1.3 http://gexf.net/1.3/gexf.xsd"
_HEX = re.compile(r"#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})?\Z")
_INT_TEXT = re.compile(r"[+-]?[0-9]+\Z")


def _is_number(v: Any) -> bool:
    v = _scalar(v)
    return isinstance(v, Real) and not isinstance(v, bool)


def _num_text(v: Any) -> str:
    v = _scalar(v)
    return repr(float(v)) if isinstance(v, float) else str(v)


def _position(value: Any) -> tuple[float, ...] | None:
    try:
        coords = tuple(float(_scalar(c)) for c in value)
    except (TypeError, ValueError):
        return None
    return coords if len(coords) in (2, 3) else None


# ---------------------------------------------------------------------- #
# writing
# ---------------------------------------------------------------------- #
def _gexf_element(g: Graph) -> ET.Element:
    root = ET.Element(
        "gexf",
        {
            "xmlns": GEXF_NS,
            "xmlns:viz": VIZ_NS,
            "xmlns:xsi": XSI_NS,
            "xmlns:aryagraph": ARYAGRAPH_NS,
            "xsi:schemaLocation": _SCHEMA,
            "version": "1.3",
        },
    )
    meta = ET.SubElement(root, "meta")
    ET.SubElement(meta, "creator").text = "AryaGraph"
    if g.name:
        ET.SubElement(meta, "description").text = _xml_text(str(g.name), "graph name")
    graph = ET.SubElement(root, "graph", {"defaultedgetype": "directed" if g.directed else "undirected", "mode": "static"})
    if isinstance(g, DAG):
        graph.set("aryagraph:dag", "true")

    def node_special(name: str, value: Any) -> bool:
        if name == "label":
            return True
        if name == "pos":
            return _position(value) is not None
        if name == "size":
            return _is_number(value)
        if name == "color":
            return isinstance(value, str) and _HEX.match(value) is not None
        return False

    def edge_special(name: str, value: Any) -> bool:
        return name == "label" or (name == "weight" and _is_number(value))

    node_attrs = _collect({k: v for k, v in d.items() if not node_special(k, v)} for d in g._node.values())
    edge_attrs = _collect({k: v for k, v in d.items() if not edge_special(k, v)} for _, _, d in g._iter_edges())
    decls: dict[tuple[str, str], tuple[str, str]] = {}
    for cls, attrs in (("node", node_attrs), ("edge", edge_attrs)):
        if not attrs:
            continue
        block = ET.SubElement(graph, "attributes", {"class": cls, "mode": "static"})
        for i, (name, values) in enumerate(attrs.items()):
            xml_type = _infer_type(values, int32="integer")
            decls[cls, name] = (str(i), xml_type)
            ET.SubElement(block, "attribute", {"id": str(i), "title": _xml_text(str(name), "attribute name"), "type": xml_type})

    def attvalues(parent: ET.Element, cls: str, attrs: Mapping[str, Any], special: Callable[[str, Any], bool]) -> None:
        items = [(k, v) for k, v in attrs.items() if v is not None and not special(k, v)]
        if not items:
            return
        block = ET.SubElement(parent, "attvalues")
        for name, value in items:
            att_id, xml_type = decls[cls, name]
            ET.SubElement(block, "attvalue", {"for": att_id, "value": _value_text(value, xml_type, f"attribute {name!r}")})

    ids = _id_table(g)
    nodes = ET.SubElement(graph, "nodes")
    for n, attrs in g._node.items():
        text, hint = ids[n]
        label = attrs.get("label")
        el = ET.SubElement(nodes, "node", {"id": text, "label": _value_text(label, "string", "label") if label is not None else text})
        if hint:
            el.set("aryagraph:type", hint)
        attvalues(el, "node", attrs, node_special)
        if "color" in attrs and node_special("color", attrs["color"]):
            r, gr, b, a = _HEX.match(attrs["color"]).groups()  # type: ignore[union-attr]
            color = {"r": str(int(r, 16)), "g": str(int(gr, 16)), "b": str(int(b, 16))}
            if a is not None:
                color["a"] = repr(round(int(a, 16) / 255, 4))
            ET.SubElement(el, "viz:color", color)
        pos = _position(attrs["pos"]) if "pos" in attrs else None
        if pos is not None:
            z = pos[2] if len(pos) == 3 else 0.0
            ET.SubElement(el, "viz:position", {"x": repr(pos[0]), "y": repr(-pos[1] + 0.0), "z": repr(z)})
        if "size" in attrs and _is_number(attrs["size"]):
            ET.SubElement(el, "viz:size", {"value": _num_text(attrs["size"])})
    edges = ET.SubElement(graph, "edges")
    for i, (u, v, attrs) in enumerate(g._iter_edges()):
        el = ET.SubElement(edges, "edge", {"id": str(i), "source": ids[u][0], "target": ids[v][0]})
        if "weight" in attrs and _is_number(attrs["weight"]):
            el.set("weight", _num_text(attrs["weight"]))
        if attrs.get("label") is not None:
            el.set("label", _value_text(attrs["label"], "string", "label"))
        attvalues(el, "edge", attrs, edge_special)
    ET.indent(root, space="  ")
    return root


def write_gexf(g: Graph, path: PathLike) -> None:
    """Write *g* as a GEXF 1.3 file (UTF-8) for Gephi; see the module docstring for the mapping."""
    text = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(_gexf_element(g), encoding="unicode") + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# ---------------------------------------------------------------------- #
# reading
# ---------------------------------------------------------------------- #
def read_gexf(path: PathLike, *, node_type: Callable[[str], Any] | None = None, dag: bool | None = None) -> Graph:
    """Read a static GEXF graph (versions 1.1–1.3).

    *node_type* and *dag* work as in :func:`~aryagraph.io.read_graphml`. Node
    labels different from the id become the ``label`` attribute; viz
    position/size/color become ``pos`` (y negated), ``size`` and ``color``.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid GEXF XML in {os.fspath(path)!r}: {exc}") from None
    if _local(root.tag) != "gexf":
        raise ValueError(f"not a GEXF document: the root element is <{_local(root.tag)}>")
    graph_el = next((el for el in root if _local(el.tag) == "graph"), None)
    if graph_el is None:
        raise ValueError("GEXF document contains no <graph>")
    edge_default = graph_el.get("defaultedgetype", "undirected")
    directed = edge_default in ("directed", "mutual", "mixed")
    make_dag = dag if dag is not None else graph_el.get(f"{{{ARYAGRAPH_NS}}}dag") == "true"
    if make_dag and not directed:
        raise ValueError("dag=True needs a directed graph (defaultedgetype='directed')")
    g: Graph = DiGraph() if directed else Graph()
    for meta in root:
        if _local(meta.tag) == "meta":
            for child in meta:
                if _local(child.tag) == "description" and child.text:
                    g.name = child.text

    decls: dict[tuple[str, str], tuple[str, str, Any]] = {}  # (class, id) -> (title, type, default)
    for block in graph_el:
        if _local(block.tag) != "attributes":
            continue
        cls = block.get("class", "node")
        for att in block:
            if _local(att.tag) != "attribute":
                continue
            att_id, title, xml_type = att.get("id"), att.get("title"), att.get("type", "string")
            if att_id is None:
                raise ValueError("<attribute> without an id")
            default = None
            for child in att:
                if _local(child.tag) == "default":
                    default = _parse_value(child.text or "", xml_type, f"default of attribute {title!r}")
            decls[cls, att_id] = (title or att_id, xml_type, default)

    def values_of(el: ET.Element, cls: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for block in el:
            if _local(block.tag) != "attvalues":
                continue
            for av in block:
                att_id = av.get("for", av.get("id"))
                if (cls, att_id) not in decls:
                    raise ValueError(f"<attvalue> refers to the undeclared {cls} attribute {att_id!r}")
                title, xml_type, _ = decls[cls, att_id]
                out[title] = _parse_value(av.get("value", ""), xml_type, f"attribute {title!r}")
        for (c, _), (title, _, default) in decls.items():
            if c == cls and default is not None and title not in out:
                out[title] = default
        return out

    ids: dict[str, Any] = {}

    def node_of(text: str, hint: str | None = None) -> Any:
        if text not in ids:
            ids[text] = node_type(text) if node_type is not None else _decode_id(text, hint)
        return ids[text]

    def read_nodes(container: ET.Element) -> None:
        for el in container:
            if _local(el.tag) != "node":
                continue
            text = el.get("id")
            if text is None:
                raise ValueError("<node> without an id")
            attrs = values_of(el, "node")
            label = el.get("label")
            if label is not None and label != text:
                attrs["label"] = label
            for viz in el:
                tag = _local(viz.tag)
                if tag == "position":
                    x, y, z = (float(viz.get(k, 0.0)) for k in "xyz")
                    attrs["pos"] = (x, -y + 0.0) if z == 0 else (x, -y + 0.0, z)
                elif tag == "size" and viz.get("value") is not None:
                    attrs["size"] = _parse_value(viz.get("value", ""), "double", "viz:size")
                elif tag == "color":
                    rgb = "".join(f"{int(float(viz.get(k, 0))):02x}" for k in "rgb")
                    alpha = viz.get("a")
                    attrs["color"] = "#" + rgb + ("" if alpha is None or float(alpha) >= 1 else f"{round(float(alpha) * 255):02x}")
            g.add_node(node_of(text, el.get(f"{{{ARYAGRAPH_NS}}}type")), **attrs)
            for sub in el:
                if _local(sub.tag) == "nodes":
                    read_nodes(sub)

    edges: list[tuple[Any, Any, dict]] = []
    for part in graph_el:
        tag = _local(part.tag)
        if tag == "nodes":
            read_nodes(part)
        elif tag == "edges":
            for el in part:
                if _local(el.tag) != "edge":
                    continue
                s, t = el.get("source"), el.get("target")
                if s is None or t is None:
                    raise ValueError("<edge> needs both a source and a target")
                attrs = values_of(el, "edge")
                weight = el.get("weight")
                if weight is not None:
                    attrs["weight"] = int(weight) if _INT_TEXT.match(weight.strip()) else float(weight)
                if el.get("label") is not None:
                    attrs["label"] = el.get("label")
                u, v = node_of(s), node_of(t)
                edges.append((u, v, attrs))
                if directed and el.get("type", edge_default) in ("mutual", "undirected"):
                    edges.append((v, u, dict(attrs)))
    for u, v, attrs in edges:
        g.add_edge(u, v, **attrs)
    return DAG(g) if make_dag else g


__all__ = ["write_gexf", "read_gexf"]
