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

"""Tests for the XML formats of aryagraph.io: GraphML and GEXF."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import networkx as nx
import pytest

import aryagraph.io as nio
from aryagraph import DAG, CycleError, DiGraph, Graph
from aryagraph.generators import project_plan, ucalgary_campus

GRAPHML = "{http://graphml.graphdrawing.org/xmlns}"


def campus_scalar() -> Graph:
    """The UCalgary campus with scalar attributes only (text and floats).

    The ``pos`` tuples and the ``sources`` list are dropped: GraphML and GEXF
    have no container types, and networkx refuses to write them.
    """
    g = ucalgary_campus()
    for n in g:
        del g.nodes[n]["pos"]
    del g.attrs["sources"]
    return g


def typed(value):
    if isinstance(value, dict):
        return {k: typed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(typed(v) for v in value)
    return (type(value).__name__, value)


def edge_dict(g) -> dict:
    return {(frozenset((u, v)) if not g.directed else (u, v)): typed(d) for u, v, d in g.edges.data()}


def xml_graph(directed: bool = False) -> Graph:
    """Every GraphML/GEXF-typed attribute kind, awkward ids, homogeneous columns."""
    g = DiGraph(name="xml & <friends>") if directed else Graph(name="xml & <friends>")
    g.add_node("گره", color="سبز", size=3, big=2**40, ratio=0.5, flag=True, label="نام")
    g.add_node("a b", color='quote " & <tag>', size=-1, big=-(2**40), ratio=-1e-9, flag=False)
    for n in ("42", 7, -7, 2.5, (1, "x", (2, 3)), "#hash", "</node>", "isolated", True):
        g.add_node(n)
    g.add_edge("گره", "a b", weight=1.5, label="یال", note="", text="multi\nline")
    g.add_edge("a b", "42", weight=2.0)
    g.add_edge("42", 7, weight=-3.25, flag=True)
    g.add_edge(7, -7)
    g.add_edge(-7, 2.5)
    g.add_edge(2.5, (1, "x", (2, 3)))
    g.add_edge((1, "x", (2, 3)), "#hash")
    g.add_edge("#hash", "</node>", flag=False)
    g.add_edge("</node>", True)
    return g


def assert_same(back, g, *, node_attrs=True):
    assert type(back) is type(g)
    assert [typed(n) for n in back] == [typed(n) for n in g]
    assert edge_dict(back) == edge_dict(g)
    if node_attrs:
        assert [typed(d) for _, d in back.nodes.data()] == [typed(d) for _, d in g.nodes.data()]


# ---------------------------------------------------------------------- #
# GraphML
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("directed", [False, True])
def test_graphml_round_trip_is_exact(tmp_path, directed):
    g = xml_graph(directed)
    path = tmp_path / "g.graphml"
    nio.write_graphml(g, path)
    back = nio.read_graphml(path)
    assert_same(back, g)
    assert back.attrs == {"name": "xml & <friends>"}
    assert_same(nio.from_graphml(nio.to_graphml(g)), g)


def test_graphml_key_types_and_markup(tmp_path):
    g = xml_graph()
    g.nodes[7]["mixed"] = 1
    g.nodes[-7]["mixed"] = 2.5
    g.nodes[2.5]["pos"] = (0.5, 1.0)
    root = ET.fromstring(nio.to_graphml(g))
    types = {(k.get("for"), k.get("attr.name")): k.get("attr.type") for k in root.iter(GRAPHML + "key")}
    assert types == {
        ("graph", "name"): "string",
        ("node", "color"): "string",
        ("node", "size"): "int",
        ("node", "big"): "long",
        ("node", "ratio"): "double",
        ("node", "flag"): "boolean",
        ("node", "label"): "string",
        ("node", "mixed"): "double",
        ("node", "pos"): "string",
        ("edge", "weight"): "double",
        ("edge", "label"): "string",
        ("edge", "note"): "string",
        ("edge", "text"): "string",
        ("edge", "flag"): "boolean",
    }
    hints = {n.get("id"): n.get("{urn:aryagraph}type") for n in root.iter(GRAPHML + "node")}
    assert hints["7"] == "int" and hints["2.5"] == "float" and hints["true"] == "bool"
    assert hints['[1, "x", [2, 3]]'] == "tuple" and hints["42"] is None and hints["</node>"] is None
    back = nio.from_graphml(nio.to_graphml(g))
    assert back.nodes[7]["mixed"] == 1.0 and type(back.nodes[7]["mixed"]) is float  # int/float column → double
    assert back.nodes[2.5]["pos"] == "[0.5, 1.0]"  # containers are stored as JSON text


def test_graphml_dag_and_errors(tmp_path):
    plan = project_plan()
    back = nio.from_graphml(nio.to_graphml(plan))
    assert isinstance(back, DAG)
    assert_same(back, plan)
    assert type(nio.from_graphml(nio.to_graphml(plan), dag=False)) is DiGraph
    with pytest.raises(ValueError, match="both be written with the id '1'"):
        nio.to_graphml(Graph([(1, "1")]))
    with pytest.raises(ValueError, match="cannot represent"):
        nio.to_graphml(Graph([("bell\x07", "b")]))
    cyclic = nio.to_graphml(DiGraph([("a", "b"), ("b", "a")]))
    with pytest.raises(CycleError):
        nio.from_graphml(cyclic, dag=True)
    with pytest.raises(ValueError, match="needs a directed graph"):
        nio.from_graphml(nio.to_graphml(Graph([(1, 2)])), dag=True)
    path = tmp_path / "u.gexf"
    nio.write_gexf(Graph([(1, 2)]), path)
    with pytest.raises(ValueError, match="needs a directed graph"):
        nio.read_gexf(path, dag=True)


def test_graphml_networkx_writes_aryagraph_reads(tmp_path):
    G = nio.to_networkx(campus_scalar())
    path = tmp_path / "nx.graphml"
    nx.write_graphml(G, path)
    g = nio.read_graphml(path)
    assert list(g.nodes.data()) == list(G.nodes(data=True))
    assert {frozenset((u, v)): d for u, v, d in g.edges.data()} == {frozenset((u, v)): d for u, v, d in G.edges(data=True)}
    assert g.name == G.graph["name"] and g.attrs == G.graph
    D = nx.DiGraph(name="typed")
    D.add_node("a", flag=True, score=0.5, count=3, tag="x")
    D.add_edge("a", "b", weight=2.5, ok=False)
    nx.write_graphml(D, path)
    d = nio.read_graphml(path)
    assert type(d) is DiGraph
    assert typed(d.nodes["a"]) == typed({"flag": True, "score": 0.5, "count": 3, "tag": "x"})
    assert typed(d.edges["a", "b"]) == typed({"weight": 2.5, "ok": False})


def test_graphml_aryagraph_writes_networkx_reads(tmp_path):
    g = campus_scalar()
    path = tmp_path / "aryagraph.graphml"
    nio.write_graphml(g, path)
    G = nx.read_graphml(path)
    assert list(G.nodes(data=True)) == list(g.nodes.data())
    assert {frozenset((u, v)): d for u, v, d in G.edges(data=True)} == {frozenset((u, v)): d for u, v, d in g.edges.data()}
    x = xml_graph(True)
    x.remove_node((1, "x", (2, 3)))
    x.remove_node(True)
    nio.write_graphml(x, path)
    X = nx.read_graphml(path)  # ids are plain strings for networkx
    assert set(X.nodes) == {str(n) for n in x.nodes}
    assert X.nodes["گره"] == x.nodes["گره"]
    assert X.edges["42", "7"] == {"weight": -3.25, "flag": True}


def test_graphml_reader_handles_foreign_files():
    text = """<?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:y="http://www.yworks.com/xml/graphml">
      <key id="k0" for="node" attr.name="color" attr.type="string"><default>gray</default></key>
      <key id="k1" for="all" attr.name="w" attr.type="float"><default>1.0</default></key>
      <key id="k2" for="node" yfiles.type="nodegraphics"/>
      <key id="k3" for="edge" attr.name="on" attr.type="boolean"/>
      <key id="k4" for="node" attr.name="n" attr.type="long"/>
      <graph id="G">
        <node id="a"><data key="k0">red</data><data key="k2"><y:ShapeNode><y:Fill color="#FF0000"/></y:ShapeNode></data></node>
        <node id="b"><data key="k4">12345678901</data>
          <graph id="inner" edgedefault="undirected"><node id="b::c"/><edge source="b::c" target="a"/></graph>
        </node>
        <edge id="e1" source="a" target="b"><data key="k3">1</data><data key="k1">2.5</data></edge>
        <edge source="a" target="b"><data key="k3">false</data></edge>
      </graph>
    </graphml>"""
    g = nio.from_graphml(text)
    assert type(g) is Graph  # edgedefault missing → undirected, as networkx
    assert g.nodes["a"] == {"color": "red", "w": 1.0}
    assert g.nodes["b"] == {"n": 12345678901, "color": "gray", "w": 1.0}
    assert "b::c" in g and ("b::c", "a") in g.edges  # nested graph flattened
    assert g.edges["a", "b"] == {"on": False, "w": 1.0}  # parallel edges merged: the later one (with its default) wins
    assert g.attrs == {"w": 1.0}


@pytest.mark.parametrize(
    "text, message",
    [
        ("<graphml><graph><hyperedge/></graph></graphml>", "hyperedges are not supported"),
        ('<graphml><graph><node id="a"><data key="nope">1</data></node></graph></graphml>', "undeclared key 'nope'"),
        ("<gexf/>", "root element is <gexf>"),
        ("<graphml></graphml>", "contains no <graph>"),
        ("<graphml><graph>", "invalid GraphML XML"),
        (
            '<graphml><key id="b" for="node" attr.name="b" attr.type="boolean"/>'
            '<graph><node id="a"><data key="b">maybe</data></node></graph></graphml>',
            "'maybe' is not a valid boolean",
        ),
        ('<graphml><graph><edge source="a"/></graph></graphml>', "needs both a source and a target"),
    ],
)
def test_graphml_reader_errors(text, message):
    with pytest.raises(ValueError, match=message):
        nio.from_graphml(text)


def test_graphml_node_type_override(tmp_path):
    path = tmp_path / "g.graphml"
    nio.write_graphml(Graph([("1", "2")]), path)
    assert list(nio.read_graphml(path, node_type=int).edges) == [(1, 2)]


# ---------------------------------------------------------------------- #
# GEXF
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("directed", [False, True])
def test_gexf_round_trip_is_exact(tmp_path, directed):
    g = xml_graph(directed)
    g.nodes["گره"].update(pos=(1.5, 2.0), size=4.5, color="#ff8000")
    g.nodes["a b"]["color"] = "#00ff0080"
    path = tmp_path / "g.gexf"
    nio.write_gexf(g, path)
    back = nio.read_gexf(path)
    expected = [typed(d) for _, d in g.nodes.data()]
    expected[1]["size"] = typed(-1.0)  # viz:size is a float in GEXF
    assert [typed(d) for _, d in back.nodes.data()] == expected
    assert_same(back, g, node_attrs=False)
    assert back.name == g.name


def test_gexf_markup(tmp_path):
    g = xml_graph()
    g.nodes["گره"].update(pos=(1.5, 2.0), size=4.5, color="#ff8000")
    path = tmp_path / "g.gexf"
    nio.write_gexf(g, path)
    root = ET.parse(path).getroot()
    ns = {"g": "http://gexf.net/1.3", "viz": "http://gexf.net/1.3/viz"}
    assert root.get("version") == "1.3"
    node = root.find(".//g:node[@id='گره']", ns)
    assert node.get("label") == "نام"
    assert node.find("viz:position", ns).attrib == {"x": "1.5", "y": "-2.0", "z": "0.0"}  # Gephi's y points up
    assert node.find("viz:color", ns).attrib == {"r": "255", "g": "128", "b": "0"}
    assert node.find("viz:size", ns).get("value") == "4.5"
    titles = {a.get("title"): a.get("type") for a in root.iterfind(".//g:attributes[@class='node']/g:attribute", ns)}
    # label, numeric size and hex colors live in GEXF's own fields, so only these are declared
    assert titles == {"color": "string", "big": "long", "ratio": "double", "flag": "boolean"}
    edge = root.find(".//g:edge[@source='گره']", ns)
    assert edge.get("weight") == "1.5" and edge.get("label") == "یال"


def test_gexf_dag(tmp_path):
    path = tmp_path / "plan.gexf"
    nio.write_gexf(project_plan(), path)
    back = nio.read_gexf(path)
    assert isinstance(back, DAG)
    assert_same(back, project_plan())


def test_gexf_networkx_interop(tmp_path):
    G = nio.to_networkx(campus_scalar())
    path = tmp_path / "nx.gexf"
    nx.write_gexf(G, path)
    g = nio.read_gexf(path)
    assert list(g.nodes.data()) == list(G.nodes(data=True))
    assert edge_dict(g) == {frozenset((u, v)): typed(d) for u, v, d in G.edges(data=True)}
    ours = tmp_path / "aryagraph.gexf"
    campus = ucalgary_campus()  # with pos, written as viz:position
    nio.write_gexf(campus, ours)
    H = nx.read_gexf(ours, version="1.3")
    assert [{k: v for k, v in d.items() if k not in ("label", "viz")} for _, d in H.nodes(data=True)] == [
        d for _, d in G.nodes(data=True)
    ]
    for n, (x, y) in campus.nodes.data("pos"):
        assert H.nodes[n]["viz"]["position"] == {"x": x, "y": -y, "z": 0.0}  # Gephi's y points up
    assert {frozenset((u, v)): {k: v for k, v in d.items() if k != "id"} for u, v, d in H.edges(data=True)} == {
        frozenset((u, v)): d for u, v, d in G.edges(data=True)
    }


def test_gexf_reader_features(tmp_path):
    path = tmp_path / "old.gexf"
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz" version="1.2">
          <graph defaultedgetype="directed" mode="static">
            <attributes class="node">
              <attribute id="0" title="kind" type="string"><default>plain</default></attribute>
              <attribute id="1" title="n" type="integer"/>
            </attributes>
            <nodes>
              <node id="a" label="Alpha"><attvalues><attvalue for="1" value="5"/></attvalues>
                <viz:color r="0" g="0" b="255" a="0.5"/>
                <nodes><node id="a1"/></nodes>
              </node>
              <node id="b" label="b"><attvalues><attvalue id="0" value="special"/></attvalues></node>
            </nodes>
            <edges>
              <edge id="0" source="a" target="b" type="mutual" weight="2.5"/>
              <edge id="1" source="b" target="a1"/>
            </edges>
          </graph>
        </gexf>""",
        encoding="utf-8",
    )
    g = nio.read_gexf(path)
    assert type(g) is DiGraph
    assert g.nodes["a"] == {"n": 5, "kind": "plain", "label": "Alpha", "color": "#0000ff80"}
    assert g.nodes["b"] == {"kind": "special"}  # label equal to the id is not stored
    assert "a1" in g and ("b", "a1") in g.edges
    assert g.edges["a", "b"] == {"weight": 2.5} and g.edges["b", "a"] == {"weight": 2.5}  # mutual


def test_gexf_reader_errors(tmp_path):
    path = tmp_path / "bad.gexf"
    path.write_text("<graphml/>", encoding="utf-8")
    with pytest.raises(ValueError, match="not a GEXF document"):
        nio.read_gexf(path)
    path.write_text('<gexf><graph><nodes><node id="a"><attvalues><attvalue for="9" value="1"/></attvalues></node></nodes></graph></gexf>', encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared node attribute '9'"):
        nio.read_gexf(path)
    path.write_text("<gexf><graph>", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid GEXF XML"):
        nio.read_gexf(path)
