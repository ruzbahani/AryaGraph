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

"""Tests for aryagraph.io: JSON, edge lists, adjacency lists and the read/write dispatcher."""

from __future__ import annotations

import datetime
import json
import math

import networkx as nx
import numpy as np
import pytest

import aryagraph.io as nio
from aryagraph import DAG, CycleError, DiGraph, Graph
from aryagraph.generators import grid_graph, les_miserables, project_plan, ucalgary_campus


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def tricky_graph(directed: bool = False) -> Graph:
    """Names and values that break naive writers: unicode, blanks, quotes, numeric-looking strings."""
    g = DiGraph(name="tricky") if directed else Graph(name="tricky")
    g.add_node("گره", color="سبز", size=3)
    g.add_node("a b", label='quote " inside')
    for n in ("42", 42, -7, 2.5, "with,comma;semi|pipe", "#hash", "true", "node", "x\\y", "isolated"):
        g.add_node(n)
    g.add_edge("گره", "a b", weight=1.5, label="یال")
    g.add_edge("a b", "42", weight=2)
    g.add_edge("42", 42, kind="same-looking")
    g.add_edge(42, -7, flag=True)
    g.add_edge(-7, 2.5, weight=0.1)
    g.add_edge(2.5, "with,comma;semi|pipe", note="")
    g.add_edge("with,comma;semi|pipe", "#hash")
    g.add_edge("#hash", "true", weight=-3)
    g.add_edge("true", "node")
    g.add_edge("node", "x\\y", text='multi\nline, "quoted"')
    return g


def campus_scalar() -> Graph:
    """The UCalgary campus with scalar attributes only (text, floats; names such as "Art Building & Art Parkade").

    The tuple-valued ``pos`` is dropped: text formats store containers as lists
    or strings (see ``test_json_preserves_dag_and_tuple_ids``), so it cannot
    round-trip exactly.
    """
    g = ucalgary_campus()
    for n in g:
        del g.nodes[n]["pos"]
    return g


def typed(value):
    """Value with its exact type, so 2 and 2.0 (or True and 1) compare different."""
    if isinstance(value, dict):
        return {k: typed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(typed(v) for v in value)
    return (type(value).__name__, value)


def edge_dict(g) -> dict:
    return {(frozenset((u, v)) if not g.directed else (u, v)): typed(d) for u, v, d in g.edges.data()}


def assert_identical(a, b, *, node_attrs: bool = True, graph_attrs: bool = True):
    assert type(a) is type(b)
    assert [typed(n) for n in a.nodes] == [typed(n) for n in b.nodes]
    assert edge_dict(a) == edge_dict(b)
    if node_attrs:
        assert [typed(d) for _, d in a.nodes.data()] == [typed(d) for _, d in b.nodes.data()]
    if graph_attrs:
        assert typed(a.attrs) == typed(b.attrs)


# ---------------------------------------------------------------------- #
# JSON
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("directed", [False, True])
def test_json_round_trip_is_exact(tmp_path, directed):
    g = tricky_graph(directed)
    path = tmp_path / "g.json"
    nio.write_json(g, path)
    assert_identical(nio.read_json(path), g)
    assert "گره" in path.read_text(encoding="utf-8")  # not \u-escaped
    assert_identical(nio.from_json(nio.to_json(g, indent=None)), g)


def test_json_preserves_dag_and_tuple_ids(tmp_path):
    plan = project_plan()
    back = nio.from_dict(nio.to_dict(plan))
    assert isinstance(back, DAG)
    assert_identical(back, plan)
    grid = grid_graph(2, 3)
    back = nio.from_json(nio.to_json(grid))
    assert list(back.nodes) == list(grid.nodes)  # tuple ids come back as tuples
    assert back.nodes[(1, 2)]["pos"] == [2.0, 1.0]  # tuple attribute values come back as lists


def test_json_layout():
    g = DAG([("a", "b", {"w": 1})], name="x")
    data = nio.to_dict(g)
    assert data == {
        "directed": True,
        "dag": True,
        "multigraph": False,
        "graph": {"name": "x"},
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [{"source": "a", "target": "b", "w": 1}],
    }
    assert nio.to_dict(g, node_key="key", source="from", target="to")["edges"] == [{"from": "a", "to": "b", "w": 1}]


def test_json_converts_unsafe_values():
    g = Graph()
    g.add_node(
        np.int64(3),
        arr=np.arange(3),
        f32=np.float32(0.5),
        s={3, 1, 2},
        t=(1, (2, 3)),
        when=datetime.date(2024, 1, 2),
        nan=math.nan,
        obj=object,
        b=np.bool_(True),
        mapping={1: "a"},
    )
    data = json.loads(nio.to_json(g))
    node = data["nodes"][0]
    assert node["id"] == 3 and type(node["id"]) is int
    assert node["arr"] == [0, 1, 2] and node["f32"] == 0.5 and node["s"] == [1, 2, 3]
    assert node["t"] == [1, [2, 3]] and node["when"] == "2024-01-02" and math.isnan(node["nan"])
    assert node["obj"] == str(object) and node["b"] is True and node["mapping"] == {"1": "a"}


def test_json_accepts_networkx_and_d3_variants():
    G = les_miserables()
    H = nx.les_miserables_graph()
    for edges_key in ("links", "edges"):
        data = nx.node_link_data(H, edges=edges_key)
        g = nio.from_dict(data)
        assert set(g.nodes) == set(H.nodes)
        assert {frozenset(e): d for *e, d in g.edges.data()} == {frozenset((u, v)): d for u, v, d in H.edges(data=True)}
    # d3 v3 index form: nodes without ids, links by position
    d3 = {"nodes": [{"name": "a"}, {"name": "b"}, {"name": "c"}], "links": [{"source": 0, "target": 2, "value": 5}]}
    g = nio.from_dict(d3)
    assert list(g.nodes.data()) == [(0, {"name": "a"}), (1, {"name": "b"}), (2, {"name": "c"})]
    assert list(g.edges.data()) == [(0, 2, {"value": 5})]
    # d3 after a simulation: endpoints replaced by node objects
    sim = {"nodes": [{"id": "a"}, {"id": "b"}], "links": [{"source": {"id": "a", "x": 1}, "target": {"id": "b"}}]}
    assert list(nio.from_dict(sim).edges) == [("a", "b")]
    # multigraph: parallel edges merge, keys dropped
    M = nx.MultiDiGraph([("a", "b", {"w": 1}), ("a", "b", {"w": 2, "c": 3})])
    g = nio.from_dict(nx.node_link_data(M, edges="links"))
    assert type(g) is DiGraph and g.edges["a", "b"] == {"w": 2, "c": 3}
    # networkx 1.x graph attributes as pairs
    assert nio.from_dict({"graph": [["name", "old"]], "nodes": [], "links": []}).name == "old"
    assert G.num_edges == H.number_of_edges()


def test_json_is_readable_by_networkx(tmp_path):
    g = campus_scalar()
    G = nx.node_link_graph(nio.to_dict(g), edges="edges")
    assert list(G.nodes(data=True)) == list(g.nodes.data())
    assert {frozenset((u, v)): d for u, v, d in G.edges(data=True)} == {frozenset((u, v)): d for u, v, d in g.edges.data()}
    assert G.graph["name"] == g.name and G.graph == dict(g.attrs)
    dg = nx.node_link_graph(nio.to_dict(project_plan()), edges="edges")
    assert dg.is_directed() and nx.is_directed_acyclic_graph(dg)


def test_json_errors(tmp_path):
    with pytest.raises(TypeError):
        nio.from_dict([1, 2])
    with pytest.raises(ValueError, match="no 'id'"):
        nio.from_dict({"nodes": [{"id": 1}, {"name": 2}], "edges": []})
    with pytest.raises(ValueError, match="no 'target'"):
        nio.from_dict({"nodes": [], "edges": [{"source": 1}]})
    with pytest.raises(ValueError, match="not a valid node index"):
        nio.from_dict({"nodes": [{"name": "a"}], "links": [{"source": 0, "target": 5}]})
    with pytest.raises(CycleError):
        nio.from_dict({"dag": True, "directed": True, "nodes": [], "edges": [{"source": 1, "target": 2}, {"source": 2, "target": 1}]})
    with pytest.raises(ValueError, match="line 1, column"):
        nio.from_json('{"nodes": [}')
    g = Graph()
    g.add_node("a", id=5)
    with pytest.raises(ValueError, match="id field"):
        nio.to_dict(g)
    with pytest.raises(TypeError):
        nio.to_dict(Graph([(frozenset({1}), 2)]))


# ---------------------------------------------------------------------- #
# edge lists
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("delimiter", [",", "\t", ";", "|"])
@pytest.mark.parametrize("directed", [False, True])
def test_edgelist_round_trip(tmp_path, delimiter, directed):
    g = tricky_graph(directed)
    path = tmp_path / "g.txt"
    nio.write_edgelist(g, path, delimiter=delimiter)
    back = nio.read_edgelist(path, directed=directed)  # delimiter and header are sniffed
    assert set(map(typed, back.nodes)) == set(map(typed, g.nodes))
    assert edge_dict(back) == edge_dict(g)


def test_edgelist_whitespace_round_trip(tmp_path):
    g = Graph()
    g.add_edge("a b", "گره", weight=2)
    g.add_edge("گره", "42", weight=0.5)
    g.add_edge("42", 42)  # trailing missing value is fine
    g.add_node("alone")
    path = tmp_path / "g.edgelist"
    nio.write_edgelist(g, path, delimiter=" ")
    back = nio.read_edgelist(path)
    assert edge_dict(back) == edge_dict(g) and set(map(typed, back)) == set(map(typed, g))
    bad = Graph([("a", "b", {"x": 1}), ("c", "d", {"y": 2})])
    with pytest.raises(ValueError, match="cannot hold empty fields"):
        nio.write_edgelist(bad, tmp_path / "bad.txt", delimiter=" ")


def test_edgelist_layout_and_quoting(tmp_path):
    g = Graph()
    g.add_edge("007", 7, w=True, s="3.5")
    g.add_edge("a", "b", s="")
    path = tmp_path / "g.csv"
    nio.write_edgelist(g, path)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "source,target,w,s",
        '"007",7,true,"3.5"',
        'a,b,,""',
    ]
    back = nio.read_edgelist(path)
    assert back.edges["007", 7] == {"w": True, "s": "3.5"}
    assert back.edges["a", "b"] == {"s": ""}
    nio.write_edgelist(g, path, attrs=False, header=False)
    assert path.read_text(encoding="utf-8").splitlines() == ['"007",7', "a,b"]


def test_edgelist_header_detection_and_named_columns(tmp_path):
    path = tmp_path / "flights.csv"
    path.write_text("origin,dest,miles,carrier\nTHR,IST,1270,IR\nIST,JFK,5000,TK\n", encoding="utf-8")
    g = nio.read_edgelist(path, source="origin", target="dest", directed=True)
    assert g.edges["THR", "IST"] == {"miles": 1270, "carrier": "IR"}
    # auto: 'miles' is numeric below a text header → header detected, source/target by position
    g = nio.read_edgelist(path, directed=True)
    assert g.edges["IST", "JFK"] == {"miles": 5000, "carrier": "TK"}
    # no header: third column is the weight, later ones col<k>
    path.write_text("1 2 0.5 x\n2 3 1.5 y\n", encoding="utf-8")
    g = nio.read_edgelist(path)
    assert g.edges[1, 2] == {"weight": 0.5, "col3": "x"}
    # forcing header=False keeps the first line as data
    path.write_text("u,v\na,b\n", encoding="utf-8")
    assert set(nio.read_edgelist(path, header=False).edges) == {("u", "v"), ("a", "b")}
    assert set(nio.read_edgelist(path).edges) == {("a", "b")}
    # target column before source
    path.write_text("to,from\nb,a\n", encoding="utf-8")
    assert list(nio.read_edgelist(path, source="from", target="to", directed=True).edges) == [("a", "b")]


def test_edgelist_types_comments_and_quotes(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text(
        '# comment line\n'
        'source,target,zip,weight\n'
        '007,1,02139,2\n'
        '"multi\nline","say ""hi"" # kept",00501,3\n'
        '   # indented comment\n'
        '\n',
        encoding="utf-8",
    )
    g = nio.read_edgelist(path, types={"zip": str, "weight": float})
    assert g.edges["007", 1] == {"zip": "02139", "weight": 2.0}
    assert g.edges["multi\nline", 'say "hi" # kept'] == {"zip": "00501", "weight": 3.0}
    auto = nio.read_edgelist(path)
    assert auto.edges["007", 1] == {"zip": "02139", "weight": 2}  # leading zeros stay text
    text_only = nio.read_edgelist(path, types=str)
    assert text_only.edges["007", "1"] == {"zip": "02139", "weight": "2"}
    ws = tmp_path / "w.txt"
    ws.write_text("a b # trailing comment\nc d\n", encoding="utf-8")
    assert nio.read_edgelist(ws).edges["a", "b"] == {}
    raw = nio.read_edgelist(ws, comments=None, header=False)
    assert raw.edges["a", "b"] == {"weight": "#", "col3": "trailing", "col4": "comment"}


def test_edgelist_directed_dag_and_isolates(tmp_path):
    plan = project_plan()
    path = tmp_path / "plan.tsv"
    nio.write_edgelist(plan, path, delimiter="\t")
    back = nio.read_edgelist(path, dag=True)
    assert isinstance(back, DAG) and edge_dict(back) == edge_dict(plan)
    path.write_text("a,b\nb,a\n", encoding="utf-8")
    with pytest.raises(CycleError):
        nio.read_edgelist(path, dag=True, header=False)
    path.write_text("a,\nb,c\n", encoding="utf-8")
    g = nio.read_edgelist(path)
    assert list(g.nodes) == ["a", "b", "c"] and g.num_edges == 1


def test_edgelist_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text('a,b\nc,"unterminated\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2, column 3: unterminated"):
        nio.read_edgelist(path)
    path.write_text('source,target\na,b,c\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: 3 fields"):
        nio.read_edgelist(path)
    path.write_text('source,target\na,b\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not in the header"):
        nio.read_edgelist(path, source="from")
    with pytest.raises(ValueError, match="without a header"):
        nio.read_edgelist(path, source="source", header=False)
    path.write_text('"a"x,b\n', encoding="utf-8")
    with pytest.raises(ValueError, match="after a quoted field"):
        nio.read_edgelist(path)
    path.write_text(',b\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing source"):
        nio.read_edgelist(path)
    with pytest.raises(ValueError, match="clashes"):
        nio.write_edgelist(Graph([(1, 2, {"source": 3})]), path)
    with pytest.raises(ValueError):
        nio.write_edgelist(Graph([(1, 2)]), path, delimiter='"')


def test_edgelist_networkx_interop(tmp_path):
    campus = ucalgary_campus()
    G = nio.to_networkx(campus)
    path = tmp_path / "nx.csv"
    nx.write_edgelist(G, path, delimiter=",", data=["weight"])
    g = nio.read_edgelist(path)
    assert {frozenset((u, v)): d for u, v, d in g.edges.data()} == {frozenset((u, v)): {"weight": w} for u, v, w in G.edges(data="weight")}
    out = tmp_path / "aryagraph.csv"
    nio.write_edgelist(campus, out, header=False, isolates=False)  # columns: kind, length, weight
    H = nx.read_edgelist(out, delimiter=",", data=[("kind", str), ("length", float), ("weight", float)])
    assert {frozenset((u, v)): d for u, v, d in H.edges(data=True)} == {frozenset((u, v)): d for u, v, d in G.edges(data=True)}


# ---------------------------------------------------------------------- #
# adjacency lists
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("directed", [False, True])
def test_adjacency_list_round_trip(tmp_path, directed):
    g = tricky_graph(directed)
    g.add_edge("گره", "گره")  # self-loop
    path = tmp_path / "g.adjlist"
    nio.write_adjacency_list(g, path)
    back = nio.read_adjacency_list(path, directed=directed)
    assert [typed(n) for n in back] == [typed(n) for n in g]  # isolated nodes survive, order kept
    assert set(edge_dict(back)) == set(edge_dict(g))


def test_adjacency_list_networkx_interop(tmp_path):
    G = nx.les_miserables_graph()
    path = tmp_path / "nx.adjlist"
    nx.write_adjlist(G, path)
    g = nio.read_adjacency_list(path)
    assert {frozenset(e) for e in g.edges} == {frozenset(e) for e in G.edges}
    nio.write_adjacency_list(les_miserables(), path)
    H = nx.read_adjlist(path)
    assert {frozenset(e) for e in H.edges} == {frozenset(e) for e in G.edges}
    dag = nio.read_adjacency_list(_write(tmp_path / "d.adjlist", "a b c\nb c\nc\n"), dag=True)
    assert isinstance(dag, DAG) and dag.topological_order() == ["a", "b", "c"]


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------- #
# dispatcher and whole-format round trips
# ---------------------------------------------------------------------- #
LOSSLESS = [".json", ".graphml", ".gexf", ".dot", ".gv"]
STRUCTURE_ONLY = [".csv", ".tsv", ".txt", ".edgelist", ".adjlist"]


@pytest.mark.parametrize("ext", LOSSLESS + STRUCTURE_ONLY + [".mmd"])
def test_read_write_dispatch_on_campus(tmp_path, ext):
    g = campus_scalar()
    path = tmp_path / f"campus{ext}"
    nio.write(g, path)
    back = nio.read(path)
    if ext == ".mmd":
        names = {n: d.get("label", n) for n, d in back.nodes.data()}
        # "END" is a Mermaid keyword, so building END is written as an n<k>["END"] node
        assert "END" not in back and list(names.values()).count("END") == 1
        assert {frozenset(names[x] for x in e) for e in back.edges} == {frozenset(e) for e in g.edges}
        return
    assert set(back.nodes) == set(g.nodes)
    if ext == ".adjlist":
        assert {frozenset(e) for e in back.edges} == {frozenset(e) for e in g.edges}
        return
    assert edge_dict(back) == edge_dict(g)
    if ext in LOSSLESS:
        assert list(back.nodes.data()) == list(g.nodes.data())
        assert back.name == g.name


@pytest.mark.parametrize("ext", [".json", ".graphml", ".gexf"])
def test_dag_survives_every_typed_format(tmp_path, ext):
    plan = project_plan()
    path = tmp_path / f"plan{ext}"
    nio.write(plan, path)
    back = nio.read(path)
    assert isinstance(back, DAG)
    assert edge_dict(back) == edge_dict(plan)
    assert [typed(d) for _, d in back.nodes.data()] == [typed(d) for _, d in plan.nodes.data()]


def test_dispatch_errors_and_format_override(tmp_path):
    g = Graph([(1, 2)])
    with pytest.raises(ValueError, match="cannot infer the format"):
        nio.write(g, tmp_path / "g.xyz")
    with pytest.raises(ValueError, match="unknown format"):
        nio.write(g, tmp_path / "g.json", format="yaml")
    nio.write(g, tmp_path / "g.data", format="graphml")
    assert list(nio.read(tmp_path / "g.data", format="graphml").edges) == [(1, 2)]
    nio.write(g, tmp_path / "g.out", format=".csv")
    assert nio.read(tmp_path / "g.out", format="csv").num_edges == 1
    assert (tmp_path / "g.TSV").exists() is False
    nio.write(g, tmp_path / "g.TSV")
    assert "\t" in (tmp_path / "g.TSV").read_text(encoding="utf-8")


def test_io_exports():
    for name in nio.__all__:
        assert getattr(nio, name) is not None
    assert {"read", "write", "to_dot", "from_networkx", "read_gexf"} <= set(nio.__all__)
