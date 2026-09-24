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

"""Tests for aryagraph.io.interop: networkx, pandas, numpy and scipy bridges."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import aryagraph.io as nio
import aryagraph.io.interop as interop
from aryagraph import DAG, CycleError, DependencyError, DiGraph, Graph, GraphTypeError, NodeNotFound
from aryagraph.generators import erdos_renyi, les_miserables, project_plan, ucalgary_campus


def typed(value):
    if isinstance(value, dict):
        return {k: typed(v) for k, v in value.items()}
    return (type(value).__name__, value)


def same_graph(g, G) -> bool:
    return (
        list(g.nodes.data()) == list(G.nodes(data=True))
        and {(frozenset((u, v)) if not g.directed else (u, v)): d for u, v, d in g.edges.data()}
        == {(frozenset((u, v)) if not G.is_directed() else (u, v)): d for u, v, d in G.edges(data=True)}
        and dict(g.attrs) == dict(G.graph)
    )


# ---------------------------------------------------------------------- #
# networkx
# ---------------------------------------------------------------------- #
def test_networkx_round_trips():
    for g in (ucalgary_campus(), les_miserables(), erdos_renyi(40, 0.1, directed=True, seed=1)):
        G = nio.to_networkx(g)
        assert G.is_directed() == g.directed and same_graph(g, G)
        back = nio.from_networkx(G)
        assert type(back) is type(g) and same_graph(back, G)
    plan = project_plan()
    G = nio.to_networkx(plan)
    assert type(G) is nx.DiGraph and nx.is_directed_acyclic_graph(G)
    dag = nio.from_networkx(G, as_dag=True)
    assert isinstance(dag, DAG) and same_graph(dag, G)
    assert type(nio.from_networkx(G)) is DiGraph


def test_from_networkx_copies_attribute_dicts():
    G = nx.Graph([(1, 2, {"w": 1})], name="n")
    G.nodes[1]["c"] = "red"
    g = nio.from_networkx(G)
    g.nodes[1]["c"] = "blue"
    g.edges[1, 2]["w"] = 5
    assert G.nodes[1]["c"] == "red" and G.edges[1, 2]["w"] == 1 and g.name == "n"


def test_from_networkx_multigraphs_and_dag_checks():
    M = nx.MultiGraph()
    M.add_edge("a", "b", w=1, x=1)
    M.add_edge("a", "b", w=2)
    M.add_edge("b", "c")
    with pytest.raises(GraphTypeError, match="multigraph"):
        nio.from_networkx(M)
    g = nio.from_networkx(M, collapse=True)
    assert type(g) is Graph and g.edges["a", "b"] == nx.Graph(M).edges["a", "b"] == {"w": 2, "x": 1}
    D = nio.from_networkx(nx.MultiDiGraph([(1, 2), (1, 2), (2, 1)]), collapse=True)
    assert type(D) is DiGraph and set(D.edges) == {(1, 2), (2, 1)}
    with pytest.raises(GraphTypeError, match="directed"):
        nio.from_networkx(nx.path_graph(3), as_dag=True)
    with pytest.raises(CycleError):
        nio.from_networkx(nx.DiGraph([(1, 2), (2, 1)]), as_dag=True)


# ---------------------------------------------------------------------- #
# numpy
# ---------------------------------------------------------------------- #
def test_to_numpy_matches_networkx():
    g = ucalgary_campus()
    G = nio.to_networkx(g)
    assert np.array_equal(nio.to_numpy(g), nx.to_numpy_array(G))
    assert np.array_equal(nio.to_numpy(g, weight=None), nx.to_numpy_array(G, weight=None))
    order = ["OVC", "MSC", "MTH", "MH"]  # holds the MTH-OVC and MSC-MH links
    assert np.array_equal(nio.to_numpy(g, nodes=order), nx.to_numpy_array(G, nodelist=order))
    d = erdos_renyi(30, 0.2, directed=True, seed=4)
    assert np.array_equal(nio.to_numpy(d), nx.to_numpy_array(nio.to_networkx(d)))
    loop = Graph([(0, 0, {"weight": 3}), (0, 1)])
    assert np.array_equal(nio.to_numpy(loop), nx.to_numpy_array(nio.to_networkx(loop)))
    assert np.array_equal(nio.to_numpy(loop, weight=lambda u, v, a: 10), [[10, 10], [10, 0]])
    assert np.isnan(nio.to_numpy(Graph([(0, 1)]), nonedge=np.nan)[0, 0])
    with pytest.raises(NodeNotFound):
        nio.to_numpy(g, nodes=["MSC", "NOPE"])
    with pytest.raises(ValueError, match="duplicates"):
        nio.to_numpy(g, nodes=["MSC", "MSC"])


def test_from_numpy():
    a = np.array([[0, 2, 0], [2, 1, 3], [0, 3, 0]])
    g = nio.from_numpy(a)
    assert type(g) is Graph
    assert {frozenset((u, v)): typed(d) for u, v, d in g.edges.data()} == {
        frozenset((0, 1)): typed({"weight": 2}),
        frozenset((1,)): typed({"weight": 1}),
        frozenset((1, 2)): typed({"weight": 3}),
    }
    G = nx.from_numpy_array(a)
    assert {frozenset(e) for e in g.edges} == {frozenset(e) for e in G.edges}
    d = nio.from_numpy(np.array([[0, 0.5], [0, 0]]), nodes=["x", "y"])
    assert type(d) is DiGraph and list(d.edges.data()) == [("x", "y", {"weight": 0.5})]
    both = nio.from_numpy(a, directed=True)
    assert type(both) is DiGraph and both.num_edges == 5
    b = nio.from_numpy(np.array([[False, True], [True, False]]))
    assert list(b.edges.data()) == [(0, 1, {})]
    assert list(nio.from_numpy(a, weight=None).edges.data())[0][2] == {}
    assert np.array_equal(nio.to_numpy(nio.from_numpy(a)), a)
    with pytest.raises(ValueError, match="not symmetric"):
        nio.from_numpy(np.array([[0, 1], [0, 0]]), directed=False)
    with pytest.raises(ValueError, match="square"):
        nio.from_numpy(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="labels"):
        nio.from_numpy(a, nodes=["a", "b"])


# ---------------------------------------------------------------------- #
# scipy
# ---------------------------------------------------------------------- #
def test_scipy_sparse_round_trip_and_networkx_agreement():
    g = ucalgary_campus()
    m = nio.to_scipy_sparse(g)
    assert m.format == "csr"
    assert np.array_equal(m.toarray(), nx.to_scipy_sparse_array(nio.to_networkx(g)).toarray())
    assert nio.to_scipy_sparse(g, format="coo").format == "coo"
    back = nio.from_scipy_sparse(m, nodes=list(g.nodes))
    assert {frozenset((u, v)): d for u, v, d in back.edges.data()} == {
        frozenset((u, v)): {"weight": w} for u, v, w in g.edges.data("weight")
    }
    assert all(type(d["weight"]) is float for *_, d in back.edges.data())
    d = erdos_renyi(25, 0.2, directed=True, seed=2)
    md = nio.to_scipy_sparse(d, weight=None, format="csc")
    assert np.array_equal(md.toarray(), nio.to_numpy(d, weight=None))
    assert set(nio.from_scipy_sparse(md).edges) == set(d.edges)
    loop = Graph([(0, 0, {"weight": 2.0}), (0, 1)])
    assert np.array_equal(nio.to_scipy_sparse(loop).toarray(), nio.to_numpy(loop))


def test_from_scipy_sparse_details():
    m = sp.coo_array((np.array([1.0, 0.0, 2.0, 3.0]), (np.array([0, 1, 1, 1]), np.array([1, 2, 0, 0]))), shape=(3, 3))
    g = nio.from_scipy_sparse(m)  # explicit zero dropped, duplicates (1,0) summed to 5
    assert type(g) is DiGraph
    assert list(g.edges.data()) == [(0, 1, {"weight": 1.0}), (1, 0, {"weight": 5.0})]
    sym = nio.from_scipy_sparse(sp.csr_matrix(np.array([[0, 1], [1, 0]])), nodes=["a", "b"])
    assert type(sym) is Graph and list(sym.edges.data()) == [("a", "b", {"weight": 1})]
    with pytest.raises(ValueError, match="not symmetric"):
        nio.from_scipy_sparse(sp.csr_array(np.array([[0, 1], [0, 0]])), directed=False)
    with pytest.raises(TypeError, match="sparse"):
        nio.from_scipy_sparse(np.eye(2))


# ---------------------------------------------------------------------- #
# pandas
# ---------------------------------------------------------------------- #
def test_to_pandas_layout():
    g = Graph(name="p")
    g.add_node("a", size=1, color="red")
    g.add_node("b", size=2)
    g.add_node("c")
    g.add_edge("a", "b", weight=1.5)
    g.add_edge("b", "c")
    nodes, edges = nio.to_pandas(g)
    assert list(nodes.columns) == ["id", "size", "color"]
    assert nodes["id"].tolist() == ["a", "b", "c"]
    assert str(nodes["size"].dtype) == "Int64" and nodes["size"].tolist()[:2] == [1, 2]
    assert list(edges.columns) == ["source", "target", "weight"]
    assert edges["weight"].tolist()[0] == 1.5 and math.isnan(edges["weight"].tolist()[1])
    empty_nodes, empty_edges = nio.to_pandas(Graph())
    assert list(empty_nodes.columns) == ["id"] and list(empty_edges.columns) == ["source", "target"]


@pytest.mark.parametrize("make", [ucalgary_campus, les_miserables, project_plan])
def test_pandas_round_trip(make):
    g = make()
    nodes, edges = nio.to_pandas(g)
    back = nio.from_pandas(nodes, edges, directed=g.directed, dag=isinstance(g, DAG))
    assert type(back) is type(g)
    assert [typed(n) for n in back] == [typed(n) for n in g]
    assert [typed(d) for _, d in back.nodes.data()] == [typed(d) for _, d in g.nodes.data()]
    key = (lambda u, v: (u, v)) if g.directed else (lambda u, v: frozenset((u, v)))
    assert {key(u, v): typed(d) for u, v, d in back.edges.data()} == {key(u, v): typed(d) for u, v, d in g.edges.data()}


def test_from_pandas_edgelist_matches_networkx():
    df = pd.DataFrame(
        {"src": ["a", "b", "c", "a"], "dst": ["b", "c", "a", "d"], "w": [1.0, 2.0, np.nan, 4.0], "tag": ["x", None, "z", "q"]}
    )
    g = nio.from_pandas_edgelist(df, "src", "dst", edge_attr=True)
    assert g.edges["a", "b"] == {"w": 1.0, "tag": "x"}
    assert g.edges["b", "c"] == {"w": 2.0}  # missing cells are skipped
    assert g.edges["c", "a"] == {"tag": "z"}
    G = nx.from_pandas_edgelist(df, "src", "dst", edge_attr="w")
    only_w = nio.from_pandas_edgelist(df, "src", "dst", edge_attr="w")
    assert {frozenset(e) for e in only_w.edges} == {frozenset(e) for e in G.edges}
    assert nio.from_pandas_edgelist(df, "src", "dst", edge_attr=["tag"]).edges["a", "d"] == {"tag": "q"}
    assert nio.from_pandas_edgelist(df, "src", "dst").edges["a", "b"] == {}
    d = nio.from_pandas_edgelist(df, "src", "dst", directed=True)
    assert type(d) is DiGraph and ("c", "a") in d.edges and ("a", "c") not in d.edges
    ints = nio.from_pandas_edgelist(pd.DataFrame({"source": [1, 2], "target": [2, 3], "n": [5, 6]}), edge_attr=True)
    assert all(type(x) is int for e in ints.edges.data() for x in (e[0], e[1], e[2]["n"]))
    with pytest.raises(CycleError):
        nio.from_pandas_edgelist(df, "src", "dst", dag=True)
    with pytest.raises(ValueError, match="column 'from'"):
        nio.from_pandas_edgelist(df, "from", "dst")
    with pytest.raises(ValueError, match="edge_attr columns"):
        nio.from_pandas_edgelist(df, "src", "dst", edge_attr=["nope"])
    with pytest.raises(ValueError, match="row 1: missing source"):
        nio.from_pandas_edgelist(pd.DataFrame({"source": ["a", None], "target": ["b", "c"]}))


def test_from_pandas_uses_index_when_no_id_column():
    nodes = pd.DataFrame({"color": ["red", None]}, index=["a", "b"])
    g = nio.from_pandas(nodes, pd.DataFrame({"source": ["a"], "target": ["c"]}))
    assert list(g.nodes.data()) == [("a", {"color": "red"}), ("b", {}), ("c", {})]
    only_nodes = nio.from_pandas(pd.DataFrame({"id": [3, 1], "x": [0.5, 1.5]}))
    assert list(only_nodes.nodes.data()) == [(3, {"x": 0.5}), (1, {"x": 1.5})]


def test_to_pandas_rejects_clashing_attributes():
    g = Graph()
    g.add_node(1, id="x")
    with pytest.raises(ValueError, match="attribute named 'id'"):
        nio.to_pandas(g)
    with pytest.raises(ValueError, match="attribute named 'source'"):
        nio.to_pandas(Graph([(1, 2, {"source": "x"})]))
    nodes, _ = nio.to_pandas(g, node_id="node")
    assert list(nodes.columns) == ["node", "id"]


# ---------------------------------------------------------------------- #
# optional dependencies
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "call, package",
    [
        (lambda: nio.to_networkx(Graph()), "networkx"),
        (lambda: nio.from_networkx(nx.Graph()), "networkx"),
        (lambda: nio.to_pandas(Graph()), "pandas"),
        (lambda: nio.from_pandas(), "pandas"),
        (lambda: nio.to_scipy_sparse(Graph()), "scipy"),
    ],
)
def test_missing_optional_dependency_raises_dependency_error(monkeypatch, call, package):
    real = interop.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] == package:
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(interop.importlib, "import_module", fake_import)
    with pytest.raises(DependencyError, match=r"aryagraph\[interop\]") as info:
        call()
    assert info.value.package == package and isinstance(info.value, ImportError)


def test_numpy_bridges_need_no_optional_packages(monkeypatch):
    def refuse(name, *args, **kwargs):
        raise ImportError(name)

    monkeypatch.setattr(interop.importlib, "import_module", refuse)
    g = nio.from_numpy(np.eye(2))
    assert np.array_equal(nio.to_numpy(g), np.eye(2))
