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

"""Tests for the DOT and Mermaid readers/writers in aryagraph.io."""

from __future__ import annotations

import re

import pytest

import aryagraph.io as nio
from aryagraph import DAG, CycleError, DiGraph, Graph
from aryagraph.generators import project_plan, ucalgary_campus
from aryagraph.io.dot import DotSyntaxError


def typed(value):
    if isinstance(value, dict):
        return {k: typed(v) for k, v in value.items()}
    return (type(value).__name__, value)


def edge_dict(g) -> dict:
    return {(frozenset((u, v)) if not g.directed else (u, v)): typed(d) for u, v, d in g.edges.data()}


def tricky_graph(directed: bool = False) -> Graph:
    g = DiGraph(name="tricky graph") if directed else Graph(name="tricky graph")
    g.add_node("گره", color="سبز", size=3)
    g.add_node("a b", label='quote " inside', note="back\\slash \\\\ pair")
    for n in ("42", 42, -7, 2.5, "007", "#hash", "true", "node", "Edge", "x\\y", "<b>html</b>", "isolated"):
        g.add_node(n)
    g.add_edge("گره", "a b", weight=1.5, label="یال")
    g.add_edge("a b", "42", weight=2)
    g.add_edge("42", 42, kind="same-looking")
    g.add_edge(42, -7, weight=-0.25)
    g.add_edge(-7, 2.5, note="")
    g.add_edge(2.5, "007")
    g.add_edge("007", "#hash", text='multi\nline, "quoted"')
    g.add_edge("#hash", "true")
    g.add_edge("true", "node", tiny=1e-7, big=1e22)
    g.add_edge("node", "Edge")
    g.nodes["Edge"]["label"] = "<<i>html</i> label>"
    g.add_edge("Edge", "x\\y")
    g.add_edge("x\\y", "<b>html</b>")
    return g


# ---------------------------------------------------------------------- #
# DOT writer
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("directed", [False, True])
def test_dot_round_trip_is_exact(tmp_path, directed):
    g = tricky_graph(directed)
    back = nio.from_dot(nio.to_dot(g))
    assert type(back) is type(g)
    assert [typed(n) for n in back] == [typed(n) for n in g]
    assert [typed(d) for _, d in back.nodes.data()] == [typed(d) for _, d in g.nodes.data()]
    assert edge_dict(back) == edge_dict(g)
    assert back.name == "tricky graph"
    path = tmp_path / "g.gv"
    nio.write_dot(g, path)
    assert edge_dict(nio.read_dot(path)) == edge_dict(g)


def test_dot_writer_quoting():
    text = nio.to_dot(tricky_graph(True), rankdir="LR")
    lines = text.splitlines()
    assert lines[0] == 'digraph "tricky graph" {'
    assert lines[1] == "  graph [rankdir=LR];"
    assert "  گره [color=سبز, size=3];" in lines
    assert '  "a b" [label="quote \\" inside", note="back\\slash \\\\ pair"];' in lines
    assert '  "42";' in lines and "  42;" in lines and "  -7;" in lines and "  2.5;" in lines
    assert '  "007";' in lines and '  "#hash";' in lines and '  "node";' in lines
    assert '  "Edge" [label=<<i>html</i> label>];' in lines  # balanced <…> is written as an HTML string
    assert "  true;" in lines and '  "<b>html</b>";' in lines  # "<b>…</b>" is not one HTML string
    assert '  true -> "node" [tiny=0.0000001, big=10000000000000000000000.0];' in lines
    assert lines[-1] == "}"


def test_dot_writer_options_and_values():
    g = Graph(name="net", rankdir="TB", meta={"x": 1})
    g.add_node(1, pos=(0.5, 2), flag=True, missing=None, data={"k": [1, 2]})
    g.add_edge(1, 2, weight=3, color="red")
    text = nio.to_dot(g, name="custom", node_attrs=["pos", "flag"], edge_attrs=False)
    assert text.splitlines() == [
        "graph custom {",
        "  graph [rankdir=TB];",
        '  1 [pos="0.5,2", flag=true];',
        "  2;",
        "  1 -- 2;",
        "}",
    ]
    assert 'data="{\\"k\\": [1, 2]}"' in nio.to_dot(g)
    assert nio.to_dot(Graph()) == "graph {\n}\n"


def test_dot_writer_rejects_unrepresentable_strings():
    for bad in ("ends with \\", 'odd \\" run', "line \\\nbreak"):
        with pytest.raises(ValueError, match="cannot be written in DOT"):
            nio.to_dot(Graph([(bad, "b")]))
    nio.to_dot(Graph([("even \\\\", 'even \\\\" ok')]))  # even runs are fine


def test_dot_clusters_round_trip():
    g = DiGraph()
    g.add_node("a", cluster="cluster_x")
    g.add_node("b", cluster="cluster_x")
    g.add_node("c", cluster=3)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.attrs["clusters"] = {"cluster_x": {"label": "Stage X", "color": "blue"}}
    text = nio.to_dot(g)
    assert "  subgraph cluster_x {" in text and '    graph [label="Stage X", color=blue];' in text
    assert "  subgraph cluster_3 {" in text
    back = nio.from_dot(text)
    assert dict(back.nodes.data("cluster")) == {"a": "cluster_x", "b": "cluster_x", "c": "cluster_3"}
    assert back.attrs["clusters"]["cluster_x"] == {"label": "Stage X", "color": "blue"}


def test_dot_dag_round_trip(tmp_path):
    plan = project_plan()
    path = tmp_path / "plan.dot"
    nio.write_dot(plan, path)
    back = nio.read_dot(path, dag=True)
    assert isinstance(back, DAG)
    assert edge_dict(back) == edge_dict(plan)
    assert [typed(d) for _, d in back.nodes.data()] == [typed(d) for _, d in plan.nodes.data()]
    assert back.attrs["description"] == plan.attrs["description"]
    with pytest.raises(CycleError):
        nio.from_dot("digraph { a -> b -> a }", dag=True)
    with pytest.raises(ValueError, match="needs a 'digraph'"):
        nio.from_dot("graph { a -- b }", dag=True)


# ---------------------------------------------------------------------- #
# DOT parser
# ---------------------------------------------------------------------- #
def test_dot_chains_and_subgraph_endpoints():
    g = nio.from_dot("digraph { a -> b -> c [color=red]; {x y} -> z; p -> {q r}; subgraph s { m n } -> o }")
    assert set(g.edges) == {("a", "b"), ("b", "c"), ("x", "z"), ("y", "z"), ("p", "q"), ("p", "r"), ("m", "o"), ("n", "o")}
    assert g.edges["a", "b"] == {"color": "red"} and g.edges["b", "c"] == {"color": "red"}
    assert list(g.nodes) == ["a", "b", "c", "x", "y", "z", "p", "q", "r", "m", "n", "o"]
    assert "cluster" not in g.nodes["m"]  # plain subgraphs are not clusters


def test_dot_default_attributes_and_scopes():
    g = nio.from_dot(
        """
        digraph G {
            node [shape=box];
            a;
            edge [style=dashed]
            subgraph cluster_0 {
                label = "Cluster A";
                node [color=red];
                b;
                subgraph cluster_inner { c [shape=circle] }
                a -> b
            }
            d;
            graph [rankdir=LR]
            a -> d [style=solid];
        }
        """
    )
    assert g.name == "G" and g.attrs["rankdir"] == "LR"
    # created before the cluster's color default; joins the cluster through "a -> b" inside it (as in Graphviz)
    assert g.nodes["a"] == {"shape": "box", "cluster": "cluster_0"}
    assert g.nodes["b"] == {"shape": "box", "color": "red", "cluster": "cluster_0"}
    assert g.nodes["c"] == {"shape": "circle", "color": "red", "cluster": "cluster_inner"}
    assert g.nodes["d"] == {"shape": "box"}  # the cluster's defaults did not leak out
    assert g.edges["a", "b"] == {"style": "dashed"}
    assert g.edges["a", "d"] == {"style": "solid"}
    assert g.attrs["clusters"] == {"cluster_0": {"label": "Cluster A"}, "cluster_inner": {}}


def test_dot_defaults_apply_only_to_later_nodes():
    g = nio.from_dot("graph { a; node [shape=box]; a; b; a [color=red] }")
    assert g.nodes["a"] == {"color": "red"} and g.nodes["b"] == {"shape": "box"}


def test_dot_comments_strings_and_html():
    g = nio.from_dot(
        '# preprocessor line\n'
        "strict digraph {  // line comment\n"
        "  /* block\n     comment -> ignored */\n"
        '  "has # hash" -> "C:\\\\path";\n'
        '  "quote \\" here" [label="line1\\\nline2", tip="abc" + "def" + "g"];\n'
        '  "raw\nnewline";\n'
        "  h [label=<<b>bold</b> &amp; <i>x</i>>];\n"
        "}\n"
    )
    assert ("has # hash", "C:\\\\path") in g.edges  # backslash pairs are kept verbatim
    assert g.nodes['quote " here'] == {"label": "line1line2", "tip": "abcdefg"}
    assert "raw\nnewline" in g
    assert g.nodes["h"]["label"] == "<<b>bold</b> &amp; <i>x</i>>"


def test_dot_numerals_ports_and_keywords():
    g = nio.from_dot(
        "DiGraph { NODE [w=1.5, n=3, neg=-2, dot=.5, trail=2., q=\"3\", z=007]; "
        "1 -> -2.5; a:p1:n -> b:sw; گره -> دیگر; \"graph\" -> x [k=v; k2=v2, k3=v3] }"
    )
    assert type(g) is DiGraph
    assert g.nodes[1] == {"w": 1.5, "n": 3, "neg": -2, "dot": 0.5, "trail": 2.0, "q": "3", "z": "007"}
    assert (1, -2.5) in g.edges
    assert g.edges["a", "b"] == {"tailport": "p1:n", "headport": "sw"}
    assert ("گره", "دیگر") in g.edges
    assert g.edges["graph", "x"] == {"k": "v", "k2": "v2", "k3": "v3"}
    u = nio.from_dot("graph { a -- b; b -- a }")
    assert type(u) is Graph and u.num_edges == 1


def test_dot_graph_attribute_statements():
    g = nio.from_dot('graph "my net" { label="Title"; bgcolor=white; size="7,5"; a }')
    assert g.attrs == {"label": "Title", "bgcolor": "white", "size": "7,5", "name": "my net"}


@pytest.mark.parametrize(
    "text, line, column, message",
    [
        ("digraph {\n  a -> b;\n  c -> ;\n}", 3, 8, "expected a node or subgraph after '->'"),
        ("graph { a -> b }", 1, 11, "'->' in an undirected graph"),
        ("digraph { a -- b }", 1, 13, "'--' in a digraph"),
        ('digraph { a [label="open] }', 1, 20, "unterminated quoted string"),
        ("digraph { /* never closed", 1, 11, "unterminated /* comment"),
        ("digraph { node }", 1, 16, "expected '[' after 'node'"),
        ("digraph { a -> b", 1, 17, "expected '}' but found end of input"),
        ("digraph { } digraph { }", 1, 13, "several graphs"),
        ("digraph { 1abc }", 1, 11, "badly delimited number"),
        ("digraph { a -> node }", 1, 16, "'node' is a keyword"),
        ("graph { a [color] }", 1, 17, "expected '=' but found ']'"),
        ("network { }", 1, 1, "expected 'graph' or 'digraph'"),
        ("digraph { a -> b } junk", 1, 20, "after the end of the graph"),
        ('digraph { a [l="x" + y] }', 1, 22, "'+' must join two quoted strings"),
        ("digraph { a @ b }", 1, 13, "unexpected character '@'"),
        ("digraph { a [x=<<b> }", 1, 16, "unterminated HTML string"),
        ("digraph { {a} [color=red] }", 1, 15, "a subgraph cannot take an attribute list"),
    ],
)
def test_dot_syntax_errors_report_position(text, line, column, message):
    with pytest.raises(DotSyntaxError, match=re.escape(message)) as info:
        nio.from_dot(text)
    assert (info.value.line, info.value.column) == (line, column)
    assert isinstance(info.value, ValueError)
    assert f"line {line}, column {column}" in str(info.value)


# ---------------------------------------------------------------------- #
# Mermaid
# ---------------------------------------------------------------------- #
def test_mermaid_writer_output():
    g = DiGraph()
    g.add_node("start", shape="stadium")
    g.add_node("is it ok?", shape="diamond")
    g.add_node("end", label='Done "#1"\nnow')
    g.add_edge("start", "is it ok?")
    g.add_edge("is it ok?", "end", label="yes")
    g.add_edge("is it ok?", "start", label="no", style="dotted")
    text = nio.to_mermaid(g, direction="lr")
    assert text.splitlines() == [
        "flowchart LR",
        '    start(["start"])',
        '    n1{"is it ok?"}',
        '    n2["Done #quot;#35;1#quot;<br>now"]',
        "    start --> n1",
        '    n1 -->|"yes"| n2',
        '    n1 -.->|"no"| start',
    ]
    u = nio.to_mermaid(Graph([(1, 2)]))
    assert u.splitlines() == ["flowchart TD", "    1", "    2", "    1 --- 2"]
    with pytest.raises(ValueError):
        nio.to_mermaid(g, direction="up")


def test_mermaid_parser_features():
    g = nio.from_mermaid(
        """
        %% a comment
        flowchart LR
            A[Start] --> B{Is it?}
            B -->|Yes| C((Done))
            B -- No way --> D(Retry)
            D -.-> A
            C ==> E[(Database)] & F>Flag]
            E --- F
            G([stadium]):::hot --o H[[sub]]
            H --x I{{hex}}
            I <--> A; J ~~~ K
            subgraph cluster1 [The group]
                L[/para/] --> M[\\alt\\]
                N[/trap\\] --> O[\\trapalt/]
            end
            click A callback
            style A fill:#f9f
            classDef hot fill:#f00
        """
    )
    assert type(g) is DiGraph and g.attrs["direction"] == "LR"
    shapes = dict(g.nodes.data("shape"))
    assert shapes == {
        "A": "rect", "B": "diamond", "C": "circle", "D": "round", "E": "cylinder", "F": "asymmetric",
        "G": "stadium", "H": "subroutine", "I": "hexagon", "J": None, "K": None, "L": "parallelogram",
        "M": "parallelogram_alt", "N": "trapezoid", "O": "trapezoid_alt",
    }  # fmt: skip
    assert g.nodes["B"]["label"] == "Is it?" and g.nodes["G"]["class"] == "hot"
    assert g.edges["B", "C"] == {"label": "Yes"}
    assert g.edges["B", "D"] == {"label": "No way"}
    assert g.edges["D", "A"] == {"style": "dotted"}
    assert g.edges["C", "E"] == {"style": "thick"} and ("C", "F") in g.edges
    assert g.edges["E", "F"] == {"arrow": "none"}
    assert g.edges["G", "H"] == {"arrow": "circle"} and g.edges["H", "I"] == {"arrow": "cross"}
    assert ("I", "A") in g.edges and ("A", "I") in g.edges
    assert ("J", "K") not in g.edges and "J" in g  # invisible link
    assert g.nodes["L"]["cluster"] == "cluster1" and "cluster" not in g.nodes["A"]


def test_mermaid_undirected_and_header_forms():
    g = nio.from_mermaid('graph TD; a --- b --- c; b ---|"x #quot;y#quot;"| d')
    assert type(g) is Graph and set(map(frozenset, g.edges)) == {frozenset("ab"), frozenset("bc"), frozenset("bd")}
    assert g.edges["b", "d"] == {"label": 'x "y"'}
    assert nio.from_mermaid("flowchart\nA-->B").attrs["direction"] == "TD"


@pytest.mark.parametrize(
    "text, message",
    [
        ("A --> B", "expected a 'flowchart' or 'graph' header"),
        ("", "empty Mermaid text"),
        ("flowchart TD\n  A -- B", "line 2, column 5"),
        ("flowchart TD\n  A[open --> B", "unterminated node shape"),
        ("flowchart TD\n  end", "without a matching 'subgraph'"),
        ("flowchart TD\n  A --> B ??", "unexpected '??'"),
        ("flowchart TD\n  A -->|oops B", "unterminated |link text|"),
    ],
)
def test_mermaid_errors(text, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        nio.from_mermaid(text)


def test_mermaid_round_trip_keeps_names_and_structure(tmp_path):
    g = DiGraph()
    for u, v in [("گره", "a b"), ("a b", 'say "hi"'), ('say "hi"', "end"), ("end", "C# code"), ("x", "گره")]:
        g.add_edge(u, v)
    g.add_node("alone", shape="circle")
    g.edges["a b", 'say "hi"']["label"] = "multi\nline"
    path = tmp_path / "g.mmd"
    nio.write_mermaid(g, path)
    back = nio.read_mermaid(path, node_names="label")
    assert list(back.nodes) == list(g.nodes)
    assert set(back.edges) == set(g.edges)
    assert back.edges["a b", 'say "hi"'] == {"label": "multi\nline"}
    assert back.nodes["alone"] == {"shape": "circle"}
    by_id = nio.from_mermaid(nio.to_mermaid(g))
    assert by_id.nodes["n0"]["label"] == "گره"


def test_mermaid_of_a_large_graph_parses():
    g = ucalgary_campus()
    back = nio.from_mermaid(nio.to_mermaid(g))
    assert type(back) is Graph and back.num_edges == 83
    plan = nio.from_mermaid(nio.to_mermaid(project_plan()), node_names="label")
    assert type(plan) is DiGraph and plan.to_dag().topological_order()[0] == "site survey"
