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

"""Structural statistics, checked against networkx on seeded random graphs."""

from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from aryagraph import DAG, DiGraph, Graph
from aryagraph.algorithms.structure import (
    attribute_assortativity,
    average_clustering,
    average_degree,
    average_shortest_path_length,
    center,
    clustering,
    core_number,
    degree_assortativity,
    degree_distribution,
    degree_histogram,
    density,
    diameter,
    eccentricity,
    global_efficiency,
    is_forest,
    is_regular,
    is_tree,
    k_core,
    local_efficiency,
    numeric_assortativity,
    onion_layers,
    periphery,
    radius,
    reciprocity,
    rich_club_coefficient,
    s_metric,
    square_clustering,
    summary,
    transitivity,
    triangles,
    wiener_index,
)
from aryagraph.core.exceptions import GraphTypeError, NotConnected
from aryagraph.core.results import NodeMap


# ---------------------------------------------------------------------- #
# fixtures
# ---------------------------------------------------------------------- #
def to_ag(G: nx.Graph) -> Graph:
    g = DiGraph() if G.is_directed() else Graph()
    g.add_nodes(G.nodes(data=True))
    g.add_edges(G.edges(data=True))
    return g


def decorate(G: nx.Graph, seed: int) -> nx.Graph:
    """Random positive weights plus a categorical and a numeric node attribute."""
    rnd = random.Random(seed)
    for u, v in G.edges:
        G[u][v]["weight"] = rnd.uniform(0.5, 4.0)
    for n in G:
        G.nodes[n]["kind"] = rnd.choice("abc")
        G.nodes[n]["size"] = rnd.random()
    return G


def with_isolates(G: nx.Graph, k: int = 3) -> nx.Graph:
    n = max(G) + 1
    G.add_nodes_from(range(n, n + k))
    return G


def with_loops(G: nx.Graph, nodes=(0, 3)) -> nx.Graph:
    G.add_edges_from((n, n, {"weight": 1.5}) for n in nodes)
    return G


UNDIRECTED = (
    [(f"gnp{s}", decorate(nx.gnp_random_graph(30, 0.15, seed=s), s)) for s in range(3)]
    + [(f"ba{s}", decorate(nx.barabasi_albert_graph(40, 2, seed=s), s)) for s in range(2)]
    + [("isolates", decorate(with_isolates(nx.gnp_random_graph(25, 0.12, seed=7)), 7))]
    + [("loops", decorate(with_loops(nx.gnp_random_graph(25, 0.2, seed=9)), 9))]
)
DIRECTED = (
    [(f"dgnp{s}", decorate(nx.gnp_random_graph(30, 0.12, seed=s, directed=True), s)) for s in range(3)]
    + [("dense", decorate(nx.gnp_random_graph(25, 0.3, seed=4, directed=True), 4))]
    + [("disolates", decorate(with_isolates(nx.gnp_random_graph(25, 0.1, seed=8, directed=True)), 8))]
    + [("dloops", decorate(with_loops(nx.gnp_random_graph(25, 0.15, seed=5, directed=True)), 5))]
)
ALL = UNDIRECTED + DIRECTED


def cases(graphs):
    return pytest.mark.parametrize("G", [G for _, G in graphs], ids=[n for n, _ in graphs])


def no_loops(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    return H


def largest_component(G: nx.Graph) -> nx.Graph:
    comps = nx.strongly_connected_components(G) if G.is_directed() else nx.connected_components(G)
    return G.subgraph(max(comps, key=len)).copy()


def assert_close(ours, theirs, tol=1e-9):
    assert set(ours) == set(theirs)
    for k in theirs:
        assert ours[k] == pytest.approx(theirs[k], abs=tol, rel=tol), k


def same_float(a: float, b: float, tol=1e-9) -> bool:
    if math.isnan(b):
        return math.isnan(a)
    return a == pytest.approx(b, abs=tol, rel=tol)


# ---------------------------------------------------------------------- #
# density and degrees
# ---------------------------------------------------------------------- #
@cases(ALL)
def test_density_and_degree_statistics(G):
    g = to_ag(G)
    assert density(g) == pytest.approx(nx.density(G))
    assert degree_histogram(g) == nx.degree_histogram(G)
    hist = nx.degree_histogram(G)
    assert degree_distribution(g) == {k: c for k, c in enumerate(hist) if c}
    assert average_degree(g) == pytest.approx(sum(d for _, d in G.degree()) / len(G))
    assert s_metric(g) == pytest.approx(nx.s_metric(G))


def test_degree_statistics_edge_cases():
    assert density(Graph()) == 0.0 and density(Graph(nodes=[1])) == 0.0
    assert degree_histogram(Graph()) == [] and degree_distribution(Graph()) == {}
    assert average_degree(Graph()) == 0.0
    assert density(Graph([(1, 2)])) == 1.0 and density(DiGraph([(1, 2)])) == 0.5
    assert degree_histogram(Graph([(1, 1)])) == [0, 0, 1]  # a self-loop adds 2


@cases(DIRECTED)
def test_reciprocity(G):
    assert reciprocity(to_ag(G)) == pytest.approx(nx.overall_reciprocity(G))


def test_reciprocity_errors():
    assert reciprocity(DiGraph([(1, 2), (2, 1), (2, 3), (3, 3)])) == pytest.approx(0.5)
    with pytest.raises(GraphTypeError):
        reciprocity(Graph([(1, 2)]))
    with pytest.raises(ValueError):
        reciprocity(DiGraph(nodes=[1, 2]))


# ---------------------------------------------------------------------- #
# triangles and clustering
# ---------------------------------------------------------------------- #
@cases(UNDIRECTED)
def test_triangles_and_transitivity(G):
    g = to_ag(G)
    res = triangles(g)
    assert isinstance(res, NodeMap) and res == nx.triangles(G)
    assert transitivity(g) == pytest.approx(nx.transitivity(G))


@cases(ALL)
def test_clustering(G):
    g = to_ag(G)
    assert_close(clustering(g), nx.clustering(G))
    assert_close(clustering(g, weight="weight"), nx.clustering(G, weight="weight"))
    for count_zeros in (True, False):
        for weight in (None, "weight"):
            assert average_clustering(g, weight, count_zeros) == pytest.approx(
                nx.average_clustering(G, weight=weight, count_zeros=count_zeros)
            )


@cases(ALL)
def test_square_clustering(G):
    assert_close(square_clustering(to_ag(G)), nx.square_clustering(G))


def test_clustering_hand_checked():
    k4 = Graph([(a, b) for a in range(4) for b in range(a + 1, 4)])
    assert clustering(k4) == dict.fromkeys(range(4), 1.0)
    assert triangles(k4) == dict.fromkeys(range(4), 3)
    assert transitivity(k4) == 1.0
    tri_tail = Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    assert clustering(tri_tail) == {0: 1.0, 1: 1.0, 2: pytest.approx(1 / 3), 3: 0.0}
    # weighted: geometric mean of max-normalised weights
    w = Graph([(0, 1, 1.0), (1, 2, 8.0), (2, 0, 1.0)])
    assert clustering(w, weight="weight")[0] == pytest.approx((1 / 8 * 1 * 1 / 8) ** (1 / 3))
    assert average_clustering(Graph()) == 0.0
    assert average_clustering(Graph([(1, 2)]), count_zeros=False) == 0.0
    with pytest.raises(GraphTypeError):
        triangles(DiGraph([(1, 2)]))
    with pytest.raises(GraphTypeError):
        transitivity(DiGraph([(1, 2)]))


# ---------------------------------------------------------------------- #
# assortativity
# ---------------------------------------------------------------------- #
@cases(ALL)
def test_degree_assortativity(G):
    g = to_ag(G)
    combos = [("out", "in"), ("in", "out"), ("in", "in"), ("out", "out")] if G.is_directed() else [("out", "in")]
    for x, y in combos:
        for weight in (None, "weight"):
            expected = nx.degree_assortativity_coefficient(G, x=x, y=y, weight=weight)
            assert same_float(degree_assortativity(g, x, y, weight), expected), (x, y, weight)


@cases(ALL)
def test_attribute_and_numeric_assortativity(G):
    g = to_ag(G)
    assert same_float(attribute_assortativity(g, "kind"), nx.attribute_assortativity_coefficient(G, "kind"))
    assert same_float(numeric_assortativity(g, "size"), nx.numeric_assortativity_coefficient(G, "size"))


def test_assortativity_edge_cases():
    cycle = Graph([(i, (i + 1) % 6) for i in range(6)])
    assert math.isnan(degree_assortativity(cycle))  # regular graph: zero variance
    assert math.isnan(degree_assortativity(Graph(nodes=[1, 2])))
    star = Graph([(0, i) for i in range(1, 6)])
    assert degree_assortativity(star) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        degree_assortativity(DiGraph([(1, 2)]), x="both")
    g = Graph([(1, 2), (2, 3)])
    g.nodes[1]["size"] = 1.0
    with pytest.raises(ValueError):
        numeric_assortativity(g, "size")
    # missing categorical values form their own category
    g.nodes[2]["kind"] = "a"
    H = nx.Graph()
    H.add_nodes_from(g.nodes(data=True))
    H.add_edges_from(g.edges)
    assert attribute_assortativity(g, "kind") == pytest.approx(nx.attribute_assortativity_coefficient(H, "kind"))
    perfect = Graph([(1, 2), (3, 4)])
    for n, k in zip([1, 2, 3, 4], "aabb"):
        perfect.nodes[n]["kind"] = k
    assert attribute_assortativity(perfect, "kind") == pytest.approx(1.0)


# ---------------------------------------------------------------------- #
# distances
# ---------------------------------------------------------------------- #
@cases(ALL)
def test_eccentricity_family(G):
    L = largest_component(G)
    g = to_ag(L)
    for weight in (None, "weight"):
        ecc = eccentricity(g, weight)
        assert ecc.name == "eccentricity"
        assert_close(ecc, nx.eccentricity(L, weight=weight))
        assert diameter(g, weight) == pytest.approx(nx.diameter(L, weight=weight))
        assert radius(g, weight) == pytest.approx(nx.radius(L, weight=weight))
        assert center(g, weight) == [v for v in g.nodes if v in set(nx.center(L, weight=weight))]
        assert periphery(g, weight) == [v for v in g.nodes if v in set(nx.periphery(L, weight=weight))]
        assert average_shortest_path_length(g, weight) == pytest.approx(
            nx.average_shortest_path_length(L, weight=weight)
        )
        assert wiener_index(g, weight) == pytest.approx(nx.wiener_index(L, weight=weight))


def _connected(G: nx.Graph) -> bool:
    return nx.is_strongly_connected(G) if G.is_directed() else nx.is_connected(G)


DISCONNECTED = [(n, G) for n, G in ALL if not _connected(G)] + [
    ("two_parts", nx.disjoint_union(nx.path_graph(4), nx.cycle_graph(5))),
    ("one_way", nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3)])),
]


@cases(DISCONNECTED)
def test_disconnected_graphs_raise(G):
    g = to_ag(G)
    for fn in (eccentricity, diameter, radius, center, periphery, average_shortest_path_length):
        with pytest.raises(NotConnected):
            fn(g)
    assert wiener_index(g) == math.inf == nx.wiener_index(G)


def test_distance_edge_cases():
    for fn in (diameter, radius, center, periphery, average_shortest_path_length):
        with pytest.raises(NotConnected):
            fn(Graph())
    assert eccentricity(Graph()) == {}
    single = Graph(nodes=["a"])
    assert average_shortest_path_length(single) == 0.0 and diameter(single) == 0
    assert wiener_index(Graph()) == 0.0
    path = Graph([(0, 1), (1, 2), (2, 3)])
    assert eccentricity(path) == {0: 3, 1: 2, 2: 2, 3: 3}
    assert center(path) == [1, 2] and periphery(path) == [0, 3]
    assert average_shortest_path_length(path) == pytest.approx(20 / 12)
    with pytest.raises(NotConnected):
        diameter(DiGraph([(0, 1), (1, 2)]))  # weakly but not strongly connected


@cases(UNDIRECTED)
def test_efficiency(G):
    g = to_ag(G)
    assert global_efficiency(g) == pytest.approx(nx.global_efficiency(G))
    H = no_loops(G)
    assert local_efficiency(to_ag(H)) == pytest.approx(nx.local_efficiency(H))


def test_efficiency_edge_cases():
    assert global_efficiency(Graph()) == 0.0 and local_efficiency(Graph()) == 0.0
    assert global_efficiency(Graph(nodes=[1])) == 0.0
    # a self-loop does not put a node into its own neighbourhood
    tri = Graph([(0, 1), (1, 2), (2, 0), (0, 0)])
    assert local_efficiency(tri) == pytest.approx(1.0)
    with pytest.raises(GraphTypeError):
        global_efficiency(DiGraph([(1, 2)]))
    with pytest.raises(GraphTypeError):
        local_efficiency(DiGraph([(1, 2)]))


# ---------------------------------------------------------------------- #
# cores
# ---------------------------------------------------------------------- #
@cases(ALL)
def test_core_number(G):
    g = to_ag(G)
    H = no_loops(G)  # networkx refuses self-loops; aryagraph ignores them
    res = core_number(g)
    assert res.name == "core_number"
    assert res == nx.core_number(H)
    assert core_number(to_ag(H)) == res


@cases(ALL)
def test_k_core(G):
    g = to_ag(G)
    H = no_loops(G)
    for k in (None, 1, 2, 3):
        ours = k_core(g, k)
        theirs = nx.k_core(H, k)
        assert set(ours.nodes) == set(theirs.nodes)
        assert set(no_loops_edges(ours)) == {e if G.is_directed() else frozenset(e) for e in theirs.edges}


def no_loops_edges(g):
    return [(u, v) if g.directed else frozenset((u, v)) for u, v in g.edges if u != v]


@cases(UNDIRECTED)
def test_onion_layers(G):
    H = no_loops(G)
    assert onion_layers(to_ag(G)) == nx.onion_layers(H)


def test_core_edge_cases():
    assert core_number(Graph()) == {}
    assert k_core(Graph()).num_nodes == 0
    assert core_number(Graph([(1, 1)], nodes=[2])) == {1: 0, 2: 0}
    k4 = Graph([(a, b) for a in range(4) for b in range(a + 1, 4)])
    k4.add_edge(3, 4)
    assert core_number(k4) == {0: 3, 1: 3, 2: 3, 3: 3, 4: 1}
    assert sorted(k_core(k4).nodes) == [0, 1, 2, 3]
    with pytest.raises(GraphTypeError):
        onion_layers(DiGraph([(1, 2)]))


@cases(UNDIRECTED)
def test_rich_club(G):
    H = no_loops(G)
    h = to_ag(H)
    ours = rich_club_coefficient(h)
    theirs = nx.rich_club_coefficient(H, normalized=False)
    assert ours.keys() == theirs.keys()
    assert_close(ours, theirs)


def test_rich_club_normalized_and_errors():
    H = nx.barabasi_albert_graph(60, 3, seed=1)
    h = to_ag(H)
    norm = rich_club_coefficient(h, normalized=True, Q=20, seed=4)
    assert norm == rich_club_coefficient(h, normalized=True, Q=20, seed=4)  # seeded
    assert norm.keys() == rich_club_coefficient(h).keys()
    finite = [v for v in norm.values() if math.isfinite(v)]
    assert finite and all(v > 0 for v in finite)
    # the randomised graph keeps the degree sequence, so low-degree levels are unchanged
    assert norm[0] == pytest.approx(1.0)
    with pytest.raises(GraphTypeError):
        rich_club_coefficient(Graph([(1, 1), (1, 2)]))
    with pytest.raises(GraphTypeError):
        rich_club_coefficient(DiGraph([(1, 2)]))
    with pytest.raises(ValueError):
        rich_club_coefficient(Graph([(1, 2), (2, 3)]), normalized=True)
    assert rich_club_coefficient(Graph(nodes=[1, 2])) == {}


# ---------------------------------------------------------------------- #
# global shape
# ---------------------------------------------------------------------- #
def trees():
    out = []
    for s in range(4):
        T = nx.barabasi_albert_graph(20 + s, 1, seed=s)  # m=1 BA graphs are trees
        out.append(T)
    return out


@pytest.mark.parametrize("T", trees())
def test_tree_and_forest_predicates(T):
    t = to_ag(T)
    assert is_tree(t) and is_forest(t)
    forest = nx.disjoint_union(T, nx.path_graph(4))
    assert not is_tree(to_ag(forest)) and is_forest(to_ag(forest))
    assert is_tree(to_ag(forest)) == nx.is_tree(forest)
    cyclic = T.copy()
    cyclic.add_edge(1, max(T))
    if not T.has_edge(1, max(T)):
        assert not is_forest(to_ag(cyclic)) and not nx.is_forest(cyclic)
    D = nx.DiGraph(T.edges)
    assert is_tree(to_ag(D)) == nx.is_tree(D)
    D.add_edge(*reversed(next(iter(T.edges))))  # a reciprocal pair is a cycle
    assert is_forest(to_ag(D)) == nx.is_forest(D) is False


@cases(ALL)
def test_predicates_on_random_graphs(G):
    g = to_ag(G)
    assert is_tree(g) == nx.is_tree(G)
    assert is_forest(g) == nx.is_forest(G)
    assert is_regular(g) == nx.is_regular(G)


def test_predicates_edge_cases():
    assert not is_tree(Graph()) and is_forest(Graph()) and is_regular(Graph())
    assert is_tree(Graph(nodes=[1]))
    assert not is_forest(Graph([(1, 1)]))
    assert is_regular(Graph([(i, (i + 1) % 5) for i in range(5)]))
    assert is_regular(DiGraph([(i, (i + 1) % 5) for i in range(5)]))
    assert not is_regular(DiGraph([(0, 1), (0, 2)]))
    dag = DAG([(0, 1), (0, 2), (2, 3)])
    assert is_tree(dag) and is_forest(dag)


# ---------------------------------------------------------------------- #
# summary
# ---------------------------------------------------------------------- #
def test_summary_undirected():
    G = with_loops(with_isolates(nx.gnp_random_graph(30, 0.15, seed=3)), nodes=(0,))
    g = to_ag(G)
    s = summary(g)
    assert s["n"] == len(G) and s["m"] == G.number_of_edges() and s["directed"] is False
    assert s["density"] == pytest.approx(nx.density(G))
    assert s["self_loops"] == 1
    assert s["avg_degree"] == pytest.approx(2 * G.number_of_edges() / len(G))
    assert s["max_degree"] == max(d for _, d in G.degree())
    assert s["isolates"] == nx.number_of_isolates(G)
    assert s["components"] == nx.number_connected_components(G)
    assert s["largest_component"] == len(max(nx.connected_components(G), key=len))
    assert s["avg_clustering"] == pytest.approx(nx.average_clustering(G))
    assert s["assortativity"] == pytest.approx(nx.degree_assortativity_coefficient(G))
    assert "is_dag" not in s and "reciprocity" not in s


def test_summary_directed():
    G = nx.gnp_random_graph(25, 0.1, seed=2, directed=True)
    g = to_ag(G)
    s = summary(g)
    assert s["directed"] is True
    assert s["components"] == nx.number_weakly_connected_components(G)
    assert s["avg_clustering"] == pytest.approx(nx.average_clustering(G.to_undirected()))
    assert s["reciprocity"] == pytest.approx(nx.overall_reciprocity(G))
    assert s["is_dag"] == nx.is_directed_acyclic_graph(G)
    assert summary(DAG([(1, 2), (2, 3)]))["is_dag"] is True


@pytest.mark.parametrize(
    "g",
    [Graph(), Graph(nodes=[1]), Graph([(1, 1)]), DiGraph(), DiGraph([(1, 1)]), Graph([(i, (i + 1) % 4) for i in range(4)])],
    ids=["empty", "single", "loop", "dempty", "dloop", "cycle"],
)
def test_summary_never_raises(g):
    s = summary(g)
    assert s["n"] == len(g) and s["m"] == g.num_edges
    assert {"density", "avg_degree", "max_degree", "components", "avg_clustering", "assortativity"} <= s.keys()
