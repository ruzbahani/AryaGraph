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

"""Centrality measures, checked against networkx on seeded random graphs."""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np
import pytest

from aryagraph import DAG, DiGraph, Graph
from aryagraph.algorithms.centrality import (
    _sample_sources,
    betweenness_centrality,
    centralities,
    closeness_centrality,
    degree_centrality,
    edge_betweenness_centrality,
    eigenvector_centrality,
    harmonic_centrality,
    hits,
    in_degree_centrality,
    katz_centrality,
    out_degree_centrality,
    pagerank,
)
from aryagraph.core.exceptions import ConvergenceError, GraphTypeError, NodeNotFound
from aryagraph.core.results import EdgeMap, NodeMap


# ---------------------------------------------------------------------- #
# fixtures
# ---------------------------------------------------------------------- #
def to_ag(G: nx.Graph) -> Graph:
    g = DiGraph() if G.is_directed() else Graph()
    g.add_nodes(G.nodes(data=True))
    g.add_edges(G.edges(data=True))
    return g


def weighted(G: nx.Graph, seed: int) -> nx.Graph:
    rnd = random.Random(seed)
    for u, v in G.edges:
        G[u][v]["weight"] = rnd.uniform(0.5, 4.0)
    return G


def with_isolates(G: nx.Graph, k: int = 3) -> nx.Graph:
    n = max(G) + 1
    G.add_nodes_from(range(n, n + k))
    return G


UNDIRECTED = (
    [(f"gnp{s}", nx.gnp_random_graph(30, 0.15, seed=s)) for s in range(3)]
    + [(f"ba{s}", nx.barabasi_albert_graph(40, 2, seed=s)) for s in range(2)]
    + [("isolates", with_isolates(nx.gnp_random_graph(25, 0.12, seed=7)))]
    + [(f"wgnp{s}", weighted(nx.gnp_random_graph(30, 0.15, seed=10 + s), s)) for s in range(2)]
)
DIRECTED = (
    [(f"dgnp{s}", nx.gnp_random_graph(30, 0.12, seed=s, directed=True)) for s in range(3)]
    + [(f"wdgnp{s}", weighted(nx.gnp_random_graph(30, 0.12, seed=20 + s, directed=True), s)) for s in range(2)]
    + [("disolates", with_isolates(nx.gnp_random_graph(25, 0.1, seed=8, directed=True)))]
)
ALL = UNDIRECTED + DIRECTED
WEIGHTED = [(n, G) for n, G in ALL if n.startswith("w")]


def cases(graphs):
    return pytest.mark.parametrize("G", [G for _, G in graphs], ids=[n for n, _ in graphs])


def assert_close(ours, theirs, tol=1e-9):
    assert set(ours) == set(theirs)
    for k in theirs:
        assert ours[k] == pytest.approx(theirs[k], abs=tol, rel=tol), k


class FixedSample(random.Random):
    """A ``random.Random`` whose ``sample`` returns a preset node list (to mirror our sampling)."""

    def __init__(self, picks):
        super().__init__(0)
        self.picks = picks

    def sample(self, population, k):
        assert k == len(self.picks)
        return list(self.picks)


# ---------------------------------------------------------------------- #
# degree
# ---------------------------------------------------------------------- #
@cases(ALL)
def test_degree_centrality(G):
    g = to_ag(G)
    res = degree_centrality(g)
    assert isinstance(res, NodeMap) and res.name == "degree_centrality"
    assert_close(res, nx.degree_centrality(G))


@cases(DIRECTED)
def test_in_out_degree_centrality(G):
    g = to_ag(G)
    assert_close(in_degree_centrality(g), nx.in_degree_centrality(G))
    assert_close(out_degree_centrality(g), nx.out_degree_centrality(G))


def test_degree_centrality_small_and_undirected_errors():
    assert degree_centrality(Graph()) == {}
    assert degree_centrality(Graph(nodes=["a"])) == {"a": 1.0}
    g = Graph([(1, 1), (1, 2)])  # a self-loop adds 2
    assert degree_centrality(g) == nx.degree_centrality(nx.Graph([(1, 1), (1, 2)]))
    with pytest.raises(GraphTypeError):
        in_degree_centrality(Graph([(1, 2)]))
    with pytest.raises(GraphTypeError):
        out_degree_centrality(Graph([(1, 2)]))


# ---------------------------------------------------------------------- #
# closeness / harmonic
# ---------------------------------------------------------------------- #
@cases(ALL)
@pytest.mark.parametrize("wf_improved", [True, False])
def test_closeness(G, wf_improved):
    g = to_ag(G)
    assert_close(closeness_centrality(g, wf_improved=wf_improved), nx.closeness_centrality(G, wf_improved=wf_improved))


@cases(WEIGHTED)
def test_closeness_weighted(G):
    g = to_ag(G)
    assert_close(closeness_centrality(g, weight="weight"), nx.closeness_centrality(G, distance="weight"))
    # a callable weight is equivalent to naming the attribute
    assert_close(closeness_centrality(g, weight=lambda u, v, d: d["weight"]), nx.closeness_centrality(G, distance="weight"))


@cases(ALL)
def test_harmonic(G):
    g = to_ag(G)
    assert_close(harmonic_centrality(g), nx.harmonic_centrality(G))
    assert_close(harmonic_centrality(g, weight="weight"), nx.harmonic_centrality(G, distance="weight"))


def test_closeness_hand_checked():
    star = Graph([(0, 1), (0, 2), (0, 3)])
    c = closeness_centrality(star)
    assert c[0] == pytest.approx(1.0) and c[1] == pytest.approx(3 / 5)
    # directed: incoming distances, so the sink of a path is the most central
    path = DiGraph([(0, 1), (1, 2)])
    c = closeness_centrality(path)
    assert c[0] == 0.0 and c[2] == pytest.approx((2 / 3) * (2 / 2))
    assert harmonic_centrality(path) == {0: 0.0, 1: 1.0, 2: 1.5}
    assert closeness_centrality(Graph(nodes=[1])) == {1: 0.0}


# ---------------------------------------------------------------------- #
# betweenness
# ---------------------------------------------------------------------- #
@cases(ALL)
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("endpoints", [False, True])
def test_betweenness(G, normalized, endpoints):
    g = to_ag(G)
    res = betweenness_centrality(g, normalized=normalized, endpoints=endpoints)
    assert res.name == "betweenness_centrality"
    assert_close(res, nx.betweenness_centrality(G, normalized=normalized, endpoints=endpoints))


@cases(WEIGHTED)
@pytest.mark.parametrize("endpoints", [False, True])
def test_betweenness_weighted(G, endpoints):
    g = to_ag(G)
    assert_close(
        betweenness_centrality(g, weight="weight", endpoints=endpoints),
        nx.betweenness_centrality(G, weight="weight", endpoints=endpoints),
    )


@pytest.mark.parametrize("graph", [UNDIRECTED[0], UNDIRECTED[3], DIRECTED[0], WEIGHTED[0]], ids=lambda p: p[0])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("endpoints", [False, True])
def test_betweenness_sampled_scaling(graph, normalized, endpoints):
    _, G = graph
    g = to_ag(G)
    weight = "weight" if graph[0].startswith("w") else None
    picks = _sample_sources(g, 7, seed=3)
    ours = betweenness_centrality(g, normalized=normalized, weight=weight, endpoints=endpoints, k=7, seed=3)
    theirs = nx.betweenness_centrality(
        G, k=7, normalized=normalized, weight=weight, endpoints=endpoints, seed=FixedSample(picks)
    )
    assert_close(ours, theirs)
    # deterministic for a fixed seed
    assert ours == betweenness_centrality(g, normalized=normalized, weight=weight, endpoints=endpoints, k=7, seed=3)


def test_betweenness_sampling_edges_and_errors():
    G = nx.gnp_random_graph(20, 0.2, seed=1)
    g = to_ag(G)
    assert betweenness_centrality(g, k=len(g)) == betweenness_centrality(g)
    picks = _sample_sources(g, 5, seed=0)
    assert_close(
        edge_betweenness_centrality(g, k=5, seed=0),
        nx.edge_betweenness_centrality(G, k=5, seed=FixedSample(picks)),
    )
    with pytest.raises(ValueError):
        betweenness_centrality(g, k=0)
    with pytest.raises(ValueError):
        betweenness_centrality(g, k=21)


@cases(ALL)
@pytest.mark.parametrize("normalized", [True, False])
def test_edge_betweenness(G, normalized):
    g = to_ag(G)
    res = edge_betweenness_centrality(g, normalized=normalized)
    assert isinstance(res, EdgeMap)
    assert list(res) == list(g.edges)
    assert_close(res, nx.edge_betweenness_centrality(G, normalized=normalized))
    if any("weight" in d for *_, d in G.edges(data=True)):
        assert_close(
            edge_betweenness_centrality(g, normalized=normalized, weight="weight"),
            nx.edge_betweenness_centrality(G, normalized=normalized, weight="weight"),
        )


def test_betweenness_hand_checked():
    path = Graph([(0, 1), (1, 2)])
    assert betweenness_centrality(path, normalized=False) == {0: 0.0, 1: 1.0, 2: 0.0}
    assert betweenness_centrality(path, normalized=True, endpoints=True) == pytest.approx({0: 2 / 3, 1: 1.0, 2: 2 / 3})
    dpath = DiGraph([(0, 1), (1, 2)])
    assert betweenness_centrality(dpath) == {0: 0.0, 1: 0.5, 2: 0.0}
    assert edge_betweenness_centrality(path, normalized=False) == {(0, 1): 2.0, (1, 2): 2.0}
    # two equal shortest paths split the load
    square = Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    assert betweenness_centrality(square, normalized=False) == pytest.approx(dict.fromkeys(range(4), 0.5))
    assert betweenness_centrality(Graph()) == {}
    assert betweenness_centrality(Graph(nodes=["x"])) == {"x": 0.0}


def test_betweenness_self_loops_and_dag():
    G = nx.gnp_random_graph(20, 0.2, seed=4)
    G.add_edges_from([(0, 0), (5, 5)])
    assert_close(betweenness_centrality(to_ag(G)), nx.betweenness_centrality(G))
    assert_close(edge_betweenness_centrality(to_ag(G)), nx.edge_betweenness_centrality(G))
    D = nx.gn_graph(25, seed=2)  # a random DAG (growing network)
    dag = DAG(list(D.edges))  # nx's EdgeView is a Mapping, which aryagraph would read as adjacency
    assert_close(betweenness_centrality(dag), nx.betweenness_centrality(D))


# ---------------------------------------------------------------------- #
# spectral
# ---------------------------------------------------------------------- #
@cases(ALL)
def test_eigenvector(G):
    g = to_ag(G)
    try:
        expected = nx.eigenvector_centrality(G)
    except nx.PowerIterationFailedConvergence:
        with pytest.raises(ConvergenceError):
            eigenvector_centrality(g)
        return
    res = eigenvector_centrality(g)
    assert res.name == "eigenvector_centrality"
    assert_close(res, expected, tol=1e-9)
    assert math.fsum(v * v for v in res.values()) == pytest.approx(1.0)


@cases(WEIGHTED)
def test_eigenvector_weighted(G):
    g = to_ag(G)
    try:
        expected = nx.eigenvector_centrality(G, weight="weight")
    except nx.PowerIterationFailedConvergence:
        with pytest.raises(ConvergenceError):
            eigenvector_centrality(g, weight="weight")
        return
    assert_close(eigenvector_centrality(g, weight="weight"), expected)


def test_eigenvector_nonconvergence_and_empty():
    with pytest.raises(ConvergenceError):
        eigenvector_centrality(to_ag(nx.gnp_random_graph(30, 0.2, seed=1)), max_iter=2)
    assert eigenvector_centrality(Graph()) == {}
    # a directed 3-cycle rotates forever under A only; A + I converges to uniform
    cyc = eigenvector_centrality(DiGraph([(0, 1), (1, 2), (2, 0)]))
    assert list(cyc.values()) == pytest.approx([1 / math.sqrt(3)] * 3)


@cases(ALL)
def test_katz(G):
    g = to_ag(G)
    assert_close(katz_centrality(g, alpha=0.05), nx.katz_centrality(G, alpha=0.05))
    assert_close(
        katz_centrality(g, alpha=0.05, normalized=False, weight="weight"),
        nx.katz_centrality(G, alpha=0.05, normalized=False, weight="weight"),
    )


def test_katz_beta_mapping_and_errors():
    G = nx.gnp_random_graph(20, 0.2, seed=3)
    g = to_ag(G)
    beta = {n: 1.0 + (n % 3) for n in G}
    assert_close(katz_centrality(g, alpha=0.05, beta=beta), nx.katz_centrality(G, alpha=0.05, beta=beta))
    with pytest.raises(ValueError):
        katz_centrality(g, beta={0: 1.0})
    with pytest.raises(ConvergenceError):
        katz_centrality(g, alpha=1.0)  # alpha above 1/λ_max diverges
    assert katz_centrality(Graph()) == {}


@cases(ALL)
@pytest.mark.parametrize("alpha", [0.85, 0.6])
def test_pagerank(G, alpha):
    g = to_ag(G)
    res = pagerank(g, alpha=alpha)
    assert res.name == "pagerank"
    assert_close(res, nx.pagerank(G, alpha=alpha))
    assert math.fsum(res.values()) == pytest.approx(1.0)
    assert_close(pagerank(g, weight=None), nx.pagerank(G, weight=None))


@cases(ALL)
def test_pagerank_personalized_and_dangling(G):
    g = to_ag(G)
    nodes = list(G)
    pers = {n: (i % 4) for i, n in enumerate(nodes)}
    dang = {n: 1.0 for n in nodes[:5]}
    start = {n: 1.0 + (i % 3) for i, n in enumerate(nodes)}
    assert_close(
        pagerank(g, personalization=pers, dangling=dang, nstart=start),
        nx.pagerank(G, personalization=pers, dangling=dang, nstart=start),
    )


def test_pagerank_errors_and_edge_cases():
    g = to_ag(nx.gnp_random_graph(15, 0.2, seed=2, directed=True))
    with pytest.raises(NodeNotFound):
        pagerank(g, personalization={"nope": 1.0})
    with pytest.raises(ValueError):
        pagerank(g, personalization={0: 0.0})
    with pytest.raises(ConvergenceError):
        pagerank(g, max_iter=1)
    assert pagerank(Graph()) == {}
    assert pagerank(Graph(nodes=["a", "b"])) == pytest.approx({"a": 0.5, "b": 0.5})
    # self-loops and a sink
    G = nx.DiGraph([(0, 0), (0, 1), (1, 2), (2, 0), (2, 3)])
    assert_close(pagerank(to_ag(G)), nx.pagerank(G))


@cases(ALL)
def test_hits(G):
    g = to_ag(G)
    hubs, auths = hits(g)
    H, A = nx.hits(G)
    assert hubs.name == "hubs" and auths.name == "authorities"
    assert_close(hubs, H, tol=1e-7)
    assert_close(auths, A, tol=1e-7)
    # unnormalised: unit authorities, hubs = A @ authorities (networkx's sign is arbitrary)
    hubs_u, auths_u = hits(g, normalized=False)
    H_u, A_u = nx.hits(G, normalized=False)
    assert_close(auths_u, {k: abs(v) for k, v in A_u.items()}, tol=1e-7)
    assert_close(hubs_u, {k: abs(v) for k, v in H_u.items()}, tol=1e-7)


def test_hits_edge_cases():
    assert hits(Graph()) == ({}, {})
    hubs, auths = hits(Graph(nodes=[1, 2, 3, 4]))
    assert hubs == auths == pytest.approx(dict.fromkeys([1, 2, 3, 4], 0.25))
    star = DiGraph([(0, 1), (0, 2), (0, 3)])
    hubs, auths = hits(star)
    assert hubs == pytest.approx({0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})
    assert auths == pytest.approx({0: 0.0, 1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
    with pytest.raises(ConvergenceError):
        hits(to_ag(nx.gnp_random_graph(30, 0.2, seed=5, directed=True)), max_iter=1)


# ---------------------------------------------------------------------- #
# centralities()
# ---------------------------------------------------------------------- #
def test_centralities_bundle():
    G = nx.gnp_random_graph(25, 0.2, seed=6)
    g = to_ag(G)
    res = centralities(g)
    assert list(res) == ["degree", "betweenness", "closeness", "pagerank", "eigenvector"]
    assert_close(res["betweenness"], nx.betweenness_centrality(G))
    res = centralities(g, kinds=("pagerank", "hubs", "authorities"), pagerank={"alpha": 0.7})
    assert_close(res["pagerank"], nx.pagerank(G, alpha=0.7))
    assert res["hubs"] == hits(g)[0] and res["authorities"] == hits(g)[1]


def test_centralities_skips_nonconvergence_and_validates():
    g = to_ag(nx.gnp_random_graph(25, 0.2, seed=6))
    res = centralities(g, kinds=("degree", "katz", "eigenvector"), katz={"alpha": 1.0}, eigenvector={"max_iter": 1})
    assert list(res) == ["degree"]
    with pytest.raises(ValueError):
        centralities(g, kinds=("nope",))
    with pytest.raises(TypeError):
        centralities(g, pagerank={"bogus": 1})
    with pytest.raises(GraphTypeError):
        centralities(g, kinds=("in_degree",))
    empty = centralities(Graph(), kinds=("degree", "betweenness", "closeness", "pagerank", "eigenvector", "katz", "hubs"))
    assert all(v == {} for v in empty.values()) and len(empty) == 7


def test_results_are_aligned_with_graph_order():
    g = Graph([("c", "a"), ("a", "b")], nodes=["z"])
    for res in (degree_centrality(g), pagerank(g), betweenness_centrality(g), closeness_centrality(g)):
        assert list(res) == list(g.nodes)
    arr = pagerank(g).to_array()
    assert isinstance(arr, np.ndarray) and arr.shape == (4,)
