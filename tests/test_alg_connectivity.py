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

"""Connectivity algorithms, checked against networkx."""

from __future__ import annotations

import random

import networkx as nx
import pytest

from aryagraph import DAG, DiGraph, Graph, GraphTypeError, NodeNotFound
from aryagraph.algorithms import connectivity as C

SEEDS = range(6)


def pair(nxg: nx.Graph) -> tuple[nx.Graph, Graph]:
    nodes = list(nxg.nodes(data=True))
    edges = list(nxg.edges(data=True))
    h = nxg.__class__()
    h.add_nodes_from(nodes)
    h.add_edges_from(edges)
    g = DiGraph() if nxg.is_directed() else Graph()
    g.add_nodes((n, dict(d)) for n, d in nodes)
    g.add_edges((u, v, dict(d)) for u, v, d in edges)
    return h, g


def rand(seed: int, n: int = 60, p: float = 0.03, directed: bool = False, loops: bool = False) -> tuple[nx.Graph, Graph]:
    nxg = nx.gnp_random_graph(n, p, seed=seed, directed=directed)
    if loops:
        rng = random.Random(seed)
        nxg.add_edges_from((v, v) for v in rng.sample(range(n), 5))
    return pair(nxg)


def as_frozen(groups) -> set[frozenset]:
    return {frozenset(c) for c in groups}


def assert_largest_first(groups: list[set], g: Graph) -> None:
    index = g.node_index()
    keys = [(-len(c), min(index[n] for n in c)) for c in groups]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------- #
# undirected components
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("loops", [False, True])
def test_connected_components_match_networkx(seed, loops):
    h, g = rand(seed, loops=loops)
    comps = C.connected_components(g)
    assert as_frozen(comps) == as_frozen(nx.connected_components(h))
    assert all(isinstance(c, set) for c in comps)
    assert_largest_first(comps, g)
    assert C.number_connected_components(g) == nx.number_connected_components(h)
    assert C.is_connected(g) == nx.is_connected(h)
    for v in list(g)[::7]:
        assert C.node_connected_component(g, v) == nx.node_connected_component(h, v)


@pytest.mark.parametrize("seed", SEEDS)
def test_is_connected_on_dense_graphs(seed):
    h, g = rand(seed, p=0.2)
    assert C.is_connected(g) == nx.is_connected(h) is True


def test_connected_components_small_and_edge_cases():
    g = Graph([(0, 1), (2, 3), (3, 4)])
    g.add_node(5)
    assert C.connected_components(g) == [{2, 3, 4}, {0, 1}, {5}]
    # ties keep first appearance in graph order
    tie = Graph([("x", "y"), ("a", "b")])
    assert C.connected_components(tie) == [{"x", "y"}, {"a", "b"}]
    empty = Graph()
    assert C.connected_components(empty) == []
    assert C.number_connected_components(empty) == 0
    assert C.is_connected(empty) is False
    single = Graph(nodes=[1])
    assert C.is_connected(single) is True
    assert C.connected_components(single) == [{1}]
    with pytest.raises(NodeNotFound):
        C.node_connected_component(g, 99)
    with pytest.raises(GraphTypeError):
        C.connected_components(DiGraph([(0, 1)]))
    with pytest.raises(GraphTypeError):
        C.is_connected(DiGraph([(0, 1)]))


# ---------------------------------------------------------------------- #
# directed components
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("p", [0.03, 0.06])
def test_strongly_connected_components_match_networkx(seed, p):
    h, g = rand(seed, p=p, directed=True, loops=True)
    comps = C.strongly_connected_components(g)
    assert as_frozen(comps) == as_frozen(nx.strongly_connected_components(h))
    assert_largest_first(comps, g)
    assert C.number_strongly_connected_components(g) == nx.number_strongly_connected_components(h)
    assert C.is_strongly_connected(g) == nx.is_strongly_connected(h)


@pytest.mark.parametrize("seed", SEEDS)
def test_weakly_connected_components_match_networkx(seed):
    h, g = rand(seed, p=0.02, directed=True)
    comps = C.weakly_connected_components(g)
    assert as_frozen(comps) == as_frozen(nx.weakly_connected_components(h))
    assert_largest_first(comps, g)
    assert C.number_weakly_connected_components(g) == nx.number_weakly_connected_components(h)
    assert C.is_weakly_connected(g) == nx.is_weakly_connected(h)


@pytest.mark.parametrize("seed", SEEDS)
def test_strong_connectivity_on_dense_graphs(seed):
    h, g = rand(seed, n=30, p=0.25, directed=True)
    assert C.is_strongly_connected(g) == nx.is_strongly_connected(h)
    assert C.is_weakly_connected(g) == nx.is_weakly_connected(h)


def test_directed_components_small_and_edge_cases():
    g = DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)])
    g.add_node(5)
    assert C.strongly_connected_components(g) == [{0, 1, 2}, {3, 4}, {5}]
    assert C.weakly_connected_components(g) == [{0, 1, 2, 3, 4}, {5}]
    assert not C.is_strongly_connected(g)
    assert not C.is_weakly_connected(g)
    empty = DiGraph()
    assert C.strongly_connected_components(empty) == []
    assert C.is_strongly_connected(empty) is False
    assert C.is_weakly_connected(empty) is False
    assert C.is_strongly_connected(DiGraph([(0, 1), (1, 0)]))
    with pytest.raises(GraphTypeError):
        C.strongly_connected_components(Graph([(0, 1)]))
    with pytest.raises(GraphTypeError):
        C.weakly_connected_components(Graph([(0, 1)]))


def test_strongly_connected_components_are_iterative():
    n = 30_000
    cycle = DiGraph((i, (i + 1) % n) for i in range(n))
    assert C.strongly_connected_components(cycle) == [set(range(n))]
    path = DiGraph((i, i + 1) for i in range(n - 1))
    assert C.number_strongly_connected_components(path) == n


# ---------------------------------------------------------------------- #
# condensation
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_condensation_matches_networkx(seed):
    h, g = rand(seed, p=0.05, directed=True, loops=True)
    cg = C.condensation(g)
    ref = nx.condensation(h)
    assert isinstance(cg, DAG)
    assert list(cg) == list(range(len(ref)))
    members = {c: frozenset(cg.nodes[c]["members"]) for c in cg}
    assert set(members.values()) == {frozenset(ref.nodes[c]["members"]) for c in ref}
    mapping = cg.attrs["mapping"]
    assert set(mapping) == set(g)
    for n, c in mapping.items():
        assert n in members[c]
    ref_members = {c: frozenset(ref.nodes[c]["members"]) for c in ref}
    assert {(members[u], members[v]) for u, v in cg.edges} == {(ref_members[u], ref_members[v]) for u, v in ref.edges}
    # ids are a topological numbering
    assert all(u < v for u, v in cg.edges)


def test_condensation_small():
    g = DiGraph([("a", "b"), ("b", "a"), ("b", "c"), ("d", "c")])
    cg = C.condensation(g)
    # topological numbering, ties by first appearance: {a, b} then {d} then {c}
    assert [cg.nodes[i]["members"] for i in cg] == [{"a", "b"}, {"d"}, {"c"}]
    assert cg.attrs["mapping"] == {"a": 0, "b": 0, "c": 2, "d": 1}
    assert set(cg.edges) == {(0, 2), (1, 2)}
    # precomputed components are accepted
    again = C.condensation(g, scc=[{"c"}, {"a", "b"}, {"d"}])
    assert list(again.edges) == list(cg.edges)
    empty = C.condensation(DiGraph())
    assert len(empty) == 0 and empty.attrs["mapping"] == {}
    with pytest.raises(ValueError):
        C.condensation(g, scc=[{"a"}, {"b"}, {"c"}, {"d"}])  # not strongly connected pieces


# ---------------------------------------------------------------------- #
# biconnectivity
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("p", [0.04, 0.08])
def test_biconnectivity_matches_networkx(seed, p):
    h, g = rand(seed, p=p, loops=True)
    h.remove_edges_from(list(nx.selfloop_edges(h)))  # networkx's routines assume no self-loops
    assert set(C.articulation_points(g)) == set(nx.articulation_points(h))
    assert {frozenset(e) for e in C.bridges(g)} == {frozenset(e) for e in nx.bridges(h)}
    comps = C.biconnected_components(g)
    assert as_frozen(comps) == as_frozen(nx.biconnected_components(h))
    assert_largest_first(comps, g)
    assert C.is_biconnected(g) == nx.is_biconnected(h)


@pytest.mark.parametrize("seed", SEEDS)
def test_is_biconnected_on_dense_graphs(seed):
    h, g = rand(seed, n=25, p=0.3)
    assert C.is_biconnected(g) == nx.is_biconnected(h)
    assert C.articulation_points(g) == [n for n in g if n in set(nx.articulation_points(h))]


def test_biconnectivity_small_and_edge_cases():
    # two triangles sharing node 2, plus a pendant edge 4-5
    g = Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5)])
    assert C.articulation_points(g) == [2, 4]
    assert C.bridges(g) == [(4, 5)]
    assert C.biconnected_components(g) == [{0, 1, 2}, {2, 3, 4}, {4, 5}]
    assert not C.is_biconnected(g)
    assert C.is_biconnected(Graph([(0, 1)]))  # networkx convention
    assert not C.is_biconnected(Graph(nodes=[0]))
    assert not C.is_biconnected(Graph())
    assert C.articulation_points(Graph()) == []
    assert C.biconnected_components(Graph(nodes=[1, 2])) == []
    looped = Graph([(0, 0), (0, 1), (1, 1)])
    assert C.bridges(looped) == [(0, 1)]
    assert C.articulation_points(looped) == []
    with pytest.raises(GraphTypeError):
        C.bridges(DiGraph([(0, 1)]))


def test_biconnectivity_is_iterative():
    n = 30_000
    path = Graph((i, i + 1) for i in range(n - 1))
    assert len(C.articulation_points(path)) == n - 2
    assert len(C.bridges(path)) == n - 1
    ring = Graph((i, (i + 1) % n) for i in range(n))
    assert C.is_biconnected(ring)
