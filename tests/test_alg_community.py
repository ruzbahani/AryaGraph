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

"""Community detection and partition quality, checked against networkx."""

from __future__ import annotations

import random
import statistics
from itertools import islice

import networkx as nx
import pytest

from aryagraph import DiGraph, Graph
from aryagraph.algorithms.community import (
    asyn_fluid_communities,
    community_labels,
    girvan_newman,
    greedy_modularity_communities,
    label_propagation_communities,
    louvain_communities,
    louvain_hierarchy,
    modularity,
    partition_quality,
)
from aryagraph.core.exceptions import GraphTypeError, NotConnected
from aryagraph.core.results import NodeMap
from aryagraph.generators import ucalgary_campus
from aryagraph.io import to_networkx


# ---------------------------------------------------------------------- #
# fixtures
# ---------------------------------------------------------------------- #
def pair(G: nx.Graph) -> tuple[Graph, nx.Graph]:
    """A aryagraph graph and an nx graph with identical node *and adjacency* order.

    Both are built by inserting the same edge sequence, so tie-breaking that
    depends on neighbour order (CNM, Girvan–Newman) sees the same input.
    """
    g = DiGraph() if G.is_directed() else Graph()
    g.add_nodes(G.nodes(data=True))
    g.add_edges(G.edges(data=True))
    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    H.add_nodes_from(g.nodes(data=True))
    H.add_edges_from(g.edges(data=True))
    return g, H


def weighted(G: nx.Graph, seed: int) -> nx.Graph:
    rnd = random.Random(seed)
    for u, v in G.edges:
        G[u][v]["weight"] = rnd.uniform(0.2, 5.0)
    return G


def sbm(sizes, p_in, p_out, seed):
    probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
    return nx.stochastic_block_model(sizes, probs, seed=seed)


def with_isolates(G: nx.Graph, k: int = 3) -> nx.Graph:
    n = max(G) + 1
    G.add_nodes_from(range(n, n + k))
    return G


def campus() -> nx.Graph:
    """The UCalgary campus (56 buildings, 83 links; ``weight`` = length in metres) as networkx."""
    return to_networkx(ucalgary_campus())


GRAPHS = (
    [("campus", campus())]
    + [(f"sbm{s}", sbm([20, 25, 30, 15], 0.3, 0.04, s)) for s in range(3)]
    + [(f"gnp{s}", nx.gnp_random_graph(60, 0.08, seed=s)) for s in range(2)]
    + [(f"ba{s}", nx.barabasi_albert_graph(80, 2, seed=s)) for s in range(2)]
    + [(f"wgnp{s}", weighted(nx.gnp_random_graph(50, 0.1, seed=10 + s), s)) for s in range(2)]
    + [("isolates", with_isolates(nx.gnp_random_graph(40, 0.1, seed=5)))]
    + [(f"dgnp{s}", nx.gnp_random_graph(50, 0.07, seed=s, directed=True)) for s in range(2)]
    + [("wdgnp", weighted(nx.gnp_random_graph(50, 0.07, seed=3, directed=True), 3))]
)
UNDIRECTED = [(n, G) for n, G in GRAPHS if not G.is_directed()]


def cases(graphs):
    return pytest.mark.parametrize("G", [G for _, G in graphs], ids=[n for n, _ in graphs])


def random_partition(G, k, seed):
    rnd = random.Random(seed)
    parts = [set() for _ in range(k)]
    for n in G:
        parts[rnd.randrange(k)].add(n)
    return [p for p in parts if p]


def as_frozen(partition):
    return {frozenset(c) for c in partition}


def assert_partition(g, comms):
    assert all(isinstance(c, set) and c for c in comms)
    assert sum(len(c) for c in comms) == len(g) and set().union(*comms) == set(g.nodes)
    sizes = [len(c) for c in comms]
    assert sizes == sorted(sizes, reverse=True)


# ---------------------------------------------------------------------- #
# modularity and quality
# ---------------------------------------------------------------------- #
@cases(GRAPHS)
@pytest.mark.parametrize("resolution", [1.0, 0.5, 1.7])
def test_modularity_matches_networkx(G, resolution):
    g, H = pair(G)
    for seed, k in [(0, 2), (1, 4), (2, 7)]:
        part = random_partition(G, k, seed)
        for weight in ("weight", None):
            assert modularity(g, part, weight=weight, resolution=resolution) == pytest.approx(
                nx.community.modularity(H, part, weight=weight, resolution=resolution), abs=1e-12
            )
    assert modularity(g, [set(G)]) == pytest.approx(nx.community.modularity(H, [set(G)]), abs=1e-12)


def test_modularity_self_loops_and_hand_checked():
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3), (0, 0)])
    D = nx.DiGraph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 3), (2, 3), (1, 1)])
    part = [{0, 1, 2}, {3, 4, 5}]
    for X in (G, D):
        g, H = pair(X)
        p = [c & set(X) for c in part]
        assert modularity(g, p) == pytest.approx(nx.community.modularity(H, p), abs=1e-12)
    assert modularity(Graph([(0, 1), (2, 3)]), [{0, 1}, {2, 3}]) == pytest.approx(0.5)
    # empty communities are allowed, as in networkx
    assert modularity(Graph([(0, 1), (2, 3)]), [{0, 1}, set(), {2, 3}]) == pytest.approx(0.5)


def test_modularity_validates_partition():
    g = Graph([(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="no community"):
        modularity(g, [{0, 1}])
    with pytest.raises(ValueError, match="communities 0 and 1"):
        modularity(g, [{0, 1}, {1, 2}])
    with pytest.raises(ValueError, match="not in the graph"):
        modularity(g, [{0, 1, 2}, {9}])
    with pytest.raises(ValueError, match="total edge weight"):
        modularity(Graph(nodes=[1, 2]), [{1}, {2}])


@cases(GRAPHS)
def test_partition_quality(G):
    g, H = pair(G)
    for seed, k in [(0, 3), (1, 6)]:
        part = random_partition(G, k, seed)
        q = partition_quality(g, part)
        coverage, performance = nx.community.partition_quality(H, part)
        assert q["coverage"] == pytest.approx(coverage)
        assert q["performance"] == pytest.approx(performance)
        assert q["modularity"] == pytest.approx(nx.community.modularity(H, part))


def test_partition_quality_errors():
    with pytest.raises(ValueError):
        partition_quality(Graph(nodes=[1, 2]), [{1}, {2}])
    with pytest.raises(ValueError):
        partition_quality(Graph([(1, 2)]), [{1}])


def test_community_labels():
    labels = community_labels([{"a"}, {"b", "c", "d"}, {"e", "f"}])
    assert isinstance(labels, NodeMap) and labels.name == "community"
    assert labels == {"b": 0, "c": 0, "d": 0, "e": 1, "f": 1, "a": 2}
    # equal sizes keep their given order
    assert community_labels([[1, 2], [3, 4]]) == {1: 0, 2: 0, 3: 1, 4: 1}
    assert community_labels([]) == {}
    with pytest.raises(ValueError):
        community_labels([{1, 2}, {2, 3}])


# ---------------------------------------------------------------------- #
# Louvain
# ---------------------------------------------------------------------- #
LOUVAIN_SEEDS = range(5)
LOUVAIN_GRAPHS = [(n, G) for n, G in GRAPHS if n != "isolates"]


@cases(LOUVAIN_GRAPHS)
def test_louvain_quality_vs_networkx(G):
    """Mean modularity over five seeds is at least 97% of networkx's Louvain (same seeds)."""
    g, H = pair(G)
    ours = [modularity(g, louvain_communities(g, seed=s)) for s in LOUVAIN_SEEDS]
    theirs = [nx.community.modularity(H, nx.community.louvain_communities(H, seed=s)) for s in LOUVAIN_SEEDS]
    assert statistics.mean(ours) >= 0.97 * statistics.mean(theirs)
    assert max(ours) >= 0.97 * max(theirs)


@cases(LOUVAIN_GRAPHS)
def test_louvain_plain_matches_networkx_quality(G):
    """Without refinement the algorithm is networkx's; its average quality must be on par."""
    g, H = pair(G)
    ours = [modularity(g, louvain_communities(g, seed=s, refine=False)) for s in LOUVAIN_SEEDS]
    theirs = [nx.community.modularity(H, nx.community.louvain_communities(H, seed=s)) for s in LOUVAIN_SEEDS]
    assert statistics.mean(ours) >= 0.97 * statistics.mean(theirs)


@cases(GRAPHS)
def test_louvain_structure(G):
    g, _ = pair(G)
    comms = louvain_communities(g, seed=1)
    assert_partition(g, comms)
    assert comms == louvain_communities(g, seed=1)  # seeded
    plain = louvain_communities(g, seed=1, refine=False)
    assert modularity(g, comms) >= modularity(g, plain) - 1e-12  # refinement never hurts
    levels = louvain_hierarchy(g, seed=1)
    assert levels[-1] == plain
    assert louvain_communities(g, seed=1, max_level=1, refine=False) == levels[0]
    for fine, coarse in zip(levels, levels[1:]):
        assert len(coarse) < len(fine)
        assert all(any(c <= d for d in coarse) for c in fine)  # each level coarsens the previous
    for level in levels:
        assert_partition(g, level)


def test_louvain_recovers_planted_blocks():
    G = sbm([15, 20, 25], 0.6, 0.01, seed=2)
    g, _ = pair(G)
    blocks = [{n for n in G if G.nodes[n]["block"] == b} for b in range(3)]
    assert as_frozen(louvain_communities(g, seed=0)) == as_frozen(blocks)


def test_louvain_resolution_and_weights():
    G = sbm([15, 20, 25], 0.5, 0.03, seed=4)
    g, _ = pair(G)
    coarse = louvain_communities(g, resolution=0.2, seed=0)
    fine = louvain_communities(g, resolution=3.0, seed=0)
    assert len(coarse) < len(fine)
    # weights steer the result: heavy edges bind 0-1 and 2-3 despite the topology
    w = Graph([(0, 1, 10.0), (1, 2, 0.1), (2, 3, 10.0), (3, 0, 0.1)])
    assert as_frozen(louvain_communities(w, seed=0)) == {frozenset({0, 1}), frozenset({2, 3})}
    assert len(louvain_communities(w, weight=None, seed=0)) >= 1


def test_louvain_edge_cases():
    assert louvain_communities(Graph()) == []
    assert louvain_communities(Graph(nodes=[3, 1, 2])) == [{3}, {1}, {2}]
    assert louvain_hierarchy(Graph(nodes=[1, 2])) == [[{1}, {2}]]
    assert louvain_communities(Graph([(1, 2, 0.0)])) == [{1}, {2}]
    with pytest.raises(ValueError):
        louvain_communities(Graph([(1, 2, -1.0)]))
    with pytest.raises(ValueError):
        louvain_communities(Graph([(1, 2)]), max_level=0)
    loops = Graph([(0, 0), (0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
    assert as_frozen(louvain_communities(loops, seed=0)) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}


# ---------------------------------------------------------------------- #
# greedy modularity (CNM)
# ---------------------------------------------------------------------- #
@cases(GRAPHS)
@pytest.mark.parametrize("weight", [None, "weight"])
@pytest.mark.parametrize("resolution", [1.0, 0.6, 1.5])
def test_greedy_modularity_matches_networkx(G, weight, resolution):
    g, H = pair(G)
    ours = greedy_modularity_communities(g, weight=weight, resolution=resolution)
    theirs = nx.community.greedy_modularity_communities(H, weight=weight, resolution=resolution)
    assert_partition(g, ours)
    q_ours = modularity(g, ours, weight=weight, resolution=resolution)
    q_theirs = nx.community.modularity(H, theirs, weight=weight, resolution=resolution)
    assert abs(q_ours - q_theirs) <= 1e-9
    assert as_frozen(ours) == as_frozen(theirs)


@pytest.mark.parametrize("name", ["campus", "sbm0", "gnp1", "dgnp0"])
@pytest.mark.parametrize("cutoff,best_n", [(3, None), (1, 2), (2, 5), (4, 4)])
def test_greedy_modularity_cutoff_best_n(name, cutoff, best_n):
    G = dict(GRAPHS)[name]
    g, H = pair(G)
    ours = greedy_modularity_communities(g, cutoff=cutoff, best_n=best_n)
    theirs = nx.community.greedy_modularity_communities(H, cutoff=cutoff, best_n=best_n)
    assert as_frozen(ours) == as_frozen(theirs)


def test_greedy_modularity_edge_cases():
    assert greedy_modularity_communities(Graph(nodes=[1, 2])) == [{1}, {2}]
    assert greedy_modularity_communities(Graph()) == []
    # disconnected: merging stops at the components
    two = Graph([(0, 1), (1, 2), (2, 0), (3, 4)])
    assert as_frozen(greedy_modularity_communities(two)) == {frozenset({0, 1, 2}), frozenset({3, 4})}
    two_h = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4)])
    assert as_frozen(greedy_modularity_communities(two, best_n=1)) == {frozenset(range(5))}
    assert as_frozen(greedy_modularity_communities(two, cutoff=2, best_n=2)) == as_frozen(
        nx.community.greedy_modularity_communities(two_h, cutoff=2, best_n=2)
    )
    for kwargs in ({"cutoff": 0}, {"cutoff": 6}, {"best_n": 0}, {"best_n": 9}, {"cutoff": 3, "best_n": 2}):
        with pytest.raises(ValueError):
            greedy_modularity_communities(two, **kwargs)


# ---------------------------------------------------------------------- #
# label propagation and fluid communities
# ---------------------------------------------------------------------- #
@cases(GRAPHS)
def test_label_propagation_partition(G):
    g, _ = pair(G)
    comms = label_propagation_communities(g, seed=3)
    assert_partition(g, comms)
    assert comms == label_propagation_communities(g, seed=3)
    assert_partition(g, label_propagation_communities(g, seed=3, weight="weight"))


def test_label_propagation_quality():
    G = sbm([20, 25, 30], 0.5, 0.01, seed=1)
    g, _ = pair(G)
    blocks = [{n for n in G if G.nodes[n]["block"] == b} for b in range(3)]
    comms = label_propagation_communities(g, seed=0)
    assert modularity(g, comms) >= 0.95 * modularity(g, blocks)
    cliques = nx.connected_caveman_graph(4, 6)
    c, _ = pair(cliques)
    assert len(label_propagation_communities(c, seed=1)) == 4


def test_label_propagation_edge_cases():
    assert label_propagation_communities(Graph()) == []
    assert label_propagation_communities(Graph(nodes=[2, 1])) == [{2}, {1}]
    d = DiGraph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert as_frozen(label_propagation_communities(d, seed=0)) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}
    g, _ = pair(nx.gnp_random_graph(40, 0.1, seed=2))
    assert_partition(g, label_propagation_communities(g, seed=0, max_iter=1))


def test_asyn_fluid():
    G = sbm([20, 20, 20], 0.5, 0.02, seed=3)
    g, H = pair(G)
    comms = asyn_fluid_communities(g, 3, seed=0)
    assert_partition(g, comms)
    assert len(comms) == 3
    assert comms == asyn_fluid_communities(g, 3, seed=0)
    ref = [modularity(g, list(nx.community.asyn_fluidc(H, 3, seed=s))) for s in range(3)]
    assert modularity(g, comms) >= 0.9 * max(ref)
    assert asyn_fluid_communities(g, 1, seed=0) == [set(G)]
    with pytest.raises(ValueError):
        asyn_fluid_communities(g, 0)
    with pytest.raises(ValueError):
        asyn_fluid_communities(g, 61)
    with pytest.raises(GraphTypeError):
        asyn_fluid_communities(DiGraph([(1, 2)]), 1)
    with pytest.raises(NotConnected):
        asyn_fluid_communities(Graph([(1, 2), (3, 4)]), 2)


# ---------------------------------------------------------------------- #
# Girvan–Newman
# ---------------------------------------------------------------------- #
GN_GRAPHS = (
    [("campus", campus())]
    + [(f"gnp{s}", nx.gnp_random_graph(25, 0.15, seed=s)) for s in range(3)]
    + [("dgnp", nx.gnp_random_graph(20, 0.15, seed=4, directed=True))]
    + [("loops", nx.Graph([(0, 0), (0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]))]
)


@cases(GN_GRAPHS)
def test_girvan_newman_matches_networkx(G):
    g, H = pair(G)
    ours = list(islice(girvan_newman(g), 6))
    theirs = list(islice(nx.community.girvan_newman(H), 6))
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        assert_partition(g, a)
        assert as_frozen(a) == as_frozen(b)
    assert g.num_edges == H.number_of_edges()  # the input is left untouched


def test_girvan_newman_edge_cases_and_custom_edge():
    assert list(girvan_newman(Graph(nodes=[1, 2]))) == [[{1}, {2}]]
    path = Graph([(0, 1), (1, 2), (2, 3)])
    splits = list(girvan_newman(path))
    assert as_frozen(splits[0]) == {frozenset({0, 1}), frozenset({2, 3})}
    assert splits[-1] == [{0}, {1}, {2}, {3}]
    last_edge = list(girvan_newman(path, most_valuable_edge=lambda h: list(h.edges)[-1]))
    assert as_frozen(last_edge[0]) == {frozenset({0, 1, 2}), frozenset({3})}
