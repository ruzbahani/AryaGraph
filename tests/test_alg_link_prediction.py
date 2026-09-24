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

"""Link prediction scores, checked against networkx."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from aryagraph import DiGraph, Graph
from aryagraph.algorithms.link_prediction import (
    adamic_adar_index,
    common_neighbors,
    jaccard_coefficient,
    predict_links,
    preferential_attachment,
    resource_allocation_index,
)
from aryagraph.core.exceptions import GraphTypeError, NodeNotFound


def to_ag(G: nx.Graph) -> Graph:
    g = Graph()
    g.add_nodes(G.nodes(data=True))
    g.add_edges(G.edges(data=True))
    return g


def with_extras(G: nx.Graph) -> nx.Graph:
    """Add a self-loop and two isolated nodes: the awkward cases for neighbourhood scores."""
    n = max(G) + 1
    G.add_edge(1, 1)
    G.add_nodes_from([n, n + 1])
    return G


GRAPHS = (
    [(f"gnp{s}", nx.gnp_random_graph(30, 0.15, seed=s)) for s in range(3)]
    + [(f"ba{s}", nx.barabasi_albert_graph(35, 2, seed=s)) for s in range(2)]
    + [("extras", with_extras(nx.gnp_random_graph(25, 0.2, seed=6)))]
)
SCORERS = [
    (jaccard_coefficient, nx.jaccard_coefficient),
    (adamic_adar_index, nx.adamic_adar_index),
    (resource_allocation_index, nx.resource_allocation_index),
    (preferential_attachment, nx.preferential_attachment),
]


def cases(graphs):
    return pytest.mark.parametrize("G", [G for _, G in graphs], ids=[n for n, _ in graphs])


def keyed(triples):
    return {frozenset((u, v)): s for u, v, s in triples}


@cases(GRAPHS)
@pytest.mark.parametrize("ours,theirs", SCORERS, ids=[f.__name__ for f, _ in SCORERS])
def test_scores_on_all_non_edges(G, ours, theirs):
    g = to_ag(G)
    got = ours(g)
    expected = keyed(theirs(G))
    assert len(got) == len(expected)
    assert keyed(got).keys() == expected.keys()
    for key, score in keyed(got).items():
        assert score == pytest.approx(expected[key], abs=1e-12)
    # graph order: u comes before v, and pairs appear in row-major order
    index = {v: i for i, v in enumerate(g.nodes)}
    positions = [(index[u], index[v]) for u, v, _ in got]
    assert all(i < j for i, j in positions) and positions == sorted(positions)


@cases(GRAPHS)
@pytest.mark.parametrize("ours,theirs", SCORERS, ids=[f.__name__ for f, _ in SCORERS])
def test_scores_on_given_pairs(G, ours, theirs):
    g = to_ag(G)
    nodes = list(G)
    pairs = [(nodes[i], nodes[-1 - i]) for i in range(len(nodes) // 2)] + list(G.edges)[:5]
    pairs = [(u, v) for u, v in pairs if u != v]
    got = ours(g, pairs)
    assert [(u, v) for u, v, _ in got] == pairs  # order and orientation preserved
    for (_, _, a), (_, _, b) in zip(got, theirs(G, pairs)):
        assert a == pytest.approx(b, abs=1e-12)


@cases(GRAPHS)
def test_common_neighbors(G):
    g = to_ag(G)
    nodes = list(G)
    for u, v in zip(nodes, nodes[1:]):
        cn = common_neighbors(g, u, v)
        assert set(cn) == set(nx.common_neighbors(G, u, v))
        assert cn == [w for w in g.adj[u] if w in set(cn)]  # u's neighbour order


def test_hand_checked_scores():
    # square 0-1-2-3 with diagonal 0-2: pair (1, 3) shares {0, 2}
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    assert common_neighbors(g, 1, 3) == [0, 2]
    assert jaccard_coefficient(g, [(1, 3)]) == [(1, 3, 1.0)]
    assert adamic_adar_index(g, [(1, 3)])[0][2] == pytest.approx(2 / math.log(3))
    assert resource_allocation_index(g, [(1, 3)])[0][2] == pytest.approx(2 / 3)
    assert preferential_attachment(g, [(1, 3)]) == [(1, 3, 4)]
    assert jaccard_coefficient(Graph(nodes=[1, 2])) == [(1, 2, 0.0)]
    assert adamic_adar_index(Graph()) == []


def test_errors():
    g = Graph([(1, 2), (2, 3)])
    with pytest.raises(NodeNotFound):
        jaccard_coefficient(g, [(1, 9)])
    with pytest.raises(NodeNotFound):
        common_neighbors(g, 1, 9)
    with pytest.raises(ValueError):
        adamic_adar_index(g, [(1, 1)])
    for fn in (jaccard_coefficient, adamic_adar_index, resource_allocation_index, preferential_attachment):
        with pytest.raises(GraphTypeError):
            fn(DiGraph([(1, 2)]))
    with pytest.raises(GraphTypeError):
        common_neighbors(DiGraph([(1, 2)]), 1, 2)
    with pytest.raises(ValueError):
        predict_links(g, method="nope")
    with pytest.raises(GraphTypeError):
        predict_links(DiGraph([(1, 2)]))


@cases(GRAPHS)
@pytest.mark.parametrize(
    "method,oracle",
    [
        ("adamic_adar", nx.adamic_adar_index),
        ("resource_allocation", nx.resource_allocation_index),
        ("jaccard", nx.jaccard_coefficient),
        ("preferential_attachment", nx.preferential_attachment),
        ("common_neighbors", lambda G: ((u, v, len(nx.common_neighbors(G, u, v))) for u, v in nx.non_edges(G))),
    ],
)
def test_predict_links_top_k(G, method, oracle):
    g = to_ag(G)
    k = 10
    top = predict_links(g, method=method, k=k)
    assert len(top) == k
    scores = [s for _, _, s in top]
    assert scores == sorted(scores, reverse=True)
    expected = sorted((s for _, _, s in oracle(G)), reverse=True)[:k]
    assert scores == pytest.approx(expected, abs=1e-12)
    assert all(not g.has_edge(u, v) and u != v for u, v, _ in top)


def test_predict_links_padding_and_ties():
    # only one two-hop pair: the rest are zero-score fillers in graph order
    g = Graph([(0, 1), (1, 2)])
    g.add_nodes([3, 4])  # graph order 0, 1, 2, 3, 4
    top = predict_links(g, "jaccard", k=4)
    assert top[0] == (0, 2, 1.0)
    assert [(u, v) for u, v, s in top[1:]] == [(0, 3), (0, 4), (1, 3)] and all(s == 0 for *_, s in top[1:])
    assert predict_links(g, k=0) == []
    assert len(predict_links(g, k=100)) == 8  # every non-edge
