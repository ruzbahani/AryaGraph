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

"""Traversal and shortest-path algorithms, checked against networkx."""

from __future__ import annotations

import itertools
import math
import random

import networkx as nx
import numpy as np
import pytest

from aryagraph import DAG, DiGraph, Graph, NegativeCycleError, NodeNotFound, NoPath
from aryagraph.algorithms import paths as P
from aryagraph.algorithms import traversal as T

SEEDS = range(6)


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def pair(nxg: nx.Graph) -> tuple[nx.Graph, Graph]:
    """The same graph twice (networkx and aryagraph), built by one replay so neighbor orders agree."""
    nodes = list(nxg.nodes(data=True))
    edges = list(nxg.edges(data=True))
    h = nxg.__class__()
    h.add_nodes_from((n, dict(d)) for n, d in nodes)
    h.add_edges_from((u, v, dict(d)) for u, v, d in edges)
    g = DiGraph() if nxg.is_directed() else Graph()
    g.add_nodes((n, dict(d)) for n, d in nodes)
    g.add_edges((u, v, dict(d)) for u, v, d in edges)
    return h, g


def rand(seed: int, n: int = 40, p: float = 0.1, directed: bool = False, weights: str | None = None) -> tuple[nx.Graph, Graph]:
    nxg = nx.gnp_random_graph(n, p, seed=seed, directed=directed)
    rng = random.Random(seed)
    if weights == "int":
        for u, v in nxg.edges:
            nxg[u][v]["weight"] = rng.randint(1, 9)
    elif weights == "float":
        for u, v in nxg.edges:
            nxg[u][v]["weight"] = rng.uniform(0.5, 10.0)
    elif weights == "potential":  # negative arcs but no negative cycle
        pot = {x: rng.randint(0, 12) for x in nxg}
        for u, v in nxg.edges:
            nxg[u][v]["weight"] = rng.randint(0, 6) + pot[u] - pot[v]
    return pair(nxg)


def path_weight(g: Graph, path: list, weight: str | None = "weight") -> float:
    total = 0
    for u, v in zip(path, path[1:]):
        assert g.has_edge(u, v), f"{u}->{v} is not an edge"
        total += 1 if weight is None else g.edges[u, v].get(weight, 1)
    return total


# ---------------------------------------------------------------------- #
# breadth-first search
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_bfs_matches_networkx(seed, directed):
    h, g = rand(seed, directed=directed)
    for s in (0, 7, 23):
        assert T.bfs_edges(g, s) == list(nx.bfs_edges(h, s))
        assert T.bfs_order(g, s) == [s, *(v for _, v in nx.bfs_edges(h, s))]
        assert T.bfs_edges(g, s, depth_limit=2) == list(nx.bfs_edges(h, s, depth_limit=2))
        if directed:
            assert T.bfs_edges(g, s, reverse=True) == list(nx.bfs_edges(h, s, reverse=True))
        tree = T.bfs_tree(g, s)
        assert isinstance(tree, DiGraph)
        assert list(tree.edges) == list(nx.bfs_tree(h, s).edges)
        assert set(tree.nodes) == set(nx.bfs_tree(h, s).nodes)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_bfs_layers_matches_networkx(seed, directed):
    h, g = rand(seed, directed=directed)
    assert T.bfs_layers(g, 0) == list(nx.bfs_layers(h, 0))
    # networkx starts from set(sources) (arbitrary order); aryagraph keeps the given order
    layers = T.bfs_layers(g, [17, 3, 9])
    assert layers[0] == [17, 3, 9]
    assert [set(x) for x in layers] == [set(x) for x in nx.bfs_layers(h, [3, 9, 17])]


def test_bfs_small_cases():
    g = Graph([(0, 1), (0, 2), (1, 3)])
    assert T.bfs_order(g, 0) == [0, 1, 2, 3]
    assert T.bfs_layers(g, 0) == [[0], [1, 2], [3]]
    single = Graph(nodes=["a"])
    assert T.bfs_order(single, "a") == ["a"]
    assert T.bfs_edges(single, "a") == []
    assert T.bfs_layers(single, "a") == [["a"]]
    # a tuple node is one source, not an iterable of sources
    tg = Graph([((0, 1), "x")])
    assert T.bfs_layers(tg, (0, 1)) == [[(0, 1)], ["x"]]
    with pytest.raises(NodeNotFound):
        T.bfs_order(g, 99)
    with pytest.raises(NodeNotFound):
        T.bfs_layers(g, [0, 99])


def test_bfs_tree_copies_attributes_and_reverse_orientation():
    g = DiGraph([("a", "b", {"w": 1}), ("c", "b", {"w": 2})], name="demo")
    g.nodes["a"]["color"] = "red"
    tree = T.bfs_tree(g, "b", reverse=True)
    assert set(tree.edges) == {("b", "a"), ("b", "c")}
    assert tree.edges["b", "c"] == {"w": 2}
    assert tree.nodes["a"] == {"color": "red"}
    assert tree.nodes["a"] is not g.nodes["a"]
    assert tree.name == "demo"


def test_bfs_self_loops_are_harmless():
    g = DiGraph([(0, 0), (0, 1), (1, 1), (1, 2)])
    assert T.bfs_order(g, 0) == [0, 1, 2]
    assert T.dfs_preorder(g, 0) == [0, 1, 2]


# ---------------------------------------------------------------------- #
# depth-first search
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_dfs_matches_networkx(seed, directed):
    h, g = rand(seed, directed=directed)
    for s in (None, 0, 11):
        assert T.dfs_preorder(g, s) == list(nx.dfs_preorder_nodes(h, s))
        assert T.dfs_postorder(g, s) == list(nx.dfs_postorder_nodes(h, s))
        assert T.dfs_edges(g, s) == list(nx.dfs_edges(h, s))
        tree = T.dfs_tree(g, s)
        ref = nx.dfs_tree(h, s)
        assert set(tree.edges) == set(ref.edges)
        assert list(tree.nodes) == T.dfs_preorder(g, s)  # aryagraph lists tree nodes in preorder
        assert set(tree.nodes) == set(ref.nodes)
    for s in (0, 5):
        assert T.dfs_preorder(g, s, depth_limit=3) == list(nx.dfs_preorder_nodes(h, s, depth_limit=3))
        assert T.dfs_edges(g, s, depth_limit=3) == list(nx.dfs_edges(h, s, depth_limit=3))


def test_dfs_postorder_depth_limit_keeps_cut_off_nodes():
    g = Graph([(0, 1), (1, 2), (2, 3)])
    assert T.dfs_preorder(g, 0, depth_limit=2) == [0, 1, 2]
    assert T.dfs_postorder(g, 0, depth_limit=2) == [2, 1, 0]
    assert T.dfs_edges(g, 0, depth_limit=2) == [(0, 1), (1, 2)]


def test_dfs_forest_and_edge_cases():
    g = Graph([(0, 1), (2, 3)])
    g.add_node(4)
    assert T.dfs_preorder(g) == [0, 1, 2, 3, 4]
    assert T.dfs_postorder(g) == [1, 0, 3, 2, 4]
    forest = T.dfs_tree(g)
    assert list(forest.nodes) == [0, 1, 2, 3, 4]
    assert list(forest.edges) == [(0, 1), (2, 3)]
    empty = Graph()
    assert T.dfs_preorder(empty) == [] and T.dfs_edges(empty) == []
    assert len(T.dfs_tree(empty)) == 0
    with pytest.raises(NodeNotFound):
        T.dfs_preorder(g, "missing")


def test_dfs_postorder_reversed_is_topological():
    dag = DAG([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    order = T.dfs_postorder(dag)[::-1]
    pos = {n: i for i, n in enumerate(order)}
    assert all(pos[u] < pos[v] for u, v in dag.edges)


def test_traversals_are_iterative():
    n = 30_000
    g = DiGraph((i, i + 1) for i in range(n - 1))
    assert len(T.dfs_preorder(g, 0)) == n
    assert T.dfs_postorder(g, 0)[0] == n - 1
    assert len(T.descendants(g, 0)) == n - 1
    assert len(T.bfs_layers(g, 0)) == n


# ---------------------------------------------------------------------- #
# reachability
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_descendants_ancestors_match_networkx(seed, directed):
    h, g = rand(seed, p=0.05, directed=directed)
    for v in list(g)[::5]:
        assert T.descendants(g, v) == nx.descendants(h, v)
        assert T.ancestors(g, v) == nx.ancestors(h, v)


def test_descendants_edge_cases():
    g = Graph([(0, 1), (1, 2)])
    g.add_node(3)
    assert T.descendants(g, 0) == {1, 2} == T.ancestors(g, 0)
    assert T.descendants(g, 3) == set()
    loop = DiGraph([(0, 0), (0, 1)])
    assert T.descendants(loop, 0) == {1}
    assert T.ancestors(loop, 0) == set()
    with pytest.raises(NodeNotFound):
        T.ancestors(g, 42)


# ---------------------------------------------------------------------- #
# single-source shortest paths
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_unweighted_shortest_paths(seed, directed):
    h, g = rand(seed, directed=directed)
    ref = nx.single_source_shortest_path_length(h, 0)
    lengths = P.shortest_path_length(g, 0)
    assert dict(lengths) == dict(ref)
    all_paths = P.shortest_path(g, 0)
    assert set(all_paths) == set(ref)
    for t, path in all_paths.items():
        assert path[0] == 0 and path[-1] == t
        assert path_weight(g, path, None) == ref[t]
    for t in list(ref)[:10]:
        assert len(P.shortest_path(g, 0, t)) - 1 == ref[t]
        assert P.shortest_path_length(g, 0, t) == ref[t]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_weighted_shortest_paths(seed, directed):
    h, g = rand(seed, directed=directed, weights="float")
    ref = nx.single_source_dijkstra_path_length(h, 0)
    got = P.shortest_path_length(g, 0, weight="weight")
    assert set(got) == set(ref)
    for t in ref:
        assert got[t] == pytest.approx(ref[t])
    for t, path in P.shortest_path(g, 0, weight="weight").items():
        assert path_weight(g, path) == pytest.approx(ref[t])
    for method in ("dijkstra", "bellman-ford", "bellman_ford"):
        assert P.shortest_path_length(g, 0, 5 if 5 in ref else 0, weight="weight", method=method) == pytest.approx(ref.get(5, 0))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_dijkstra_predecessors_match_networkx(seed, directed):
    h, g = rand(seed, directed=directed, weights="int")
    ref_pred, ref_dist = nx.dijkstra_predecessor_and_distance(h, 0)
    dist, pred = P.dijkstra(g, 0)
    assert dict(dist) == dict(ref_dist)
    assert {v: set(p) for v, p in pred.items()} == {v: set(p) for v, p in ref_pred.items()}
    # cutoff
    ref_cut = nx.single_source_dijkstra_path_length(h, 0, cutoff=8)
    dist_cut, _ = P.dijkstra(g, 0, cutoff=8)
    assert dict(dist_cut) == dict(ref_cut)


@pytest.mark.parametrize("seed", SEEDS)
def test_dijkstra_with_target_keeps_complete_predecessors(seed):
    h, g = rand(seed, directed=True, weights="int")
    ref_pred, ref_dist = nx.dijkstra_predecessor_and_distance(h, 0)
    for t in list(ref_dist)[1:8]:
        dist, pred = P.dijkstra(g, 0, target=t)
        assert dist[t] == ref_dist[t]
        assert set(pred[t]) == set(ref_pred[t])


def test_dijkstra_zero_weight_ties_are_all_recorded():
    g = DiGraph([("s", "a", 1), ("s", "b", 1), ("a", "b", 0), ("b", "t", 1), ("a", "t", 1)])
    dist, pred = P.dijkstra(g, "s", target="t")
    assert dist["t"] == 2
    assert set(pred["t"]) == {"a", "b"}
    assert set(pred["b"]) == {"s", "a"}
    assert sorted(P.all_shortest_paths(g, "s", "t", weight="weight")) == [["s", "a", "b", "t"], ["s", "a", "t"], ["s", "b", "t"]]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_bellman_ford_matches_networkx(seed, directed):
    weights = "potential" if directed else "int"
    h, g = rand(seed, directed=directed, weights=weights)
    ref_pred, ref_dist = nx.bellman_ford_predecessor_and_distance(h, 0)
    dist, pred = P.bellman_ford(g, 0)
    assert dict(dist) == dict(ref_dist)
    assert {v: set(p) for v, p in pred.items()} == {v: set(p) for v, p in ref_pred.items()}
    # 'auto' picks Bellman–Ford when a weight is negative
    auto = P.shortest_path_length(g, 0, weight="weight")
    assert dict(auto) == dict(ref_dist)
    for t, path in P.shortest_path(g, 0, weight="weight").items():
        assert path_weight(g, path) == ref_dist[t]


def _assert_negative_cycle(g: Graph, cycle: list, weight: str = "weight") -> None:
    assert len(cycle) >= 2 and cycle[0] == cycle[-1]
    assert len(set(cycle[:-1])) == len(cycle) - 1, "cycle must be simple"
    assert path_weight(g, cycle, weight) < 0


@pytest.mark.parametrize("seed", range(12))
def test_negative_cycles_are_reported_exactly(seed):
    rng = random.Random(seed)
    nxg = nx.gnp_random_graph(25, 0.12, seed=seed, directed=True)
    for u, v in nxg.edges:
        nxg[u][v]["weight"] = rng.randint(-3, 12)
    h, g = pair(nxg)
    reachable_negative = nx.negative_edge_cycle(h.subgraph(nx.descendants(h, 0) | {0}).copy())
    if reachable_negative:
        with pytest.raises(NegativeCycleError) as info:
            P.bellman_ford(g, 0)
        _assert_negative_cycle(g, info.value.cycle)
    else:
        dist, _ = P.bellman_ford(g, 0)
        assert dict(dist) == nx.single_source_bellman_ford_path_length(h, 0)
    if nx.negative_edge_cycle(h):
        with pytest.raises(NegativeCycleError) as info:
            P.floyd_warshall(g)
        _assert_negative_cycle(g, info.value.cycle)
        with pytest.raises(NegativeCycleError):
            P.all_pairs_shortest_path_length(g, weight="weight")


def test_negative_cycle_small_cases():
    tri = DiGraph([("a", "b", 1), ("b", "c", -3), ("c", "a", 1), ("s", "a", 1)])
    with pytest.raises(NegativeCycleError) as info:
        P.bellman_ford(tri, "s")
    assert info.value.cycle[:-1] in (["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"])
    loop = DiGraph([(0, 1, 1), (1, 1, -1)])
    with pytest.raises(NegativeCycleError) as info:
        P.bellman_ford(loop, 0)
    assert info.value.cycle == [1, 1]
    und = Graph([(0, 1, 2), (1, 2, -1)])  # an undirected negative edge is a 2-cycle
    with pytest.raises(NegativeCycleError) as info:
        P.shortest_path(und, 0, 2, weight="weight")
    assert sorted(info.value.cycle[:-1]) == [1, 2]
    with pytest.raises(NegativeCycleError):
        P.floyd_warshall(und)
    # a negative cycle not reachable from the source is irrelevant
    far = DiGraph([("s", "t", 1), ("x", "y", -2), ("y", "x", 1)])
    assert P.bellman_ford(far, "s")[0] == {"s": 0, "t": 1}


def test_path_errors():
    g = DiGraph([(0, 1), (1, 2)])
    g.add_node(3)
    with pytest.raises(NodeNotFound):
        P.shortest_path(g, 0, 9)
    with pytest.raises(NodeNotFound):
        P.shortest_path(g, 9)
    with pytest.raises(NoPath):
        P.shortest_path(g, 2, 0)
    with pytest.raises(NoPath):
        P.shortest_path_length(g, 0, 3)
    with pytest.raises(NoPath):
        P.astar_path(g, 0, 3)
    with pytest.raises(ValueError):
        P.shortest_path(g, 0, 2, method="magic")
    with pytest.raises(ValueError):
        P.shortest_path(g, 0, 2, weight="weight", method="bfs")
    neg = DiGraph([(0, 1, -1)])
    with pytest.raises(ValueError):
        P.dijkstra(neg, 0)
    assert P.shortest_path(g, 0, 0) == [0]
    assert P.shortest_path_length(g, 3) == {3: 0}


def test_weight_callable_can_hide_edges():
    g = DiGraph([("a", "b", {"blocked": True}), ("a", "c", 1), ("c", "b", 1)])
    w = lambda u, v, d: None if d.get("blocked") else d.get("weight", 1)  # noqa: E731
    assert P.shortest_path(g, "a", "b", weight=w) == ["a", "c", "b"]
    assert P.dijkstra(g, "a", weight=w)[0]["b"] == 2
    assert P.bellman_ford(g, "a", weight=w)[0]["b"] == 2


# ---------------------------------------------------------------------- #
# A*
# ---------------------------------------------------------------------- #
def test_astar_on_grid_with_manhattan_heuristic():
    nxg = nx.grid_2d_graph(12, 12)
    rng = random.Random(3)
    for u, v in nxg.edges:
        nxg[u][v]["weight"] = rng.randint(1, 4)
    h, g = pair(nxg)
    manhattan = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])  # noqa: E731
    for s, t in [((0, 0), (11, 11)), ((3, 7), (10, 1)), ((5, 5), (5, 5))]:
        path = P.astar_path(g, s, t, manhattan)
        assert path[0] == s and path[-1] == t
        assert path_weight(g, path) == nx.astar_path_length(h, s, t, manhattan)


@pytest.mark.parametrize("seed", SEEDS)
def test_astar_with_inconsistent_admissible_heuristic(seed):
    h, g = rand(seed, n=50, p=0.12, directed=True, weights="int")
    rng = random.Random(seed)
    target = 49
    to_target = nx.single_source_dijkstra_path_length(h.reverse(), target)
    if 0 not in to_target:
        with pytest.raises(NoPath):
            P.astar_path(g, 0, target)
        return
    scale = {v: rng.random() for v in h}
    heuristic = lambda u, t: to_target.get(u, 0) * scale[u]  # noqa: E731  admissible, not consistent
    path = P.astar_path(g, 0, target, heuristic)
    assert path_weight(g, path) == to_target[0]
    assert path_weight(g, P.astar_path(g, 0, target)) == to_target[0]


# ---------------------------------------------------------------------- #
# path enumeration
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_all_shortest_paths_match_networkx(seed, directed):
    h, g = rand(seed, n=30, p=0.15, directed=directed)
    hw, gw = rand(seed, n=30, p=0.15, directed=directed, weights="int")
    for t in (7, 19, 29):
        if nx.has_path(h, 0, t):
            assert {tuple(p) for p in P.all_shortest_paths(g, 0, t)} == {tuple(p) for p in nx.all_shortest_paths(h, 0, t)}
            got = {tuple(p) for p in P.all_shortest_paths(gw, 0, t, weight="weight")}
            assert got == {tuple(p) for p in nx.all_shortest_paths(hw, 0, t, weight="weight")}
        else:
            with pytest.raises(NoPath):
                P.all_shortest_paths(g, 0, t)


def test_all_shortest_paths_trivial_and_errors():
    g = Graph([(0, 1), (1, 2), (0, 3), (3, 2)])
    assert sorted(P.all_shortest_paths(g, 0, 2)) == [[0, 1, 2], [0, 3, 2]]
    assert list(P.all_shortest_paths(g, 1, 1)) == [[1]]
    with pytest.raises(NodeNotFound):
        P.all_shortest_paths(g, 0, 9)  # raised eagerly, before iteration


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_all_simple_paths_match_networkx(seed, directed):
    h, g = rand(seed, n=11, p=0.3, directed=directed)
    for s, t in [(0, 10), (2, 7), (5, 5)]:
        for cutoff in (None, 3):
            got = list(P.all_simple_paths(g, s, t, cutoff=cutoff))
            ref = list(nx.all_simple_paths(h, s, t, cutoff=cutoff))
            assert sorted(map(tuple, got)) == sorted(map(tuple, ref))
            assert len({tuple(p) for p in got}) == len(got)


def test_all_simple_paths_small():
    g = DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 3)])
    assert list(P.all_simple_paths(g, 0, 3)) == [[0, 1, 2, 3], [0, 1, 3], [0, 2, 3]]
    assert list(P.all_simple_paths(g, 0, 3, cutoff=2)) == [[0, 1, 3], [0, 2, 3]]
    assert list(P.all_simple_paths(g, 3, 0)) == []
    assert list(P.all_simple_paths(g, 1, 1)) == [[1]]
    with pytest.raises(NodeNotFound):
        P.all_simple_paths(g, 0, 99)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_k_shortest_paths_match_networkx(seed, directed):
    h, g = rand(seed, n=25, p=0.2, directed=directed, weights="int")
    for t in (9, 24):
        if not nx.has_path(h, 0, t):
            with pytest.raises(NoPath):
                P.k_shortest_paths(g, 0, t, 5)
            continue
        got = P.k_shortest_paths(g, 0, t, 8)
        ref = list(itertools.islice(nx.shortest_simple_paths(h, 0, t, weight="weight"), 8))
        assert [path_weight(g, p) for p in got] == [path_weight(g, p) for p in ref]
        assert len({tuple(p) for p in got}) == len(got)
        for p in got:
            assert p[0] == 0 and p[-1] == t and len(set(p)) == len(p)
        unweighted = P.k_shortest_paths(g, 0, t, 4, weight=None)
        ref_u = list(itertools.islice(nx.shortest_simple_paths(h, 0, t), 4))
        assert [len(p) for p in unweighted] == [len(p) for p in ref_u]


def test_k_shortest_paths_small():
    g = DiGraph([("s", "a", 1), ("a", "t", 1), ("s", "t", 3), ("s", "b", 2), ("b", "t", 2)])
    assert P.k_shortest_paths(g, "s", "t", 10) == [["s", "a", "t"], ["s", "t"], ["s", "b", "t"]]
    assert P.k_shortest_paths(g, "s", "s", 3) == [["s"]]
    with pytest.raises(ValueError):
        P.k_shortest_paths(g, "s", "t", 0)


# ---------------------------------------------------------------------- #
# all pairs
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_all_pairs_lengths_match_networkx(seed, directed):
    h, g = rand(seed, n=25, p=0.12, directed=directed)
    got = P.all_pairs_shortest_path_length(g)
    assert {u: dict(d) for u, d in got.items()} == dict(nx.all_pairs_shortest_path_length(h))
    hw, gw = rand(seed, n=25, p=0.12, directed=directed, weights="float")
    got = P.all_pairs_shortest_path_length(gw, weight="weight")
    ref = dict(nx.all_pairs_dijkstra_path_length(hw))
    assert set(got) == set(ref)
    for u in ref:
        assert set(got[u]) == set(ref[u])
        for v in ref[u]:
            assert got[u][v] == pytest.approx(ref[u][v])


@pytest.mark.parametrize("seed", SEEDS)
def test_all_pairs_with_negative_weights_uses_johnson(seed):
    h, g = rand(seed, n=25, p=0.15, directed=True, weights="potential")
    got = P.all_pairs_shortest_path_length(g, weight="weight")
    ref = dict(nx.all_pairs_bellman_ford_path_length(h))
    assert {u: dict(d) for u, d in got.items()} == ref


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weights", [None, "float", "potential"])
def test_floyd_warshall_matches_networkx(seed, directed, weights):
    if weights == "potential" and not directed:
        weights = "int"
    h, g = rand(seed, n=30, p=0.1, directed=directed, weights=weights)
    dmat, nodes = P.floyd_warshall(g, weight="weight" if weights else None)
    assert nodes == list(h)
    ref = nx.floyd_warshall_numpy(h, nodelist=nodes, weight="weight" if weights else "none")
    assert np.allclose(dmat, ref)


def test_floyd_warshall_edge_cases():
    dmat, nodes = P.floyd_warshall(Graph())
    assert dmat.shape == (0, 0) and nodes == []
    g = DiGraph([(0, 1, 2), (1, 1, 5)])
    g.add_node(2)
    dmat, _ = P.floyd_warshall(g)
    assert dmat.tolist() == [[0, 2, math.inf], [math.inf, 0, math.inf], [math.inf, math.inf, 0]]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_has_path_matches_networkx(seed, directed):
    h, g = rand(seed, p=0.04, directed=directed)
    for s, t in itertools.product(range(0, 40, 7), range(0, 40, 5)):
        assert P.has_path(g, s, t) == nx.has_path(h, s, t)
    with pytest.raises(NodeNotFound):
        P.has_path(g, 0, "nope")
