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

"""Flow, spanning trees, matchings and coloring, checked against networkx."""

from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from aryagraph import DiGraph, Graph, GraphTypeError, NodeNotFound
from aryagraph.algorithms import coloring as K
from aryagraph.algorithms import flow as F
from aryagraph.algorithms import matching as M
from aryagraph.algorithms import spanning as S

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


def weighted(seed: int, n: int = 40, p: float = 0.1, directed: bool = False, attr: str = "weight", kind: str = "int", lo: int = 1, hi: int = 20) -> tuple[nx.Graph, Graph]:
    nxg = nx.gnp_random_graph(n, p, seed=seed, directed=directed)
    rng = random.Random(seed)
    for u, v in nxg.edges:
        nxg[u][v][attr] = rng.randint(lo, hi) if kind == "int" else rng.uniform(lo, hi)
    return pair(nxg)


# ---------------------------------------------------------------------- #
# maximum flow / minimum cut
# ---------------------------------------------------------------------- #
def assert_valid_flow(g: Graph, s, t, result: F.FlowResult, capacity: str = "capacity") -> None:
    flow = result.flow
    for u, v, d in g.edges.data():
        if u == v:
            continue
        cap = d.get(capacity, math.inf)
        if g.directed:
            assert -1e-9 <= flow[u][v] <= cap + 1e-9
        else:
            assert flow[u][v] >= 0 and flow[v][u] >= 0 and min(flow[u][v], flow[v][u]) == 0
            assert max(flow[u][v], flow[v][u]) <= cap + 1e-9
    net = {n: 0.0 for n in g}
    for u, row in flow.items():
        for v, f in row.items():
            net[u] -= f
            net[v] += f
    for n in g:
        if n not in (s, t):
            assert net[n] == pytest.approx(0, abs=1e-9)
    assert net[t] == pytest.approx(result.value)
    assert -net[s] == pytest.approx(result.value)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("directed", [False, True])
def test_maximum_flow_matches_networkx(seed, directed):
    h, g = weighted(seed, n=40, p=0.12, directed=directed, attr="capacity")
    s, t = 0, 39
    result = F.maximum_flow(g, s, t)
    assert result.value == nx.maximum_flow_value(h, s, t)
    assert isinstance(result.value, int)
    assert_valid_flow(g, s, t, result)
    cut = F.minimum_cut(g, s, t)
    ref_value, (ref_s, ref_t) = nx.minimum_cut(h, s, t)
    assert cut.value == ref_value
    assert s in cut.source_side and t in cut.sink_side
    assert cut.source_side | cut.sink_side == set(g) and not cut.source_side & cut.sink_side
    assert sum(g.edges[u, v]["capacity"] for u, v in cut.cut_edges) == cut.value
    for u, v in cut.cut_edges:
        assert u in cut.source_side and v in cut.sink_side


@pytest.mark.parametrize("seed", SEEDS)
def test_maximum_flow_float_capacities(seed):
    h, g = weighted(seed, n=35, p=0.15, directed=True, attr="capacity", kind="float", lo=0, hi=5)
    result = F.maximum_flow(g, 0, 34)
    assert result.value == pytest.approx(nx.maximum_flow_value(h, 0, 34))
    assert_valid_flow(g, 0, 34, result)
    cut = F.minimum_cut(g, 0, 34)
    assert sum(g.edges[u, v]["capacity"] for u, v in cut.cut_edges) == pytest.approx(result.value)


@pytest.mark.parametrize("seed", SEEDS)
def test_maximum_flow_missing_capacity_is_infinite(seed):
    h, g = weighted(seed, n=30, p=0.15, directed=True, attr="capacity")
    rng = random.Random(seed)
    for u, v in list(h.edges):
        if rng.random() < 0.3:
            del h[u][v]["capacity"]
            del g.edges[u, v]["capacity"]
    try:
        ref = nx.maximum_flow_value(h, 0, 29)
    except nx.NetworkXUnbounded:
        with pytest.raises(ValueError, match="unbounded"):
            F.maximum_flow(g, 0, 29)
        return
    result = F.maximum_flow(g, 0, 29)
    assert result.value == ref
    assert_valid_flow(g, 0, 29, result)
    assert F.minimum_cut(g, 0, 29).value == ref


def test_maximum_flow_small_cases():
    g = DiGraph([("s", "a", {"capacity": 3}), ("s", "b", {"capacity": 2}), ("a", "b", {"capacity": 1}),
                 ("a", "t", {"capacity": 2}), ("b", "t", {"capacity": 3})])
    r = F.maximum_flow(g, "s", "t")
    assert r.value == 5
    assert r.flow["s"] == {"a": 3, "b": 2}
    cut = F.minimum_cut(g, "s", "t")
    assert cut.value == 5 and cut.source_side == {"s"} and cut.cut_edges == [("s", "a"), ("s", "b")]
    # antiparallel arcs never both carry flow
    anti = DiGraph([(0, 1, {"capacity": 4}), (1, 0, {"capacity": 4}), (1, 2, {"capacity": 2})])
    r = F.maximum_flow(anti, 0, 2)
    assert r.value == 2 and r.flow[1][0] == 0
    # undirected edges carry flow either way
    und = Graph([("s", "a", {"capacity": 2}), ("a", "b", {"capacity": 5}), ("t", "b", {"capacity": 3})])
    r = F.maximum_flow(und, "s", "t")
    assert r.value == 2 and r.flow["b"]["t"] == 2 and r.flow["t"]["b"] == 0
    assert F.minimum_cut(und, "t", "s").cut_edges == [("a", "s")]
    # disconnected sink, self-loops and a capacity callable
    lone = DiGraph([(0, 0, {"capacity": 9}), (0, 1, {"capacity": 1})])
    lone.add_node(2)
    assert F.maximum_flow(lone, 0, 2).value == 0
    assert F.maximum_flow(lone, 0, 1, capacity=lambda u, v, d: 7).value == 7
    assert "value=5" in repr(F.maximum_flow(g, "s", "t"))


def test_flow_errors():
    g = DiGraph([(0, 1), (1, 2, {"capacity": 1})])
    with pytest.raises(ValueError, match="unbounded"):
        F.maximum_flow(DiGraph([(0, 1), (1, 2)]), 0, 2)
    assert F.maximum_flow(g, 0, 2).value == 1  # the infinite arc is not the bottleneck
    with pytest.raises(ValueError):
        F.maximum_flow(g, 0, 0)
    with pytest.raises(NodeNotFound):
        F.maximum_flow(g, 0, 9)
    with pytest.raises(ValueError):
        F.maximum_flow(DiGraph([(0, 1, {"capacity": -1})]), 0, 1)
    with pytest.raises(ValueError):
        F.minimum_cut(DiGraph([(0, 1, {"capacity": float("nan")})]), 0, 1)


# ---------------------------------------------------------------------- #
# spanning trees
# ---------------------------------------------------------------------- #
def total_weight(g: Graph, weight: str = "weight") -> float:
    return sum(d.get(weight, 1) for _, _, d in g.edges.data())


def assert_spanning_forest(tree: Graph, g: Graph) -> None:
    """Same nodes, only edges of g, acyclic, one tree per component of g."""
    assert list(tree) == list(g)
    for u, v in tree.edges:
        assert g.has_edge(u, v) and u != v
    assert tree.num_edges == len(g) - nx.number_connected_components(_nx(g))
    assert nx.number_connected_components(_nx(tree)) == nx.number_connected_components(_nx(g))


def _nx(g: Graph) -> nx.Graph:
    h = nx.Graph()
    h.add_nodes_from(g)
    h.add_edges_from(g.edges)
    return h


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("algorithm", ["kruskal", "prim"])
@pytest.mark.parametrize("kind", ["int", "float"])
def test_spanning_trees_match_networkx(seed, algorithm, kind):
    h, g = weighted(seed, n=40, p=0.08, kind=kind)
    g.add_edge(3, 3, weight=-100)  # self-loops never belong to a tree
    mst = S.minimum_spanning_tree(g, algorithm=algorithm)
    assert_spanning_forest(mst, g)
    assert total_weight(mst) == pytest.approx(total_weight(nx.minimum_spanning_tree(h)))
    xst = S.maximum_spanning_tree(g, algorithm=algorithm)
    assert_spanning_forest(xst, g)
    assert total_weight(xst) == pytest.approx(total_weight(nx.maximum_spanning_tree(h)))
    edges = S.minimum_spanning_edges(g, algorithm=algorithm)
    assert sorted(map(frozenset, ((u, v) for u, v, _ in edges)), key=sorted) == sorted(map(frozenset, mst.edges), key=sorted)
    if algorithm == "kruskal":
        ws = [d["weight"] for _, _, d in edges]
        assert ws == sorted(ws)
    for u, v, d in edges:
        assert d is g.edges[u, v]


def test_spanning_tree_details():
    g = Graph([("a", "b", 4), ("b", "c", 1), ("a", "c", 2), ("c", "d", {"weight": 3, "label": "x"})], name="net")
    g.add_node("e", color="blue")
    for algorithm in ("kruskal", "prim"):
        t = S.minimum_spanning_tree(g, algorithm=algorithm)
        assert set(map(frozenset, t.edges)) == {frozenset("bc"), frozenset("ac"), frozenset("cd")}
        assert t.edges["c", "d"] == {"weight": 3, "label": "x"}
        assert t.edges["c", "d"] is not g.edges["c", "d"]
        assert t.nodes["e"] == {"color": "blue"} and t.name == "net"
        assert len(S.minimum_spanning_tree(g, weight=None, algorithm=algorithm).edges) == 3
    assert S.minimum_spanning_edges(Graph()) == []
    assert len(S.minimum_spanning_tree(Graph())) == 0
    hidden = S.minimum_spanning_edges(g, weight=lambda u, v, d: None if d.get("label") else d["weight"])
    assert len(hidden) == 2
    with pytest.raises(GraphTypeError):
        S.minimum_spanning_tree(DiGraph([(0, 1)]))
    with pytest.raises(ValueError):
        S.minimum_spanning_tree(g, algorithm="boruvka!")
    with pytest.raises(ValueError):
        S.minimum_spanning_tree(Graph([(0, 1, float("nan"))]))


def test_union_find():
    uf = S._UnionFind(range(10))
    assert uf.union(0, 1) and uf.union(2, 3) and uf.union(1, 3)
    assert not uf.union(0, 2)
    assert uf.find(3) == uf.find(0) != uf.find(4)


# ---------------------------------------------------------------------- #
# matchings
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_hopcroft_karp_matches_networkx(seed):
    nxg = nx.bipartite.random_graph(25, 30, 0.08, seed=seed)
    h, g = pair(nxg)
    top = [n for n, d in h.nodes(data=True) if d["bipartite"] == 0]
    ref = nx.bipartite.hopcroft_karp_matching(h, top_nodes=top)
    got = M.hopcroft_karp(g, top)
    assert len(got) == len(ref)
    assert all(got[v] == u for u, v in got.items())
    assert M.is_matching(g, got)
    assert list(got) == [n for n in g if n in got]
    # sides are optional; the size is the same
    assert len(M.bipartite_maximum_matching(g)) == len(ref)


def test_hopcroft_karp_small_and_errors():
    g = Graph([("a", 1), ("a", 2), ("b", 1), ("c", 1)])
    m = M.hopcroft_karp(g, ["a", "b", "c"])
    assert len(m) == 4  # two pairs, listed both ways
    assert m["a"] == 2 and m[2] == "a"  # 'a' must take 2, the only node nobody else reaches
    assert m[1] in ("b", "c") and m[m[1]] == 1
    assert M.hopcroft_karp(Graph(), []) == {}
    with pytest.raises(GraphTypeError):
        M.hopcroft_karp(Graph([(0, 1), (1, 2), (2, 0)]))
    with pytest.raises(GraphTypeError):
        M.hopcroft_karp(g, ["a", 1])
    with pytest.raises(NodeNotFound):
        M.hopcroft_karp(g, ["zzz"])
    with pytest.raises(GraphTypeError):
        M.hopcroft_karp(DiGraph([(0, 1)]), [0])


@pytest.mark.parametrize("seed", SEEDS)
def test_maximal_matching(seed):
    _, g = weighted(seed, p=0.1)
    m = M.maximal_matching(g)
    assert M.is_matching(g, m)
    assert M.is_maximal_matching(g, m)
    assert all(g.has_edge(u, v) for u, v in m)


def test_matching_predicates():
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    assert M.is_matching(g, {(0, 1), (2, 3)})
    assert M.is_perfect_matching(g, {(0, 1), (2, 3)})
    assert M.is_matching(g, {0: 1, 1: 0})
    assert not M.is_maximal_matching(g, {0: 1})  # (2, 3) could still be added
    assert M.is_maximal_matching(g, iter([(0, 1), (2, 3)]))
    assert not M.is_matching(g, {(0, 1), (1, 2)})  # node 1 twice
    assert not M.is_matching(g, {(0, 2)})  # not an edge
    assert not M.is_matching(g, {0: 1, 1: 2})  # inconsistent dict
    assert not M.is_perfect_matching(g, {(0, 1)})
    with pytest.raises(NodeNotFound):
        M.is_matching(g, {(0, 9)})


def matching_weight(g: Graph, m, weight: str = "weight") -> float:
    return sum(g.edges[u, v].get(weight, 1) for u, v in m)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("maxcardinality", [False, True])
@pytest.mark.parametrize("kind", ["int", "float"])
def test_max_weight_matching_matches_networkx(seed, maxcardinality, kind):
    h, g = weighted(seed, n=40, p=0.12, kind=kind, lo=1, hi=30)
    got = M.max_weight_matching(g, maxcardinality=maxcardinality)
    ref = nx.max_weight_matching(h, maxcardinality=maxcardinality)
    assert M.is_matching(g, got)
    assert matching_weight(g, got) == pytest.approx(matching_weight(g, ref))
    if maxcardinality:
        assert len(got) == len(ref) == len(nx.max_weight_matching(h, maxcardinality=True, weight="__none__"))
    for u, v in got:
        assert (u, v) in set(g.edges)  # oriented as g.edges reports


@pytest.mark.parametrize("seed", range(150))
def test_max_weight_matching_stress_small_graphs(seed):
    """Many small dense graphs with few distinct weights: blossoms nest and expand constantly."""
    rng = random.Random(seed)
    n = rng.randint(4, 14)
    nxg = nx.gnp_random_graph(n, rng.uniform(0.25, 0.8), seed=seed)
    for u, v in nxg.edges:
        nxg[u][v]["weight"] = rng.choice([1, 2, 3, 5, 8]) if seed % 3 else rng.randint(-5, 40)
    h, g = pair(nxg)
    for maxcard in (False, True):
        got = M.max_weight_matching(g, maxcardinality=maxcard)
        ref = nx.max_weight_matching(h, maxcardinality=maxcard)
        assert M.is_matching(g, got)
        assert matching_weight(g, got) == matching_weight(g, ref)
        if maxcard:
            assert len(got) == len(ref)


def test_max_weight_matching_small_cases():
    g = Graph([(1, 2, 6), (1, 3, 2), (2, 3, 1), (2, 4, 7), (3, 5, 9), (4, 5, 3)])
    assert M.max_weight_matching(g) == {(2, 4), (3, 5)}
    # classic blossom: odd cycle with a stem
    blossom = Graph([(1, 2, 8), (1, 3, 9), (2, 3, 10), (3, 4, 7)])
    assert M.max_weight_matching(blossom) == {(1, 2), (3, 4)}
    assert M.max_weight_matching(Graph()) == set()
    assert M.max_weight_matching(Graph([(0, 0, 5)])) == set()
    neg = Graph([(0, 1, -2), (1, 2, -1)])
    assert M.max_weight_matching(neg) == set()
    assert len(M.max_weight_matching(neg, maxcardinality=True)) == 1
    assert M.max_weight_matching(neg, weight=None) in ({(0, 1)}, {(1, 2)})
    with pytest.raises(GraphTypeError):
        M.max_weight_matching(DiGraph([(0, 1)]))


@pytest.mark.parametrize("seed", SEEDS)
def test_min_weight_matching_matches_networkx(seed):
    h, g = weighted(seed, n=30, p=0.15, lo=1, hi=30)
    got = M.min_weight_matching(g)
    ref = nx.min_weight_matching(h)
    assert M.is_matching(g, got)
    assert len(got) == len(ref)
    assert matching_weight(g, got) == matching_weight(g, ref)


def test_max_weight_matching_large_blossoms():
    n = 1501  # one huge odd cycle: a single blossom with n vertices
    ring = Graph((i, (i + 1) % n, 1) for i in range(n))
    m = M.max_weight_matching(ring)
    assert M.is_matching(ring, m) and len(m) == n // 2
    # a tower of 5-cycles linked base to base: blossoms inside blossoms
    rng = random.Random(0)
    h = nx.Graph()
    for k in range(60):
        ring5 = [5 * k + i for i in range(5)]
        for i in range(5):
            h.add_edge(ring5[i], ring5[(i + 1) % 5], weight=rng.randint(5, 9))
        if k:
            h.add_edge(5 * k, 5 * k - 3, weight=rng.randint(5, 9))
    h, g = pair(h)
    for maxcard in (False, True):
        got = M.max_weight_matching(g, maxcardinality=maxcard)
        assert M.is_matching(g, got)
        assert matching_weight(g, got) == matching_weight(g, nx.max_weight_matching(h, maxcardinality=maxcard))


# ---------------------------------------------------------------------- #
# coloring
# ---------------------------------------------------------------------- #
STRATEGIES = [
    "largest_first",
    "smallest_last",
    "dsatur",
    "saturation_largest_first",
    "independent_set",
    "random_sequential",
    "connected_sequential_bfs",
    "connected_sequential_dfs",
    "insertion",
]


def assert_proper(g: Graph, colors) -> None:
    assert set(colors) == set(g)
    for u, v in g.edges:
        if u != v:
            assert colors[u] != colors[v]
    assert all(isinstance(c, int) and c >= 0 for c in colors.values())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_greedy_color_is_proper(seed, strategy):
    h, g = weighted(seed, n=50, p=0.15)
    g.add_edge(4, 4)  # self-loops are ignored
    colors = K.greedy_color(g, strategy, seed=seed)
    assert_proper(g, colors)
    assert list(colors) == list(g)
    max_deg = max(len(set(g._succ[n]) - {n}) for n in g)
    assert max(colors.values()) + 1 <= max_deg + 1
    if strategy == "smallest_last":
        assert max(colors.values()) + 1 <= max(nx.core_number(h).values()) + 1


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("strategy", ["largest_first", "saturation_largest_first"])
def test_greedy_color_equals_networkx_for_deterministic_orders(seed, strategy):
    h, g = weighted(seed, n=50, p=0.15)
    assert dict(K.greedy_color(g, strategy)) == nx.greedy_color(h, strategy)


@pytest.mark.parametrize("seed", SEEDS)
def test_greedy_color_directed_graphs_use_both_directions(seed):
    h, g = weighted(seed, n=40, p=0.1, directed=True)
    for strategy in ("largest_first", "dsatur", "smallest_last"):
        colors = K.greedy_color(g, strategy)
        assert_proper(g, colors)


def test_greedy_color_small_and_errors():
    bip = Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    assert max(K.greedy_color(bip, "dsatur").values()) == 1
    assert K.greedy_color(Graph()) == {}
    assert K.greedy_color(bip, "random_sequential", seed=3) == K.greedy_color(bip, "random_sequential", seed=3)
    order = K.greedy_color(bip, lambda g: [3, 2, 1, 0])
    assert order[3] == 0 and order[2] == 1
    assert K.greedy_color(bip).name == "color"
    with pytest.raises(ValueError):
        K.greedy_color(bip, "rainbow")
    with pytest.raises(ValueError):
        K.greedy_color(bip, lambda g: [0, 1])
    crown = Graph([(0, 1), (2, 3), (0, 3), (2, 1), (4, 5), (0, 5), (4, 1)])
    assert max(K.greedy_color(crown, "dsatur").values()) == 1  # DSatur is exact on bipartite graphs


@pytest.mark.parametrize("seed", SEEDS)
def test_bipartite_detection_matches_networkx(seed):
    cases = [
        pair(nx.bipartite.random_graph(15, 20, 0.1, seed=seed)),
        pair(nx.gnp_random_graph(30, 0.05, seed=seed)),
        pair(nx.random_labeled_tree(25, seed=seed)),
        pair(nx.gnp_random_graph(20, 0.3, seed=seed, directed=True)),
    ]
    for h, g in cases:
        assert K.is_bipartite(g) == nx.is_bipartite(h)
        if nx.is_bipartite(h):
            top, bottom = K.bipartite_sets(g)
            assert top | bottom == set(g) and not top & bottom
            for u, v in g.edges:
                assert (u in top) != (v in top)
        else:
            with pytest.raises(GraphTypeError, match="odd cycle"):
                K.bipartite_sets(g)


def test_bipartite_small_cases():
    g = Graph([(0, 1), (2, 3)])
    g.add_node(4)
    assert K.bipartite_sets(g) == ({0, 2, 4}, {1, 3})  # first node of each component goes on top
    assert K.is_bipartite(Graph())
    assert not K.is_bipartite(Graph([(0, 0)]))
    assert K._two_coloring(Graph([(0, 0)]))[1] == [0, 0]
    pent = Graph([(i, (i + 1) % 5) for i in range(5)])
    odd = K._two_coloring(pent)[1]
    assert odd[0] == odd[-1] and len(odd) == 6 and all(pent.has_edge(u, v) for u, v in zip(odd, odd[1:]))
