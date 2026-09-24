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

"""Tests for aryagraph.generators: structure, formulas, determinism and networkx equality."""

from __future__ import annotations

import itertools
import math
from collections import Counter

import networkx as nx
import numpy as np
import pytest

from aryagraph import DAG, DiGraph, Graph
from aryagraph import generators as gen
from aryagraph.generators.random import _bernoulli_indices, _pair_index


def edge_set(g) -> set:
    """Edges as a comparable set: frozensets when undirected, tuples when directed."""
    if g.is_directed():
        return set(g.edges())
    return {frozenset(e) for e in g.edges()}


def assert_same_as_nx(g, G, *, ordered: bool = True):
    if ordered:
        assert list(g.nodes) == list(G.nodes)
    else:  # networkx inserts some nodes in edge order; aryagraph numbers them ascending
        assert set(g.nodes) == set(G.nodes)
    assert edge_set(g) == edge_set(G)
    assert g.num_edges == G.number_of_edges()


def is_weakly_connected(g) -> bool:
    if len(g) == 0:
        return True
    start = next(iter(g.nodes))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        nbrs = set(g.succ[u]) | set(g.pred[u])
        for v in nbrs - seen:
            seen.add(v)
            stack.append(v)
    return len(seen) == len(g)


# ---------------------------------------------------------------------- #
# classic families
# ---------------------------------------------------------------------- #
CLASSIC_CASES = [
    (gen.empty_graph, (5,), nx.empty_graph, (5,)),
    (gen.path_graph, (7,), nx.path_graph, (7,)),
    (gen.cycle_graph, (6,), nx.cycle_graph, (6,)),
    (gen.cycle_graph, (1,), nx.cycle_graph, (1,)),
    (gen.cycle_graph, (2,), nx.cycle_graph, (2,)),
    (gen.complete_graph, (6,), nx.complete_graph, (6,)),
    (gen.complete_bipartite_graph, (3, 4), nx.complete_bipartite_graph, (3, 4)),
    (gen.star_graph, (5,), nx.star_graph, (5,)),
    (gen.star_graph, (0,), nx.star_graph, (0,)),
    (gen.balanced_tree, (2, 3), nx.balanced_tree, (2, 3)),
    (gen.balanced_tree, (3, 2), nx.balanced_tree, (3, 2)),
    (gen.balanced_tree, (1, 4), nx.balanced_tree, (1, 4)),
    (gen.binomial_tree, (4,), nx.binomial_tree, (4,)),
    (gen.ladder_graph, (5,), nx.ladder_graph, (5,)),
    (gen.circular_ladder_graph, (5,), nx.circular_ladder_graph, (5,)),
    (gen.circular_ladder_graph, (1,), nx.circular_ladder_graph, (1,)),
    (gen.lollipop_graph, (4, 3), nx.lollipop_graph, (4, 3)),
    (gen.lollipop_graph, (3, 0), nx.lollipop_graph, (3, 0)),
    (gen.barbell_graph, (4, 2), nx.barbell_graph, (4, 2)),
    (gen.barbell_graph, (3, 0), nx.barbell_graph, (3, 0)),
    (gen.petersen_graph, (), nx.petersen_graph, ()),
    (gen.complete_multipartite_graph, (2, 3, 1), nx.complete_multipartite_graph, (2, 3, 1)),
    (gen.turan_graph, (10, 3), nx.turan_graph, (10, 3)),
    (gen.turan_graph, (5, 5), nx.turan_graph, (5, 5)),
] + [(gen.wheel_graph, (n,), nx.wheel_graph, (n,)) for n in range(7)]


@pytest.mark.parametrize("fn, args, nx_fn, nx_args", CLASSIC_CASES, ids=lambda x: getattr(x, "__name__", str(x)))
def test_classic_matches_networkx(fn, args, nx_fn, nx_args):
    g = fn(*args)
    assert type(g) is Graph
    assert_same_as_nx(g, nx_fn(*nx_args), ordered=fn is not gen.binomial_tree)
    assert g.name.startswith(fn.__name__)


DIRECTED_CASES = [
    (gen.path_graph, (5,), nx.path_graph),
    (gen.cycle_graph, (5,), nx.cycle_graph),
    (gen.cycle_graph, (2,), nx.cycle_graph),
    (gen.complete_graph, (4,), nx.complete_graph),
    (gen.star_graph, (4,), nx.star_graph),
    (gen.balanced_tree, (2, 3), nx.balanced_tree),
    (gen.binomial_tree, (3,), nx.binomial_tree),
    (gen.empty_graph, (3,), nx.empty_graph),
]


@pytest.mark.parametrize("fn, args, nx_fn", DIRECTED_CASES, ids=lambda x: getattr(x, "__name__", str(x)))
def test_classic_directed_matches_networkx(fn, args, nx_fn):
    g = fn(*args, directed=True)
    assert type(g) is DiGraph
    assert_same_as_nx(g, nx_fn(*args, create_using=nx.DiGraph), ordered=fn is not gen.binomial_tree)


def test_classic_directed_orientations_without_networkx_counterpart():
    # networkx refuses create_using=DiGraph for these; aryagraph orients them naturally
    assert set(gen.complete_bipartite_graph(2, 2, directed=True).edges) == {(0, 2), (0, 3), (1, 2), (1, 3)}
    wheel = gen.wheel_graph(5, directed=True)
    assert set(wheel.edges) == {(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)}
    ladder = gen.ladder_graph(3, directed=True)
    assert set(ladder.edges) == {(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)}
    assert ladder.is_dag()


def test_grid_graph_matches_networkx_and_has_positions():
    for rows, cols, periodic in [(3, 4, False), (4, 5, True), (2, 5, True), (1, 1, False), (0, 3, False), (3, 3, (True, False))]:
        g = gen.grid_graph(rows, cols, periodic=periodic)
        G = nx.grid_2d_graph(rows, cols, periodic=periodic)
        assert_same_as_nx(g, G)
        for (r, c), attrs in g.nodes.data():
            assert attrs["pos"] == (float(c), float(r))


def test_grid_graph_edge_count_formulas():
    r, c = 4, 6
    assert gen.grid_graph(r, c).num_edges == 2 * r * c - r - c
    assert gen.grid_graph(r, c, periodic=True).num_edges == 2 * r * c


def test_grid_graph_directed_points_right_and_down():
    g = gen.grid_graph(3, 3, directed=True)
    assert g.is_dag()
    assert all(v[0] >= u[0] and v[1] >= u[1] for u, v in g.edges)
    assert g.num_edges == 12


def test_hypercube_matches_networkx_under_bit_labels():
    for d in range(5):
        g = gen.hypercube_graph(d)
        G = nx.hypercube_graph(d)
        to_bits = {i: tuple((i >> (d - 1 - k)) & 1 for k in range(d)) for i in g.nodes}
        if d == 1:  # networkx labels the 1-cube with plain integers
            to_bits = {0: 0, 1: 1}
        assert {frozenset(map(to_bits.get, e)) for e in g.edges} == {frozenset(e) for e in G.edges}
        assert g.num_edges == d * 2 ** (d - 1) if d else g.num_edges == 0
        assert all(g.nodes[i]["bits"] == format(i, "b").zfill(d) for i in g.nodes if d)


def test_hypercube_directed_is_boolean_lattice():
    g = gen.hypercube_graph(3, directed=True)
    assert g.is_dag()
    assert all(v == u | v and bin(u ^ v).count("1") == 1 for u, v in g.edges)
    assert g.in_degree(0) == 0 and g.out_degree(7) == 0


def test_tree_node_counts():
    assert len(gen.balanced_tree(2, 4)) == 31
    assert len(gen.balanced_tree(3, 3)) == 40
    assert len(gen.balanced_tree(0, 5)) == 1
    t = gen.binomial_tree(5)
    assert len(t) == 32 and t.num_edges == 31
    assert gen.balanced_tree(2, 3, directed=True).to_dag().sources() == [0]


def test_bipartite_and_multipartite_attributes():
    g = gen.complete_bipartite_graph(2, 3)
    assert [g.nodes[n]["bipartite"] for n in g] == [0, 0, 1, 1, 1]
    m = gen.complete_multipartite_graph(1, 2, 3)
    G = nx.complete_multipartite_graph(1, 2, 3)
    assert dict(m.nodes.data("subset")) == dict(G.nodes(data="subset"))
    assert gen.complete_multipartite_graph(2, 2, directed=True).is_dag()


def test_turan_graph_is_densest_without_clique():
    for n, r in [(7, 3), (12, 4), (6, 1)]:
        g = gen.turan_graph(n, r)
        expected = int((1 - 1 / r) * n * n / 2) if n % r == 0 else nx.turan_graph(n, r).number_of_edges()
        assert g.num_edges == expected
        assert Counter(d["subset"] for _, d in g.nodes.data()) == Counter(
            d["subset"] for _, d in nx.turan_graph(n, r).nodes(data=True)
        )


def test_petersen_properties():
    g = gen.petersen_graph()
    assert set(g.degree().values()) == {3}
    assert nx.girth(nx.Graph(list(g.edges))) == 5


def test_circular_ladder_directed_rails_are_cycles():
    g = gen.circular_ladder_graph(4, directed=True)
    assert (3, 0) in g.edges and (7, 4) in g.edges and (0, 4) in g.edges
    assert g.num_edges == 12


@pytest.mark.parametrize(
    "call",
    [
        lambda: gen.path_graph(-1),
        lambda: gen.lollipop_graph(1, 3),
        lambda: gen.barbell_graph(1, 2),
        lambda: gen.turan_graph(3, 4),
        lambda: gen.turan_graph(3, 0),
        lambda: gen.complete_multipartite_graph(2, -1),
    ],
)
def test_classic_rejects_bad_sizes(call):
    with pytest.raises(ValueError):
        call()


def test_classic_rejects_non_integers():
    with pytest.raises(TypeError):
        gen.path_graph(2.5)
    with pytest.raises(TypeError):
        gen.grid_graph("3", 3)


def test_zero_and_one_node_edge_cases():
    for fn in (gen.empty_graph, gen.path_graph, gen.complete_graph, gen.wheel_graph, gen.ladder_graph):
        assert len(fn(0)) == 0
    assert gen.path_graph(1).num_edges == 0
    assert gen.complete_graph(1).num_edges == 0
    assert len(gen.hypercube_graph(0)) == 1


# ---------------------------------------------------------------------- #
# sampling primitives
# ---------------------------------------------------------------------- #
def test_pair_index_decodes_exactly_even_for_huge_indices():
    rng = np.random.default_rng(0)
    k = np.concatenate([np.arange(2000), rng.integers(0, 5 * 10**11, size=5000), [5 * 10**11 - 1]])
    w, v = _pair_index(k)
    assert np.all((w >= 0) & (w < v))
    assert np.array_equal(v * (v - 1) // 2 + w, k)
    w, v = _pair_index(np.arange(15), loops=True)
    assert list(zip(w.tolist(), v.tolist()))[:4] == [(0, 0), (0, 1), (1, 1), (0, 2)]
    assert np.all(w <= v)


def test_bernoulli_indices_are_sorted_distinct_and_unbiased():
    rng = np.random.default_rng(3)
    counts = np.zeros(50)
    for _ in range(4000):
        idx = _bernoulli_indices(rng, 50, 0.2)
        assert np.all(np.diff(idx) > 0) and (idx.size == 0 or (idx[0] >= 0 and idx[-1] < 50))
        counts[idx] += 1
    # each position is kept with probability 0.2: mean 800, sd ≈ 25
    assert np.all(np.abs(counts - 800) < 130)
    assert _bernoulli_indices(rng, 10, 0.0).size == 0
    assert _bernoulli_indices(rng, 10, 1.0).tolist() == list(range(10))


# ---------------------------------------------------------------------- #
# Erdős–Rényi
# ---------------------------------------------------------------------- #
def test_erdos_renyi_extremes_and_determinism():
    assert gen.erdos_renyi(10, 0.0, seed=1).num_edges == 0
    assert edge_set(gen.erdos_renyi(12, 1.0, seed=1)) == edge_set(nx.complete_graph(12))
    assert edge_set(gen.erdos_renyi(7, 1.0, directed=True, seed=1)) == edge_set(nx.complete_graph(7, nx.DiGraph))
    a, b = gen.erdos_renyi(200, 0.05, seed=42), gen.erdos_renyi(200, 0.05, seed=42)
    assert list(a.edges) == list(b.edges)
    assert list(a.edges) != list(gen.erdos_renyi(200, 0.05, seed=43).edges)
    rng = np.random.default_rng(5)
    assert isinstance(gen.erdos_renyi(20, 0.3, seed=rng), Graph)


@pytest.mark.parametrize("directed", [False, True])
def test_erdos_renyi_edge_count_is_binomial(directed):
    n, p = 400, 0.02
    total = n * (n - 1) // (1 if directed else 2)
    for seed in range(5):
        g = gen.erdos_renyi(n, p, directed=directed, seed=seed)
        assert not g.selfloops()
        assert abs(g.num_edges - total * p) < 5 * math.sqrt(total * p * (1 - p))
        assert g.is_directed() == directed


def test_erdos_renyi_pairs_are_uniform():
    counts = Counter()
    for seed in range(3000):
        counts.update(frozenset(e) for e in gen.erdos_renyi(5, 0.3, seed=seed).edges)
    assert len(counts) == 10
    assert all(abs(c - 900) < 125 for c in counts.values())  # sd ≈ 25


def test_erdos_renyi_validates_probability():
    with pytest.raises(ValueError):
        gen.erdos_renyi(5, 1.5)
    with pytest.raises(ValueError):
        gen.erdos_renyi(5, -0.1)


def test_erdos_renyi_is_fast_for_sparse_large_graphs():
    g = gen.erdos_renyi(200_000, 2e-5, seed=0)
    assert len(g) == 200_000
    expected = 2e-5 * 200_000 * 199_999 / 2
    assert abs(g.num_edges - expected) < 5 * math.sqrt(expected)


def test_gnm_random_graph_has_exact_size():
    for directed in (False, True):
        g = gen.gnm_random_graph(30, 100, directed=directed, seed=7)
        assert g.num_edges == 100 and len(g) == 30 and not g.selfloops()
    full = gen.gnm_random_graph(6, 15, seed=0)
    assert edge_set(full) == edge_set(nx.complete_graph(6))
    assert gen.gnm_random_graph(6, 0, seed=0).num_edges == 0
    assert list(gen.gnm_random_graph(50, 80, seed=3).edges) == list(gen.gnm_random_graph(50, 80, seed=3).edges)
    with pytest.raises(ValueError):
        gen.gnm_random_graph(5, 11)


# ---------------------------------------------------------------------- #
# growth and small-world models
# ---------------------------------------------------------------------- #
def test_barabasi_albert_size_and_degrees():
    for n, m in [(50, 1), (200, 3), (10, 9)]:
        g = gen.barabasi_albert(n, m, seed=n)
        assert len(g) == n and g.num_edges == m * (n - m)
        assert min(g.degree().values()) >= 1
        assert all(g.degree(v) >= m for v in range(m + 1, n))
        assert not g.selfloops()
    big = gen.barabasi_albert(3000, 2, seed=1)
    assert max(big.degree().values()) > 40  # heavy tail: hubs emerge
    assert list(gen.barabasi_albert(100, 2, seed=9).edges) == list(gen.barabasi_albert(100, 2, seed=9).edges)
    with pytest.raises(ValueError):
        gen.barabasi_albert(5, 5)
    with pytest.raises(ValueError):
        gen.barabasi_albert(5, 0)


def test_powerlaw_cluster_adds_exactly_m_edges_per_node():
    for p in (0.0, 0.5, 1.0):
        g = gen.powerlaw_cluster(300, 3, p, seed=11)
        assert len(g) == 300 and g.num_edges == 3 * (300 - 3)
        assert all(g.degree(v) >= 3 for v in range(3, 300))
        assert not g.selfloops()
    high = nx.average_clustering(nx.Graph(list(gen.powerlaw_cluster(500, 3, 1.0, seed=2).edges)))
    low = nx.average_clustering(nx.Graph(list(gen.powerlaw_cluster(500, 3, 0.0, seed=2).edges)))
    assert high > 2 * low
    with pytest.raises(ValueError):
        gen.powerlaw_cluster(10, 3, 1.5)


def test_watts_strogatz():
    g0 = gen.watts_strogatz(20, 4, 0.0, seed=1)
    assert edge_set(g0) == edge_set(nx.watts_strogatz_graph(20, 4, 0.0))
    for p in (0.1, 0.5, 1.0):
        g = gen.watts_strogatz(60, 6, p, seed=3)
        assert g.num_edges == 60 * 3 and not g.selfloops()
    assert edge_set(gen.watts_strogatz(5, 5, 0.3, seed=0)) == edge_set(nx.complete_graph(5))
    assert gen.watts_strogatz(10, 5, 0.0).num_edges == 20  # odd k uses k // 2
    with pytest.raises(ValueError):
        gen.watts_strogatz(5, 6, 0.1)


# ---------------------------------------------------------------------- #
# block models
# ---------------------------------------------------------------------- #
def test_stochastic_block_model_structure():
    g = gen.stochastic_block_model([3, 4], [[1.0, 0.0], [0.0, 1.0]], seed=0)
    assert [g.nodes[n]["block"] for n in g] == [0, 0, 0, 1, 1, 1, 1]
    assert edge_set(g) == {frozenset(e) for e in itertools.combinations(range(3), 2)} | {
        frozenset(e) for e in itertools.combinations(range(3, 7), 2)
    }
    full = gen.stochastic_block_model([2, 3], [[0.0, 1.0], [1.0, 0.0]], seed=0)
    assert full.num_edges == 6
    d = gen.stochastic_block_model([2, 2], [[1.0, 1.0], [0.0, 1.0]], directed=True, seed=0)
    assert set(d.edges) == {(0, 1), (1, 0), (2, 3), (3, 2), (0, 2), (0, 3), (1, 2), (1, 3)}
    loops = gen.stochastic_block_model([3], [[1.0]], selfloops=True, seed=0)
    assert len(loops.selfloops()) == 3 and loops.num_edges == 6
    dloops = gen.stochastic_block_model([3], [[1.0]], directed=True, selfloops=True, seed=0)
    assert dloops.num_edges == 9


def test_stochastic_block_model_densities():
    sizes, p = [150, 150], [[0.2, 0.01], [0.01, 0.1]]
    g = gen.stochastic_block_model(sizes, p, seed=4)
    within0 = sum(1 for u, v in g.edges if u < 150 and v < 150)
    across = sum(1 for u, v in g.edges if (u < 150) != (v < 150))
    assert abs(within0 - 0.2 * 150 * 149 / 2) < 5 * math.sqrt(0.2 * 0.8 * 150 * 149 / 2)
    assert abs(across - 0.01 * 150 * 150) < 5 * math.sqrt(0.01 * 150 * 150)


def test_stochastic_block_model_validation():
    with pytest.raises(ValueError):
        gen.stochastic_block_model([2, 2], [[0.5, 0.1], [0.2, 0.5]])  # asymmetric
    with pytest.raises(ValueError):
        gen.stochastic_block_model([2, 2], [[0.5, 0.1]])
    with pytest.raises(ValueError):
        gen.stochastic_block_model([2], [[1.2]])


def test_planted_partition():
    g = gen.planted_partition(3, 4, 1.0, 0.0, seed=1)
    assert len(g) == 12 and g.num_edges == 3 * 6
    assert {g.nodes[n]["block"] for n in g} == {0, 1, 2}
    assert gen.planted_partition(2, 3, 0.0, 1.0, seed=1).num_edges == 9


# ---------------------------------------------------------------------- #
# spatial and degree-constrained models
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("n, radius", [(300, 0.1), (100, 0.35), (50, 1.5), (40, 0.0), (1, 0.2)])
def test_random_geometric_matches_brute_force(n, radius):
    g = gen.random_geometric(n, radius, seed=8)
    pos = np.array([g.nodes[i]["pos"] for i in range(n)])
    assert np.all((pos >= 0) & (pos < 1))
    expected = {
        frozenset((i, j))
        for i, j in itertools.combinations(range(n), 2)
        if radius > 0 and np.sum((pos[i] - pos[j]) ** 2) <= radius * radius
    }
    assert edge_set(g) == expected


@pytest.mark.parametrize("d, n", [(3, 10), (4, 51), (0, 5), (5, 6), (7, 10), (2, 3), (10, 30)])
def test_random_regular(d, n):
    g = gen.random_regular(d, n, seed=d * 100 + n)
    assert len(g) == n and set(g.degree().values()) <= {d}
    assert not g.selfloops() and g.num_edges == n * d // 2


def test_random_regular_validation():
    with pytest.raises(ValueError):
        gen.random_regular(3, 5)  # n·d odd
    with pytest.raises(ValueError):
        gen.random_regular(5, 5)


def test_random_tree_is_a_uniform_labeled_tree():
    for n in (1, 2, 3, 10, 200):
        t = gen.random_tree(n, seed=n)
        assert len(t) == n and t.num_edges == n - 1 and is_weakly_connected(t)
    counts = Counter(frozenset(frozenset(e) for e in gen.random_tree(4, seed=s).edges) for s in range(3200))
    assert len(counts) == 16  # Cayley: 4^2 labeled trees, all reachable
    assert all(abs(c - 200) < 70 for c in counts.values())
    with pytest.raises(ValueError):
        gen.random_tree(0)


def test_random_tree_directed_is_an_arborescence():
    t = gen.random_tree(40, directed=True, seed=2)
    assert t.in_degree(0) == 0
    assert all(t.in_degree(v) == 1 for v in range(1, 40))
    assert t.is_dag()


def test_configuration_model_simple_graph():
    degrees = [3, 3, 2, 2, 2, 1, 1]
    g = gen.configuration_model(degrees, seed=5)
    assert not g.selfloops()
    assert all(g.degree(i) <= d for i, d in enumerate(degrees))
    big = gen.configuration_model([3] * 1000, seed=1)
    assert sum(big.degree().values()) >= 0.98 * 3000
    with pytest.raises(ValueError):
        gen.configuration_model([1, 1, 1])


# ---------------------------------------------------------------------- #
# random DAGs
# ---------------------------------------------------------------------- #
def test_random_dag():
    d = gen.random_dag(30, p=1.0, seed=3)
    assert isinstance(d, DAG) and d.num_edges == 30 * 29 // 2
    order = d.topological_order()
    assert order != list(range(30))  # the hidden order is shuffled
    pos = {n: i for i, n in enumerate(order)}
    assert all(pos[u] < pos[v] for u, v in d.edges)
    assert gen.random_dag(20, p=0.0, seed=1).num_edges == 0
    assert gen.random_dag(25, m=40, seed=2).num_edges == 40
    assert list(gen.random_dag(40, seed=6).edges) == list(gen.random_dag(40, seed=6).edges)
    default = gen.random_dag(1000, seed=0)
    assert abs(default.num_edges - 1500) < 5 * math.sqrt(1500)
    with pytest.raises(ValueError):
        gen.random_dag(5, p=0.5, m=3)
    with pytest.raises(ValueError):
        gen.random_dag(5, m=11)


@pytest.mark.parametrize("seed", range(6))
def test_layered_dag_connected(seed):
    layers = [3, 5, 4, 6, 2]
    d = gen.layered_dag(layers, p=0.15, seed=seed)
    assert isinstance(d, DAG) and len(d) == sum(layers)
    layer = dict(d.nodes.data("layer"))
    assert Counter(layer.values()) == Counter({k: s for k, s in enumerate(layers)})
    assert all(layer[v] == layer[u] + 1 for u, v in d.edges)
    assert all(d.in_degree(v) >= 1 for v in d if layer[v] > 0)
    assert all(d.out_degree(v) >= 1 for v in d if layer[v] < len(layers) - 1)
    assert dict(d.levels()) == layer
    assert is_weakly_connected(d)
    assert [d.nodes[n]["index"] for n in d if layer[n] == 1] == list(range(5))


def test_layered_dag_unconnected_and_validation():
    assert gen.layered_dag([4, 4], p=0.0, connected=False, seed=0).num_edges == 0
    assert gen.layered_dag([3, 3], p=1.0, seed=0).num_edges == 9
    assert gen.layered_dag([5], seed=0).num_edges == 0
    with pytest.raises(ValueError):
        gen.layered_dag([3, 0, 2])


def test_random_task_dag():
    for n in (1, 2, 15, 80):
        d = gen.random_task_dag(n, seed=n)
        assert isinstance(d, DAG) and len(d) == n
        assert all(u < v for u, v in d.edges)  # labels are a topological order
        assert d.sources() == [0] and d.sinks() == [n - 1]
        for _, a in d.nodes.data():
            assert 0 < a["min"] <= a["mode"] <= a["max"] and a["duration"] == a["mode"]
            assert all((2 * a[k]) == int(2 * a[k]) for k in ("min", "mode", "max"))  # half hours
    a, b = gen.random_task_dag(30, seed=4), gen.random_task_dag(30, seed=4)
    assert list(a.edges) == list(b.edges) and dict(a.nodes.data("mode")) == dict(b.nodes.data("mode"))


# ---------------------------------------------------------------------- #
# hand-crafted DAGs
# ---------------------------------------------------------------------- #
EXAMPLES = [
    (gen.ml_pipeline, 16),
    (gen.software_build, 20),
    (gen.project_plan, 18),
    (gen.data_warehouse_etl, 25),
    (gen.course_prerequisites, 22),
]


@pytest.mark.parametrize("fn, size", EXAMPLES, ids=[f.__name__ for f, _ in EXAMPLES])
def test_example_dags(fn, size):
    d = fn()
    assert isinstance(d, DAG) and len(d) == size and d.name
    assert is_weakly_connected(d)
    assert nx.is_directed_acyclic_graph(nx.DiGraph(list(d.edges)))
    for node, a in d.nodes.data():
        assert isinstance(node, str)
        assert a["min"] <= a["mode"] <= a["max"] and a["duration"] == a["mode"] > 0
        assert isinstance(a["kind"], str) and isinstance(a["team"], str)
    assert len(d.generations()) >= 4
    d.add_node("extra")
    assert "extra" not in fn()  # every call builds a fresh graph


def test_example_dag_details():
    plan = gen.project_plan()
    assert plan.edges["foundation", "framing"]["lag"] == 72.0
    assert plan.sources() == ["site survey"] and plan.sinks() == ["final inspection"]
    ml = gen.ml_pipeline()
    assert ml.edges["evaluate models", "package model"]["artifact"] == "best model"
    courses = gen.course_prerequisites()
    assert courses.nodes["CS370"]["title"] == "Machine Learning"
    assert courses.nodes["CS370"]["level"] == 300 and courses.nodes["MATH101"]["department"] == "MATH"
    assert {r for *_, r in courses.edges.data("requirement")} == {"required", "recommended"}
    etl = gen.data_warehouse_etl()
    assert etl.in_degree("fact_orders") == 5 and etl.nodes["dim_customer"]["table"] == "dw.dim_customer"


# ---------------------------------------------------------------------- #
# bundled datasets
# ---------------------------------------------------------------------- #
DATASETS = [
    (gen.les_miserables, nx.les_miserables_graph),
    (gen.florentine_families, nx.florentine_families_graph),
    (gen.davis_southern_women, nx.davis_southern_women_graph),
]


@pytest.mark.parametrize("fn, nx_fn", DATASETS, ids=[f.__name__ for f, _ in DATASETS])
def test_dataset_equals_networkx(fn, nx_fn):
    g, G = fn(), nx_fn()
    assert type(g) is Graph
    assert list(g.nodes) == list(G.nodes)
    assert {n: dict(d) for n, d in g.nodes.data()} == {n: dict(d) for n, d in G.nodes(data=True)}
    assert g.num_edges == G.number_of_edges()
    assert edge_set(g) == edge_set(G)
    for u, v, d in G.edges(data=True):
        assert g.edges[u, v] == d
    for key, value in G.graph.items():
        assert g.attrs[key] == value
    assert g.name and g.attrs["citation"]


def test_datasets_return_fresh_graphs():
    d = gen.davis_southern_women()
    d.nodes["E1"]["bipartite"] = "changed"
    d.remove_node("Evelyn Jefferson")
    d.attrs["top"].append("x")
    fresh = gen.davis_southern_women()
    assert fresh.nodes["E1"]["bipartite"] == 1 and "Evelyn Jefferson" in fresh
    assert "x" not in fresh.attrs["top"]
    m = gen.les_miserables()
    m.edges["Valjean", "Javert"]["weight"] = 0
    assert gen.les_miserables().edges["Valjean", "Javert"]["weight"] == 17


# ---------------------------------------------------------------------- #
# University of Calgary campus dataset
# ---------------------------------------------------------------------- #
INDOOR_KINDS = {"tunnel", "pedway", "attached"}
BUILDING_KINDS = {
    "academic", "residence", "student-life", "athletics", "arts", "library", "research", "administration", "services",
}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6_371_008.8 * math.asin(math.sqrt(h))


def component_of(g, start) -> set:
    seen, stack = {start}, [start]
    while stack:
        for v in set(g[stack.pop()]) - seen:
            seen.add(v)
            stack.append(v)
    return seen


def test_ucalgary_campus_loads():
    g = gen.ucalgary_campus()
    assert type(g) is Graph and not g.is_directed()
    assert g.name == "University of Calgary main campus"
    assert len(g) == 56 and g.num_edges == 83
    assert list(g.nodes) == sorted(g.nodes)  # sorted by building code
    assert g.attrs["description"] and "OpenStreetMap" in g.attrs["attribution"]
    sources = g.attrs["sources"]
    assert sources and all(s.startswith("https://") for s in sources)
    assert any("ucalgary.ca" in s for s in sources) and any("openstreetmap.org" in s for s in sources)
    assert Counter(k for *_, k in g.edges.data("kind")) == {"outdoor": 43, "attached": 26, "pedway": 11, "tunnel": 3}
    assert is_weakly_connected(g)


def test_ucalgary_campus_nodes_have_names_kinds_and_coordinates():
    g = gen.ucalgary_campus()
    assert {"MSC", "MH", "TFDL", "ICT", "SS", "ST", "KNA", "OO", "HNSC", "MT", "SH", "ENA", "ENG"} <= set(g.nodes)
    assert g.nodes["TFDL"]["name"] == "Taylor Family Digital Library"
    assert g.nodes["MSC"]["kind"] == "student-life" and g.nodes["TFDL"]["kind"] == "library"
    assert g.nodes["CR"]["kind"] == "residence" and g.nodes["OO"]["kind"] == "athletics"
    # current official names: Kinesiology A/B were renamed in February 2026
    assert g.nodes["KNA"]["name"] == "Dr. Roger Jackson Kinesiology Complex (Block A)"
    assert g.nodes["KNB"]["name"] == "Dr. Roger Jackson Kinesiology Complex (Block B)"
    assert g.nodes["EDC"]["name"] == "Education Classrooms"
    # Trailer B holds general-assignment classrooms, so it is academic, not a service building
    assert g.nodes["TRB"]["kind"] == "academic"
    services = {n for n, k in g.nodes.data("kind") if k == "services"}
    assert services == {"CC", "GR", "HP", "OVC", "PP"}
    assert {k for _, k in g.nodes.data("kind")} == BUILDING_KINDS
    for code, a in g.nodes.data():
        assert isinstance(code, str) and code.isupper()
        assert isinstance(a["name"], str) and a["name"]
        assert a["kind"] in BUILDING_KINDS
        # main campus; the southern bound admits OVC at McMahon Stadium (lat 51.0714)
        assert 51.070 < a["lat"] < 51.084 and -114.145 < a["lon"] < -114.120
        assert len(a["pos"]) == 2 and all(isinstance(c, float) for c in a["pos"])
    # OVC is the one building south of 24 Avenue NW (lat ~51.075 there); its only link is MTH
    assert min(g.nodes, key=lambda n: g.nodes[n]["lat"]) == "OVC"
    assert list(g["OVC"]) == ["MTH"] and g.edges["MTH", "OVC"]["kind"] == "outdoor"
    assert g.edges["MTH", "OVC"]["length"] == max(d["length"] for *_, d in g.edges.data())


def test_ucalgary_campus_edge_lengths_are_haversine_distances():
    g = gen.ucalgary_campus()
    for u, v, d in g.edges.data():
        assert d["kind"] in INDOOR_KINDS | {"outdoor"}
        a, b = g.nodes[u], g.nodes[v]
        expected = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
        assert d["length"] > 0 and d["weight"] == d["length"]
        assert d["length"] == pytest.approx(expected, abs=0.051)  # rounded to 0.1 m
    # the local plane is in metres: planar distances agree with great-circle ones
    for u, v, d in g.edges.data():
        (x1, y1), (x2, y2) = g.nodes[u]["pos"], g.nodes[v]["pos"]
        assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(d["length"], rel=0.005, abs=0.2)
    # known links
    assert g.edges["MH", "SB"]["kind"] == "tunnel" and g.edges["KNA", "DC"]["kind"] == "tunnel"
    assert g.edges["ES", "MS"]["kind"] == "pedway" and g.edges["TFDL", "HNSC"]["kind"] == "pedway"
    assert g.edges["MSC", "MH"]["kind"] == "attached"
    # nearest-neighbour walking links stay within 250 m; only three longer links join outlying buildings
    outdoor = [d["length"] for *_, d in g.edges.data() if d["kind"] == "outdoor"]
    assert sum(1 for x in outdoor if x > 250) == 3


def test_ucalgary_campus_positions_are_y_down():
    g = gen.ucalgary_campus()
    pos = dict(g.nodes.data("pos"))
    lat = dict(g.nodes.data("lat"))
    lon = dict(g.nodes.data("lon"))
    # Mechanical Engineering is north of the Olympic Volunteer Centre, so it sits higher on screen
    assert lat["MEB"] > lat["OVC"] and pos["MEB"][1] < pos["OVC"][1]
    # Scurfield Hall is east of Physical Plant, so it sits further right
    assert lon["SH"] > lon["PP"] and pos["SH"][0] > pos["PP"][0]
    for u, v in itertools.combinations(g.nodes, 2):
        if abs(lat[u] - lat[v]) > 1e-5:
            assert (lat[u] > lat[v]) == (pos[u][1] < pos[v][1])
        if abs(lon[u] - lon[v]) > 1e-5:
            assert (lon[u] > lon[v]) == (pos[u][0] > pos[v][0])
    xs, ys = zip(*pos.values())
    assert abs(sum(xs) / len(xs)) < 1 and abs(sum(ys) / len(ys)) < 1  # centred on the campus


def test_ucalgary_campus_indoor_only():
    full = gen.ucalgary_campus()
    indoor = gen.ucalgary_campus(indoor_only=True)
    assert list(indoor.nodes) == list(full.nodes)  # every building is kept
    assert {k for *_, k in indoor.edges.data("kind")} == INDOOR_KINDS
    expected = {frozenset((u, v)) for u, v, k in full.edges.data("kind") if k in INDOOR_KINDS}
    assert edge_set(indoor) == expected and indoor.num_edges == 40
    for u, v, d in indoor.edges.data():
        assert d == full.edges[u, v]
    # one indoor network links the student centre, library, engineering and business school ...
    core = component_of(indoor, "MSC")
    assert {"TFDL", "ICT", "SS", "ENF", "SH", "MTH", "OO", "IH", "AU", "AB", "RC"} <= core
    assert len(core) == 38
    # ... while the stand-alone residences have no indoor link at all
    assert all(indoor.degree(r) == 0 for r in ("CR", "GL", "OL", "KA", "YA", "RU", "CD"))


def test_ucalgary_campus_is_deterministic_and_fresh():
    a, b = gen.ucalgary_campus(), gen.ucalgary_campus()
    assert list(a.nodes.data()) == list(b.nodes.data())
    assert list(a.edges.data()) == list(b.edges.data())
    assert a.attrs == b.attrs
    a.nodes["MSC"]["name"] = "changed"
    a.attrs["sources"].append("x")
    a.remove_node("TFDL")
    fresh = gen.ucalgary_campus()
    assert fresh.nodes["MSC"]["name"] == "MacEwan Student Centre" and "TFDL" in fresh
    assert "x" not in fresh.attrs["sources"]


def test_generators_package_exports_everything():
    for name in gen.__all__:
        assert callable(getattr(gen, name))
    assert len(gen.__all__) == len(set(gen.__all__)) == 41
