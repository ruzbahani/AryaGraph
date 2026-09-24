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

"""Hierarchical (Sugiyama) layout: layering, ordering, coordinates, routes."""

from __future__ import annotations

import importlib
import itertools

import networkx as nx
import numpy as np
import pytest

from aryagraph import DAG, DiGraph, Graph
from aryagraph.layout import hierarchical

H = importlib.import_module("aryagraph.layout.hierarchical")


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def gnp_dag(n: int, deg: float, seed: int) -> DiGraph:
    G = nx.gnp_random_graph(n, deg / n, seed=seed, directed=True)
    g = DiGraph()
    g.add_nodes(range(n))
    g.add_edges([(u, v) for u, v in G.edges if u < v])
    return g


def layered_dag(n: int = 30, layers: int = 6, seed: int = 3) -> DAG:
    """Random DAG with a hidden layering, mostly short edges and a few long ones."""
    rng = np.random.default_rng(seed)
    lay = sorted(rng.integers(0, layers, n))
    g = DAG()
    g.add_nodes(range(n))
    for i in range(n):
        for j in range(n):
            if lay[j] == lay[i] + 1 and rng.random() < 0.24:
                g.add_edge(i, j)
            elif lay[j] > lay[i] + 1 and rng.random() < 0.04:
                g.add_edge(i, j)
    for j in range(n):  # give every non-top node a parent
        if lay[j] > 0 and not list(g.predecessors(j)):
            cands = [i for i in range(n) if lay[i] == lay[j] - 1]
            if cands:
                g.add_edge(int(rng.choice(cands)), j)
    return g


def sizes_for(g, seed=0):
    rng = np.random.default_rng(seed)
    return {v: (float(rng.integers(20, 90)), float(rng.integers(16, 50))) for v in g.nodes}


def rank_axis(lay):
    return 1 if lay.meta["orientation"] in ("TB", "BT") else 0


def geometric_crossings(g, lay) -> int:
    """Crossings counted from the drawing itself (segments between adjacent layers)."""
    ax = rank_axis(lay)
    other = 1 - ax
    rp = {round(p, 6): k for k, p in enumerate(lay.meta["rank_positions"])}
    by_pair: dict[int, list[tuple[float, float]]] = {}
    for u, v in g.edges:
        if u == v or (u, v) in lay.meta["flat_edges"]:
            continue
        pts = [lay[u], *lay.routes.get((u, v), []), lay[v]]
        for p, q in zip(pts, pts[1:]):
            kp, kq = rp[round(float(p[ax]), 6)], rp[round(float(q[ax]), 6)]
            assert abs(kp - kq) == 1, "segments must join adjacent layers"
            if kp > kq:
                p, q, kp = q, p, kq
            by_pair.setdefault(kp, []).append((float(p[other]), float(q[other])))
    total = 0
    for segs in by_pair.values():
        for (a, b), (c, d) in itertools.combinations(segs, 2):
            if (a - c) * (b - d) < 0:
                total += 1
    return total


def assert_layer_separation(g, lay, sizes, node_sep):
    ax = rank_axis(lay)
    other = 1 - ax
    extent_idx = 0 if other == 0 else 1  # width for TB/BT, height for LR/RL
    layers: dict[int, list] = {}
    for v in g.nodes:
        layers.setdefault(lay.meta["layers"][v], []).append(v)
    for members in layers.values():
        members.sort(key=lambda v: lay[v][other])
        for a, b in zip(members, members[1:]):
            need = (sizes[a][extent_idx] + sizes[b][extent_idx]) / 2 + node_sep
            assert lay[b][other] - lay[a][other] >= need - 1e-6, (a, b)


def lp_optimum(n, tails, heads):
    """min Σ (r_h − r_t) s.t. r_h − r_t ≥ 1, via the dual min-cost flow in networkx."""
    G = nx.DiGraph()
    c = np.zeros(n)
    for t, h in zip(tails, heads):
        c[h] += 1
        c[t] -= 1
    for v in range(n):
        G.add_node(v, demand=int(c[v]))
    for t, h in zip(tails, heads):
        G.add_edge(t, h, weight=-1)
    cost, _ = nx.network_simplex(G)
    return -cost


# ---------------------------------------------------------------------- #
# layering & direction
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(5))
def test_edges_point_down_and_layers_are_consistent(seed):
    g = gnp_dag(60, 2.5, seed)
    lay = hierarchical(g)
    L = lay.meta["layers"]
    assert lay.meta["reversed_edges"] == []
    rp = lay.meta["rank_positions"]
    for u, v in g.edges:
        assert L[v] > L[u]
        assert lay[v][1] > lay[u][1]
    for v in g.nodes:
        assert lay[v][1] == pytest.approx(rp[L[v]])
    assert all(b > a for a, b in zip(rp, rp[1:]))


@pytest.mark.parametrize("seed", range(6))
def test_network_simplex_layering_is_optimal(seed):
    g = gnp_dag(45, 2.2, 100 + seed)
    lay = hierarchical(g)
    L = lay.meta["layers"]
    total = sum(L[v] - L[u] for u, v in g.edges)
    nodes = list(g.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    opt = lp_optimum(len(nodes), [idx[u] for u, _ in g.edges], [idx[v] for _, v in g.edges])
    assert total == opt


def test_network_simplex_cut_values_are_maintained_incrementally():
    g = gnp_dag(80, 3.0, 7)
    edges = list(g.edges)
    n = len(g)
    t = [u for u, _ in edges]
    h = [v for _, v in edges]
    comp = max(nx.weakly_connected_components(nx.DiGraph(edges)), key=len)
    keep = [(a, b) for a, b in zip(t, h) if a in comp]
    remap = {v: i for i, v in enumerate(sorted(comp))}
    t = [remap[a] for a, _ in keep]
    h = [remap[b] for _, b in keep]
    ns = H._NetworkSimplex(len(remap), t, h, [1] * len(t), [1.0] * len(t), H._longest_path(len(remap), t, h))
    ns.solve(100000)
    fresh = H._NetworkSimplex.__new__(H._NetworkSimplex)
    fresh.__dict__.update(ns.__dict__)
    post = fresh._dfs_range(0, -1, 1)
    saved = ns.cut.copy()
    fresh._init_cut_values(post)
    tree = np.array(ns.tree_e)
    np.testing.assert_allclose(fresh.cut[tree], saved[tree])
    assert np.all(saved[tree] >= -1e-9)  # optimality certificate
    assert tree.sum() == len(remap) - 1 <= n - 1


def test_balance_moves_free_nodes_to_sparse_layers():
    # s → a → t (long path) and s → x → y → t: node "b" with 1 in / 1 out could sit anywhere
    g = DAG([("s", "a"), ("a", "b"), ("b", "c"), ("c", "t"), ("s", "m"), ("m", "t")])
    L = hierarchical(g).meta["layers"]
    assert L["t"] - L["s"] == 4
    assert L["m"] in (1, 2, 3)
    counts = np.bincount([L[v] for v in g.nodes])
    assert counts.max() <= 2


def test_longest_path_layering():
    g = DAG([("a", "b"), ("b", "c"), ("a", "c"), ("d", "c"), ("c", "e")])
    L = hierarchical(g, layering="longest_path").meta["layers"]
    assert (L["a"], L["b"], L["c"], L["d"], L["e"]) == (0, 1, 2, 0, 3)


def test_coffman_graham_respects_width():
    g = DAG([("s", f"x{i}") for i in range(9)] + [(f"x{i}", "t") for i in range(9)])
    lay = hierarchical(g, layering="coffman_graham", max_width=3)
    L = lay.meta["layers"]
    counts = np.bincount([L[v] for v in g.nodes])
    assert counts.max() <= 3
    for u, v in g.edges:
        assert L[v] > L[u]


def test_unknown_layering_and_orientation_raise():
    g = DAG([(0, 1)])
    with pytest.raises(ValueError):
        hierarchical(g, layering="magic")
    with pytest.raises(ValueError):
        hierarchical(g, orientation="up")


# ---------------------------------------------------------------------- #
# cycles, undirected graphs, user ranks
# ---------------------------------------------------------------------- #
def test_cycles_are_broken_with_few_reversals():
    g = DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 3), (5, 6), (6, 6)])
    lay = hierarchical(g)
    rev = set(lay.meta["reversed_edges"])
    assert len(rev) == 2
    assert (6, 6) not in rev
    L = lay.meta["layers"]
    for u, v in g.edges:
        if u == v:
            continue
        if (u, v) in rev:
            assert L[u] > L[v]
        else:
            assert L[v] > L[u]


def test_edges_between_sccs_are_never_reversed():
    # two cycles joined by a bridge: only intra-cycle arcs may be reversed
    g = DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])
    rev = set(hierarchical(g).meta["reversed_edges"])
    assert (2, 3) not in rev
    assert len(rev) == 2


def test_reversed_long_edge_route_runs_from_u_to_v():
    g = DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    lay = hierarchical(g)
    assert lay.meta["reversed_edges"] == [(4, 0)]
    L = lay.meta["layers"]
    route = lay.routes[(4, 0)]
    assert len(route) == L[4] - L[0] - 1 == 3
    ys = route[:, 1]
    assert np.all(np.diff(ys) < 0)  # from 4 (bottom) up to 0 (top)
    rp = lay.meta["rank_positions"]
    np.testing.assert_allclose(ys, [rp[k] for k in range(L[4] - 1, L[0], -1)])


def test_two_cycle_gets_one_reversal():
    g = DiGraph([("a", "b"), ("b", "a")])
    lay = hierarchical(g)
    assert len(lay.meta["reversed_edges"]) == 1
    assert lay.meta["layers"]["a"] != lay.meta["layers"]["b"]


def test_undirected_graph_is_oriented_by_bfs_depth():
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 4)])
    lay = hierarchical(g)
    L = lay.meta["layers"]
    assert lay.meta["reversed_edges"] == []
    assert sorted(L.values()) == [0, 1, 2, 3, 4]
    assert {L[0], L[4]} == {0, 4}  # starts from a peripheral node


def test_undirected_long_edge_route_keyed_by_reported_edge():
    g = Graph([(0, 1), (1, 2), (2, 3), (0, 3), (3, 4)])
    g.add_edge(4, 0)
    lay = hierarchical(g)
    L = lay.meta["layers"]
    for (u, v), pts in lay.routes.items():
        assert (u, v) in g.edges
        assert len(pts) == abs(L[u] - L[v]) - 1
        # ordered from u to v
        du = np.hypot(*(pts[0] - lay[u]))
        dv = np.hypot(*(pts[0] - lay[v]))
        assert du <= dv + 1e-9


def test_user_ranks_override_with_backward_and_flat_edges():
    g = DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("a", "d"), ("d", "c")])
    ranks = {"a": 0, "b": 1, "c": 3, "d": 0}
    lay = hierarchical(g, ranks=ranks)
    L = lay.meta["layers"]
    assert dict(L) == ranks
    assert ("c", "a") in lay.meta["reversed_edges"]
    assert ("a", "d") in lay.meta["flat_edges"]
    assert len(lay.routes[("c", "a")]) == 2  # spans 3 layers upwards
    assert len(lay.routes[("b", "c")]) == 1
    rp = lay.meta["rank_positions"]
    for v in g.nodes:
        assert lay[v][1] == pytest.approx(rp[L[v]])


def test_user_ranks_are_normalised_and_must_cover_all_nodes():
    g = DAG([(1, 2)])
    lay = hierarchical(g, ranks={1: 5, 2: 9})
    assert dict(lay.meta["layers"]) == {1: 0, 2: 4}
    assert len(lay.routes[(1, 2)]) == 3
    with pytest.raises(ValueError):
        hierarchical(g, ranks={1: 0})


# ---------------------------------------------------------------------- #
# crossings
# ---------------------------------------------------------------------- #
def test_bjm_counter_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = int(rng.integers(1, 12))
        south = rng.integers(0, q, int(rng.integers(0, 30))).tolist()
        brute = sum(1 for i in range(len(south)) for j in range(i + 1, len(south)) if south[i] > south[j])
        assert H._bjm(south, q) == brute


@pytest.mark.parametrize("seed", range(6))
def test_reported_crossings_match_the_drawing(seed):
    g = gnp_dag(50, 2.5, 200 + seed)
    lay = hierarchical(g, sizes=sizes_for(g, seed))
    assert lay.meta["crossings"] == geometric_crossings(g, lay)


def test_crossings_much_lower_than_a_naive_drawing():
    g = layered_dag(60, 8, seed=11)
    lay = hierarchical(g)
    # naive drawing with the same layers: nodes by id within a layer, long
    # edges as straight lines (bend points interpolated)
    L = lay.meta["layers"]
    rank_in_layer = {}
    for k in set(L.values()):
        for i, v in enumerate(sorted(v for v in g.nodes if L[v] == k)):
            rank_in_layer[v] = float(i)
    segs: dict[int, list[tuple[float, float]]] = {}
    for u, v in g.edges:
        span = L[v] - L[u]
        xs = [rank_in_layer[u] + (rank_in_layer[v] - rank_in_layer[u]) * t / span for t in range(span + 1)]
        for t in range(span):
            segs.setdefault(L[u] + t, []).append((xs[t], xs[t + 1]))
    naive = sum(1 for s in segs.values() for (a, b), (c, d) in itertools.combinations(s, 2) if (a - c) * (b - d) < 0)
    assert lay.meta["crossings"] <= 0.5 * naive
    m = g.num_edges
    assert lay.meta["crossings"] <= m * (m - 1) // 2


def test_known_crossing_count_on_layered_dag():
    g = layered_dag()
    assert hierarchical(g).meta["crossings"] <= 16


def _planar_cases():
    rng = np.random.default_rng(5)
    T = nx.random_labeled_tree(40, seed=3) if hasattr(nx, "random_labeled_tree") else nx.random_tree(40, seed=3)
    tree = DiGraph(list(nx.bfs_tree(T, 0).edges))
    diamonds = DAG()
    for i in range(6):
        a, b, c, d = f"t{i}", f"l{i}", f"r{i}", f"t{i + 1}"
        diamonds.add_edges([(a, b), (a, c), (b, d), (c, d)])
    grid = DiGraph()
    for i in range(5):
        for j in range(5):
            if i < 4:
                grid.add_edge((i, j), (i + 1, j))
            if j < 4:
                grid.add_edge((i, j), (i, j + 1))
    series_parallel = DAG([("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("c", "e"), ("c", "f"), ("d", "t"), ("e", "t"), ("f", "t"), ("s", "t")])
    binary = DiGraph(list(nx.balanced_tree(2, 5, create_using=nx.DiGraph).edges))
    del rng
    return {"tree": tree, "diamonds": diamonds, "grid": grid, "series_parallel": series_parallel, "binary": binary}


@pytest.mark.parametrize("name", sorted(_planar_cases()))
def test_zero_crossings_on_planar_layered_graphs(name):
    g = _planar_cases()[name]
    lay = hierarchical(g)
    assert lay.meta["crossings"] == 0
    assert geometric_crossings(g, lay) == 0


def test_siblings_keep_graph_order_when_free():
    g = DAG([("root", f"c{i}") for i in range(6)] + [(f"c{i}", f"g{i}") for i in range(6)])
    lay = hierarchical(g)
    xs = [lay[f"c{i}"][0] for i in range(6)]
    assert xs == sorted(xs)


# ---------------------------------------------------------------------- #
# coordinates
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("orientation", ["TB", "BT", "LR", "RL"])
def test_no_overlap_within_layers_with_variable_sizes(orientation, compact):
    g = layered_dag(40, 7, seed=2)
    sizes = sizes_for(g, 3)
    lay = hierarchical(g, sizes=sizes, orientation=orientation, compact=compact, node_sep=20)
    assert_layer_separation(g, lay, sizes, 20)


def test_layer_gap_uses_tallest_node():
    g = DAG([("a", "b"), ("b", "c")])
    sizes = {"a": (40, 20), "b": (40, 100), "c": (40, 30)}
    lay = hierarchical(g, sizes=sizes, rank_sep=50)
    rp = lay.meta["rank_positions"]
    assert rp[1] - rp[0] == pytest.approx(10 + 50 + 50)
    assert rp[2] - rp[1] == pytest.approx(50 + 50 + 15)
    x0, y0, _, _ = lay.bounds()
    assert y0 == pytest.approx(10)  # boxes start at 0


@pytest.mark.parametrize("orientation,axis,sign", [("TB", 1, 1), ("BT", 1, -1), ("LR", 0, 1), ("RL", 0, -1)])
def test_orientations_put_layers_along_the_right_axis(orientation, axis, sign):
    g = gnp_dag(30, 2.5, 4)
    lay = hierarchical(g, orientation=orientation)
    L = lay.meta["layers"]
    rp = lay.meta["rank_positions"]
    for u, v in g.edges:
        assert sign * (lay[v][axis] - lay[u][axis]) > 0
    for v in g.nodes:
        assert lay[v][axis] == pytest.approx(rp[L[v]])
    assert lay.meta["orientation"] == orientation
    x0, y0, _, _ = lay.bounds(include_routes=True)
    assert x0 >= 0 and y0 >= 0  # node centres and bend points in the positive quadrant


def test_lr_uses_heights_for_in_layer_spacing():
    g = DAG([("s", "a"), ("s", "b"), ("s", "c")])
    sizes = {v: (200, 20) for v in g.nodes}
    lay = hierarchical(g, sizes=sizes, orientation="LR", node_sep=10)
    ys = sorted(lay[v][1] for v in "abc")
    assert ys[1] - ys[0] == pytest.approx(30, abs=1)
    xs = {round(lay[v][0], 6) for v in "abc"}
    assert len(xs) == 1
    assert lay["a"][0] - lay["s"][0] == pytest.approx(200 + 72)


def test_long_edges_get_routes_with_span_minus_one_points():
    g = gnp_dag(50, 3.0, 9)
    lay = hierarchical(g)
    L = lay.meta["layers"]
    rp = lay.meta["rank_positions"]
    for u, v in g.edges:
        span = L[v] - L[u]
        if span > 1:
            pts = lay.routes[(u, v)]
            assert pts.shape == (span - 1, 2)
            np.testing.assert_allclose(pts[:, 1], [rp[k] for k in range(L[u] + 1, L[v])])
        else:
            assert (u, v) not in lay.routes


@pytest.mark.parametrize("compact", [True, False])
def test_inner_segments_of_long_edges_are_straight(compact):
    g = DAG([(f"a{i}", f"a{i + 1}") for i in range(6)] + [("a0", "a6"), ("a1", "a5"), ("b", "a3"), ("a0", "b")])
    lay = hierarchical(g, compact=compact)
    for key in (("a0", "a6"), ("a1", "a5")):
        pts = lay.routes[key]
        assert len(pts) >= 3
        assert np.ptp(pts[:, 0]) == pytest.approx(0, abs=1e-6)  # every dummy–dummy segment is vertical


@pytest.mark.parametrize("compact", [True, False])
def test_inner_segments_mostly_straight_on_random_dags(compact):
    straight = total = 0
    for seed in range(4):
        g = gnp_dag(60, 2.5, 300 + seed)
        lay = hierarchical(g, compact=compact)
        for pts in lay.routes.values():
            for p, q in zip(pts, pts[1:]):
                total += 1
                straight += abs(p[0] - q[0]) < 1e-6
    assert total > 20
    assert straight / total >= 0.9


def test_parent_centred_over_symmetric_children():
    g = DAG([("p", "a"), ("p", "b"), ("p", "c")])
    lay = hierarchical(g)
    assert lay["p"][0] == pytest.approx(lay["b"][0], abs=1.0)
    assert lay["p"][0] == pytest.approx((lay["a"][0] + lay["c"][0]) / 2, abs=1.0)


def test_chain_is_vertical():
    g = DAG([(i, i + 1) for i in range(5)])
    lay = hierarchical(g)
    assert np.ptp(lay.xy[:, 0]) == pytest.approx(0)


# ---------------------------------------------------------------------- #
# components, isolated nodes, loops, meta
# ---------------------------------------------------------------------- #
def test_components_side_by_side_sharing_layer_zero():
    g = DAG([("a", "b"), ("b", "c"), ("x", "y"), ("x", "z")])
    g.add_node("lonely")
    lay = hierarchical(g)
    L = lay.meta["layers"]
    assert L["a"] == L["x"] == 0
    assert lay["a"][1] == pytest.approx(lay["x"][1])
    comp1 = np.array([lay[v][0] for v in "abc"])
    comp2 = np.array([lay[v][0] for v in "xyz"])
    assert comp1.max() + 36 <= comp2.min() or comp2.max() + 36 <= comp1.min()
    assert lay.meta["components"] == 3


def test_isolated_nodes_are_packed_into_rows():
    g = DAG([("a", "b"), ("b", "c"), ("c", "d")])
    g.add_nodes(range(9))
    lay = hierarchical(g)
    L = lay.meta["layers"]
    rows = {L[i] for i in range(9)}
    assert len(rows) == 3
    sizes = {v: (36, 36) for v in g.nodes}
    assert_layer_separation(g, lay, sizes, 28)
    flat = hierarchical(g, compact=False).meta["layers"]
    assert {flat[i] for i in range(9)} == {0}


def test_self_loops_are_ignored():
    g = DiGraph([("a", "a"), ("a", "b"), ("b", "b")])
    lay = hierarchical(g)
    assert lay.meta["reversed_edges"] == []
    assert lay.routes == {}
    assert lay.meta["layers"]["b"] == 1


def test_meta_contents():
    g = DAG([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    lay = hierarchical(g, rank_sep=40, node_sep=10)
    m = lay.meta
    assert m["orientation"] == "TB" and m["rank_sep"] == 40 and m["node_sep"] == 10
    assert dict(m["layers"]) == {"a": 0, "b": 1, "c": 1, "d": 2}
    assert m["crossings"] == 0
    assert m["reversed_edges"] == [] and m["flat_edges"] == []
    assert len(m["rank_positions"]) == 3
    assert lay.metric and lay.method == "hierarchical"


def test_empty_and_single():
    lay = hierarchical(DAG())
    assert lay.nodes == [] and lay.meta["crossings"] == 0
    lay = hierarchical(DAG(nodes=["x"]))
    np.testing.assert_allclose(lay["x"], [18, 18])


def test_moderately_large_dag_runs():
    g = gnp_dag(400, 2.5, 1)
    lay = hierarchical(g)
    assert np.all(np.isfinite(lay.xy))
    assert lay.meta["crossings"] == geometric_crossings(g, lay)
