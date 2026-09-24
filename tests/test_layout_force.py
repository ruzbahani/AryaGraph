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

"""Distance- and force-based layouts: stress, Fruchterman–Reingold, ForceAtlas2, spectral."""

from __future__ import annotations

import importlib

import networkx as nx
import numpy as np
import pytest

from aryagraph import DiGraph, Graph
from aryagraph.generators import ucalgary_campus
from aryagraph.layout import force_atlas2, fruchterman_reingold, spectral, stress

S = importlib.import_module("aryagraph.layout.stress")
F = importlib.import_module("aryagraph.layout.force")


def from_nx(G) -> Graph:
    g = DiGraph() if G.is_directed() else Graph()
    g.add_nodes(G.nodes)
    g.add_edges(G.edges(data=True))
    return g


def pairwise(xy):
    d = xy[:, None, :] - xy[None, :, :]
    return np.hypot(d[..., 0], d[..., 1])


def normalised_stress(xy, dist):
    """Stress after the optimal uniform scaling (scale-free comparison)."""
    iu = np.triu_indices(len(xy), 1)
    e = pairwise(xy)[iu]
    d = dist[iu]
    w = 1.0 / d**2
    s = (w * e * d).sum() / (w * e * e).sum()
    return float((w * (s * e - d) ** 2).sum())


def edge_vs_nonedge(g, lay):
    idx = {v: i for i, v in enumerate(lay.nodes)}
    d = pairwise(lay.xy)
    mask = np.zeros_like(d, dtype=bool)
    for u, v in g.edges:
        if u != v:
            mask[idx[u], idx[v]] = mask[idx[v], idx[u]] = True
    off = ~np.eye(len(d), dtype=bool)
    return d[mask].mean(), d[off & ~mask].mean()


# ---------------------------------------------------------------------- #
# distances
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(4))
def test_hop_distances_match_networkx(seed):
    G = nx.gnp_random_graph(60, 0.05, seed=seed)
    g = from_nx(G)
    d = S.graph_distances(g)
    nodes = list(g.nodes)
    ref = dict(nx.all_pairs_shortest_path_length(G))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            expect = ref[u].get(v, np.inf)
            assert d[i, j] == expect


def test_weighted_distances_match_networkx():
    G = nx.gnp_random_graph(30, 0.15, seed=4)
    rng = np.random.default_rng(0)
    for u, v in G.edges:
        G.edges[u, v]["weight"] = float(rng.uniform(0.5, 3))
    g = from_nx(G)
    d = S.graph_distances(g, "weight")
    ref = dict(nx.all_pairs_dijkstra_path_length(G))
    nodes = list(g.nodes)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            assert d[i, j] == pytest.approx(ref[u].get(v, np.inf))


# ---------------------------------------------------------------------- #
# stress
# ---------------------------------------------------------------------- #
def test_stress_reproduces_grid_geometry():
    g = from_nx(nx.grid_2d_graph(7, 5))
    lay = stress(g)
    d = S.graph_distances(g)
    iu = np.triu_indices(len(g), 1)
    geo = pairwise(lay.xy)[iu]
    # hop distance on a grid is Manhattan: even the ideal drawing reaches only ≈ 0.968
    assert np.corrcoef(d[iu], geo)[0, 1] > 0.95
    ideal = pairwise(np.array(list(g.nodes), dtype=float))[iu]
    assert np.corrcoef(ideal, geo)[0, 1] > 0.99
    lengths = [np.hypot(*(lay[u] - lay[v])) for u, v in g.edges]
    assert 0.9 < np.median(lengths) < 1.3  # unit edge length (a grid stretches a little: diagonals want 2)
    # principal axis horizontal: the 7-long side lies along x
    assert np.ptp(lay.xy[:, 0]) > np.ptp(lay.xy[:, 1])


@pytest.mark.parametrize("seed", range(3))
def test_stress_beats_random_and_other_starts(seed):
    g = from_nx(nx.connected_watts_strogatz_graph(80, 4, 0.1, seed=seed))
    d = S.graph_distances(g)
    rnd = np.random.default_rng(seed).random((len(g), 2))
    lay = stress(g, seed=seed)
    assert normalised_stress(lay.xy, d) < 0.25 * normalised_stress(rnd, d)
    from_random = stress(g, seed=seed, init="random")
    assert normalised_stress(from_random.xy, d) < 0.5 * normalised_stress(rnd, d)
    assert lay.meta["stress"] >= 0


def test_stress_init_options():
    g = from_nx(nx.cycle_graph(10))
    start = {v: (np.cos(v), np.sin(v)) for v in g.nodes}
    lay = stress(g, init=start)
    assert np.all(np.isfinite(lay.xy))
    with pytest.raises(ValueError):
        stress(g, init="bogus")


def test_stress_weights_set_edge_lengths():
    g = Graph([("a", "b", 3.0), ("b", "c", 1.0), ("c", "d", 1.0)])
    lay = stress(g, weight="weight")
    ab = np.hypot(*(lay["a"] - lay["b"]))
    bc = np.hypot(*(lay["b"] - lay["c"]))
    assert ab / bc == pytest.approx(3.0, rel=0.05)


def test_pivot_mds_approximates_classical_mds():
    g = from_nx(nx.grid_2d_graph(12, 8))
    d = S.graph_distances(g)
    rng = np.random.default_rng(0)
    full = S._classical_mds(d, rng)
    piv = S._classical_mds(d, rng, pivots=12)
    iu = np.triu_indices(len(g), 1)
    for x in (full, piv):
        assert np.corrcoef(d[iu], pairwise(x)[iu])[0, 1] > 0.95
    assert np.corrcoef(pairwise(full)[iu], pairwise(piv)[iu])[0, 1] > 0.98


def test_stress_collinear_start_escapes():
    # MDS of a cycle is a circle, of a path a line; a star-of-paths MDS start is
    # nearly degenerate, so the seeded jitter must let SMACOF unfold it
    g = Graph([(0, i) for i in range(1, 4)] + [(i, i + 3) for i in range(1, 4)])
    lay = stress(g)
    assert np.ptp(lay.xy[:, 1]) > 0.5


def test_stress_directed_graph_ignores_direction():
    g = DiGraph([(0, 1), (1, 2), (2, 3)])
    lay = stress(g)
    assert np.hypot(*(lay[0] - lay[3])) == pytest.approx(3.0, rel=0.05)


# ---------------------------------------------------------------------- #
# Fruchterman–Reingold
# ---------------------------------------------------------------------- #
def test_fr_edges_shorter_than_non_edges():
    g = ucalgary_campus()
    lay = fruchterman_reingold(g)
    e, ne = edge_vs_nonedge(g, lay)
    assert e < 0.6 * ne


def test_fr_fixed_nodes_do_not_move():
    g = from_nx(nx.path_graph(8))
    init = {v: (float(v), 0.0) for v in g.nodes}
    lay = fruchterman_reingold(g, init=init, fixed=[0, 7], iterations=100)
    np.testing.assert_allclose(lay[0], [0, 0])
    np.testing.assert_allclose(lay[7], [7, 0])
    assert not np.allclose(lay[3], [3, 0])


def test_fr_gravity_keeps_components_close():
    g = Graph([(0, 1), (2, 3), (4, 5)])
    loose = fruchterman_reingold(g, iterations=200)
    tight = fruchterman_reingold(g, iterations=200, gravity=2.0)
    assert np.ptp(tight.xy) < np.ptp(loose.xy)


def test_fr_unknown_fixed_node_raises():
    from aryagraph.core.exceptions import NodeNotFound

    with pytest.raises(NodeNotFound):
        fruchterman_reingold(Graph([(0, 1)]), fixed=["zz"])


# ---------------------------------------------------------------------- #
# ForceAtlas2
# ---------------------------------------------------------------------- #
def test_barnes_hut_matches_exact_repulsion():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 2)) * 50
    mass = rng.integers(1, 6, 400).astype(float)
    exact = F._exact_repulsion(x, 2.0, mass)
    brute = np.zeros_like(x)
    for i in range(len(x)):
        d = x[i] - x
        d2 = (d**2).sum(axis=1)
        d2[i] = np.inf
        brute[i] = ((2.0 * mass[i] * mass / d2)[:, None] * d).sum(axis=0)
    np.testing.assert_allclose(exact, brute, rtol=1e-6, atol=1e-9 * np.abs(brute).max())
    tree = F._QuadTree(x, mass)
    near_exact = tree.repulsion(x, mass, 2.0, 1e-3, None)
    np.testing.assert_allclose(near_exact, exact, rtol=1e-6, atol=1e-6 * np.abs(exact).max())
    approx = tree.repulsion(x, mass, 2.0, 1.0, None)
    err = np.linalg.norm(approx - exact, axis=1) / np.linalg.norm(exact, axis=1)
    assert np.median(err) < 0.05


def test_barnes_hut_handles_coincident_points():
    x = np.zeros((20, 2))
    x[10:] = 1.0
    mass = np.ones(20)
    out = F._QuadTree(x, mass).repulsion(x, mass, 1.0, 1.2, None)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("barnes_hut", [False, True])
def test_fa2_quality(barnes_hut):
    g = from_nx(nx.connected_watts_strogatz_graph(150, 4, 0.05, seed=2))
    lay = force_atlas2(g, barnes_hut=barnes_hut, iterations=300)
    assert lay.meta["barnes_hut"] is barnes_hut
    e, ne = edge_vs_nonedge(g, lay)
    assert e < 0.3 * ne


@pytest.mark.parametrize(
    "opts",
    [dict(lin_log=True), dict(strong_gravity=True), dict(dissuade_hubs=True), dict(scaling=10.0, gravity=0.1), dict(weight=None)],
)
def test_fa2_options_run(opts):
    g = ucalgary_campus()
    lay = force_atlas2(g, iterations=150, **opts)
    assert np.all(np.isfinite(lay.xy))
    e, ne = edge_vs_nonedge(g, lay)
    assert e < ne


def test_fa2_prevent_overlap_reduces_overlaps():
    g = ucalgary_campus()
    sizes = {v: (30, 30) for v in g.nodes}

    def overlaps(lay):
        d = pairwise(lay.xy)
        np.fill_diagonal(d, np.inf)
        return int((d < 30).sum() // 2)

    plain = force_atlas2(g, iterations=300, scaling=0.2)
    spaced = force_atlas2(g, iterations=300, scaling=0.2, prevent_overlap=True, sizes=sizes)
    assert overlaps(spaced) < overlaps(plain)


def test_fa2_edge_weights_pull_harder():
    g = Graph([("a", "b", 10.0), ("b", "c", 1.0), ("c", "a", 1.0), ("c", "d", 1.0)])
    lay = force_atlas2(g, iterations=400)
    assert np.hypot(*(lay["a"] - lay["b"])) < np.hypot(*(lay["b"] - lay["c"]))


def test_fa2_init_is_respected():
    g = from_nx(nx.path_graph(5))
    init = {v: (100.0 * v, 0.0) for v in g.nodes}
    lay = force_atlas2(g, init=init, iterations=1)
    assert np.all(np.diff(lay.xy[:, 0]) > 0)


# ---------------------------------------------------------------------- #
# spectral
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("normalized", [True, False])
def test_spectral_path_is_monotone(normalized):
    g = from_nx(nx.path_graph(12))
    lay = spectral(g, normalized=normalized)
    x = lay.xy[:, 0]
    assert np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)


def test_spectral_grid_neighbours_close():
    g = from_nx(nx.grid_2d_graph(6, 6))
    lay = spectral(g)
    e, ne = edge_vs_nonedge(g, lay)
    assert e < 0.5 * ne


def test_spectral_degenerate_falls_back_to_circle():
    lay = spectral(from_nx(nx.complete_graph(6)))
    assert lay.meta["fallback"] == "circular"
    r = np.hypot(*lay.xy.T)
    assert np.ptp(r) == pytest.approx(0, abs=1e-9)
    tiny = spectral(Graph([(0, 1)]))
    assert tiny.meta["fallback"] == "circular"
