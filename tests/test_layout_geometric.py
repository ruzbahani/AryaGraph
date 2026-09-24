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

"""Geometric layouts and node-overlap removal."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from aryagraph import DiGraph, Graph
from aryagraph.core.exceptions import NodeNotFound
from aryagraph.layout import Layout, arc, bipartite, circular, grid, random, remove_overlaps, shell, spiral


def from_nx(Gx) -> Graph:
    g = DiGraph() if Gx.is_directed() else Graph()
    g.add_nodes(Gx.nodes)
    g.add_edges(Gx.edges)
    return g


def chord_crossings(lay, g, order_key):
    pos = {v: k for k, v in enumerate(order_key)}
    chords = [tuple(sorted((pos[u], pos[v]))) for u, v in g.edges if u != v]
    c = 0
    for i, (a, b) in enumerate(chords):
        for x, y in chords[i + 1 :]:
            if a < x < b < y or x < a < y < b:
                c += 1
    return c


# ---------------------------------------------------------------------- #
# circular / arc
# ---------------------------------------------------------------------- #
def test_circular_geometry_and_start_angle():
    g = Graph([(i, (i + 1) % 8) for i in range(8)])
    lay = circular(g)
    r = np.hypot(*lay.xy.T)
    assert np.ptp(r) == pytest.approx(0, abs=1e-12)
    assert lay[0][0] == pytest.approx(0, abs=1e-12) and lay[0][1] < 0  # 12 o'clock, y down
    assert lay[2][0] > 0 and abs(lay[2][1]) < 1e-9  # clockwise on screen: 3 o'clock next
    neighbour = np.hypot(*(lay[0] - lay[1]))
    assert neighbour == pytest.approx(1.0, rel=0.05)
    right = circular(g, start_angle=0)
    assert right[0][1] == pytest.approx(0, abs=1e-12) and right[0][0] > 0


def test_circular_explicit_order():
    g = Graph([(0, 1), (1, 2), (2, 3)])
    lay = circular(g, order=[3, 1])
    assert lay.meta["order"] == [3, 1, 0, 2]
    with pytest.raises(NodeNotFound):
        circular(g, order=["zz"])
    with pytest.raises(ValueError):
        circular(g, order="best")


@pytest.mark.parametrize("seed", range(4))
def test_circular_auto_order_uncrosses_trees(seed):
    T = nx.random_labeled_tree(30, seed=seed) if hasattr(nx, "random_labeled_tree") else nx.random_tree(30, seed=seed)
    g = from_nx(T)
    lay = circular(g, order="auto")
    assert chord_crossings(lay, g, lay.meta["order"]) == 0
    base = circular(g)
    assert chord_crossings(base, g, base.meta["order"]) >= 0


def test_circular_auto_order_beats_graph_order():
    rng = np.random.default_rng(0)
    perm = rng.permutation(24)
    Gx = nx.relabel_nodes(nx.cycle_graph(24), dict(enumerate(perm.tolist())))
    g = Graph()
    g.add_nodes(range(24))
    g.add_edges(Gx.edges)
    auto = circular(g, order="auto")
    assert chord_crossings(auto, g, auto.meta["order"]) == 0
    assert chord_crossings(circular(g), g, list(range(24))) > 0


def test_arc_is_one_line():
    g = from_nx(nx.path_graph(6))
    lay = arc(g, order="auto")
    assert lay.meta["arc"] is True
    np.testing.assert_allclose(lay.xy[:, 1], 0)
    assert sorted(lay.xy[:, 0].tolist()) == [0, 1, 2, 3, 4, 5]
    assert chord_crossings(lay, g, lay.meta["order"]) == 0


# ---------------------------------------------------------------------- #
# shell / grid / random / spiral
# ---------------------------------------------------------------------- #
def test_shell_given_shells():
    g = Graph([(0, i) for i in range(1, 5)] + [(i, i + 4) for i in range(1, 5)])
    lay = shell(g, shells=[[0], [1, 2, 3, 4]])
    np.testing.assert_allclose(lay[0], [0, 0])
    r1 = [np.hypot(*lay[v]) for v in (1, 2, 3, 4)]
    r2 = [np.hypot(*lay[v]) for v in (5, 6, 7, 8)]  # left-over nodes form an outer shell
    assert np.ptp(r1) == pytest.approx(0) and np.ptp(r2) == pytest.approx(0)
    assert min(r2) > max(r1)
    with pytest.raises(NodeNotFound):
        shell(g, shells=[["zz"]])


def test_shell_default_is_bfs_rings_and_uncrossed_on_trees():
    g = from_nx(nx.balanced_tree(3, 2))
    lay = shell(g)
    assert lay.meta["shells"][0] == [0]
    # children sit next to their parent's angle: no edge crossings
    segs = [(lay[u], lay[v]) for u, v in g.edges]

    def cross(p1, p2, p3, p4):
        d = lambda a, b, c: np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return d(p1, p2, p3) * d(p1, p2, p4) < 0 and d(p3, p4, p1) * d(p3, p4, p2) < 0

    for i, (a, b) in enumerate(segs):
        for c, d_ in segs[i + 1 :]:
            assert not cross(a, b, c, d_)


def test_grid():
    g = Graph(nodes=range(10))
    lay = grid(g, columns=4)
    assert lay.meta["columns"] == 4
    np.testing.assert_allclose(lay[5], [1, 1])
    assert len({tuple(p) for p in lay.xy.tolist()}) == 10
    default = grid(g)
    assert default.meta["columns"] == 4


def test_random_in_unit_square():
    g = Graph(nodes=range(50))
    lay = random(g, seed=7)
    assert (lay.xy >= 0).all() and (lay.xy < 1).all()
    np.testing.assert_array_equal(lay.xy, random(g, seed=7).xy)
    assert not np.allclose(lay.xy, random(g, seed=8).xy)


def test_spiral_even_spacing():
    g = Graph(nodes=range(200))
    lay = spiral(g)
    steps = np.hypot(*np.diff(lay.xy, axis=0).T)
    assert steps[20:].min() > 0.7 and steps[20:].max() < 1.3
    r = np.hypot(*lay.xy.T)
    assert np.all(np.diff(r) >= -1e-9)  # moving outwards
    few = spiral(g, turns=2)
    assert np.all(np.isfinite(few.xy))


# ---------------------------------------------------------------------- #
# bipartite
# ---------------------------------------------------------------------- #
def test_bipartite_two_lines_and_coloring():
    Gx = nx.complete_bipartite_graph(3, 4)
    g = from_nx(Gx)
    lay = bipartite(g)
    top = set(lay.meta["top"])
    assert top in ({0, 1, 2}, {3, 4, 5, 6})
    xs = {round(float(lay[v][0]), 9) for v in top}
    assert len(xs) == 1
    horiz = bipartite(g, top=[0, 1, 2], align="horizontal")
    assert {round(float(horiz[v][1]), 9) for v in (0, 1, 2)} == {0.0}
    assert len({round(float(horiz[v][1]), 9) for v in (3, 4, 5, 6)}) == 1
    with pytest.raises(ValueError):
        bipartite(g, align="diagonal")


def test_bipartite_barycenter_uncrosses_a_zigzag():
    # a path alternating sides has a crossing-free two-line drawing
    rng = np.random.default_rng(1)
    labels = rng.permutation(16).tolist()
    g = Graph()
    g.add_nodes(sorted(labels))
    g.add_edges([(labels[i], labels[i + 1]) for i in range(15)])
    lay = bipartite(g)
    top = set(lay.meta["top"])
    segs = []
    for u, v in g.edges:
        a, b = (u, v) if u in top else (v, u)
        segs.append((lay[a][1], lay[b][1]))
    crossings = sum(1 for i, (a, b) in enumerate(segs) for c, d in segs[i + 1 :] if (a - c) * (b - d) < 0)
    assert crossings == 0


# ---------------------------------------------------------------------- #
# overlap removal
# ---------------------------------------------------------------------- #
def count_overlaps(xy, sizes, padding=0.0, tol=1e-6):
    n = len(xy)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            ox = (sizes[i][0] + sizes[j][0]) / 2 + padding - abs(xy[i, 0] - xy[j, 0])
            oy = (sizes[i][1] + sizes[j][1]) / 2 + padding - abs(xy[i, 1] - xy[j, 1])
            if ox > tol and oy > tol:
                c += 1
    return c


@pytest.mark.parametrize("seed", range(5))
def test_remove_overlaps_leaves_no_overlap(seed):
    rng = np.random.default_rng(seed)
    n = 120
    xy = rng.random((n, 2)) * 300
    nodes = list(range(n))
    sizes = {v: (float(rng.integers(10, 50)), float(rng.integers(10, 30))) for v in nodes}
    lay = Layout(nodes, xy, method="stress")
    out = remove_overlaps(lay, sizes, padding=4.0)
    arr = [sizes[v] for v in nodes]
    assert count_overlaps(xy, arr, 4.0) > 0
    assert count_overlaps(out.xy, arr, 4.0 - 1e-6) == 0
    assert out.nodes == nodes and out.method == "stress"
    np.testing.assert_array_equal(out.xy, remove_overlaps(lay, sizes, padding=4.0).xy)


def test_remove_overlaps_minimal_on_simple_cases():
    # two boxes overlapping horizontally move apart symmetrically, a far one stays
    lay = Layout(["a", "b", "far"], [[0, 0], [10, 0], [500, 500]])
    out = remove_overlaps(lay, {"a": (20, 20), "b": (20, 20), "far": (20, 20)}, padding=0)
    np.testing.assert_allclose(out["a"], [-5, 0])
    np.testing.assert_allclose(out["b"], [15, 0])
    np.testing.assert_allclose(out["far"], [500, 500])
    # a tall pair is cheaper to separate horizontally, a wide pair vertically
    wide = Layout([0, 1], [[0, 0], [2, 1]])
    out = remove_overlaps(wide, (100, 10), padding=0)
    assert out[1][1] - out[0][1] == pytest.approx(10)
    assert out[0][0] == pytest.approx(0) and out[1][0] == pytest.approx(2)


def test_remove_overlaps_preserves_order():
    rng = np.random.default_rng(3)
    xs = np.sort(rng.random(30) * 100)
    lay = Layout(list(range(30)), np.column_stack([xs, np.zeros(30)]))
    out = remove_overlaps(lay, (12, 12), padding=2)
    assert np.all(np.diff(out.xy[:, 0]) >= 14 - 1e-6)
    assert count_overlaps(out.xy, [(12, 12)] * 30, 2 - 1e-6) == 0


def test_remove_overlaps_identical_positions_and_trivial():
    lay = Layout(list("abcd"), np.zeros((4, 2)))
    out = remove_overlaps(lay, (10, 10))
    assert count_overlaps(out.xy, [(10, 10)] * 4, 4 - 1e-6) == 0
    single = remove_overlaps(Layout(["x"], [[1, 2]]), None)
    np.testing.assert_allclose(single["x"], [1, 2])
    empty = remove_overlaps(Layout([], np.zeros((0, 2))), None)
    assert empty.nodes == []


def test_remove_overlaps_drops_only_affected_routes():
    lay = Layout(["a", "b", "c", "d"], [[0, 0], [5, 0], [300, 0], [300, 300]], routes={("a", "b"): [[1, 1]], ("c", "d"): [[300, 150]]}, metric=True)
    out = remove_overlaps(lay, (20, 20))
    assert ("c", "d") in out.routes and ("a", "b") not in out.routes
    assert out.metric


def test_remove_overlaps_is_fast_enough_for_the_renderer():
    rng = np.random.default_rng(0)
    n = 1500
    lay = Layout(list(range(n)), rng.random((n, 2)) * 800)
    out = remove_overlaps(lay, (20, 20), padding=3)
    assert count_overlaps(out.xy[:200], [(20, 20)] * 200, 3 - 1e-6) == 0
