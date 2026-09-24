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

"""Tidy tree (Walker / Buchheim–Jünger–Leipert) and radial tree layouts."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from aryagraph import DAG, DiGraph, Graph
from aryagraph.core.exceptions import NodeNotFound
from aryagraph.layout import radial, tree


def random_tree(n: int, seed: int) -> Graph:
    T = nx.random_labeled_tree(n, seed=seed) if hasattr(nx, "random_labeled_tree") else nx.random_tree(n, seed=seed)
    return Graph(list(T.edges))


def sizes_for(g, seed=0):
    rng = np.random.default_rng(seed)
    return {v: (float(rng.integers(16, 80)), float(rng.integers(16, 40))) for v in g.nodes}


def children_of(lay):
    kids: dict = {}
    for v, p in lay.meta["parent"].items():
        if p is not None:
            kids.setdefault(p, []).append(v)
    return kids


def check_tidy(g, lay, sizes, node_sep, orientation="TB"):
    across_ax = 0 if orientation in ("TB", "BT") else 1
    size_idx = 0 if orientation in ("TB", "BT") else 1
    depth = lay.meta["depth"]
    # 1. spanning forest of g
    for v, p in lay.meta["parent"].items():
        if p is not None:
            assert g.has_edge(p, v) or g.has_edge(v, p)
            assert depth[v] == depth[p] + 1
    # 2. no overlaps within a level (all trees of the forest together)
    levels: dict[int, list] = {}
    for v in g.nodes:
        levels.setdefault(depth[v], []).append(v)
    for members in levels.values():
        members.sort(key=lambda v: lay[v][across_ax])
        for a, b in zip(members, members[1:]):
            need = (sizes[a][size_idx] + sizes[b][size_idx]) / 2 + node_sep
            assert lay[b][across_ax] - lay[a][across_ax] >= need - 1e-6
    # 3. parents centred over their first and last child
    for p, kids in children_of(lay).items():
        xs = sorted(lay[c][across_ax] for c in kids)
        assert lay[p][across_ax] == pytest.approx((xs[0] + xs[-1]) / 2, abs=1e-6)
    # 4. every level on its own line
    rp = lay.meta["rank_positions"]
    for v in g.nodes:
        assert lay[v][1 - across_ax] == pytest.approx(rp[depth[v]])


@pytest.mark.parametrize("seed", range(6))
def test_random_trees_are_tidy(seed):
    g = random_tree(80, seed)
    sizes = sizes_for(g, seed)
    lay = tree(g, sizes=sizes, node_sep=12)
    assert lay.metric and lay.method == "tree"
    check_tidy(g, lay, sizes, 12)


@pytest.mark.parametrize("orientation", ["TB", "BT", "LR", "RL"])
def test_orientations(orientation):
    g = random_tree(40, 1)
    sizes = sizes_for(g, 2)
    lay = tree(g, sizes=sizes, orientation=orientation, node_sep=10)
    check_tidy(g, lay, sizes, 10, orientation)
    depth = lay.meta["depth"]
    axis = 1 if orientation in ("TB", "BT") else 0
    sign = 1 if orientation in ("TB", "LR") else -1
    for v, p in lay.meta["parent"].items():
        if p is not None:
            assert sign * (lay[v][axis] - lay[p][axis]) > 0
    assert depth[lay.meta["roots"][0]] == 0
    assert (lay.xy.min(axis=0) >= 0).all()


def test_level_spacing_uses_tallest_box():
    g = DAG([("r", "a"), ("r", "b"), ("a", "c")])
    sizes = {"r": (30, 20), "a": (30, 60), "b": (30, 10), "c": (30, 30)}
    lay = tree(g, sizes=sizes, rank_sep=40)
    rp = lay.meta["rank_positions"]
    assert rp[1] - rp[0] == pytest.approx(10 + 40 + 30)
    assert rp[2] - rp[1] == pytest.approx(30 + 40 + 15)


def test_root_selection():
    assert tree(DiGraph([("b", "c"), ("a", "b"), ("a", "d")])).meta["roots"] == ["a"]
    path = Graph([(i, i + 1) for i in range(6)])
    assert tree(path).meta["roots"] == [3]
    star = Graph([(0, i) for i in range(1, 6)])
    assert tree(star).meta["roots"] == [0]
    assert tree(path, root=0).meta["roots"] == [0]
    with pytest.raises(NodeNotFound):
        tree(path, root="missing")


def test_directed_graph_follows_arcs_first():
    # a unique source whose BFS along arcs reaches everything
    g = DiGraph([("s", "a"), ("s", "x"), ("x", "y"), ("y", "z"), ("z", "a")])
    lay = tree(g)
    assert lay.meta["parent"]["a"] == "s"
    assert lay.meta["parent"]["z"] == "y"


def test_non_tree_graph_uses_bfs_spanning_tree():
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (2, 4)])
    lay = tree(g, root=0)
    parents = lay.meta["parent"]
    assert sum(p is not None for p in parents.values()) == len(g) - 1
    assert lay.routes == {}
    check_tidy(g, lay, {v: (36, 36) for v in g.nodes}, 24)


def test_forest_side_by_side_sharing_levels():
    g = Graph([(0, 1), (0, 2), (10, 11), (11, 12), (11, 13)], nodes=["iso"])
    lay = tree(g)
    roots = lay.meta["roots"]
    assert len(roots) == 3
    ys = {round(float(lay[r][1]), 6) for r in roots}
    assert len(ys) == 1
    check_tidy(g, lay, {v: (36, 36) for v in g.nodes}, 24)
    # trees do not interleave
    spans = []
    for comp in ([0, 1, 2], [10, 11, 12, 13], ["iso"]):
        xs = [lay[v][0] for v in comp]
        spans.append((min(xs), max(xs)))
    spans.sort()
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))


def test_deep_path_is_fine_and_straight():
    g = DiGraph([(i, i + 1) for i in range(3000)])
    lay = tree(g)
    assert np.ptp(lay.xy[:, 0]) == pytest.approx(0)
    assert lay.meta["depth"][3000] == 3000


def test_symmetric_subtrees_are_mirror_images():
    g = DiGraph(list(nx.balanced_tree(3, 3, create_using=nx.DiGraph).edges))
    lay = tree(g)
    root_x = lay[0][0]
    xs = sorted(lay[v][0] - root_x for v in g.nodes if lay.meta["depth"][v] == 3)
    np.testing.assert_allclose(xs, [-x for x in reversed(xs)], atol=1e-6)


def test_self_loops_and_empty():
    lay = tree(Graph([(0, 0), (0, 1)]))
    assert lay.meta["parent"][1] == 0
    empty = tree(Graph())
    assert empty.nodes == [] and empty.metric
    single = tree(Graph(nodes=["x"]))
    np.testing.assert_allclose(single["x"], [18, 18])


# ---------------------------------------------------------------------- #
# radial
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(4))
def test_radial_rings_and_spacing(seed):
    g = random_tree(60, seed)
    sizes = sizes_for(g, seed)
    lay = radial(g, sizes=sizes, node_sep=8)
    assert lay.metric and lay.method == "radial"
    root = lay.meta["roots"][0]
    radii = lay.meta["radii"][0]
    depth = lay.meta["depth"]
    center = lay[root]
    for v in g.nodes:
        assert np.hypot(*(lay[v] - center)) == pytest.approx(radii[depth[v]], abs=1e-6)
    ext = {v: max(sizes[v]) for v in g.nodes}
    for d in range(1, len(radii)):
        ring = [v for v in g.nodes if depth[v] == d]
        for i, a in enumerate(ring):
            for b in ring[i + 1 :]:
                assert np.hypot(*(lay[a] - lay[b])) >= (ext[a] + ext[b]) / 2 + 8 - 1e-6
    assert all(b > a for a, b in zip(radii, radii[1:]))


def test_radial_subtrees_occupy_disjoint_wedges():
    T = nx.balanced_tree(2, 4, create_using=nx.DiGraph)
    lay = radial(DiGraph(list(T.edges)))
    cx = lay[0][0]
    # wedges start at 12 o'clock and run clockwise: the first subtree takes the
    # right half-plane, the second the left one
    assert all(lay[v][0] > cx + 1e-6 for v in nx.descendants(T, 1) | {1})
    assert all(lay[v][0] < cx - 1e-6 for v in nx.descendants(T, 2) | {2})
    # within a subtree, grandchildren keep their parent's angular order
    for p in T.nodes:
        kids = list(T.successors(p))
        if len(kids) == 2 and p != 0:
            a = [math.atan2(*(lay[k] - lay[0])[::-1]) for k in kids]
            assert a[0] != a[1]


def test_radial_forest_is_packed_without_overlap():
    g = Graph([(0, 1), (0, 2), (0, 3), (10, 11), (11, 12)], nodes=["iso"])
    lay = radial(g)
    assert len(lay.meta["roots"]) == 3
    d = np.hypot(*(lay.xy[:, None, :] - lay.xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 36 - 1e-6
    assert "orientation" not in lay.meta
