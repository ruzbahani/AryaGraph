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

"""Layout contract (every engine) and the compute() dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from aryagraph import DAG, DiGraph, Graph
from aryagraph.core.exceptions import AryaGraphError
from aryagraph.layout import (
    METHODS,
    Layout,
    auto_method,
    compute,
    hierarchical,
    pack_components,
    stress,
)

CANONICAL = {
    "stress": "stress",
    "kamada_kawai": "stress",
    "force": "fruchterman_reingold",
    "fruchterman_reingold": "fruchterman_reingold",
    "spring": "fruchterman_reingold",
    "forceatlas2": "forceatlas2",
    "force_atlas2": "forceatlas2",
    "spectral": "spectral",
    "circular": "circular",
    "shell": "shell",
    "grid": "grid",
    "random": "random",
    "spiral": "spiral",
    "bipartite": "bipartite",
    "arc": "arc",
    "tree": "tree",
    "radial": "radial",
    "hierarchical": "hierarchical",
    "layered": "hierarchical",
    "sugiyama": "hierarchical",
}
NAMES = sorted(CANONICAL)


def _graphs() -> dict[str, object]:
    disconnected = Graph([(1, 2), (2, 3), (3, 1), (4, 5), (6, 6)], nodes=[7, 8])
    return {
        "empty": Graph(),
        "single": Graph(nodes=["a"]),
        "loop": Graph([("a", "a")]),
        "pair": DiGraph([("a", "b")]),
        "disconnected": disconnected,
        "cyclic_digraph": DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 3), (4, 5)]),
        "dag": DAG([("s", "a"), ("s", "b"), ("a", "t"), ("b", "t"), ("s", "t")]),
        "path": Graph([(i, i + 1) for i in range(12)]),
    }


GRAPHS = _graphs()


def test_method_table_matches_expected_names():
    assert set(METHODS) == set(CANONICAL)


@pytest.mark.parametrize("gname", sorted(GRAPHS))
@pytest.mark.parametrize("method", NAMES)
def test_contract_every_method(method, gname):
    g = GRAPHS[gname]
    lay = compute(g, method, seed=3)
    assert isinstance(lay, Layout)
    assert lay.nodes == list(g.nodes)
    assert lay.xy.shape == (len(g), 2)
    assert np.all(np.isfinite(lay.xy))
    assert lay.method == CANONICAL[method]
    for pts in lay.routes.values():
        assert np.all(np.isfinite(pts))
    again = compute(g, method, seed=3)
    np.testing.assert_array_equal(lay.xy, again.xy)
    assert set(lay.routes) == set(again.routes)


@pytest.mark.parametrize("method", NAMES)
def test_direct_call_handles_disconnected_and_loops(method):
    g = GRAPHS["disconnected"]
    lay = METHODS[method](g, seed=1)
    assert lay.nodes == list(g.nodes)
    assert np.all(np.isfinite(lay.xy))


@pytest.mark.parametrize("method", ["random", "fruchterman_reingold", "forceatlas2"])
def test_seed_changes_randomised_layouts(method):
    g = GRAPHS["path"]
    a = compute(g, method, seed=1).xy
    b = compute(g, method, seed=2).xy
    assert not np.allclose(a, b)


@pytest.mark.parametrize("method", NAMES)
def test_positions_are_distinct_on_small_graph(method):
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (4, 0)])
    lay = compute(g, method, seed=0)
    d = np.hypot(*(lay.xy[:, None, :] - lay.xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    assert d.min() > 1e-6


# ---------------------------------------------------------------------- #
# compute(): dispatch, auto rules, passthrough, packing
# ---------------------------------------------------------------------- #
def test_unknown_method_lists_valid_names():
    with pytest.raises(ValueError) as exc:
        compute(GRAPHS["path"], "nope")
    msg = str(exc.value)
    for name in ("auto", "stress", "hierarchical", "forceatlas2", "sugiyama", "bipartite"):
        assert name in msg


def test_method_names_are_case_insensitive():
    assert compute(GRAPHS["path"], "Stress").method == "stress"
    assert compute(GRAPHS["path"], "force-atlas2").method == "forceatlas2"


def test_auto_rules():
    assert auto_method(GRAPHS["dag"]) == "hierarchical"
    assert auto_method(DiGraph([(0, 1), (1, 2)])) == "hierarchical"  # acyclic DiGraph, not a DAG instance
    assert auto_method(GRAPHS["cyclic_digraph"]) == "stress"
    assert auto_method(DiGraph([(0, 0)])) == "stress"  # a self-loop is a cycle
    assert auto_method(GRAPHS["path"]) == "stress"
    big_dag = DiGraph()
    big_dag.add_nodes(range(2001))
    assert auto_method(big_dag) == "stress"
    big = Graph()
    big.add_nodes(range(3001))
    assert auto_method(big) == "forceatlas2"
    assert compute(GRAPHS["dag"]).method == "hierarchical"
    assert compute(GRAPHS["cyclic_digraph"]).method == "stress"


def test_auto_ignores_options_the_engine_does_not_take():
    lay = compute(GRAPHS["cyclic_digraph"], "auto", orientation="LR", rank_sep=10)
    assert lay.method == "stress"
    lay = compute(GRAPHS["dag"], "auto", orientation="LR")
    assert lay.meta["orientation"] == "LR"


def test_explicit_method_rejects_unknown_options():
    with pytest.raises(TypeError):
        compute(GRAPHS["path"], "stress", orientation="LR")


def test_sizes_are_forwarded_only_to_size_aware_engines():
    g = GRAPHS["dag"]
    small = compute(g, "hierarchical", sizes={v: (10, 10) for v in g.nodes})
    big = compute(g, "hierarchical", sizes={v: (200, 10) for v in g.nodes})
    assert np.ptp(big.xy[:, 0]) > np.ptp(small.xy[:, 0])
    lay = compute(GRAPHS["path"], "stress", sizes={0: (100, 100)})  # ignored, no error
    assert lay.method == "stress"


def test_passthrough_layout_and_mapping():
    g = GRAPHS["dag"]
    pos = {v: (i, 2 * i) for i, v in enumerate(reversed(list(g.nodes)))}
    pos["extra"] = (99, 99)
    lay = compute(g, pos)
    assert lay.nodes == list(g.nodes)
    for v in g.nodes:
        np.testing.assert_allclose(lay[v], pos[v])
    base = Layout.from_positions(pos, method="mine", metric=True)
    lay2 = compute(g, base)
    assert lay2.nodes == list(g.nodes)
    assert lay2.method == "mine" and lay2.metric
    np.testing.assert_allclose(lay2.xy, lay.xy)


def test_passthrough_missing_nodes_raise():
    g = GRAPHS["dag"]
    with pytest.raises(AryaGraphError):
        compute(g, {"s": (0, 0)})
    with pytest.raises(AryaGraphError):
        compute(g, Layout.from_positions({"s": (0, 0)}))


def test_callable_method():
    g = GRAPHS["path"]
    lay = compute(g, lambda gg, **kw: {v: (v, 0) for v in gg.nodes})
    assert lay.nodes == list(g.nodes)
    np.testing.assert_allclose(lay.xy[:, 0], list(g.nodes))


def _boxes_disjoint(lay: Layout, groups: list[list]) -> bool:
    boxes = []
    for grp in groups:
        pts = np.array([lay[v] for v in grp])
        boxes.append((pts.min(axis=0), pts.max(axis=0)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (a0, a1), (b0, b1) = boxes[i], boxes[j]
            if np.all(a0 <= b1) and np.all(b0 <= a1):
                return False
    return True


@pytest.mark.parametrize("method", ["stress", "force", "forceatlas2", "spectral"])
def test_components_are_packed_apart(method):
    g = Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 3), (7, 8)], nodes=[9])
    lay = compute(g, method)
    assert lay.meta.get("components") == 4
    assert _boxes_disjoint(lay, [[0, 1, 2], [3, 4, 5, 6], [7, 8], [9]])
    # comparable scale: median edge length of every component ≈ 1
    for comp in ([0, 1, 2], [3, 4, 5, 6]):
        pts = np.array([lay[v] for v in comp])
        lens = np.hypot(*(pts - np.roll(pts, 1, axis=0)).T)
        assert 0.5 < np.median(lens) < 2.0


def test_components_none_lays_out_whole_graph():
    g = Graph([(0, 1), (2, 3)])
    lay = compute(g, "stress", components=None)
    assert lay.nodes == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        compute(g, "stress", components="bogus")


def test_pack_components_reexported():
    a = Layout([1, 2], [[0, 0], [1, 0]])
    b = Layout([3], [[0, 0]])
    packed = pack_components([a, b], gap=1.0)
    assert sorted(packed.nodes) == [1, 2, 3]


def test_graph_layout_shortcut_uses_compute():
    g = GRAPHS["dag"]
    assert g.layout().method == "hierarchical"
    assert g.layout("stress").method == "stress"


def test_every_module_defines_all():
    import importlib

    for mod in ("force", "stress", "spectral", "geometric", "tree", "hierarchical", "overlap"):
        m = importlib.import_module(f"aryagraph.layout.{mod}")
        assert m.__all__
        for name in m.__all__:
            assert hasattr(m, name)
    import aryagraph.layout as pkg

    for name in pkg.__all__:
        assert hasattr(pkg, name)


def test_hierarchical_and_stress_exposed():
    assert hierarchical(GRAPHS["dag"]).metric
    assert not stress(GRAPHS["path"]).metric
