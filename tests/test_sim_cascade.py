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

"""Tests for aryagraph.sim cascades: IC, LT, spread estimation and greedy influence maximisation.

Expected spreads are computed exactly by enumerating every live-edge graph of
small instances (Kempe, Kleinberg & Tardos 2003).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from aryagraph import DiGraph, Graph
from aryagraph.core.exceptions import NodeNotFound
import aryagraph.sim as sim

nx = pytest.importorskip("networkx")


def small_digraph() -> DiGraph:
    g = DiGraph()
    arcs = [(0, 1, 0.5, 1.0), (0, 2, 0.3, 2.0), (1, 2, 0.6, 1.0), (2, 3, 0.4, 1.0),
            (1, 3, 0.7, 3.0), (3, 4, 0.9, 1.0), (4, 1, 0.2, 1.0)]
    for u, v, p, w in arcs:
        g.add_edge(u, v, p=p, weight=w)
    return g


def reach(arcs: list[tuple[int, int]], seeds: set[int]) -> set[int]:
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        u = stack.pop()
        for a, b in arcs:
            if a == u and b not in seen:
                seen.add(b)
                stack.append(b)
    return seen


def exact_ic(g: DiGraph, seeds: set[int]) -> float:
    arcs = [(u, v, d["p"]) for u, v, d in g.edges.data()]
    total = 0.0
    for live in itertools.product([False, True], repeat=len(arcs)):
        prob = math.prod(p if keep else 1 - p for (_, _, p), keep in zip(arcs, live))
        total += prob * len(reach([(u, v) for (u, v, _), keep in zip(arcs, live) if keep], seeds))
    return total


def exact_lt(g: DiGraph, seeds: set[int]) -> float:
    choices = []
    for v in g.nodes:
        ins = [(u, g.edges[u, v]["weight"]) for u in g.pred[v]]
        tot = sum(w for _, w in ins)
        choices.append([((u, v), w / tot) for u, w in ins] if ins else [(None, 1.0)])
    total = 0.0
    for combo in itertools.product(*choices):
        prob = math.prod(p for _, p in combo)
        total += prob * len(reach([a for a, _ in combo if a is not None], seeds))
    return total


def gnp(n: int, p: float, seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    g.add_nodes(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(i, j)
    return g


class TestIndependentCascade:
    def test_frames_and_meta(self):
        g = gnp(60, 0.08, 1)
        r = sim.independent_cascade(g, [0, 5], p=0.3, seed=3)
        assert r.states == ["inactive", "active"] and r.kind == "categorical"
        assert r.roles == {"inactive": "neutral", "active": "accent"}
        assert np.all(np.diff(r.values, axis=0) >= 0)  # progressive
        assert r.meta["newly_active"][0] == [0, 5]
        for t in range(1, r.T):
            new = set(r.meta["newly_active"][t])
            assert new == {n for n in g if r.frame(t)[n] == "active" and r.frame(t - 1)[n] == "inactive"}
            # each newly active node is credited to an edge from a node activated one step before
            assert {v for _, v in r.edge_activity[t]} == new
            for u, v in r.edge_activity[t]:
                assert g.has_edge(u, v) and u in r.meta["newly_active"][t - 1]
        assert r.meta["spread"] == r.counts()["active"][-1]
        assert np.array_equal(r.first_time("active").to_array() == 0, np.isin(np.arange(60), [0, 5]))

    def test_reproducible_and_monotone_in_seeds(self):
        g = gnp(80, 0.06, 2)
        a = sim.independent_cascade(g, [1], p=0.25, seed=9)
        b = sim.independent_cascade(g, [1], p=0.25, seed=9)
        assert np.array_equal(a.values, b.values) and a.edge_activity == b.edge_activity
        for s in range(10):
            small = sim.independent_cascade(g, [1], p=0.25, seed=s).values[-1]
            big = sim.independent_cascade(g, [1, 7, 30], p=0.25, seed=s).values[-1]
            assert np.all(big >= small)

    def test_extreme_probabilities(self):
        g = small_digraph()
        r = sim.independent_cascade(g, [2], p=1.0, seed=0)
        assert {n for n, s in r.final().items() if s == "active"} == {2} | nx.descendants(nx.DiGraph(list(g.edges)), 2)
        # activation step = hop distance from the seed when every coin succeeds
        dist = nx.single_source_shortest_path_length(nx.DiGraph(list(g.edges)), 2)
        assert {n: int(t) for n, t in r.first_time("active").items() if not math.isnan(t)} == dist
        r = sim.independent_cascade(g, [2], p=0.0, seed=0)
        assert r.T == 1 and r.meta["spread"] == 1

    def test_max_steps(self):
        g = Graph([(i, i + 1) for i in range(10)])
        r = sim.independent_cascade(g, [0], p=1.0, max_steps=3)
        assert r.T == 4 and r.meta["spread"] == 4

    def test_edge_attribute_and_errors(self):
        g = small_digraph()
        r = sim.independent_cascade(g, 0, p="p", seed=1)
        assert r.params["p"] == "p" and r.params["seeds"] == [0]
        g.add_edge(4, 0)
        with pytest.raises(ValueError, match="no 'p' attribute"):
            sim.independent_cascade(g, 0, p="p")
        with pytest.raises(ValueError):
            sim.independent_cascade(g, 0, p=1.5)
        with pytest.raises(ValueError):
            sim.independent_cascade(Graph([(0, 1, {"q": 2.0})]), 0, p="q")
        with pytest.raises(NodeNotFound):
            sim.independent_cascade(g, [99])
        with pytest.raises(ValueError):
            sim.independent_cascade(g, 0, max_steps=-1)

    def test_matches_exact_expectation(self):
        g = small_digraph()
        exact = exact_ic(g, {0})
        runs = 4000
        sizes = [sim.independent_cascade(g, [0], p="p", seed=s).meta["spread"] for s in range(runs)]
        se = np.std(sizes, ddof=1) / math.sqrt(runs)
        assert abs(np.mean(sizes) - exact) < 4 * se


class TestLinearThreshold:
    def test_progressive_and_thresholds(self):
        g = gnp(60, 0.1, 3)
        r = sim.linear_threshold(g, [0, 1, 2], seed=4)
        assert np.all(np.diff(r.values, axis=0) >= 0)
        th = r.meta["thresholds"]
        assert all(0 < x <= 1 for x in th.values())
        # the activation rule holds at every step
        for t in range(1, r.T):
            prev, cur = r.frame(t - 1), r.frame(t)
            for v in g:
                if prev[v] == "inactive":
                    share = sum(1 for u in g.adj[v] if prev[u] == "active") / len(g.adj[v]) if len(g.adj[v]) else 0
                    assert (cur[v] == "active") == (share > 0 and share >= th[v] * (1 - 1e-12))

    def test_threshold_one_reachable_despite_rounding(self):
        g = DiGraph()
        for k in range(10):
            g.add_edge(k, "t", weight=0.1)
        r = sim.linear_threshold(g, list(range(10)), thresholds=1.0, weight="weight")
        assert r.final()["t"] == "active"
        r = sim.linear_threshold(g, list(range(9)), thresholds=1.0, weight="weight")
        assert r.final()["t"] == "inactive"

    def test_threshold_specs(self):
        g = DiGraph([("a", "b"), ("c", "b"), ("b", "d")])
        g.nodes["b"]["th"] = 0.5
        g.nodes["d"]["th"] = 0.9
        g.nodes["a"]["th"] = g.nodes["c"]["th"] = 1.0
        r = sim.linear_threshold(g, ["a"], thresholds="th")
        assert r.final() == {"a": "active", "b": "active", "c": "inactive", "d": "active"}
        assert r.edge_activity[1] == [("a", "b")] and r.edge_activity[2] == [("b", "d")]
        r = sim.linear_threshold(g, ["a"], thresholds={"a": 0, "b": 0.6, "c": 0, "d": 0})
        assert r.final()["b"] == "inactive"
        # zero threshold still needs one active in-neighbour
        r = sim.linear_threshold(g, ["d"], thresholds=0.0)
        assert r.meta["spread"] == 1
        with pytest.raises(ValueError):
            sim.linear_threshold(g, ["a"], thresholds={"a": 0.1})
        with pytest.raises(TypeError):
            sim.linear_threshold(g, ["a"], thresholds=[0.1, 0.2])

    def test_monotone_under_fixed_seed(self):
        g = gnp(70, 0.08, 5)
        for s in range(10):
            small = sim.linear_threshold(g, [3], seed=s).values[-1]
            big = sim.linear_threshold(g, [3, 10, 40], seed=s).values[-1]
            assert np.all(big >= small)

    def test_direct_simulation_matches_live_edge_exact(self):
        """Kempe's equivalence: random-threshold LT has the live-edge spread distribution."""
        g = small_digraph()
        exact = exact_lt(g, {0})
        runs = 4000
        sizes = [sim.linear_threshold(g, [0], weight="weight", seed=s).meta["spread"] for s in range(runs)]
        se = np.std(sizes, ddof=1) / math.sqrt(runs)
        assert abs(np.mean(sizes) - exact) < 4 * se


class TestSpreadAndGreedy:
    @pytest.mark.parametrize("seeds", [{0}, {2}, {1, 4}])
    def test_influence_spread_exact(self, seeds):
        g = small_digraph()
        ic = sim.influence_spread(g, seeds, "ic", 20000, p="p", seed=1)
        assert abs(ic.mean - exact_ic(g, seeds)) < 4 * ic.stderr
        lt = sim.influence_spread(g, seeds, "lt", 20000, weight="weight", seed=2)
        assert abs(lt.mean - exact_lt(g, seeds)) < 4 * lt.stderr
        lo, hi = ic.confidence_interval(0.95)
        assert lo < ic.mean < hi and float(ic) == ic.mean
        assert ic.sizes.shape == (20000,) and ic.sizes.min() >= len(seeds)

    def test_undirected_uses_both_directions(self):
        g = Graph([(0, 1), (1, 2)])
        assert sim.influence_spread(g, [2], "ic", 50, p=1.0, seed=0).mean == 3
        # LT: node 1 keeps one of its two in-arcs, so it (and then 2) is reached half of the time
        est = sim.influence_spread(g, [0], "lt", 4000, seed=0)
        assert abs(est.mean - (1 + 1 * 0.5 + 1 * 0.5 * 1.0)) < 4 * est.stderr

    def test_monotone_with_common_random_numbers(self):
        g = gnp(50, 0.08, 6)
        for model in ("ic", "lt"):
            a = sim.influence_spread(g, [0], model, 300, p=0.2, seed=4)
            b = sim.influence_spread(g, [0, 9], model, 300, p=0.2, seed=4)
            assert np.all(b.sizes >= a.sizes)

    @pytest.mark.parametrize("model", ["ic", "lt"])
    def test_celf_equals_plain_greedy(self, model):
        g = gnp(25, 0.12, 7)
        runs, seed, k = 150, 11, 3
        im = sim.greedy_influence_maximization(g, k, model, runs, p=0.15, seed=seed)
        chosen: list = []
        for i in range(k):
            # the estimator with the same seed reuses the exact same live-edge samples
            best = max(
                (v for v in g if v not in chosen),
                key=lambda v: (sim.influence_spread(g, chosen + [v], model, runs, p=0.15, seed=seed).sizes.sum(),
                               -list(g).index(v)),
            )
            chosen.append(best)
            assert im.seeds[i] == best
            est = sim.influence_spread(g, chosen, model, runs, p=0.15, seed=seed).mean
            assert im.spread[i] == pytest.approx(est)
        assert np.all(np.diff(im.gains) <= 1e-12)  # submodular: gains never increase
        assert im.evaluations < k * len(g)
        assert "seeds=" in repr(im)

    def test_greedy_edge_cases(self):
        g = gnp(10, 0.3, 8)
        assert sim.greedy_influence_maximization(g, 0).seeds == []
        im = sim.greedy_influence_maximization(g, 2, candidates=[3, 4, 5], runs=20, seed=1)
        assert set(im.seeds) <= {3, 4, 5}
        with pytest.raises(ValueError):
            sim.greedy_influence_maximization(g, 11)
        with pytest.raises(ValueError):
            sim.greedy_influence_maximization(g, 2, model="sir")
        with pytest.raises(ValueError):
            sim.influence_spread(g, [0], runs=0)
