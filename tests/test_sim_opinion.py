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

"""Tests for aryagraph.sim opinion dynamics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aryagraph import DiGraph, Graph
from aryagraph.core.exceptions import NodeNotFound
import aryagraph.sim as sim


def star(k: int) -> Graph:
    return Graph([("c", i) for i in range(k)])


def complete(n: int) -> Graph:
    g = Graph()
    g.add_nodes(range(n))
    g.add_edges((i, j) for i in range(n) for j in range(i + 1, n))
    return g


class TestVoter:
    def test_consensus_is_absorbing(self):
        r = sim.voter_model(complete(12), steps=500, seed=1)
        assert r.meta["consensus"] in ("0", "1") and r.meta["consensus_step"] == r.T - 1
        c = r.counts()
        assert c[r.meta["consensus"]][-1] == 12
        assert np.all(c["0"] + c["1"] == 12)
        # changes are copies along edges that switched the head's opinion
        for t in range(1, r.T):
            for u, v in r.edge_activity[t]:
                assert u != v
        # already at consensus: a single frame
        r = sim.voter_model(complete(5), opinions={i: "x" for i in range(5)}, steps=10)
        assert r.T == 1 and r.meta["consensus"] == "x"

    def test_exit_probability_is_degree_weighted(self):
        """Degree-weighted opinion density is a martingale: P(all 1) = Σ_{x_v=1} d_v / 2m."""
        g = star(4)
        start = {"c": 1, 0: 1, 1: 0, 2: 0, 3: 0}
        expected = (4 + 1) / 8
        runs = 1500
        wins = sum(sim.voter_model(g, opinions=start, steps=10000, seed=s).meta["consensus"] == "1"
                   for s in range(runs))
        assert abs(wins / runs - expected) < 4 * math.sqrt(expected * (1 - expected) / runs)

    def test_labels_and_directed(self):
        g = DiGraph([("a", "b"), ("b", "c")])
        r = sim.voter_model(g, opinions={"a": "red", "b": "blue", "c": "blue"}, steps=50, seed=2)
        assert r.states == ["blue", "red"] and r.meta["consensus"] == "red"
        assert r.final() == {"a": "red", "b": "red", "c": "red"}  # the source never changes
        assert set(r.roles.values()) <= set(sim.ROLES)
        with pytest.raises(ValueError):
            sim.voter_model(g, opinions={"a": 1})
        with pytest.raises(NodeNotFound):
            sim.voter_model(g, opinions={"a": 1, "b": 1, "c": 1, "z": 0})
        with pytest.raises(ValueError):
            sim.voter_model(g, opinions={"a": 1, "b": "1", "c": 1})

    def test_reproducible(self):
        g = complete(20)
        a = sim.voter_model(g, opinions=3, steps=5, seed=4)
        b = sim.voter_model(g, opinions=3, steps=5, seed=4)
        assert np.array_equal(a.values, b.values) and a.states == ["0", "1", "2"]


class TestMajority:
    def test_complete_graph_adopts_majority(self):
        g = complete(9)
        start = {i: (1 if i < 5 else 0) for i in range(9)}
        r = sim.majority_rule(g, opinions=start, include_self=True)
        assert r.meta["fixed_point"] and r.counts()["1"][-1] == 9 and r.T == 2
        moved = {v for _, v in r.edge_activity[1]}
        assert moved == {5, 6, 7, 8}

    def test_ties_keep_current_opinion(self):
        g = Graph([(0, 1), (0, 2)])
        r = sim.majority_rule(g, opinions={0: "a", 1: "a", 2: "b"}, seed=0, steps=5)
        # node 0 sees a tie (a vs b) and keeps "a"; the leaves copy node 0
        assert r.final() == {0: "a", 1: "a", 2: "a"}

    def test_two_cycle_runs_to_steps(self):
        g = Graph([(0, 1)])
        r = sim.majority_rule(g, opinions={0: 0, 1: 1}, steps=6)
        assert not r.meta["fixed_point"] and r.T == 7
        assert r.frame(1) == {0: "1", 1: "0"}


class TestDeGroot:
    def test_converges_to_left_eigenvector_consensus(self):
        g = DiGraph()
        arcs = [(0, 1, 1.0), (1, 2, 2.0), (2, 0, 1.0), (2, 3, 0.5), (3, 0, 3.0), (1, 3, 1.0)]
        for u, v, w in arcs:
            g.add_edge(u, v, weight=w)
        x0 = {0: 1.0, 1: -2.0, 2: 5.0, 3: 0.5}
        r = sim.degroot(g, x0, steps=400, weight="weight", self_weight=0.5)
        # independent construction of W and its left Perron vector
        n = 4
        w = np.eye(n) * 0.5
        for u, v, wt in arcs:
            w[v, u] += wt
        w /= w.sum(axis=1, keepdims=True)
        vals, vecs = np.linalg.eig(w.T)
        pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
        pi /= pi.sum()
        consensus = float(pi @ np.array([x0[i] for i in range(n)]))
        assert r.values[-1] == pytest.approx(np.full(n, consensus), abs=1e-10)
        assert r.meta["consensus"] == pytest.approx(consensus, abs=1e-12)
        assert r.meta["influence"].to_array(list(range(n))) == pytest.approx(pi, abs=1e-12)
        assert r.values[1] == pytest.approx(w @ r.values[0])
        assert r.kind == "continuous" and r.meta["domain"] == (-2.0, 5.0) and r.meta["diverging"]

    def test_undirected_consensus_is_degree_weighted_mean(self):
        g = star(3)
        x0 = {"c": 0.0, 0: 1.0, 1: 1.0, 2: 4.0}
        r = sim.degroot(g, x0, steps=300)
        # symmetric trust + self weight 1: π ∝ degree + 1
        weights = {"c": 4, 0: 2, 1: 2, 2: 2}
        expected = sum(weights[k] * x0[k] for k in x0) / sum(weights.values())
        assert r.meta["consensus"] == pytest.approx(expected)
        assert r.values[-1] == pytest.approx(np.full(4, expected))

    def test_convex_hull_tol_and_stuck_nodes(self):
        g = DiGraph([("s", "a"), ("a", "b"), ("b", "a")])
        r = sim.degroot(g, {"s": 1.0, "a": 0.0, "b": 0.0}, steps=1000, self_weight=0.0, tol=1e-12)
        assert r.meta["converged"] and r.T < 1001
        assert r.values.min() >= 0.0 and r.values.max() <= 1.0
        assert r.values[:, 0] == pytest.approx(1.0)  # "s" listens to nobody and keeps its value
        assert r.meta["consensus"] == pytest.approx(1.0)

    def test_self_loop_is_extra_self_trust(self):
        g = Graph([(0, 1), (1, 2)])
        looped = Graph([(0, 1), (1, 2), (1, 1, {"weight": 2.0})])
        x0 = [0.0, 1.0, 3.0]
        a = sim.degroot(looped, x0, steps=5, weight="weight", self_weight=1.0)
        own = np.array([1.0, 3.0, 1.0])  # node 1 trusts itself 1 + 2
        w = np.diag(own) + np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        w /= w.sum(axis=1, keepdims=True)
        assert a.values[5] == pytest.approx(np.linalg.matrix_power(w, 5) @ x0)
        assert not np.allclose(a.values, sim.degroot(g, x0, steps=5).values)

    def test_no_consensus(self):
        # two stubborn sources: two closed classes
        g = DiGraph([("a", "m"), ("b", "m")])
        r = sim.degroot(g, {"a": 0.0, "b": 1.0, "m": 0.3}, steps=50)
        assert r.meta["consensus"] is None and r.values[-1] == pytest.approx([0.0, 0.5, 1.0], abs=1e-9)
        # periodic listening structure without self-trust: oscillates, no consensus
        r = sim.degroot(Graph([(0, 1)]), [0.0, 1.0], steps=4, self_weight=0.0)
        assert r.meta["consensus"] is None and r.values[-1].tolist() == [0.0, 1.0]

    def test_errors(self):
        g = Graph([(0, 1)])
        with pytest.raises(ValueError):
            sim.degroot(g, {0: 1.0})
        with pytest.raises(ValueError):
            sim.degroot(g, [1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            sim.degroot(g, [1.0, 2.0], self_weight=-1)
        with pytest.raises(ValueError):
            sim.degroot(g, [1.0, math.nan])


class TestBoundedConfidence:
    def test_undirected_conserves_the_mean(self):
        g = complete(30)
        r = sim.bounded_confidence(g, epsilon=0.3, mu=0.3, steps=40, seed=1)
        assert r.values.sum(axis=1) == pytest.approx(np.full(r.T, r.values[0].sum()))
        assert r.values.min() >= 0.0 and r.values.max() <= 1.0
        assert r.meta["domain"] == (0.0, 1.0)

    def test_wide_confidence_reaches_one_cluster(self):
        g = complete(25)
        r = sim.bounded_confidence(g, epsilon=1.01, mu=0.5, steps=200, seed=2)
        assert len(r.meta["clusters"]) == 1
        assert r.meta["clusters"][0] == pytest.approx(r.values[0].mean())
        assert np.ptp(r.values[-1]) < 1e-6

    def test_narrow_confidence_freezes(self):
        g = complete(6)
        x0 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        r = sim.bounded_confidence(g, x0, epsilon=0.15, steps=20, seed=3)
        assert r.meta["settled"] and r.T == 1 and len(r.meta["clusters"]) == 6

    def test_directed_only_heads_move(self):
        g = DiGraph([("a", "b"), ("b", "c")])
        r = sim.bounded_confidence(g, {"a": 0.5, "b": 0.6, "c": 0.7}, epsilon=0.5, steps=30, seed=4)
        assert np.all(r.values[:, 0] == 0.5)
        assert r.values[-1] == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)
        for frame in r.edge_activity:
            assert all(e in {("a", "b"), ("b", "c")} for e in frame)

    def test_errors(self):
        g = complete(3)
        with pytest.raises(ValueError):
            sim.bounded_confidence(g, [0.1, 0.2, 1.3])
        with pytest.raises(ValueError):
            sim.bounded_confidence(g, mu=0.7)
        with pytest.raises(ValueError):
            sim.bounded_confidence(g, epsilon=0)
