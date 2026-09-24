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

"""Tests for aryagraph.sim random walks and their stationary distribution."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aryagraph import DiGraph, Graph
from aryagraph.core.exceptions import NodeNotFound, NotConnected
import aryagraph.sim as sim

nx = pytest.importorskip("networkx")


def connected_gnp(n: int, p: float, seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    while True:
        g = Graph()
        g.add_nodes(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    g.add_edge(i, j, weight=float(rng.integers(1, 4)))
        if nx.is_connected(nx.Graph(list(g.edges))) and len(nx.Graph(list(g.edges))) == n:
            return g


class TestRandomWalk:
    def test_frames_states_and_moves(self):
        g = connected_gnp(20, 0.2, 1)
        r = sim.random_walk(g, walkers=3, steps=60, seed=2)
        traj = r.meta["trajectories"]
        assert traj.shape == (3, 61) and r.T == 61
        assert r.states == ["unvisited", "visited", "current"]
        visited = set()
        for t in range(r.T):
            here = {r.nodes[i] for i in traj[:, t]}
            visited |= here
            f = r.frame(t)
            assert {n for n, s in f.items() if s == "current"} == here
            assert {n for n, s in f.items() if s != "unvisited"} == visited
            if t:
                moves = sorted(r.edge_activity[t])
                expect = sorted((r.nodes[a], r.nodes[b]) for a, b in zip(traj[:, t - 1], traj[:, t]))
                assert moves == expect and all(g.has_edge(u, v) for u, v in moves)
        assert r.edge_activity[0] == []
        visits = r.meta["visits"]
        assert sum(visits.values()) == pytest.approx(1.0)
        assert visits.to_array(r.nodes) == pytest.approx(np.bincount(traj.ravel(), minlength=20) / traj.size)

    def test_reproducible(self):
        g = connected_gnp(15, 0.3, 3)
        a = sim.random_walk(g, walkers=4, steps=30, seed=7, restart=0.2)
        b = sim.random_walk(g, walkers=4, steps=30, seed=7, restart=0.2)
        assert np.array_equal(a.meta["trajectories"], b.meta["trajectories"])
        assert np.array_equal(a.values, b.values) and a.seed == 7

    def test_visit_frequencies_converge_to_degree_share(self):
        g = connected_gnp(15, 0.3, 4)
        r = sim.random_walk(g, walkers=20, steps=20000, seed=5, n_frames=50)
        deg = np.array([g.degree(n) for n in r.nodes], dtype=float)
        target = deg / deg.sum()
        assert np.max(np.abs(r.meta["visits"].to_array(r.nodes) - target)) < 0.005
        # weighted walk: weighted degree share
        r = sim.random_walk(g, walkers=20, steps=20000, weight="weight", seed=6, n_frames=50)
        wdeg = np.array([g.degree(n, weight="weight") for n in r.nodes], dtype=float)
        assert np.max(np.abs(r.meta["visits"].to_array(r.nodes) - wdeg / wdeg.sum())) < 0.005

    def test_weighted_transition_probabilities(self):
        g = Graph([("c", "light", {"w": 1.0}), ("c", "heavy", {"w": 3.0})])
        r = sim.random_walk(g, walkers=4000, steps=1, start="c", weight="w", seed=1)
        share = np.mean(r.meta["trajectories"][:, 1] == r.nodes.index("heavy"))
        assert abs(share - 0.75) < 4 * math.sqrt(0.75 * 0.25 / 4000)

    def test_dangling_teleports_uniformly(self):
        g = DiGraph([(0, 1)])
        g.add_nodes([2, 3])
        r = sim.random_walk(g, walkers=8000, steps=1, start=1, seed=2)
        counts = np.bincount(r.meta["trajectories"][:, 1], minlength=4) / 8000
        assert np.all(np.abs(counts - 0.25) < 4 * math.sqrt(0.25 * 0.75 / 8000))
        assert r.meta["teleports"] == 8000 and r.edge_activity[1] == []

    def test_restart(self):
        g = connected_gnp(12, 0.3, 5)
        r = sim.random_walk(g, walkers=2, steps=20, start=[0, 5], restart=1.0, seed=3)
        traj = r.meta["trajectories"]
        assert np.all(traj[0] == 0) and np.all(traj[1] == r.nodes.index(5))
        assert r.meta["restarts"] == 40 and sum(map(len, r.edge_activity)) == 0

    def test_self_loop_lets_walker_stay(self):
        g = Graph([(0, 0), (0, 1)])
        r = sim.random_walk(g, walkers=4000, steps=1, start=0, seed=4)
        assert abs(np.mean(r.meta["trajectories"][:, 1] == 0) - 0.5) < 0.04

    def test_thinned_frames_keep_all_moves(self):
        g = connected_gnp(10, 0.4, 6)
        full = sim.random_walk(g, walkers=2, steps=40, seed=8)
        thin = sim.random_walk(g, walkers=2, steps=40, seed=8, n_frames=5)
        assert thin.times.tolist() == [0, 10, 20, 30, 40]
        assert sum(map(len, thin.edge_activity)) == sum(map(len, full.edge_activity))
        for i, t in enumerate(thin.times):
            assert thin.frame(i) == full.at(t)

    def test_errors(self):
        g = connected_gnp(5, 0.6, 7)
        with pytest.raises(ValueError):
            sim.random_walk(Graph())
        with pytest.raises(ValueError):
            sim.random_walk(g, walkers=0)
        with pytest.raises(ValueError):
            sim.random_walk(g, restart=1.5)
        with pytest.raises(NodeNotFound):
            sim.random_walk(g, start="zz")
        with pytest.raises(ValueError):
            sim.random_walk(g, walkers=2, start=[0, 1, 2])


class TestStationary:
    def test_undirected_is_degree_share(self):
        g = connected_gnp(12, 0.3, 8)
        pi = sim.stationary_distribution(g)
        deg = g.degree()
        assert all(pi[n] == pytest.approx(deg[n] / (2 * g.num_edges)) for n in g)
        piw = sim.stationary_distribution(g, weight="weight")
        wdeg = g.degree(weight="weight")
        total = sum(wdeg.values())
        assert all(piw[n] == pytest.approx(wdeg[n] / total) for n in g)
        assert piw.name == "stationary"

    def test_directed_solves_balance_equations(self):
        g = DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 1), (1, 4)])
        g.add_edge(4, 4)
        g.add_edge(4, 0, weight=2.0)
        pi = sim.stationary_distribution(g, weight="weight")
        p = sim.transition_matrix(g, weight="weight")
        v = pi.to_array(list(g))
        assert v @ p == pytest.approx(v, abs=1e-12) and v.sum() == pytest.approx(1.0)
        # networkx oracle: PageRank without damping is the stationary distribution
        G = nx.DiGraph()
        G.add_weighted_edges_from((u, w, d.get("weight", 1.0)) for u, w, d in g.edges.data())
        ref = nx.pagerank(G, alpha=1.0, weight="weight", tol=1e-13, max_iter=100000)
        assert all(pi[n] == pytest.approx(ref[n], abs=1e-8) for n in g)

    def test_dangling_and_transient_nodes(self):
        g = DiGraph([("s", "a"), ("a", "b"), ("b", "a"), ("b", "d")])  # d dangling, s transient
        pi = sim.stationary_distribution(g)
        G = nx.DiGraph(list(g.edges))
        ref = nx.pagerank(G, alpha=1.0, tol=1e-13, max_iter=100000)
        assert all(pi[n] == pytest.approx(ref[n], abs=1e-8) for n in g)
        assert pi["s"] > 0  # the uniform teleport from d reaches s

    def test_walk_frequencies_match_directed_stationary(self):
        g = DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 1), (1, 4), (4, 0), (4, 4)])
        pi = sim.stationary_distribution(g).to_array(list(g))
        r = sim.random_walk(g, walkers=20, steps=20000, seed=9, n_frames=10)
        assert np.max(np.abs(r.meta["visits"].to_array(list(g)) - pi)) < 0.006

    def test_not_unique(self):
        g = Graph([(0, 1), (2, 3)])
        with pytest.raises(NotConnected):
            sim.stationary_distribution(g)
        with pytest.raises(NotConnected):
            sim.stationary_distribution(DiGraph([(0, 1), (1, 1), (0, 2), (2, 2)]))
        assert sim.stationary_distribution(Graph()) == {}
        # a bipartite (periodic) chain still has a unique stationary distribution
        pi = sim.stationary_distribution(Graph([(0, 1), (1, 2)]))
        assert pi == {0: 0.25, 1: 0.5, 2: 0.25}
