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

"""Tests for aryagraph.sim heat diffusion and Kuramoto oscillators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aryagraph import DiGraph, Graph
import aryagraph.sim as sim


def weighted_graph(n: int, p: float, seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    g = Graph()
    g.add_nodes(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(i, j, weight=float(rng.uniform(0.5, 2.0)))
    return g


def dense(g: Graph, weight: str | None = "weight") -> np.ndarray:
    nodes = list(g)
    idx = {v: i for i, v in enumerate(nodes)}
    a = np.zeros((len(nodes), len(nodes)))
    for u, v, d in g.edges.data():
        if u != v:
            w = d.get(weight, 1.0) if weight else 1.0
            a[idx[u], idx[v]] = a[idx[v], idx[u]] = w
    return a


class TestHeat:
    def test_two_nodes_closed_form(self):
        g = Graph([("a", "b", 0.7)])
        r = sim.heat_diffusion(g, {"a": 3.0, "b": 1.0}, rate=1.5, t_max=2.0, n_frames=21, weight="weight")
        t = r.times
        mean, half = 2.0, 1.0
        decay = np.exp(-2 * 1.5 * 0.7 * t)
        assert r.values[:, 0] == pytest.approx(mean + half * decay, abs=1e-12)
        assert r.values[:, 1] == pytest.approx(mean - half * decay, abs=1e-12)
        assert r.kind == "continuous" and r.meta["domain"] == pytest.approx((1.0, 3.0))

    @pytest.mark.parametrize("normalized", [False, True])
    def test_conserves_total_heat(self, normalized):
        g = weighted_graph(25, 0.2, 1)
        g.add_node("isolated")
        x0 = {v: float(i % 5) - 1.0 for i, v in enumerate(g)}
        r = sim.heat_diffusion(g, x0, t_max=3.0, weight="weight", normalized=normalized)
        total = sum(x0.values())
        assert r.meta["total"] == pytest.approx(np.full(r.T, total), abs=1e-10)
        assert r.values[:, -1] == pytest.approx(x0["isolated"])  # nothing flows to an isolated node
        assert r.meta["diverging"]

    def test_matches_matrix_exponential(self):
        linalg = pytest.importorskip("scipy.linalg")
        g = weighted_graph(15, 0.3, 2)
        x0 = np.random.default_rng(0).normal(size=15)
        a = dense(g)
        lap = np.diag(a.sum(1)) - a
        r = sim.heat_diffusion(g, x0, rate=0.8, t_max=2.0, n_frames=5, weight="weight")
        for t, row in zip(r.times, r.values):
            assert row == pytest.approx(linalg.expm(-0.8 * t * lap) @ x0, abs=1e-10)
        rn = sim.heat_diffusion(g, x0, rate=0.8, t_max=2.0, n_frames=5, weight="weight", normalized=True)
        deg = a.sum(1)
        inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)  # isolated nodes: zero column
        for t, row in zip(rn.times, rn.values):
            assert row == pytest.approx(linalg.expm(-0.8 * t * lap @ np.diag(inv)) @ x0, abs=1e-10)

    def test_steady_state(self):
        g = Graph([(0, 1), (1, 2), (3, 4)])
        x0 = {0: 3.0, 3: 1.0}
        r = sim.heat_diffusion(g, x0, t_max=60.0)
        ss = r.meta["steady_state"]
        assert ss == pytest.approx({0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5, 4: 0.5})
        assert r.values[-1] == pytest.approx(ss.to_array(r.nodes), abs=1e-9)
        rn = sim.heat_diffusion(g, x0, t_max=80.0, normalized=True)
        # random-walk Laplacian: steady state ∝ degree within each component
        assert rn.meta["steady_state"] == pytest.approx({0: 0.75, 1: 1.5, 2: 0.75, 3: 0.5, 4: 0.5})
        assert rn.values[-1] == pytest.approx(rn.meta["steady_state"].to_array(rn.nodes), abs=1e-9)

    def test_directed_is_symmetrised_and_unit_source(self):
        d = DiGraph([(0, 1), (1, 2)])
        u = Graph([(0, 1), (1, 2)])
        a = sim.heat_diffusion(d, 0, t_max=1.0)
        b = sim.heat_diffusion(u, 0, t_max=1.0)
        assert np.allclose(a.values, b.values) and a.values[0].tolist() == [1.0, 0.0, 0.0]
        # maximum principle for the combinatorial Laplacian
        assert a.values.min() >= -1e-12 and a.values.max() <= 1 + 1e-12

    def test_self_loops_carry_no_heat(self):
        g = Graph([(0, 1), (1, 2)])
        looped = Graph([(0, 1), (1, 2), (1, 1)])
        for normalized in (False, True):
            a = sim.heat_diffusion(g, 0, normalized=normalized)
            b = sim.heat_diffusion(looped, 0, normalized=normalized)
            assert np.allclose(a.values, b.values)

    def test_errors_and_empty(self):
        g = Graph([(0, 1)])
        with pytest.raises(ValueError):
            sim.heat_diffusion(g, [1.0])
        with pytest.raises(ValueError):
            sim.heat_diffusion(g, 0, t_max=0)
        with pytest.raises(ValueError):
            sim.heat_diffusion(g, 0, rate=-1)
        with pytest.raises(ValueError):
            sim.heat_diffusion(g, 0, n_frames=1)
        r = sim.heat_diffusion(Graph(), {}, t_max=1.0, n_frames=3)
        assert r.values.shape == (3, 0)


class TestKuramoto:
    def test_two_oscillators_closed_form(self):
        """Equal frequencies: tan(φ/2) = tan(φ0/2)·exp(-2Kt) for the phase difference φ."""
        g = Graph([(0, 1)])
        phi0, k = 2.0, 0.8
        r = sim.kuramoto(g, coupling=k, frequencies=0.3, initial=[0.0, phi0], t_max=3.0, dt=0.01)
        raw = np.unwrap(r.values, axis=0)
        phi = raw[:, 1] - raw[:, 0]
        expected = 2 * np.arctan(math.tan(phi0 / 2) * np.exp(-2 * k * r.times))
        assert phi == pytest.approx(expected, abs=1e-8)
        # the common drift is the shared frequency
        assert (raw[:, 0] + raw[:, 1]) / 2 == pytest.approx(phi0 / 2 + 0.3 * r.times, abs=1e-8)

    def test_phase_locking(self):
        g = Graph([(0, 1)])
        k, dw = 1.0, 0.6
        r = sim.kuramoto(g, coupling=k, frequencies={0: 0.0, 1: dw}, initial=[0.0, 0.0], t_max=30.0, dt=0.02)
        raw = np.unwrap(r.values, axis=0)
        assert raw[-1, 1] - raw[-1, 0] == pytest.approx(math.asin(dw / (2 * k)), abs=1e-8)

    def test_synchronises_and_order_parameter(self):
        g = weighted_graph(20, 0.4, 3)
        r = sim.kuramoto(g, coupling=2.0, frequencies=0.0, t_max=15.0, dt=0.05, seed=4)
        order = r.meta["order_parameter"]
        z = np.exp(1j * r.values).mean(axis=1)
        assert order == pytest.approx(np.abs(z))
        assert order[-1] > 0.999
        assert np.all((r.values >= 0) & (r.values < 2 * math.pi))
        assert r.meta["domain"] == (0.0, 2 * math.pi) and r.meta["cyclic"]

    def test_reproducible_thinning_and_normalize(self):
        g = weighted_graph(10, 0.5, 5)
        a = sim.kuramoto(g, seed=1, t_max=2.0, dt=0.1)
        b = sim.kuramoto(g, seed=1, t_max=2.0, dt=0.1, n_frames=5)
        assert a.T == 21 and b.T == 5
        for i, t in enumerate(b.times):
            assert b.values[i] == pytest.approx(a.values[a.index_at(t + 1e-9)])
        assert a.meta["frequencies"] == b.meta["frequencies"]
        c = sim.kuramoto(g, seed=1, t_max=2.0, dt=0.1, normalize=True, coupling=3.0)
        assert not np.allclose(c.values, a.values)

    def test_directed_source_is_unaffected(self):
        g = DiGraph([(0, 1)])
        r = sim.kuramoto(g, coupling=1.0, frequencies=[0.5, 0.0], initial=[0.0, 1.0], t_max=2.0, dt=0.01)
        assert r.values[:, 0] == pytest.approx(np.mod(0.5 * r.times, 2 * math.pi), abs=1e-12)
