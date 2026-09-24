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

"""Tests for aryagraph.sim.run_ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from aryagraph import Graph
import aryagraph.sim as sim


def ring(n: int) -> Graph:
    g = Graph([(i, (i + 1) % n) for i in range(n)])
    g.add_edges((i, (i + 3) % n) for i in range(0, n, 2))
    return g


class TestEnsemble:
    def test_shapes_bands_and_conservation(self):
        g = ring(40)
        ens = sim.run_ensemble(sim.sir, 30, g=g, beta=0.8, gamma=0.3, initial={"I": 2}, method="gillespie",
                               seed=1, n_points=51)
        assert ens.times.shape == (51,) and ens.samples.shape == (30, 51, 3)
        assert np.all(ens.samples.sum(axis=2) == 40)
        assert ens.states == ["S", "I", "R"] and ens.roles["I"] == "critical"
        for s in ens.states:
            q = ens.quantiles[s]
            assert list(q) == [5, 25, 50, 75, 95]
            assert np.all(q[5] <= q[25]) and np.all(q[25] <= q[50]) and np.all(q[50] <= q[75])
            assert np.all(q[75] <= q[95])
            assert ens.mean[s] == pytest.approx(ens.samples[:, :, ens.states.index(s)].mean(axis=0))
            assert np.all(ens.std[s] >= 0)
        lo, hi = ens.band("I")
        assert np.all(lo <= hi)
        assert sum(ens.fractions().values()) == pytest.approx(np.ones(51))
        assert ens.final_sizes["R"].shape == (30,) and np.all(ens.final_sizes["I"] == 0)
        assert "30 runs" in repr(ens) and callable(ens.plot)

    def test_reproducible_and_replayable(self):
        g = ring(30)
        kw = dict(g=g, beta=0.5, gamma=0.4, initial={"I": 1}, method="gillespie")
        a = sim.run_ensemble(sim.sir, 12, seed=7, keep=True, **kw)
        b = sim.run_ensemble(sim.sir, 12, seed=7, **kw)
        c = sim.run_ensemble(sim.sir, 12, seed=8, **kw)
        assert np.array_equal(a.samples, b.samples) and a.seeds == b.seeds
        assert a.seeds != c.seeds and len(set(a.seeds)) == 12
        assert b.results is None and len(a.results) == 12
        for i in (0, 5, 11):
            run = sim.sir(seed=a.seeds[i], **kw)
            assert np.array_equal(run.values, a.results[i].values)
            assert run.counts()["R"][-1] == a.final_sizes["R"][i]

    def test_step_interpolation_is_exact(self):
        g = ring(30)
        ens = sim.run_ensemble(sim.sis, 5, g=g, beta=0.6, gamma=0.5, initial={"I": 5}, method="gillespie",
                               t_max=10, record="events", seed=3, keep=True)
        for k, res in enumerate(ens.results):
            for j, t in enumerate(ens.times):
                frame = res.at(t)
                assert ens.samples[k, j, 1] == sum(1 for s in frame.values() if s == "I")

    def test_linear_interpolation_and_discrete_runs(self):
        g = ring(20)
        ens = sim.run_ensemble(sim.sir, 8, g=g, beta=0.5, gamma=0.5, initial={"I": 2}, seed=2,
                               interpolation="linear", n_points=11)
        assert np.all(np.abs(ens.samples.sum(axis=2) - 20) < 1e-9)
        # runs that ended early hold their final counts on the common grid
        assert np.all(ens.samples[:, -1, :] == np.stack([ens.final_sizes[s] for s in ens.states], axis=1))

    def test_other_simulators(self):
        g = ring(25)
        ens = sim.run_ensemble(sim.independent_cascade, 50, g=g, seeds=[0], p=0.5, seed=4)
        est = sim.influence_spread(g, [0], "ic", 4000, p=0.5, seed=5)
        assert abs(ens.final_sizes["active"].mean() - est.mean) < 4 * np.hypot(
            ens.final_sizes["active"].std(ddof=1) / np.sqrt(50), est.stderr)
        model = sim.SEIR(0.6, 0.5, 0.3)
        ens = sim.run_ensemble(model.simulate, 5, g=g, method="gillespie", seed=1, quantiles=(10, 90))
        assert list(ens.quantiles["E"]) == [10, 90] and ens.model == "SEIR"

    def test_errors(self):
        g = ring(10)
        with pytest.raises(TypeError, match="categorical"):
            sim.run_ensemble(sim.bounded_confidence, 2, g=g, steps=2)
        with pytest.raises(ValueError):
            sim.run_ensemble(sim.sir, 0, g=g, beta=1, gamma=1)
        with pytest.raises(ValueError):
            sim.run_ensemble(sim.sir, 2, g=g, beta=1, gamma=1, n_points=1)
        with pytest.raises(ValueError):
            sim.run_ensemble(sim.sir, 2, g=g, beta=1, gamma=1, interpolation="cubic")
