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

"""Tests for aryagraph.sim: SimulationResult and compartmental models.

Statistical checks compare seeded Monte-Carlo estimates with exact values
(computed here independently) using tolerances of about four standard errors.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aryagraph import DiGraph, Graph
from aryagraph.core.exceptions import NodeNotFound
import aryagraph.sim as sim
from aryagraph.sim import SimulationResult


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def complete(n: int) -> Graph:
    g = Graph()
    g.add_nodes(range(n))
    g.add_edges((i, j) for i in range(n) for j in range(i + 1, n))
    return g


def gnp(n: int, p: float, seed: int, directed: bool = False) -> Graph:
    rng = np.random.default_rng(seed)
    g = DiGraph() if directed else Graph()
    g.add_nodes(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and (directed or i < j) and rng.random() < p:
                g.add_edge(i, j)
    return g


def empty(n: int) -> Graph:
    g = Graph()
    g.add_nodes(range(n))
    return g


def sir_final_size_exact(n: int, beta: float, gamma: float, i0: int = 1) -> tuple[float, float]:
    """Mean and std of the number ever infected in the Markovian SIR on K_n (embedded-chain DP)."""
    p = np.zeros((n + 1, n + 2))
    p[n - i0, i0] = 1.0
    m1 = m2 = 0.0
    for s in range(n - i0, -1, -1):
        for i in range(n - s, 0, -1):
            mass = p[s, i]
            if mass == 0.0:
                continue
            q = beta * s / (beta * s + gamma)
            if s > 0:
                p[s - 1, i + 1] += mass * q
            p[s, i - 1] += mass * (1.0 - q)
        m1 += p[s, 0] * (n - s)
        m2 += p[s, 0] * (n - s) ** 2
    return m1, math.sqrt(m2 - m1 * m1)


def sis_qsd_mean(n: int, beta: float, gamma: float) -> float:
    """Mean infected count under the quasi-stationary distribution of SIS on K_n."""
    q = np.zeros((n, n))  # states i = 1..n
    for i in range(1, n + 1):
        up, down = beta * i * (n - i), gamma * i
        k = i - 1
        q[k, k] = -(up + down)
        if i < n:
            q[k, k + 1] = up
        if i > 1:
            q[k, k - 1] = down
    vals, vecs = np.linalg.eig(q.T)
    v = np.real(vecs[:, np.argmax(np.real(vals))])
    v = v / v.sum()
    return float(v @ np.arange(1, n + 1))


def time_average(res: SimulationResult, state: str, t0: float) -> float:
    """Exact time average of the count in *state* over [t0, t_end] (events-recorded result)."""
    c = res.counts()[state].astype(float)
    t = res.times
    lo = np.maximum(t[:-1], t0)
    hi = np.maximum(t[1:], t0)
    return float(np.sum(c[:-1] * (hi - lo)) / (t[-1] - t0))


# ---------------------------------------------------------------------- #
# SimulationResult
# ---------------------------------------------------------------------- #
def make_result() -> SimulationResult:
    g = Graph([("a", "b"), ("b", "c")])
    values = np.array([[0, 1, 0], [0, 1, 1], [2, 2, 1], [2, 2, 2]])
    return SimulationResult(
        graph=g, nodes=["a", "b", "c"], times=[0.0, 1.0, 2.5, 4.0], values=values, kind="categorical",
        states=["S", "I", "R"], roles={}, model="toy", params={"beta": 1}, seed=3,
        edge_activity=[[], [("b", "c")], [], []],
    )


class TestSimulationResult:
    def test_shape_and_frames(self):
        r = make_result()
        assert (r.T, r.N) == (4, 3)
        assert r.values.dtype == np.int8
        assert r.frame(1) == {"a": "S", "b": "I", "c": "I"}
        assert r.frame(-1) == r.final() == {"a": "R", "b": "R", "c": "R"}
        assert r.history("c") == ["S", "I", "I", "R"]
        with pytest.raises(NodeNotFound):
            r.history("zz")

    def test_at_uses_last_frame_not_after(self):
        r = make_result()
        assert r.at(0.0) == r.frame(0)
        assert r.at(2.4999) == r.frame(1)
        assert r.at(2.5) == r.frame(2)
        assert r.at(100) == r.frame(3)
        with pytest.raises(ValueError):
            r.at(-0.1)

    def test_counts_fractions_summary(self):
        r = make_result()
        c = r.counts()
        assert c["S"].tolist() == [2, 1, 0, 0]
        assert c["I"].tolist() == [1, 2, 1, 0]
        assert c["R"].tolist() == [0, 0, 2, 3]
        total = sum(c.values())
        assert np.all(total == 3)
        assert np.allclose(r.fractions()["R"], [0, 0, 2 / 3, 1])
        s = r.summary()
        assert s["I"] == {"initial": 1, "final": 0, "peak": 2, "peak_time": 1.0}

    def test_first_time_and_peak(self):
        r = make_result()
        ft = r.first_time("I")
        assert ft["b"] == 0.0 and ft["c"] == 1.0 and math.isnan(ft["a"])
        assert r.first_time("R")["c"] == 4.0
        assert r.peak("I") == (1.0, 2)
        with pytest.raises(ValueError):
            r.peak("X")

    def test_records_and_pandas(self):
        r = make_result()
        rec = r.to_records()
        assert len(rec) == r.T * r.N
        assert rec[4] == {"frame": 1, "time": 1.0, "node": "b", "state": "I"}
        pd = pytest.importorskip("pandas")
        df = r.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["frame", "time", "node", "state"]
        assert len(df) == 12

    def test_roles_default_and_repr(self):
        r = make_result()
        assert r.roles == {"S": "neutral", "I": "critical", "R": "good"}
        text = repr(r)
        assert "toy" in text and "3 nodes" in text and "4 frames" in text and "R=3" in text

    def test_continuous(self):
        g = Graph([(0, 1)])
        r = SimulationResult(
            graph=g, nodes=[0, 1], times=[0, 1], values=[[0.0, 1.0], [-0.5, 0.5]], kind="continuous",
            states=[], roles={}, model="c", params={}, seed=None,
        )
        assert r.meta["domain"] == (-0.5, 1.0) and r.meta["diverging"] is False
        s = r.summary()
        assert np.allclose(s["mean"], [0.5, 0.0]) and np.allclose(s["max"], [1.0, 0.5])
        assert r.frame(1) == {0: -0.5, 1: 0.5}
        assert r.to_records()[0] == {"frame": 0, "time": 0.0, "node": 0, "value": 0.0}
        with pytest.raises(ValueError):
            r.counts()
        assert "mean=" in repr(r)

    @pytest.mark.parametrize(
        "change",
        [
            {"times": [1.0, 2.0, 3.0, 4.0]},
            {"times": [0.0, 1.0, 1.0, 4.0]},
            {"values": np.zeros((3, 3))},
            {"values": np.full((4, 3), 5)},
            {"roles": {"S": "loud"}},
            {"roles": {"X": "good"}},
            {"kind": "fuzzy"},
            {"edge_activity": [[]]},
        ],
    )
    def test_validation(self, change):
        base = dict(
            graph=Graph([("a", "b")]), nodes=["a", "b", "c"], times=[0.0, 1.0, 2.5, 4.0],
            values=np.zeros((4, 3), dtype=int), kind="categorical", states=["S", "I", "R"], roles={},
            model="toy", params={}, seed=None,
        )
        base.update(change)
        with pytest.raises(ValueError):
            SimulationResult(**base)

    def test_hooks_exist(self):
        r = make_result()
        assert callable(r.animate) and callable(r.plot)


# ---------------------------------------------------------------------- #
# model definition
# ---------------------------------------------------------------------- #
class TestModel:
    def test_factories(self):
        assert sim.SIR(0.3, 0.1).states == ["S", "I", "R"]
        assert sim.SIS(0.3, 0.1).name == "SIS" and sim.SIS(0.3, 0.1).states == ["S", "I"]
        assert sim.SEIR(1, 2, 3).params == {"beta": 1, "sigma": 2, "gamma": 3}
        assert sim.SEIRD(1, 2, 3, 4).roles["D"] == "muted"
        assert sim.SIRS(1, 1, 1).name == "SIRS"
        assert "S→I" in repr(sim.SIR(0.3, 0.1))

    def test_terminates(self):
        assert sim.SIR(1, 1).terminates and sim.SEIRD(1, 1, 1, 1).terminates and sim.SI(1).terminates
        assert not sim.SIS(1, 1).terminates and not sim.SIRS(1, 1, 1).terminates
        assert sim.SIRS(1, 1, 0.0).terminates  # zero-rate transitions never fire

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"states": ["S", "S"]},
            {"states": ["S", "I"], "spontaneous": [("I", "X", 1.0)]},
            {"states": ["S", "I"], "spontaneous": [("I", "I", 1.0)]},
            {"states": ["S", "I"], "spontaneous": [("I", "S", -1.0)]},
            {"states": ["S", "I"], "induced": [("S", "I", 1.0)]},
            {"states": ["S", "I"], "roles": {"S": "loud"}},
        ],
    )
    def test_invalid_models(self, kwargs):
        with pytest.raises(ValueError):
            sim.CompartmentalModel(**kwargs)

    def test_custom_model_roles(self):
        m = sim.CompartmentalModel(["healthy", "sick"], induced=[("healthy", "sick", "sick", 1.0)],
                                   roles={"sick": "serious"})
        assert m.name == "healthy-sick"
        assert m.roles == {"healthy": "accent", "sick": "serious"}


# ---------------------------------------------------------------------- #
# initial conditions and options
# ---------------------------------------------------------------------- #
class TestInitial:
    def test_counts_fractions_and_lists(self):
        g = empty(20)
        m = sim.SIR(1.0, 0.0)
        r = m.simulate(g, initial={"I": 3, "R": [0, 1]}, steps=0, seed=1)
        c = r.counts()
        assert (c["S"][0], c["I"][0], c["R"][0]) == (15, 3, 2)
        assert r.frame(0)[0] == "R" and r.frame(0)[1] == "R"
        r = m.simulate(g, initial={"I": 0.25}, steps=0, seed=1)
        assert r.counts()["I"][0] == 5
        r = m.simulate(g, initial={"I": 0.025}, steps=0, seed=1)  # 0.5 rounds half up
        assert r.counts()["I"][0] == 1
        r = m.simulate(g, initial={"S": [3], "I": 0.5}, steps=0, seed=1)  # explicit nodes go first
        assert r.counts()["I"][0] == 10 and r.frame(0)[3] == "S"
        with pytest.raises(ValueError):  # a fraction is of all nodes, and node 3 is taken
            m.simulate(g, initial={"S": [3], "I": 1.0}, steps=0, seed=1)

    def test_int_is_count_node_via_list(self):
        g = empty(10)
        r = sim.SIR(1, 0).simulate(g, initial={"I": 4}, steps=0, seed=0)
        assert r.counts()["I"][0] == 4
        r = sim.SIR(1, 0).simulate(g, initial={"I": [4]}, steps=0, seed=0)
        assert r.frame(0)[4] == "I" and r.counts()["I"][0] == 1

    def test_node_mapping(self):
        g = Graph([("a", "b"), ("b", "c")])
        r = sim.SIR(1, 1).simulate(g, initial={"a": "I", "c": "R"}, steps=0)
        assert r.frame(0) == {"a": "I", "b": "S", "c": "R"}

    def test_default_initial_is_one_infected(self):
        r = sim.SIR(0.5, 0.5).simulate(complete(10), steps=0, seed=3)
        assert r.counts()["I"][0] == 1

    @pytest.mark.parametrize(
        "initial, exc",
        [
            ({"I": 11}, ValueError),
            ({"I": 1.5}, ValueError),
            ({"I": [0], "R": [0]}, ValueError),
            ({"I": [99]}, NodeNotFound),
            ({"I": True}, TypeError),
            ({"X": 1, 0: "I"}, ValueError),
            ({0: "X"}, ValueError),
            ([0, 1], TypeError),
        ],
    )
    def test_invalid_initial(self, initial, exc):
        with pytest.raises(exc):
            sim.SIR(1, 1).simulate(empty(10), initial=initial, steps=1)

    def test_option_validation(self):
        g = complete(5)
        with pytest.raises(ValueError):
            sim.sir(g, 1, 1, method="euler")
        with pytest.raises(ValueError):
            sim.sir(g, 1, 1, record="all")
        with pytest.raises(ValueError):
            sim.sir(g, 1, 1, dt=0)
        with pytest.raises(ValueError):
            sim.sir(g, 1, 1, t_max=5, steps=5)
        with pytest.raises(ValueError):
            sim.sir(g, 1, 1, steps=-1)
        with pytest.raises(ValueError):
            sim.sir(g, 1, 1, n_frames=1)
        with pytest.raises(ValueError):
            sim.sir(Graph([(0, 1, {"weight": -1})]), 1, 1, weight="weight")

    def test_empty_and_single_node(self):
        for method in ("discrete", "gillespie"):
            r = sim.sir(Graph(), 1, 1, method=method)
            assert (r.T, r.N) == (1, 0) and r.meta["absorbed"]
            g = empty(1)
            r = sim.sir(g, 1, 1, method=method, seed=0)
            assert r.final() == {0: "R"} and r.meta["absorbed"]


# ---------------------------------------------------------------------- #
# dynamics: invariants
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["discrete", "gillespie"])
class TestInvariants:
    def test_reproducible(self, method):
        g = gnp(60, 0.08, 1)
        a = sim.sir(g, 0.4, 0.2, initial={"I": 3}, method=method, seed=11)
        b = sim.sir(g, 0.4, 0.2, initial={"I": 3}, method=method, seed=11)
        c = sim.sir(g, 0.4, 0.2, initial={"I": 3}, method=method, seed=12)
        assert np.array_equal(a.values, b.values) and np.array_equal(a.times, b.times)
        assert a.edge_activity == b.edge_activity and a.seed == 11
        assert not (np.array_equal(a.values[-1], c.values[-1]) and np.array_equal(a.times, c.times))

    def test_conservation_and_monotone_sir(self, method):
        g = gnp(80, 0.06, 2)
        r = sim.sir(g, 0.5, 0.2, initial={"I": 4}, method=method, dt=0.5, seed=5)
        c = r.counts()
        assert np.all(c["S"] + c["I"] + c["R"] == 80)
        assert np.all(np.diff(c["S"]) <= 0) and np.all(np.diff(c["R"]) >= 0)
        assert np.all(np.diff(r.times) > 0) and r.times[0] == 0

    def test_absorbs_and_stops(self, method):
        g = gnp(50, 0.1, 3)
        r = sim.sir(g, 0.5, 0.5, initial={"I": 2}, method=method, seed=2)
        assert r.meta["absorbed"] and r.counts()["I"][-1] == 0
        assert r.times[-1] == pytest.approx(r.meta["t_end"])
        # SI saturates the component of the seeds, then stops
        r = sim.si(complete(15), 0.3, method=method, seed=4)
        assert r.meta["absorbed"] and r.counts()["I"][-1] == 15

    def test_event_log_is_consistent(self, method):
        """Every logged event is a legal transition and every source is an infectious neighbour."""
        g = gnp(40, 0.12, 4, directed=True)
        model = sim.SEIRD(0.8, 0.7, 0.3, 0.2)
        r = model.simulate(g, initial={"I": 3}, method=method, dt=0.2, seed=9, log_events=True, record="events")
        legal = {(t.source, t.target) for t in model.transitions}
        state = r.frame(0)
        last_t = 0.0
        for t, node, old, new, src in r.meta["events"]:
            assert t >= last_t
            last_t = t
            assert state[node] == old and (old, new) in legal
            if (old, new) == ("S", "E"):
                assert src is not None and g.has_edge(src, node)  # influence travels along arcs
            else:
                assert src is None
            if method == "gillespie" and src is not None:
                assert state[src] == "I"
            state[node] = new
        assert state == r.final()
        # for synchronous updates the source must be infectious at the start of the step
        if method == "discrete":
            for t, node, old, new, src in r.meta["events"]:
                if src is not None:
                    before = r.frame(r.index_at(t) - 1)
                    assert before[src] == "I"

    def test_record_modes_agree(self, method):
        g = gnp(50, 0.1, 5)
        kw = dict(initial={"I": 2}, method=method, dt=0.25, seed=21)
        ev = sim.sir(g, 0.6, 0.3, record="events", **kw)
        fr = sim.sir(g, 0.6, 0.3, record="frames", **kw)
        fin = sim.sir(g, 0.6, 0.3, record="final", **kw)
        for i, t in enumerate(fr.times):
            assert ev.at(t) == fr.frame(i)
        assert fin.T == 2 and fin.final() == ev.final() == fr.final()
        assert fin.times[-1] == ev.times[-1] == fr.times[-1]
        # transmissions are conserved across recording modes
        n_tx = sum(len(a) for a in ev.edge_activity)
        assert n_tx == sum(len(a) for a in fr.edge_activity) == sum(len(a) for a in fin.edge_activity)
        assert n_tx == ev.counts()["R"][-1] + ev.counts()["I"][-1] - 2

    def test_directed_spreads_along_arcs_only(self, method):
        g = DiGraph([(0, 1), (1, 2), (2, 3)])
        r = sim.si(g, 5.0, initial={"I": [2]}, method=method, seed=0)
        assert r.final() == {0: "S", 1: "S", 2: "I", 3: "I"}
        assert [e for f in r.edge_activity for e in f] == [(2, 3)]

    def test_self_loops_do_not_act_on_their_node(self, method):
        # "A → B driven by A-neighbours": a lone A node with a self-loop has no neighbour to act on it
        model = sim.CompartmentalModel("BA", induced=[("A", "B", "A", 5.0)])  # default state B
        g = Graph([(0, 0), (1, 2)])
        r = model.simulate(g, initial={"A": [0, 1]}, method=method, seed=0, t_max=20)
        assert r.final() == {0: "A", 1: "A", 2: "B"} and r.meta["absorbed"]
        r = model.simulate(g, initial={"A": [0, 1, 2]}, method=method, seed=0, t_max=20)
        final = r.final()
        assert final[0] == "A" and r.meta["absorbed"] and "B" in (final[1], final[2])
        if method == "gillespie":  # one conversion leaves the other node without an A-neighbour
            assert sorted([final[1], final[2]]) == ["A", "B"]

    def test_zero_weight_never_transmits(self, method):
        g = Graph([(0, 1, {"w": 0.0}), (1, 2, {"w": 1.0})])
        r = sim.si(g, 10.0, initial={"I": [1]}, method=method, weight="w", seed=0, t_max=50)
        assert r.final() == {0: "S", 1: "I", 2: "I"}

    def test_default_horizon(self, method):
        r = sim.sis(complete(20), 0.2, 0.5, initial={"I": 5}, method=method, seed=1)
        assert r.params["t_max"] == sim.DEFAULT_T_MAX
        assert r.times[-1] == pytest.approx(sim.DEFAULT_T_MAX)
        r = sim.sir(complete(20), 0.2, 0.5, initial={"I": 5}, method=method, seed=1)
        assert r.params["t_max"] == math.inf and r.meta["absorbed"]


def test_discrete_steps_and_thinning():
    g = complete(30)
    r = sim.sis(g, 0.05, 0.1, initial={"I": 10}, steps=40, dt=0.5, seed=1)
    assert r.T == 41 and r.times[-1] == pytest.approx(20.0)
    assert np.allclose(np.diff(r.times), 0.5)
    t = sim.sis(g, 0.05, 0.1, initial={"I": 10}, steps=40, dt=0.5, seed=1, n_frames=11)
    assert t.T == 11 and np.allclose(t.times, np.linspace(0, 20, 11))
    for i, time in enumerate(t.times):
        assert t.frame(i) == r.at(time)
    assert sum(map(len, t.edge_activity)) == sum(map(len, r.edge_activity))


def test_gillespie_grid():
    g = complete(30)
    r = sim.sis(g, 0.05, 0.1, initial={"I": 10}, t_max=20, method="gillespie", seed=1, n_frames=41)
    assert np.allclose(r.times, np.linspace(0, 20, 41))
    # an absorbing run stops at the absorption time, which is the last frame
    r = sim.sir(g, 0.001, 1.0, initial={"I": 2}, t_max=100, method="gillespie", seed=1)
    assert r.meta["absorbed"] and r.times[-1] < 100
    assert r.times[-1] == r.meta["t_end"] and np.allclose(np.diff(r.times[:-1]), 1.0)


# ---------------------------------------------------------------------- #
# dynamics: statistics against exact values
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["discrete", "gillespie"])
def test_competing_transitions_split_by_rate(method):
    """I→R at γ and I→D at μ: the share of deaths is μ/(γ+μ) for both solvers."""
    n = 4000
    model = sim.CompartmentalModel("IRD", spontaneous=[("I", "R", 0.3), ("I", "D", 0.1)])
    r = model.simulate(empty(n), initial={"I": 1.0}, method=method, dt=1.0, seed=3)
    assert r.meta["absorbed"]
    share = r.counts()["D"][-1] / n
    se = math.sqrt(0.25 * 0.75 / n)
    assert abs(share - 0.25) < 4 * se


def test_discrete_step_probabilities():
    """Spontaneous: 1 - exp(-γ dt); induced: 1 - exp(-β dt Σw) with independent neighbours."""
    n = 5000
    r = sim.sir(empty(n), 1.0, 0.7, initial={"I": 1.0}, steps=1, dt=0.5, seed=2)
    p = 1 - math.exp(-0.7 * 0.5)
    assert abs(r.counts()["R"][1] / n - p) < 4 * math.sqrt(p * (1 - p) / n)
    # stars: a susceptible centre with infected leaves of weights 1, 2, 3
    g = Graph()
    m = 3000
    for k in range(m):
        for leaf, w in ((1, 1.0), (2, 2.0), (3, 3.0)):
            g.add_edge(("c", k), (leaf, k), weight=w)
    infected = [(leaf, k) for k in range(m) for leaf in (1, 2, 3)]
    r = sim.si(g, 0.1, initial={"I": infected}, steps=1, dt=1.0, weight="weight", seed=4)
    p = 1 - math.exp(-0.1 * 6.0)
    got = (r.counts()["I"][1] - 3 * m) / m
    assert abs(got - p) < 4 * math.sqrt(p * (1 - p) / m)
    # the credited source is drawn in proportion to its weight: 1/6, 2/6, 3/6
    src = np.array([s[0] for f in r.edge_activity for s, _ in f])
    for leaf in (1, 2, 3):
        share, q = np.mean(src == leaf), leaf / 6
        assert abs(share - q) < 4 * math.sqrt(q * (1 - q) / src.size)


def test_gillespie_waiting_times_are_exponential():
    n = 3000
    r = sim.sir(empty(n), 1.0, 0.5, initial={"I": 1.0}, method="gillespie", record="events", seed=6)
    t = r.first_time("R").to_array()
    assert abs(t.mean() - 2.0) < 4 * 2.0 / math.sqrt(n)
    assert abs(np.median(t) - 2.0 * math.log(2)) < 0.1


def test_gillespie_source_proportional_to_weight():
    g = Graph()
    m = 2000
    for k in range(m):
        g.add_edge(("c", k), ("light", k), weight=1.0)
        g.add_edge(("c", k), ("heavy", k), weight=3.0)
    infected = [(kind, k) for k in range(m) for kind in ("light", "heavy")]
    r = sim.si(g, 1.0, initial={"I": infected}, method="gillespie", weight="weight", seed=8)
    src = [s[0] for f in r.edge_activity for s, _ in f]
    assert len(src) == m
    share = src.count("heavy") / m
    assert abs(share - 0.75) < 4 * math.sqrt(0.75 * 0.25 / m)


def test_sir_final_size_matches_exact_distribution():
    """Complete graph: the mean final size equals the exact embedded-chain value,
    for Gillespie (exact) and for the discrete solver with a small step."""
    n, beta, gamma, runs = 40, 2.0 / 39, 1.0, 400
    mean, std = sir_final_size_exact(n, beta, gamma)
    g = complete(n)
    tol = 4 * std / math.sqrt(runs)
    gil = sim.run_ensemble(sim.sir, runs, g=g, beta=beta, gamma=gamma, initial={"I": 1},
                           method="gillespie", seed=1)
    dis = sim.run_ensemble(sim.sir, runs, g=g, beta=beta, gamma=gamma, initial={"I": 1},
                           method="discrete", dt=0.05, seed=2)
    m_gil = gil.final_sizes["R"].mean()
    m_dis = dis.final_sizes["R"].mean()
    assert abs(m_gil - mean) < tol
    assert abs(m_dis - mean) < tol + 0.5  # O(dt) discretisation bias allowance
    assert abs(m_gil - m_dis) < math.sqrt(2) * tol + 0.5
    assert abs(gil.final_sizes["R"].std(ddof=1) - std) < 0.15 * std


def test_sir_major_outbreak_matches_mean_field():
    """Among major outbreaks the final fraction approaches z = 1 - exp(-R0 z) (loose)."""
    n, r0 = 100, 2.0
    beta = r0 / (n - 1)
    z = 0.5
    for _ in range(200):
        z = 1 - math.exp(-r0 * z)
    ens = sim.run_ensemble(sim.sir, 120, g=complete(n), beta=beta, gamma=1.0, initial={"I": 1},
                           method="gillespie", seed=5)
    finals = ens.final_sizes["R"] / n
    major = finals[finals > 0.25]
    assert major.size > 40
    assert abs(major.mean() - z) < 0.06
    # minor outbreaks happen with probability ≈ 1/R0 in a branching approximation
    assert abs(np.mean(finals <= 0.25) - 1 / r0) < 0.15


@pytest.mark.parametrize("method", ["gillespie", "discrete"])
def test_sis_endemic_level(method):
    n, beta, gamma = 50, 0.06, 1.0
    exact = sis_qsd_mean(n, beta, gamma) / n
    mean_field = 1 - gamma / (beta * (n - 1))
    assert abs(exact - mean_field) < 0.03  # sanity of the oracle itself
    levels = []
    for s in range(5):
        r = sim.sis(complete(n), beta, gamma, initial={"I": 25}, t_max=60, method=method, dt=0.02,
                    record="events", seed=s)
        levels.append(time_average(r, "I", 10.0) / n)
    assert abs(np.mean(levels) - exact) < 0.02
