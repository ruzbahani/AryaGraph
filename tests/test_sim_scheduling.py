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

"""Tests for aryagraph.sim DAG scheduling (discrete-event simulation and Monte Carlo)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aryagraph import DAG, DiGraph, Graph
from aryagraph.core.exceptions import CycleError, GraphTypeError
import aryagraph.sim as sim
from aryagraph.sim.scheduling import STATES


def random_dag(n: int, p: float, seed: int, *, zero: bool = False) -> DAG:
    rng = np.random.default_rng(seed)
    dag = DAG()
    for i in range(n):
        d = 0.0 if zero and rng.random() < 0.2 else float(rng.integers(1, 9)) / 2
        dag.add_node(i, duration=d)
    dag.add_edges([(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p])
    return dag


def cpm_length(dag: DAG, dur: dict) -> float:
    finish: dict = {}
    for v in dag.topological_order():
        finish[v] = max((finish[u] for u in dag.pred[v]), default=0.0) + dur[v]
    return max(finish.values(), default=0.0)


def durations(dag: DAG) -> dict:
    return {v: dag.nodes[v].get("duration", 1.0) for v in dag}


def check_schedule(dag: DAG, res: sim.ScheduleResult, workers: int | None, capacity: dict | None = None) -> None:
    """Precedence, one task per worker at a time, frames consistent with the task log.

    Work conservation: while a worker is idle, no ready task fits the free
    resources (*capacity*, demands from node attributes).
    """
    done = {r.node: r for r in res.tasks if r.status == "done"}
    assert set(done) == set(dag) and len(res.tasks) == len(dag)
    for v, run in done.items():
        assert run.end - run.start == pytest.approx(dag.nodes[v]["duration"])
        for u in dag.pred[v]:
            assert run.start >= done[u].end
    by_worker: dict = {}
    for run in res.tasks:
        by_worker.setdefault(run.worker, []).append(run)
        if workers is not None:
            assert 0 <= run.worker < workers
    for runs in by_worker.values():
        runs.sort(key=lambda r: (r.start, r.end))
        for a, b in zip(runs, runs[1:]):
            assert b.start >= a.end
    assert res.makespan == max((r.end for r in res.tasks), default=0.0)
    assert np.all(np.diff(res.times) > 0) and res.times[0] == 0 and res.times[-1] == res.makespan
    running = res.values == STATES.index("running")
    ready = res.values == STATES.index("ready")
    nodes = res.nodes
    for i in range(res.T):
        idle = workers is None or running[i].sum() < workers
        if workers is not None:
            assert running[i].sum() <= workers
        if not idle:
            continue
        used = {r: sum(dag.nodes[nodes[k]].get(r, 0) for k in np.flatnonzero(running[i])) for r in capacity or {}}
        for k in np.flatnonzero(ready[i]):
            need = {r: dag.nodes[nodes[k]].get(r, 0) for r in capacity or {}}
            assert any(used[r] + need[r] > capacity[r] for r in need), "a ready task could have started"
    for i, t in enumerate(res.times):
        frame = res.frame(i)
        for v in dag:
            run = done[v]
            if run.end <= t:  # frames hold the state after every event at their time
                assert frame[v] == "done"
            elif run.start <= t < run.end:
                assert frame[v] == "running"
    assert all(s == "done" for s in res.final().values())


class TestSimulateSchedule:
    @pytest.mark.parametrize("seed", range(5))
    def test_unlimited_workers_is_cpm(self, seed):
        dag = random_dag(25, 0.15, seed, zero=seed % 2 == 0)
        res = sim.simulate_schedule(dag, workers=None)
        assert res.makespan == cpm_length(dag, durations(dag))  # same arithmetic: exact
        check_schedule(dag, res, None)
        path = res.critical_path
        assert all(dag.has_edge(a, b) for a, b in zip(path, path[1:]))
        assert sum(dag.nodes[v]["duration"] for v in path) == pytest.approx(res.makespan)
        assert not dag.pred[path[0]] and path[-1] in {r.node for r in res.tasks if r.end == res.makespan}

    @pytest.mark.parametrize("seed", range(3))
    def test_one_worker_is_sum_of_durations(self, seed):
        dag = random_dag(20, 0.2, seed)
        res = sim.simulate_schedule(dag, workers=1)
        assert res.makespan == pytest.approx(sum(durations(dag).values()))
        assert res.utilization[0] == pytest.approx(1.0) and res.idle_time == pytest.approx(0.0)
        check_schedule(dag, res, 1)

    @pytest.mark.parametrize("policy", list(sim.scheduling.POLICIES) + [lambda v: -v])
    @pytest.mark.parametrize("workers", [1, 2, 3])
    def test_policies_give_valid_schedules(self, policy, workers):
        for seed in range(3):
            dag = random_dag(18, 0.2, 10 + seed)
            res = sim.simulate_schedule(dag, workers=workers, policy=policy, seed=seed)
            check_schedule(dag, res, workers)
            lower = max(cpm_length(dag, durations(dag)), sum(durations(dag).values()) / workers)
            assert res.makespan >= lower - 1e-9
            busy = sum(r.duration for r in res.tasks)
            assert res.idle_time == pytest.approx(workers * res.makespan - busy)
            assert res.utilization["overall"] == pytest.approx(busy / (workers * res.makespan))

    def test_policy_hand_checked_example(self):
        dag = DAG()
        dag.add_node("B", duration=1)
        dag.add_node("D", duration=1)
        dag.add_node("A", duration=3)
        dag.add_node("C", duration=3)
        dag.add_edge("A", "C")
        assert sim.simulate_schedule(dag, workers=2, policy="critical_path").makespan == 6
        assert sim.simulate_schedule(dag, workers=2, policy="longest_first").makespan == 6
        assert sim.simulate_schedule(dag, workers=2, policy="fifo").makespan == 7
        assert sim.simulate_schedule(dag, workers=2, policy="shortest_first").makespan == 7
        res = sim.simulate_schedule(dag, workers=2, policy="critical_path")
        assert res.critical_path == ["A", "C"]
        # FIFO: B and D take both workers, A waits for the one B frees
        res = sim.simulate_schedule(dag, workers=2, policy="fifo")
        assert res.critical_path == ["B", "A", "C"]
        assert res.frame(0) == {"B": "running", "D": "running", "A": "ready", "C": "pending"}
        assert res.edge_activity[res.index_at(4)] == [("A", "C")]

    def test_deterministic_under_seed(self):
        dag = random_dag(20, 0.2, 3)
        kw = dict(workers=3, policy="random", jitter=("uniform", 0.4), failure_rate=0.2, max_retries=2)
        a = sim.simulate_schedule(dag, seed=5, **kw)
        b = sim.simulate_schedule(dag, seed=5, **kw)
        c = sim.simulate_schedule(dag, seed=6, **kw)
        assert a.tasks == b.tasks and np.array_equal(a.values, b.values) and a.seed == 5
        assert a.tasks != c.tasks

    def test_failures_retries_and_blocking(self):
        dag = DAG([("a", "b"), ("b", "c"), ("a", "d"), ("x", "d")])
        for v in dag:
            dag.nodes[v]["duration"] = 2.0
        res = sim.simulate_schedule(dag, workers=2, failure_rate={"b": 1.0}, max_retries=2, seed=0)
        runs_b = res.runs_of("b")
        assert [r.attempt for r in runs_b] == [1, 2, 3] and all(r.status == "failed" for r in runs_b)
        assert runs_b[1].start == runs_b[0].end  # re-queued immediately (a worker is free)
        assert res.blocked == ["c"] and res.final()["c"] == "failed" and res.runs_of("c") == []
        assert res.final()["d"] == "done"
        # the blocked task turns failed at the moment b fails for good
        t_fail = runs_b[-1].end
        assert res.at(t_fail)["c"] == "failed" and res.at(t_fail - 1e-9)["c"] == "pending"
        assert res.makespan == t_fail == 8.0
        assert "failed" in repr(res)

    def test_failure_attribute_and_retry_statistics(self):
        dag = DAG()
        dag.add_node("t", duration=1.0, flaky=0.3)
        mc = sim.monte_carlo_schedule(dag, runs=4000, workers=1, failure_rate="flaky", max_retries=50, seed=2)
        # attempts ~ Geometric(0.7): mean makespan 1/0.7
        assert abs(mc.mean - 1 / 0.7) < 4 * mc.stderr
        assert mc.makespans.min() == 1.0 and set(np.unique(mc.makespans)) <= set(range(1, 52))

    def test_resources_are_respected(self):
        dag = random_dag(16, 0.1, 4)
        rng = np.random.default_rng(0)
        for v in dag:
            dag.nodes[v]["gpu"] = int(rng.integers(0, 3))
        res = sim.simulate_schedule(dag, workers=4, resources={"gpu": 2})
        check_schedule(dag, res, 4, {"gpu": 2})
        assert (res.values == STATES.index("ready")).any()  # the constraint did bind
        for i in range(res.T):
            running = [v for v, s in res.frame(i).items() if s == "running"]
            assert sum(dag.nodes[v]["gpu"] for v in running) <= 2
        with pytest.raises(ValueError, match="could never run"):
            sim.simulate_schedule(dag, resources={"gpu": 1})

    def test_zero_durations_and_trivial_graphs(self):
        dag = DAG([("a", "b"), ("b", "c")])
        res = sim.simulate_schedule(dag, duration=0.0)
        assert res.T == 1 and res.makespan == 0 and res.final() == {"a": "done", "b": "done", "c": "done"}
        assert res.critical_path == ["a", "b", "c"]
        empty = sim.simulate_schedule(DAG())
        assert empty.makespan == 0 and empty.tasks == [] and empty.critical_path == []

    def test_duration_specs_and_digraph_input(self):
        g = DiGraph([("a", "b"), ("a", "c")])
        assert sim.simulate_schedule(g, duration=2.0, workers=None).makespan == 4.0
        assert sim.simulate_schedule(g, duration={"a": 1, "b": 5, "c": 2}, workers=None).makespan == 6.0
        assert sim.simulate_schedule(g, duration=lambda v, d: len(v) * 3.0, workers=1).makespan == 9.0
        assert sim.simulate_schedule(g, duration="missing", workers=None).makespan == 2.0  # default 1
        res = sim.simulate_schedule(g, workers=None, jitter=lambda v, d, rng: d + 1.0)
        assert res.makespan == 4.0

    def test_errors(self):
        with pytest.raises(CycleError) as info:
            sim.simulate_schedule(DiGraph([(0, 1), (1, 2), (2, 0)]))
        assert info.value.cycle[0] == info.value.cycle[-1]
        with pytest.raises(CycleError):
            sim.monte_carlo_schedule(DiGraph([(0, 1), (1, 0)]))
        with pytest.raises(GraphTypeError):
            sim.simulate_schedule(Graph([(0, 1)]))
        dag = DAG([(0, 1)])
        for kw in (
            {"workers": 0},
            {"policy": "magic"},
            {"jitter": ("gauss", 0.1)},
            {"jitter": ("uniform", 2.0)},
            {"duration": -1.0},
            {"failure_rate": 1.5},
            {"max_retries": -1},
            {"resources": {"cpu": 0}},
        ):
            with pytest.raises(ValueError):
                sim.simulate_schedule(dag, **kw)
        dag.nodes[0].update(min=3, mode=2, max=4)
        with pytest.raises(ValueError):
            sim.simulate_schedule(dag, jitter=("triangular", "min", "mode", "max"))

    def test_result_contract(self):
        dag = random_dag(8, 0.3, 5)
        res = sim.simulate_schedule(dag, workers=2)
        assert isinstance(res, sim.SimulationResult)
        assert res.states == ["pending", "ready", "running", "done", "failed"]
        assert res.roles == {"pending": "muted", "ready": "warning", "running": "accent", "done": "good",
                             "failed": "critical"}
        assert len(res.edge_activity) == res.T and callable(res.gantt)
        for i in range(res.T):
            for u, v in res.edge_activity[i]:
                assert dag.has_edge(u, v) and res.frame(i)[u] == "done"
        assert "makespan" in repr(res)


class TestMonteCarlo:
    def test_deterministic_durations(self):
        dag = random_dag(15, 0.2, 6)
        mc = sim.monte_carlo_schedule(dag, runs=10)
        length = cpm_length(dag, durations(dag))
        assert np.all(mc.makespans == length) and mc.cpm_length == length and mc.std == 0
        crit = [v for v, c in mc.criticality.items() if c == 1.0]
        assert crit and all(c in (0.0, 1.0) for c in mc.criticality.values())
        res = sim.simulate_schedule(dag, workers=None)
        assert set(res.critical_path) <= set(crit)

    def test_parallel_tasks_exact_expectations(self):
        dag = DAG()
        dag.add_node("a", duration=1.0)
        dag.add_node("b", duration=1.0)
        runs = 20000
        mc = sim.monte_carlo_schedule(dag, runs=runs, jitter=("uniform", 0.5), seed=3)
        # max of two U(0.5, 1.5): mean 0.5 + 2/3, P80 = 0.5 + sqrt(0.8)
        assert abs(mc.mean - (0.5 + 2 / 3)) < 4 * mc.stderr
        assert mc.percentiles["P80"] == pytest.approx(0.5 + math.sqrt(0.8), abs=0.01)
        assert mc.criticality["a"] + mc.criticality["b"] == pytest.approx(1.0)
        assert abs(mc.criticality["a"] - 0.5) < 4 * math.sqrt(0.25 / runs)
        assert mc.percentiles["P50"] <= mc.percentiles["P80"] <= mc.percentiles["P95"]
        assert mc.probability(mc.percentiles["P80"]) == pytest.approx(0.8, abs=1e-3)
        assert mc.percentile(95) == mc.percentiles["P95"]

    @pytest.mark.parametrize("kind, mean", [("triangular", (1 + 2 + 6) / 3), ("pert", (1 + 4 * 2 + 6) / 6)])
    def test_three_point_estimates(self, kind, mean):
        dag = DAG()
        dag.add_node("t", lo=1.0, mid=2.0, hi=6.0)
        mc = sim.monte_carlo_schedule(dag, runs=20000, jitter=(kind, "lo", "mid", "hi"), seed=4)
        assert abs(mc.mean - mean) < 4 * mc.stderr
        assert mc.makespans.min() >= 1.0 and mc.makespans.max() <= 6.0
        assert mc.cpm_length == 2.0  # the mode is the nominal duration

    def test_fast_and_event_paths_agree(self):
        dag = random_dag(12, 0.25, 7)
        kw = dict(runs=3000, jitter=("uniform", 0.3))
        fast = sim.monte_carlo_schedule(dag, seed=1, **kw)
        slow = sim.monte_carlo_schedule(dag, seed=2, resources={"cpu": 1e9}, **kw)  # forces full simulation
        se = math.hypot(fast.stderr, slow.stderr)
        assert abs(fast.mean - slow.mean) < 4 * se
        for v in dag:
            p = fast.criticality[v]
            assert abs(p - slow.criticality[v]) < 4 * math.sqrt(max(p * (1 - p), 1e-4) * 2 / 3000) + 1e-9

    def test_limited_workers_and_reproducible(self):
        dag = random_dag(12, 0.2, 8)
        a = sim.monte_carlo_schedule(dag, runs=200, workers=1, jitter=("uniform", 0.2), seed=9)
        b = sim.monte_carlo_schedule(dag, runs=200, workers=1, jitter=("uniform", 0.2), seed=9)
        assert np.array_equal(a.makespans, b.makespans)
        assert all(c == 1.0 for c in a.criticality.values())  # one worker: every task is on the chain
        total = sum(durations(dag).values())
        assert abs(a.mean - total) < 4 * a.stderr
        assert a.workers == 1 and "runs" in repr(a) and callable(a.plot)
