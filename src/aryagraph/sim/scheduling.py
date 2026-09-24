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

"""DAG task execution as a discrete-event simulation.

Tasks are the nodes of a DAG and arcs are precedences: a task becomes *ready*
when all its predecessors are *done*, and a pool of workers runs ready tasks
(non-preemptive list scheduling). The dispatcher is work-conserving: whenever
a worker is free and a ready task fits the free resources, one starts; the
*policy* decides which.

A task moves through ``pending → ready → running → done``. An attempt fails
with the task's failure probability (the failure shows at the end of the
attempt, which occupies its worker for the whole drawn duration); a failed task
is re-queued immediately while it has retries left, otherwise it is *failed*
for good and every descendant becomes *failed* too, at the same instant, because
they can never run (see :attr:`ScheduleResult.blocked`).

>>> res = aryagraph.sim.simulate_schedule(dag, workers=2, jitter=("uniform", 0.2), seed=1)
>>> res.makespan, res.critical_path
>>> mc = aryagraph.sim.monte_carlo_schedule(dag, runs=2000, jitter=("pert", "min", "mode", "max"))
>>> mc.percentiles["P80"], mc.criticality.top(3)
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Hashable

import numpy as np

from ..core.dag import DAG, _kahn
from ..core.exceptions import CycleError, GraphTypeError
from ..core.results import NodeMap
from ..core.utils import make_rng
from .base import SimulationResult, _seed_record

STATES = ["pending", "ready", "running", "done", "failed"]
PENDING, READY, RUNNING, DONE, FAILED = range(5)
POLICIES = ("critical_path", "fifo", "longest_first", "shortest_first", "random")
_RES_TOL = 1e-9


@dataclass(frozen=True)
class TaskRun:
    """One attempt of a task: which worker ran it, when, and how it ended (``"done"``/``"failed"``).

    ``attempt`` counts from 1.
    """

    node: Hashable
    worker: int
    start: float
    end: float
    attempt: int
    status: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(repr=False)
class ScheduleResult(SimulationResult):
    """A simulated schedule: frames at every event time plus the task log.

    Attributes
    ----------
    tasks:
        Every attempt as a :class:`TaskRun`, in dispatch order.
    makespan:
        Time at which the last task finished (or failed).
    utilization:
        Busy fraction of the makespan per worker id, plus ``"overall"``.
    critical_path:
        The chain of tasks that determined the makespan, found by backtracking
        from the last task to finish. Each attempt is linked to what released
        its start: its own failed previous attempt, a predecessor finishing, or
        (when it had to wait) the attempt that freed its worker or resources. With
        unlimited workers this is a longest path of the realised durations.
    idle_time:
        Total idle worker-time, ``workers × makespan − busy time`` (with
        unlimited workers, over the worker ids actually used).
    blocked:
        Tasks that never ran because an ancestor failed for good.
    """

    tasks: list[TaskRun] = field(default_factory=list)
    makespan: float = 0.0
    utilization: dict = field(default_factory=dict)
    critical_path: list = field(default_factory=list)
    idle_time: float = 0.0
    blocked: list = field(default_factory=list)

    def runs_of(self, node: Hashable) -> list[TaskRun]:
        """Attempts of *node*, in order."""
        return [r for r in self.tasks if r.node == node]

    def gantt(self, **kwargs: Any):
        """Gantt chart of :attr:`tasks` by worker; see :func:`aryagraph.charts.gantt.gantt`."""
        from ..charts.gantt import gantt

        return gantt(self, **kwargs)

    def __repr__(self) -> str:
        done = sum(1 for r in self.tasks if r.status == "done")
        workers = self.params.get("workers")
        return (
            f"<ScheduleResult: {self.N} tasks, workers={workers if workers is not None else 'unlimited'}, "
            f"makespan={self.makespan:.4g}, utilization={self.utilization.get('overall', 0.0):.1%}, "
            f"{done} done, {self.N - done} failed>"
        )


# ---------------------------------------------------------------------- #
# static structure
# ---------------------------------------------------------------------- #
@dataclass
class _Plan:
    nodes: list
    index: dict
    preds: list[list[int]]
    succ: list[list[int]]
    topo: list[int]
    pos: list[int]


def _plan(dag: Any) -> _Plan:
    if not getattr(dag, "directed", False):
        raise GraphTypeError("scheduling needs a DAG or an acyclic DiGraph")
    if isinstance(dag, DAG):
        order = dag.topological_order()
    else:
        order, cycle = _kahn(dag)
        if cycle is not None:
            raise CycleError("cannot schedule a graph with the cycle " + " → ".join(map(repr, cycle)), cycle)
    nodes = list(dag._node)
    index = {v: i for i, v in enumerate(nodes)}
    preds = [[index[u] for u in dag._pred[v]] for v in nodes]
    succ = [[index[u] for u in dag._succ[v]] for v in nodes]
    topo = [index[v] for v in order]
    pos = [0] * len(nodes)
    for k, i in enumerate(topo):
        pos[i] = k
    return _Plan(nodes, index, preds, succ, topo, pos)


def _check_nonneg(values: np.ndarray, nodes: list, what: str) -> np.ndarray:
    bad = ~(np.isfinite(values) & (values >= 0))
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        raise ValueError(f"{what} of task {nodes[k]!r} must be finite and non-negative, got {values[k]!r}")
    return values


def _base_durations(dag: Any, nodes: list, duration: Any) -> np.ndarray:
    if isinstance(duration, str):
        vals = [dag._node[v].get(duration, 1.0) for v in nodes]
    elif isinstance(duration, Mapping):
        missing = [v for v in nodes if v not in duration]
        if missing:
            raise ValueError(f"no duration for task {missing[0]!r}")
        vals = [duration[v] for v in nodes]
    elif callable(duration):
        vals = [duration(v, dag._node[v]) for v in nodes]
    elif isinstance(duration, Real) and not isinstance(duration, bool):
        vals = [duration] * len(nodes)
    else:
        raise TypeError("duration must be an attribute name, a number, a {node: duration} mapping or a callable")
    return _check_nonneg(np.array(vals, dtype=float).reshape(len(nodes)), nodes, "duration")


class _Durations:
    """Nominal durations and a sampler of realised ones.

    *jitter*: ``None``; ``("uniform", frac)`` for nominal × U(1 − frac, 1 + frac);
    ``("triangular", lo, mode, hi)`` / ``("pert", lo, mode, hi)`` with the
    node-attribute names of a three-point estimate (a missing attribute falls
    back to the nominal duration; the mode becomes the nominal); or a callable
    ``f(node, nominal, rng) -> duration``.
    """

    def __init__(self, dag: Any, nodes: list, duration: Any, jitter: Any) -> None:
        self.nodes = nodes
        self.nominal = _base_durations(dag, nodes, duration)
        self.kind: str | None = None
        self.desc: Any = None
        if jitter is None:
            return
        if callable(jitter):
            self.kind, self.fn, self.desc = "callable", jitter, "callable"
            return
        spec = (jitter,) if isinstance(jitter, str) else tuple(jitter)
        name = str(spec[0]).lower() if spec else ""
        self.desc = spec
        if name == "uniform":
            if len(spec) != 2:
                raise ValueError("uniform jitter is ('uniform', frac)")
            frac = float(spec[1])
            if not 0.0 <= frac <= 1.0:
                raise ValueError(f"uniform jitter fraction must lie in [0, 1], got {frac!r}")
            self.kind, self.frac = "uniform", frac
        elif name in ("triangular", "pert"):
            attrs = spec[1:] or ("min", "mode", "max")
            if len(attrs) != 3:
                raise ValueError(f"{name} jitter is ({name!r}, min_attr, mode_attr, max_attr)")
            cols = [
                np.array([dag._node[v].get(a, d) for v, d in zip(nodes, self.nominal.tolist())], dtype=float)
                for a in attrs
            ]
            lo, mode, hi = (_check_nonneg(c, nodes, f"{name} {a!r}") for c, a in zip(cols, attrs))
            bad = (lo > mode) | (mode > hi)
            if bad.any():
                k = int(np.flatnonzero(bad)[0])
                raise ValueError(f"task {nodes[k]!r} needs min <= mode <= max, got {lo[k]}, {mode[k]}, {hi[k]}")
            self.kind, self.lo, self.mode, self.hi = name, lo, mode, hi
            self.spread = hi > lo
            if name == "pert":
                width = np.where(self.spread, hi - lo, 1.0)
                self.alpha = 1.0 + 4.0 * (mode - lo) / width
                self.beta = 1.0 + 4.0 * (hi - mode) / width
            self.nominal = mode.copy()
        else:
            raise ValueError(
                f"unknown jitter {jitter!r}; use ('uniform', frac), ('triangular'|'pert', lo, mode, hi) or a callable"
            )

    def draw(self, i: int, rng: np.random.Generator) -> float:
        kind = self.kind
        if kind is None:
            return float(self.nominal[i])
        if kind == "uniform":
            return float(self.nominal[i] * (1.0 + self.frac * (2.0 * rng.random() - 1.0)))
        if kind == "callable":
            x = float(self.fn(self.nodes[i], float(self.nominal[i]), rng))
            if not (math.isfinite(x) and x >= 0):
                raise ValueError(f"jitter returned an invalid duration {x!r} for task {self.nodes[i]!r}")
            return x
        if not self.spread[i]:
            return float(self.lo[i])
        if kind == "triangular":
            return float(rng.triangular(self.lo[i], self.mode[i], self.hi[i]))
        return float(self.lo[i] + (self.hi[i] - self.lo[i]) * rng.beta(self.alpha[i], self.beta[i]))

    def draw_all(self, rng: np.random.Generator, runs: int) -> np.ndarray:
        """``(runs, n)`` independent draws."""
        n = self.nominal.size
        kind = self.kind
        if kind is None:
            return np.tile(self.nominal, (runs, 1))
        if kind == "uniform":
            return self.nominal * (1.0 + self.frac * (2.0 * rng.random((runs, n)) - 1.0))
        if kind == "callable":
            return np.array([[self.draw(i, rng) for i in range(n)] for _ in range(runs)]).reshape(runs, n)
        out = np.tile(self.lo, (runs, 1))
        s = self.spread
        k = int(s.sum())
        if k:
            lo, hi = self.lo[s], self.hi[s]
            if kind == "triangular":
                out[:, s] = rng.triangular(lo, self.mode[s], hi, size=(runs, k))
            else:
                out[:, s] = lo + (hi - lo) * rng.beta(self.alpha[s], self.beta[s], size=(runs, k))
        return out


def _failure_probs(dag: Any, nodes: list, failure_rate: Any) -> np.ndarray:
    if isinstance(failure_rate, str):
        vals = np.array([float(dag._node[v].get(failure_rate, 0.0)) for v in nodes])
    elif isinstance(failure_rate, Mapping):
        vals = np.array([float(failure_rate.get(v, 0.0)) for v in nodes])
    elif isinstance(failure_rate, Real) and not isinstance(failure_rate, bool):
        vals = np.full(len(nodes), float(failure_rate))
    else:
        raise TypeError("failure_rate must be a probability, a node-attribute name or a {node: probability} mapping")
    bad = ~((vals >= 0) & (vals <= 1))
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        raise ValueError(f"failure probability of task {nodes[k]!r} must lie in [0, 1], got {vals[k]!r}")
    return vals


def _resources(dag: Any, nodes: list, resources: Mapping | None) -> tuple[list[float], list[list[float]]] | None:
    if not resources:
        return None
    names = list(resources)
    cap = [float(resources[r]) for r in names]
    for r, c in zip(names, cap):
        if not (math.isfinite(c) and c > 0):
            raise ValueError(f"capacity of resource {r!r} must be positive, got {c!r}")
    demand = []
    for v in nodes:
        row = [float(dag._node[v].get(r, 0.0)) for r in names]
        for r, need, c in zip(names, row, cap):
            if not (math.isfinite(need) and need >= 0):
                raise ValueError(f"task {v!r} needs an invalid amount {need!r} of {r!r}")
            if need > c + _RES_TOL:
                raise ValueError(f"task {v!r} needs {need:g} {r!r} but only {c:g} exist; it could never run")
        demand.append(row)
    return cap, demand


def _check_policy(policy: Any) -> None:
    if not callable(policy) and policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES} or a key function, got {policy!r}")


def _check_workers(workers: Any) -> int | None:
    if workers is None:
        return None
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError(f"workers must be a positive integer or None (unlimited), got {workers!r}")
    return int(workers)


# ---------------------------------------------------------------------- #
# the event loop
# ---------------------------------------------------------------------- #
@dataclass
class _Outcome:
    times: list[float]
    frames: list[list[int]]
    fired: list[list[tuple[int, int]]]
    records: list[tuple]  # (task, worker, start, end, attempt, status, start_seq, end_seq)
    busy: list[float]
    lanes: int
    blocked: list[int]
    chain: list[int]

    @property
    def makespan(self) -> float:
        return self.times[-1]


def _run(
    plan: _Plan,
    durations: _Durations,
    fail_p: np.ndarray,
    workers: int | None,
    max_retries: int,
    policy: Any,
    res: tuple | None,
    rng: np.random.Generator,
) -> _Outcome:
    n = len(plan.nodes)
    succ, preds, pos = plan.succ, plan.preds, plan.pos
    nominal = durations.nominal.tolist()
    fail = fail_p.tolist()
    state = [PENDING] * n
    remaining = [len(p) for p in preds]
    attempts = [0] * n
    unlimited = workers is None

    if policy == "critical_path":
        level = [0.0] * n
        for i in reversed(plan.topo):
            level[i] = nominal[i] + max((level[s] for s in succ[i]), default=0.0)
        static_key = [(-level[i], pos[i]) for i in range(n)]
    elif policy == "longest_first":
        static_key = [(-nominal[i], pos[i]) for i in range(n)]
    elif policy == "shortest_first":
        static_key = [(nominal[i], pos[i]) for i in range(n)]
    elif callable(policy):
        static_key = [(policy(plan.nodes[i]), pos[i]) for i in range(n)]
    else:
        static_key = []
    fifo = policy == "fifo"
    randomized = policy == "random"
    ready_heap: list = []
    ready_list: list[int] = []
    ready_seq = itertools.count()

    def make_ready(i: int, t: float) -> None:
        state[i] = READY
        if randomized:
            ready_list.append(i)
        elif fifo:
            heapq.heappush(ready_heap, ((t, next(ready_seq)), i))
        else:
            heapq.heappush(ready_heap, (static_key[i], i))

    free: list[int] = [] if unlimited else list(range(workers))  # type: ignore[arg-type]
    lanes = 0 if unlimited else int(workers)  # type: ignore[arg-type]
    busy = [0.0] * lanes
    if res is not None:
        avail, demand = list(res[0]), res[1]
    else:
        avail, demand = None, None
    running: list[tuple] = []
    records: list[tuple] = []
    blocked: list[int] = []
    seq = itertools.count()

    def fits(i: int) -> bool:
        return avail is None or all(a >= d - _RES_TOL for a, d in zip(avail, demand[i]))  # type: ignore[index]

    def start(i: int, t: float) -> None:
        nonlocal lanes
        if free:
            w = heapq.heappop(free)
        else:  # only reachable with unlimited workers
            w = lanes
            lanes += 1
            busy.append(0.0)
        attempts[i] += 1
        d = durations.draw(i, rng)
        fails = fail[i] > 0.0 and rng.random() < fail[i]
        state[i] = RUNNING
        if avail is not None:
            for k, need in enumerate(demand[i]):  # type: ignore[index]
                avail[k] -= need
        heapq.heappush(running, (t + d, next(seq), i, w, t, attempts[i], fails))

    def dispatch(t: float) -> None:
        skipped: list = []
        if randomized:
            while ready_list and (unlimited or free):
                k = int(rng.integers(len(ready_list)))
                i = ready_list[k]
                ready_list[k] = ready_list[-1]
                ready_list.pop()
                if fits(i):
                    start(i, t)
                else:
                    skipped.append(i)
            ready_list.extend(skipped)
        else:
            while ready_heap and (unlimited or free):
                item = heapq.heappop(ready_heap)
                if fits(item[1]):
                    start(item[1], t)
                else:
                    skipped.append(item)
            for item in skipped:
                heapq.heappush(ready_heap, item)

    def complete(entry: tuple, fired: list) -> None:
        end, s_seq, i, w, begun, attempt, fails = entry
        e_seq = next(seq)
        heapq.heappush(free, w)
        busy[w] += end - begun
        if avail is not None:
            for k, need in enumerate(demand[i]):  # type: ignore[index]
                avail[k] += need
        if fails:
            records.append((i, w, begun, end, attempt, "failed", s_seq, e_seq))
            if attempt <= max_retries:
                make_ready(i, end)
                return
            state[i] = FAILED
            stack = list(succ[i])
            newly = []
            while stack:
                v = stack.pop()
                if state[v] == PENDING:
                    state[v] = FAILED
                    newly.append(v)
                    stack.extend(succ[v])
            blocked.extend(sorted(newly, key=pos.__getitem__))
            return
        records.append((i, w, begun, end, attempt, "done", s_seq, e_seq))
        state[i] = DONE
        for v in succ[i]:
            fired.append((i, v))
            remaining[v] -= 1
            if remaining[v] == 0 and state[v] == PENDING:
                make_ready(v, end)

    times: list[float] = []
    frames: list[list[int]] = []
    activity: list[list[tuple[int, int]]] = []

    def settle(t: float) -> None:
        fired: list[tuple[int, int]] = []
        while True:
            while running and running[0][0] <= t:
                complete(heapq.heappop(running), fired)
            dispatch(t)
            if not (running and running[0][0] <= t):
                break
        times.append(t)
        frames.append(list(state))
        activity.append(fired)

    for i in plan.topo:
        if remaining[i] == 0:
            make_ready(i, 0.0)
    settle(0.0)
    while running:
        settle(running[0][0])
    stuck = [plan.nodes[i] for i in range(n) if state[i] in (PENDING, READY)]
    if stuck:  # pragma: no cover - guarded by the resource validation
        raise RuntimeError(f"tasks {stuck!r} could not be scheduled")
    return _Outcome(times, frames, activity, records, busy, lanes, blocked, _chain(records, preds))


def _chain(records: list[tuple], preds: list[list[int]]) -> list[int]:
    """Backtrack the attempts that released each other's start, from the last to finish."""
    if not records:
        return []
    by_end: dict[float, list[tuple]] = {}
    for r in records:
        by_end.setdefault(r[3], []).append(r)
    cur = max(records, key=lambda r: (r[3], r[7]))
    chain = [cur]
    while True:
        task, worker, begun, _, attempt, _, s_seq, _ = cur
        # only attempts completed before this one was dispatched can have released it
        cands = [r for r in by_end.get(begun, ()) if r[7] < s_seq]
        if not cands:
            break
        parents = set(preds[task])
        prev = (
            next((r for r in cands if r[0] == task and r[4] == attempt - 1), None)
            or next((r for r in cands if r[5] == "done" and r[0] in parents), None)
            or next((r for r in cands if r[1] == worker), None)
            or cands[-1]
        )
        chain.append(prev)
        cur = prev
    out: list[int] = []
    for r in reversed(chain):
        if not out or out[-1] != r[0]:
            out.append(r[0])
    return out


# ---------------------------------------------------------------------- #
# public API
# ---------------------------------------------------------------------- #
def simulate_schedule(
    dag: Any,
    *,
    duration: str | float | Mapping | Callable = "duration",
    workers: int | None = 2,
    policy: str | Callable = "critical_path",
    failure_rate: float | str | Mapping = 0.0,
    max_retries: int = 0,
    jitter: Any = None,
    seed: int | np.random.Generator | None = None,
    resources: Mapping[str, float] | None = None,
) -> ScheduleResult:
    """Simulate running the tasks of *dag* on a pool of workers.

    Parameters
    ----------
    dag:
        A :class:`~aryagraph.DAG` or an acyclic :class:`~aryagraph.DiGraph`.
    duration:
        Node-attribute name (missing ⇒ 1), a number, ``{node: duration}`` or
        ``f(node, attrs)``.
    workers:
        Pool size; ``None`` for unlimited (every ready task starts at once).
    policy:
        Which ready task a free worker takes: ``"critical_path"`` (longest
        remaining path of nominal durations first, i.e. HLFET), ``"fifo"``
        (first ready first), ``"longest_first"``, ``"shortest_first"``,
        ``"random"`` (uniform among the ready tasks that fit), or a key function
        ``key(node)`` (smallest first). Ties go to topological order.
    failure_rate:
        Per-attempt failure probability: a number, a node-attribute name
        (missing ⇒ 0) or ``{node: probability}``.
    max_retries:
        Extra attempts allowed after a failure.
    jitter:
        Random durations: ``None``, ``("uniform", frac)``,
        ``("triangular", min_attr, mode_attr, max_attr)``,
        ``("pert", min_attr, mode_attr, max_attr)`` or ``f(node, nominal, rng)``.
        One duration is drawn per attempt.
    resources:
        ``{name: capacity}``; a task needs the amount given by its node
        attribute of that name (missing ⇒ 0). A lower-priority task may start
        when the preferred one does not fit.

    Returns
    -------
    ScheduleResult
        Categorical frames (``pending``/``ready``/``running``/``done``/``failed``)
        at every event time (state after all events at that time);
        ``edge_activity`` lists the precedence arcs released by tasks finishing
        in that frame. With unlimited workers and no failures the makespan
        equals the critical-path length; with one worker, the sum of durations.

    Raises
    ------
    CycleError
        If the graph has a directed cycle.
    """
    plan = _plan(dag)
    workers = _check_workers(workers)
    _check_policy(policy)
    if isinstance(max_retries, bool) or int(max_retries) != max_retries or max_retries < 0:
        raise ValueError(f"max_retries must be a non-negative integer, got {max_retries!r}")
    durations = _Durations(dag, plan.nodes, duration, jitter)
    fail_p = _failure_probs(dag, plan.nodes, failure_rate)
    res = _resources(dag, plan.nodes, resources)
    rng = make_rng(seed)
    out = _run(plan, durations, fail_p, workers, int(max_retries), policy, res, rng)
    return _result(dag, plan, out, workers, policy, durations, failure_rate, max_retries, resources, seed)


def _result(dag, plan, out, workers, policy, durations, failure_rate, max_retries, resources, seed) -> ScheduleResult:
    nodes = plan.nodes
    makespan = out.makespan
    total_busy = math.fsum(out.busy)
    utilization: dict = {w: (b / makespan if makespan > 0 else 0.0) for w, b in enumerate(out.busy)}
    utilization["overall"] = total_busy / (out.lanes * makespan) if makespan > 0 and out.lanes else 0.0
    ordered = sorted(out.records, key=lambda r: r[6])
    tasks = [TaskRun(nodes[r[0]], r[1], r[2], r[3], r[4], r[5]) for r in ordered]
    params = {
        "workers": workers,
        "policy": policy if isinstance(policy, str) else "callable",
        "failure_rate": failure_rate if isinstance(failure_rate, (Real, str)) else "mapping",
        "max_retries": int(max_retries),
        "jitter": durations.desc,
        "resources": dict(resources) if resources else None,
    }
    return ScheduleResult(
        graph=dag,
        nodes=nodes,
        times=np.array(out.times),
        values=np.array(out.frames, dtype=np.int8).reshape(len(out.times), len(nodes)),
        kind="categorical",
        states=list(STATES),
        roles={},
        model="schedule",
        params=params,
        seed=_seed_record(seed),
        edge_activity=[[(nodes[u], nodes[v]) for u, v in f] for f in out.fired],
        meta={"lanes": out.lanes, "busy": list(out.busy)},
        tasks=tasks,
        makespan=makespan,
        utilization=utilization,
        critical_path=[nodes[i] for i in out.chain],
        idle_time=out.lanes * makespan - total_busy,
        blocked=[nodes[i] for i in out.blocked],
    )


def _cpm_batch(plan: _Plan, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Makespans and zero-float masks of many duration vectors at once (CPM)."""
    runs, n = d.shape
    ef = np.zeros((runs, n))
    for i in plan.topo:
        p = plan.preds[i]
        ef[:, i] = (ef[:, p].max(axis=1) if p else 0.0) + d[:, i]
    makespan = ef.max(axis=1) if n else np.zeros(runs)
    lf = np.empty((runs, n))
    for i in reversed(plan.topo):
        s = plan.succ[i]
        lf[:, i] = (lf[:, s] - d[:, s]).min(axis=1) if s else makespan
    tol = 1e-9 * np.maximum(1.0, makespan)
    return makespan, (lf - ef) <= tol[:, None]


@dataclass(repr=False)
class MonteCarloResult:
    """Distribution of the makespan over many simulated schedules.

    Attributes
    ----------
    makespans:
        ``(runs,)`` makespan of every run.
    mean:
        Sample mean of :attr:`makespans`.
    std:
        Sample standard deviation of :attr:`makespans` (``ddof=1``; 0 for a
        single run).
    percentiles:
        ``{"P50": …, "P80": …, "P95": …}`` (linear interpolation).
    criticality:
        Share of runs in which each task was critical: zero total float with
        unlimited workers (every tied critical path counts), membership of the
        run's :attr:`ScheduleResult.critical_path` otherwise.
    cpm_length:
        Critical-path length of the nominal durations (the deterministic plan).
    """

    makespans: np.ndarray
    mean: float
    std: float
    percentiles: dict[str, float]
    criticality: NodeMap
    runs: int
    workers: int | None
    cpm_length: float
    seed: int | None
    graph: Any = None
    params: dict = field(default_factory=dict)

    @property
    def stderr(self) -> float:
        """Standard error of :attr:`mean`."""
        return self.std / math.sqrt(self.runs) if self.runs > 1 else math.nan

    def percentile(self, q: float) -> float:
        """Makespan not exceeded in *q* percent of the runs."""
        return float(np.percentile(self.makespans, q))

    def probability(self, deadline: float) -> float:
        """Share of runs finishing by *deadline*."""
        return float(np.mean(self.makespans <= deadline))

    def plot(self, **kwargs: Any):
        """Histogram of makespans with percentile markers; see :func:`aryagraph.charts.sim.plot_monte_carlo`."""
        from ..charts.sim import plot_monte_carlo

        return plot_monte_carlo(self, **kwargs)

    def __repr__(self) -> str:
        pct = ", ".join(f"{k}={v:.4g}" for k, v in self.percentiles.items())
        return f"<MonteCarloResult: {self.runs} runs, makespan {self.mean:.4g} ± {self.std:.3g}; {pct}>"


def monte_carlo_schedule(
    dag: Any,
    *,
    runs: int = 1000,
    workers: int | None = None,
    duration: str | float | Mapping | Callable = "duration",
    jitter: Any = "auto",
    policy: str | Callable = "critical_path",
    failure_rate: float | str | Mapping = 0.0,
    max_retries: int = 0,
    resources: Mapping[str, float] | None = None,
    percentiles: tuple[float, ...] = (50, 80, 95),
    seed: int | np.random.Generator | None = None,
) -> MonteCarloResult:
    """Makespan distribution and task criticality by Monte Carlo.

    With ``workers=None``, no resources and no failures this is a pure
    critical-path (CPM/PERT) Monte Carlo, vectorised over the runs (O(runs ·
    (n + m))). Otherwise every run is a full :func:`simulate_schedule`.

    ``jitter="auto"`` (the default) uses PERT durations when every task has
    ``min`` / ``mode`` / ``max`` attributes and no jitter otherwise; without
    jitter or failures all runs are identical.
    """
    runs = int(runs)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if isinstance(jitter, str) and jitter == "auto":
        has_estimates = len(dag) > 0 and all(all(k in d for k in ("min", "mode", "max")) for _, d in dag.nodes.data())
        jitter = ("pert", "min", "mode", "max") if has_estimates else None
    plan = _plan(dag)
    workers = _check_workers(workers)
    _check_policy(policy)
    durations = _Durations(dag, plan.nodes, duration, jitter)
    fail_p = _failure_probs(dag, plan.nodes, failure_rate)
    res = _resources(dag, plan.nodes, resources)
    rng = make_rng(seed)
    n = len(plan.nodes)
    if workers is None and res is None and not fail_p.any():
        makespans, critical = _cpm_batch(plan, durations.draw_all(rng, runs))
        counts = critical.sum(axis=0)
    else:
        makespans = np.empty(runs)
        counts = np.zeros(n, dtype=np.int64)
        for r in range(runs):
            out = _run(plan, durations, fail_p, workers, int(max_retries), policy, res, rng)
            makespans[r] = out.makespan
            counts[sorted(set(out.chain))] += 1  # a retried task may occur twice in a chain
    cpm_length = float(_cpm_batch(plan, durations.nominal[None, :])[0][0])
    return MonteCarloResult(
        makespans=makespans,
        mean=float(makespans.mean()),
        std=float(makespans.std(ddof=1)) if runs > 1 else 0.0,
        percentiles={f"P{q:g}": float(np.percentile(makespans, q)) for q in percentiles},
        criticality=NodeMap(zip(plan.nodes, (counts / runs).tolist()), name="criticality"),
        runs=runs,
        workers=workers,
        cpm_length=cpm_length,
        seed=_seed_record(seed),
        graph=dag,
        params={
            "policy": policy if isinstance(policy, str) else "callable",
            "jitter": durations.desc,
            "failure_rate": failure_rate if isinstance(failure_rate, (Real, str)) else "mapping",
            "max_retries": int(max_retries),
            "resources": dict(resources) if resources else None,
        },
    )


__all__ = [
    "simulate_schedule",
    "monte_carlo_schedule",
    "ScheduleResult",
    "MonteCarloResult",
    "TaskRun",
    "POLICIES",
]
