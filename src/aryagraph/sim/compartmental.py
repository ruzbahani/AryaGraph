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

"""Compartmental (epidemic-style) models on networks.

A :class:`CompartmentalModel` is a list of states and two kinds of transitions:

* **spontaneous** ``X → Y`` at a constant rate (recovery, waning immunity, …);
* **induced** ``X → Y`` driven by neighbours in a *via* state (infection): the
  rate is ``rate × Σ w(u, v)`` over the neighbours ``u`` of ``v`` that are in the
  via state. On directed graphs influence travels along arcs ``u → v``, so the
  in-neighbours of ``v`` count. Self-loops are ignored (a node does not act on
  itself).

Two solvers:

``method="gillespie"``
    Exact stochastic simulation of the continuous-time Markov chain
    (Gillespie's direct method). Node event rates live in a binary sum tree, so
    drawing the next event costs O(log N) and an event only updates the node
    and its out-neighbours.
``method="discrete"``
    Synchronous updates every ``dt``. A node whose state has total hazard ``H``
    changes state with probability ``1 - exp(-H·dt)`` and picks the transition
    in proportion to its hazard (competing risks). For an induced transition
    this is exactly independent transmission by each neighbour with probability
    ``1 - exp(-rate·w·dt)``. It converges to the continuous-time chain as
    ``dt → 0`` with an O(dt) bias: e.g. a node stays infectious for a whole
    number of steps, on average ``dt / (1 - exp(-γ·dt)) ≈ 1/γ + dt/2``.

>>> res = aryagraph.sim.sir(g, beta=0.3, gamma=0.1, initial={"I": 3}, method="gillespie", seed=1)
>>> res.counts()["R"][-1]          # final epidemic size
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from ..core.utils import WeightSpec, make_rng
from .base import (
    ROLES,
    SimulationResult,
    _arcs,
    _code_dtype,
    _is_node,
    _node_indices,
    _seed_record,
    default_roles,
)

#: Horizon used when a model can cycle forever (SIS, SIRS, …) and none is given.
DEFAULT_T_MAX = 100.0
_METHODS = ("discrete", "gillespie")
_RECORDS = ("frames", "events", "final")


@dataclass(frozen=True)
class Transition:
    """``source → target`` at *rate*; induced by neighbours in state *via* when given."""

    source: str
    target: str
    rate: float
    via: str | None = None

    @property
    def induced(self) -> bool:
        return self.via is not None

    def __str__(self) -> str:
        if self.via is None:
            return f"{self.source}→{self.target} ({self.rate:g})"
        return f"{self.source}→{self.target} ({self.rate:g} × {self.via})"


class CompartmentalModel:
    """A network compartmental model.

    Parameters
    ----------
    states:
        State names; ``states[0]`` is the default initial state.
    spontaneous:
        ``(from, to, rate)`` triples.
    induced:
        ``(from, to, via_state, rate)`` quadruples: rate per unit edge weight to
        a neighbour in *via_state*.
    name:
        Display name (defaults to the concatenated state names, e.g. ``"SIR"``).
    roles:
        Overrides of the colour role of some states (see :data:`aryagraph.sim.ROLES`).
    params:
        Named parameters recorded on every result (the factories set these).

    Examples
    --------
    >>> sir = CompartmentalModel("SIR", spontaneous=[("I", "R", 0.1)], induced=[("S", "I", "I", 0.3)])
    >>> res = sir.simulate(g, initial={"I": 0.05}, method="gillespie", seed=0)
    """

    def __init__(
        self,
        states: Iterable[str],
        spontaneous: Iterable[tuple] = (),
        induced: Iterable[tuple] = (),
        *,
        name: str | None = None,
        roles: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        states = list(states)
        if not states:
            raise ValueError("a compartmental model needs at least one state")
        if any(not isinstance(s, str) or not s for s in states):
            raise TypeError("state names must be non-empty strings")
        if len(set(states)) != len(states):
            raise ValueError(f"duplicate state names in {states!r}")
        self.states = states
        transitions = []
        for item in spontaneous:
            if len(item) != 3:
                raise ValueError(f"spontaneous transitions are (from, to, rate), got {item!r}")
            transitions.append(self._transition(item[0], item[1], item[2], None))
        for item in induced:
            if len(item) != 4:
                raise ValueError(f"induced transitions are (from, to, via, rate), got {item!r}")
            transitions.append(self._transition(item[0], item[1], item[3], item[2]))
        self.transitions: tuple[Transition, ...] = tuple(transitions)
        self.name = name or ("".join(states) if all(len(s) == 1 for s in states) else "-".join(states))
        full = default_roles(states)
        for s, r in (roles or {}).items():
            if s not in full:
                raise ValueError(f"role given for unknown state {s!r}")
            if r not in ROLES:
                raise ValueError(f"unknown role {r!r}; use one of {ROLES}")
            full[s] = r
        self.roles = full
        self.params = dict(params or {})

    def _transition(self, source: Any, target: Any, rate: Any, via: Any) -> Transition:
        for s in (source, target) + ((via,) if via is not None else ()):
            if s not in self.states:
                raise ValueError(f"unknown state {s!r}; states are {self.states!r}")
        if source == target:
            raise ValueError(f"transition {source!r} → {target!r} does not change state")
        rate = float(rate)
        if not (math.isfinite(rate) and rate >= 0):
            raise ValueError(f"rate of {source!r} → {target!r} must be finite and non-negative, got {rate!r}")
        return Transition(source, target, rate, via)

    # ------------------------------------------------------------------ #
    @property
    def spontaneous(self) -> tuple[Transition, ...]:
        return tuple(t for t in self.transitions if t.via is None)

    @property
    def induced(self) -> tuple[Transition, ...]:
        return tuple(t for t in self.transitions if t.via is not None)

    @property
    def terminates(self) -> bool:
        """True when every run must reach an absorbing state.

        That is the case when the graph of positive-rate transitions between
        states has no cycle (SIR yes, SIS no).
        """
        succ: dict[str, set[str]] = {s: set() for s in self.states}
        for t in self.transitions:
            if t.rate > 0:
                succ[t.source].add(t.target)
        indeg = {s: 0 for s in self.states}
        for s in succ:
            for x in succ[s]:
                indeg[x] += 1
        queue = [s for s, k in indeg.items() if k == 0]
        seen = 0
        while queue:
            s = queue.pop()
            seen += 1
            for x in succ[s]:
                indeg[x] -= 1
                if indeg[x] == 0:
                    queue.append(x)
        return seen == len(self.states)

    def __repr__(self) -> str:
        body = ", ".join(str(t) for t in self.transitions)
        return f"<CompartmentalModel {self.name}: {body}>"

    # ------------------------------------------------------------------ #
    def simulate(
        self,
        g: Any,
        *,
        initial: Mapping | None = None,
        t_max: float | None = None,
        steps: int | None = None,
        method: str = "discrete",
        dt: float = 1.0,
        weight: WeightSpec = None,
        seed: int | np.random.Generator | None = None,
        record: str = "frames",
        n_frames: int | None = None,
        log_events: bool = False,
    ) -> SimulationResult:
        """Run the model on *g*.

        Parameters
        ----------
        initial:
            Either ``{state: spec}`` where *spec* is a node, a list of nodes, an
            ``int`` count or a ``float`` fraction of all nodes (rounded half up),
            or a complete ``{node: state}`` assignment. With ``{state: spec}``,
            explicit nodes are placed first and counts and fractions are then
            drawn at random from the remaining nodes. Unassigned nodes start in
            ``states[0]``. Default: one random node in the via state of the
            first induced transition.
        t_max, steps:
            Horizon, as a time or as a number of ``dt`` steps (``t_max = steps·dt``,
            also for Gillespie). Default: until absorption when the model
            :attr:`terminates`, else :data:`DEFAULT_T_MAX`. The run always stops
            early once no transition can fire.
        method:
            ``"discrete"`` (synchronous, step *dt*) or ``"gillespie"`` (exact).
        weight:
            Edge weight scaling induced rates (``None``: all 1).
        record:
            ``"frames"``: every step (discrete) or a regular grid of *n_frames*
            points (Gillespie, default 101); ``"events"``: one frame per distinct
            event time (exact piecewise-constant trajectory); ``"final"``: first
            and last frame only. A Gillespie grid is cut at the absorption time,
            which becomes the last frame.
        n_frames:
            Grid size for Gillespie; for the discrete method, thins the steps to
            about *n_frames* evenly spaced frames.
        log_events:
            Store every transition in ``meta["events"]`` as
            ``(time, node, old_state, new_state, source_node_or_None)``.

        Returns
        -------
        SimulationResult
            Categorical; ``edge_activity`` lists the transmissions ``u → v`` in
            each frame (the transmitting neighbour is drawn in proportion to edge
            weight). ``meta`` holds ``absorbed``, ``t_end`` and ``n_events``.
        """
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        if record not in _RECORDS:
            raise ValueError(f"record must be one of {_RECORDS}, got {record!r}")
        dt = float(dt)
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be positive, got {dt!r}")
        horizon = self._horizon(t_max, steps, dt)
        if n_frames is not None:
            n_frames = int(n_frames)
            if n_frames < 2:
                raise ValueError("n_frames must be at least 2")
        arcs = _arcs(g, weight)
        rng = make_rng(seed)
        state0 = self._initial_state(initial, arcs.nodes, arcs.index, rng)
        if method == "gillespie":
            run = _gillespie(self, state0, arcs, horizon, rng)
        else:
            run = _discrete(self, state0, arcs, dt, horizon, rng)
        grid = _frame_grid(run, method, record, n_frames, horizon, dt)
        track = any(t.via is not None for t in self.transitions)
        values, activity = _replay(state0, run, grid, arcs.nodes, track)

        params = dict(self.params)
        params.update(method=method, t_max=horizon)
        if method == "discrete":
            params["dt"] = dt
        if weight is None or isinstance(weight, str):
            params["weight"] = weight
        meta: dict[str, Any] = {
            "absorbed": run.absorbed,
            "t_end": run.t_end,
            "n_events": int(run.times.size),
            "transitions": [str(t) for t in self.transitions],
        }
        if log_events:
            nodes, states = arcs.nodes, self.states
            meta["events"] = [
                (t, nodes[i], states[o], states[nw], nodes[s] if s >= 0 else None)
                for t, i, o, nw, s in zip(
                    run.times.tolist(), run.node.tolist(), run.old.tolist(), run.new.tolist(), run.src.tolist()
                )
            ]
        return SimulationResult(
            graph=g,
            nodes=arcs.nodes,
            times=grid,
            values=values,
            kind="categorical",
            states=list(self.states),
            roles=dict(self.roles),
            model=self.name,
            params=params,
            seed=_seed_record(seed),
            edge_activity=activity,
            meta=meta,
        )

    # ------------------------------------------------------------------ #
    def _horizon(self, t_max: float | None, steps: int | None, dt: float) -> float:
        if steps is not None:
            if t_max is not None:
                raise ValueError("give either t_max or steps, not both")
            if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
                raise ValueError(f"steps must be a non-negative integer, got {steps!r}")
            return int(steps) * dt
        if t_max is None:
            return math.inf if self.terminates else DEFAULT_T_MAX
        t_max = float(t_max)
        if math.isnan(t_max) or t_max < 0:
            raise ValueError(f"t_max must be non-negative, got {t_max!r}")
        return t_max

    def _initial_state(self, initial: Any, nodes: list, index: dict, rng: np.random.Generator) -> np.ndarray:
        n = len(nodes)
        code = {s: i for i, s in enumerate(self.states)}
        state = np.zeros(n, dtype=_code_dtype(len(self.states)))
        if initial is None:
            via = next((t.via for t in self.transitions if t.via is not None), None)
            if via is None or n == 0:
                return state
            initial = {via: 1}
        if not isinstance(initial, Mapping):
            raise TypeError("initial must be a mapping {state: nodes|count|fraction} or {node: state}")
        keys = list(initial)
        if not all(isinstance(k, str) and k in code for k in keys):
            if all(_is_node(k, index) for k in keys):
                for node, s in initial.items():
                    if s not in code:
                        raise ValueError(f"unknown state {s!r} for node {node!r}")
                    state[index[node]] = code[s]
                return state
            raise ValueError(
                "initial must map state names to nodes/counts/fractions, or nodes to state names; "
                f"states are {self.states!r}"
            )
        assigned = np.zeros(n, dtype=bool)
        drawn = []
        for s, spec in initial.items():
            if isinstance(spec, (bool, np.bool_)):
                raise TypeError(f"initial[{s!r}] must be nodes, a count or a fraction, got {spec!r}")
            if isinstance(spec, Real):
                drawn.append((s, spec))
                continue
            idx = _node_indices(spec, index, f"initial[{s!r}]")
            clash = [i for i in idx if assigned[i]]
            if clash:
                raise ValueError(f"node {nodes[clash[0]]!r} is given two initial states")
            assigned[idx] = True
            state[idx] = code[s]
        for s, spec in drawn:
            if isinstance(spec, Integral):
                k = int(spec)
            else:
                frac = float(spec)
                if not 0.0 <= frac <= 1.0:
                    raise ValueError(f"initial fraction for {s!r} must lie in [0, 1], got {frac!r}")
                k = int(math.floor(frac * n + 0.5))
            pool = np.flatnonzero(~assigned)
            if k < 0 or k > pool.size:
                raise ValueError(f"cannot place {k} nodes in {s!r}: only {pool.size} unassigned nodes left")
            if k:
                chosen = rng.choice(pool, size=k, replace=False)
                assigned[chosen] = True
                state[chosen] = code[s]
        return state


# ---------------------------------------------------------------------- #
# factories
# ---------------------------------------------------------------------- #
def SI(beta: float) -> CompartmentalModel:
    """Susceptible → Infected at ``beta`` per infected neighbour; no recovery."""
    return CompartmentalModel("SI", induced=[("S", "I", "I", beta)], name="SI", params={"beta": beta})


def SIS(beta: float, gamma: float) -> CompartmentalModel:
    """SI plus recovery back to susceptible at ``gamma`` (no immunity; endemic above threshold)."""
    return CompartmentalModel(
        "SI", spontaneous=[("I", "S", gamma)], induced=[("S", "I", "I", beta)], name="SIS",
        params={"beta": beta, "gamma": gamma},
    )


def SIR(beta: float, gamma: float) -> CompartmentalModel:
    """Susceptible → Infected (``beta`` per infected neighbour) → Recovered (``gamma``)."""
    return CompartmentalModel(
        "SIR", spontaneous=[("I", "R", gamma)], induced=[("S", "I", "I", beta)], name="SIR",
        params={"beta": beta, "gamma": gamma},
    )


def SEIR(beta: float, sigma: float, gamma: float) -> CompartmentalModel:
    """SIR with a latent Exposed state left at rate ``sigma`` (mean incubation ``1/sigma``)."""
    return CompartmentalModel(
        "SEIR", spontaneous=[("E", "I", sigma), ("I", "R", gamma)], induced=[("S", "E", "I", beta)],
        name="SEIR", params={"beta": beta, "sigma": sigma, "gamma": gamma},
    )


def SIRS(beta: float, gamma: float, xi: float) -> CompartmentalModel:
    """SIR whose immunity wanes at rate ``xi`` (R → S)."""
    return CompartmentalModel(
        "SIR", spontaneous=[("I", "R", gamma), ("R", "S", xi)], induced=[("S", "I", "I", beta)],
        name="SIRS", params={"beta": beta, "gamma": gamma, "xi": xi},
    )


def SEIRD(beta: float, sigma: float, gamma: float, mu: float) -> CompartmentalModel:
    """SEIR where infected nodes recover at ``gamma`` or die at ``mu`` (case fatality ``mu/(gamma+mu)``)."""
    return CompartmentalModel(
        "SEIRD", spontaneous=[("E", "I", sigma), ("I", "R", gamma), ("I", "D", mu)],
        induced=[("S", "E", "I", beta)], name="SEIRD",
        params={"beta": beta, "sigma": sigma, "gamma": gamma, "mu": mu},
    )


def si(g: Any, beta: float, **kwargs: Any) -> SimulationResult:
    """Simulate :func:`SI` on *g*; keyword arguments go to :meth:`CompartmentalModel.simulate`."""
    return SI(beta).simulate(g, **kwargs)


def sis(g: Any, beta: float, gamma: float, **kwargs: Any) -> SimulationResult:
    """Simulate :func:`SIS` on *g* (see :meth:`CompartmentalModel.simulate`)."""
    return SIS(beta, gamma).simulate(g, **kwargs)


def sir(g: Any, beta: float, gamma: float, **kwargs: Any) -> SimulationResult:
    """Simulate :func:`SIR` on *g* (see :meth:`CompartmentalModel.simulate`)."""
    return SIR(beta, gamma).simulate(g, **kwargs)


def seir(g: Any, beta: float, sigma: float, gamma: float, **kwargs: Any) -> SimulationResult:
    """Simulate :func:`SEIR` on *g* (see :meth:`CompartmentalModel.simulate`)."""
    return SEIR(beta, sigma, gamma).simulate(g, **kwargs)


def sirs(g: Any, beta: float, gamma: float, xi: float, **kwargs: Any) -> SimulationResult:
    """Simulate :func:`SIRS` on *g* (see :meth:`CompartmentalModel.simulate`)."""
    return SIRS(beta, gamma, xi).simulate(g, **kwargs)


def seird(g: Any, beta: float, sigma: float, gamma: float, mu: float, **kwargs: Any) -> SimulationResult:
    """Simulate :func:`SEIRD` on *g* (see :meth:`CompartmentalModel.simulate`)."""
    return SEIRD(beta, sigma, gamma, mu).simulate(g, **kwargs)


# ---------------------------------------------------------------------- #
# solvers
# ---------------------------------------------------------------------- #
@dataclass
class _Run:
    """Event log of one run: transitions sorted by time."""

    times: np.ndarray
    node: np.ndarray
    old: np.ndarray
    new: np.ndarray
    src: np.ndarray
    t_end: float
    absorbed: bool
    n_steps: int = 0


class _SumTree:
    """Binary sum tree over non-negative leaves.

    Internal nodes are recomputed from their children (never updated by
    deltas), so sums carry no accumulated rounding: when every leaf is zero the
    total is exactly zero.
    """

    __slots__ = ("size", "tree")

    def __init__(self, values: list[float]) -> None:
        size = 2
        while size < len(values):
            size *= 2
        tree = [0.0] * (2 * size)
        tree[size : size + len(values)] = values
        for p in range(size - 1, 0, -1):
            tree[p] = tree[2 * p] + tree[2 * p + 1]
        self.size = size
        self.tree = tree

    @property
    def total(self) -> float:
        return self.tree[1]

    def update(self, items: list[int], values: list[float]) -> None:
        """Set leaves ``items`` to ``values[i]`` and refresh their ancestors once each."""
        tree, size = self.tree, self.size
        parents = set()
        for i in items:
            p = i + size
            tree[p] = values[i]
            parents.add(p >> 1)
        while parents:
            nxt = set()
            for p in parents:
                tree[p] = tree[2 * p] + tree[2 * p + 1]
                if p > 1:
                    nxt.add(p >> 1)
            parents = nxt

    def find(self, u: float) -> int:
        """Leaf ``i`` with ``prefix(i) <= u < prefix(i + 1)``; never a zero leaf."""
        tree, size = self.tree, self.size
        p = 1
        while p < size:
            left = 2 * p
            if u < tree[left] or tree[left + 1] <= 0.0:
                p = left
            else:
                u -= tree[left]
                p = left + 1
        return p - size


def _gillespie(
    model: CompartmentalModel, state0: np.ndarray, arcs: Any, horizon: float, rng: np.random.Generator
) -> _Run:
    n = arcs.n
    code = {s: i for i, s in enumerate(model.states)}
    n_states = len(model.states)
    via_codes = sorted({code[t.via] for t in model.transitions if t.via is not None and t.rate > 0})
    slot = [-1] * n_states
    for q, v in enumerate(via_codes):
        slot[v] = q
    spont: list[list[tuple[int, float]]] = [[] for _ in range(n_states)]
    induced: list[list[tuple[int, int, float]]] = [[] for _ in range(n_states)]
    uses: list[set[int]] = [set() for _ in range(n_states)]
    for t in model.transitions:
        if t.rate <= 0:
            continue
        a, b = code[t.source], code[t.target]
        if t.via is None:
            spont[a].append((b, t.rate))
        else:
            q = slot[code[t.via]]
            induced[a].append((b, q, t.rate))
            uses[a].add(q)
    spont_total = [math.fsum(r for _, r in spont[s]) for s in range(n_states)]

    out_nb: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    in_nb: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, j, w in zip(arcs.src.tolist(), arcs.dst.tolist(), arcs.w.tolist()):
        out_nb[i].append((j, w))
        in_nb[j].append((i, w))

    state = state0.tolist()
    # W[q][i]: weight of i's in-neighbours in via state q; C[q][i]: how many of them.
    # The count lets a sum that should be zero be reset to exactly zero.
    weight_in = [[0.0] * n for _ in via_codes]
    count_in = [[0] * n for _ in via_codes]
    for j in range(n):
        q = slot[state[j]]
        if q >= 0:
            wq, cq = weight_in[q], count_in[q]
            for i, w in out_nb[j]:
                wq[i] += w
                cq[i] += 1

    def rate_of(i: int) -> float:
        s = state[i]
        r = spont_total[s]
        for _, q, beta in induced[s]:
            r += beta * weight_in[q][i]
        return r

    rates = [rate_of(i) for i in range(n)]
    tree = _SumTree(rates)
    rand = rng.random
    expo = rng.standard_exponential
    ev_t, ev_node, ev_old, ev_new, ev_src = array("d"), array("q"), array("i"), array("i"), array("q")
    t = 0.0
    absorbed = False
    while True:
        total = tree.total
        if total <= 0.0:
            absorbed = True
            break
        t_next = t + expo() / total
        if t_next > horizon:
            break
        t = t_next
        i = tree.find(rand() * total)
        s = state[i]
        # which transition: proportional to its hazard
        u = rand() * rates[i]
        target, q_via, last = -1, -1, (-1, -1)
        for b, a in spont[s]:
            last = (b, -1)
            if u < a:
                target = b
                break
            u -= a
        if target < 0:
            for b, q, beta in induced[s]:
                h = beta * weight_in[q][i]
                if h > 0.0:
                    last = (b, q)
                    if u < h:
                        target, q_via = b, q
                        break
                    u -= h
        if target < 0:  # rounding ran past the end: the last positive hazard wins
            target, q_via = last
        # who transmitted: an in-neighbour in the via state, in proportion to weight
        source = -1
        if q_via >= 0:
            via = via_codes[q_via]
            x = rand() * weight_in[q_via][i]
            for j, w in in_nb[i]:
                if state[j] == via:
                    source = j
                    if x < w:
                        break
                    x -= w
        state[i] = target
        ev_t.append(t)
        ev_node.append(i)
        ev_old.append(s)
        ev_new.append(target)
        ev_src.append(source)
        dirty = [i]
        q_old, q_new = slot[s], slot[target]
        if q_old >= 0 or q_new >= 0:
            for k, w in out_nb[i]:
                if q_old >= 0:
                    ck = count_in[q_old]
                    ck[k] -= 1
                    if ck[k] == 0:
                        weight_in[q_old][k] = 0.0
                    else:
                        weight_in[q_old][k] -= w
                if q_new >= 0:
                    count_in[q_new][k] += 1
                    weight_in[q_new][k] += w
                used = uses[state[k]]
                if used and (q_old in used or q_new in used):
                    dirty.append(k)
        for k in dirty:
            rates[k] = rate_of(k)
        tree.update(dirty, rates)
    return _Run(
        times=np.frombuffer(ev_t, dtype=np.float64).copy(),
        node=np.frombuffer(ev_node, dtype=np.int64).astype(np.intp),
        old=np.frombuffer(ev_old, dtype=np.int32).astype(state0.dtype),
        new=np.frombuffer(ev_new, dtype=np.int32).astype(state0.dtype),
        src=np.frombuffer(ev_src, dtype=np.int64).astype(np.intp),
        t_end=t if absorbed else horizon,
        absorbed=absorbed,
    )


def _discrete(
    model: CompartmentalModel, state0: np.ndarray, arcs: Any, dt: float, horizon: float, rng: np.random.Generator
) -> _Run:
    n = arcs.n
    code = {s: i for i, s in enumerate(model.states)}
    trans = [t for t in model.transitions if t.rate > 0]
    k_tr = len(trans)
    frm = np.array([code[t.source] for t in trans], dtype=np.intp)
    to = np.array([code[t.target] for t in trans], dtype=state0.dtype)
    via = np.array([code[t.via] if t.via is not None else -1 for t in trans], dtype=np.intp)
    rate = np.array([t.rate for t in trans], dtype=float)
    via_codes = sorted({int(v) for v in via if v >= 0})
    src, dst, w = arcs.src, arcs.dst, arcs.w
    max_steps = None if math.isinf(horizon) else int(math.floor(horizon / dt + 1e-9))

    state = state0.copy()
    chunks: list[tuple] = []
    k = 0
    absorbed = False
    while max_steps is None or k < max_steps:
        if k_tr == 0 or n == 0:
            absorbed = True
            break
        pressure = {}
        for v in via_codes:
            m = state[src] == v
            pressure[v] = np.bincount(dst[m], weights=w[m], minlength=n)
        hazard = np.zeros((n, k_tr))
        for c in range(k_tr):
            m = state == frm[c]
            if m.any():
                hazard[m, c] = rate[c] if via[c] < 0 else rate[c] * pressure[int(via[c])][m]
        total = hazard.sum(axis=1)
        if not (total > 0).any():
            absorbed = True
            break
        k += 1
        t = k * dt
        fired = np.flatnonzero(rng.random(n) < -np.expm1(-total * dt))
        if fired.size == 0:
            continue
        hz = hazard[fired]
        cum = np.cumsum(hz, axis=1)
        u = rng.random(fired.size) * total[fired]
        choice = np.count_nonzero(cum <= u[:, None], axis=1)
        last_positive = k_tr - 1 - np.argmax(hz[:, ::-1] > 0, axis=1)
        choice = np.minimum(choice, last_positive)
        new = to[choice]
        # transmitting neighbour: the first of independent exponential clocks
        # with rates ∝ weight, i.e. chosen in proportion to weight
        sources = np.full(fired.size, -1, dtype=np.intp)
        via_of = via[choice]
        induced = via_of >= 0
        if induced.any():
            want = np.full(n, -1, dtype=np.intp)
            want[fired[induced]] = via_of[induced]
            cand = np.flatnonzero((want[dst] >= 0) & (state[src] == want[dst]))
            key = rng.standard_exponential(cand.size) / w[cand]
            cd, cs = dst[cand], src[cand]
            order = np.lexsort((key, cd))
            cd, cs = cd[order], cs[order]
            first = np.ones(cd.size, dtype=bool)
            first[1:] = cd[1:] != cd[:-1]
            src_of = np.full(n, -1, dtype=np.intp)
            src_of[cd[first]] = cs[first]
            sources = src_of[fired]
        old = state[fired].copy()
        state[fired] = new
        chunks.append((np.full(fired.size, t), fired, old, new, sources))
    if chunks:
        times, node, old, new, srcs = (np.concatenate(parts) for parts in zip(*chunks))
    else:
        times = np.zeros(0)
        node = np.zeros(0, dtype=np.intp)
        old = new = np.zeros(0, dtype=state0.dtype)
        srcs = np.zeros(0, dtype=np.intp)
    return _Run(times, node.astype(np.intp), old, new, srcs.astype(np.intp), t_end=k * dt, absorbed=absorbed, n_steps=k)


def _frame_grid(run: _Run, method: str, record: str, n_frames: int | None, horizon: float, dt: float) -> np.ndarray:
    t_end = run.t_end
    if record == "final":
        return np.array([0.0]) if t_end == 0 else np.array([0.0, t_end])
    if record == "events":
        pts = np.unique(np.concatenate(([0.0], run.times, [t_end])))
        return pts
    if method == "discrete":
        n = run.n_steps
        if n_frames is None or n_frames >= n + 1:
            k = np.arange(n + 1)
        else:
            k = np.unique(np.rint(np.linspace(0, n, n_frames)).astype(np.int64))
        return k * dt
    m = 101 if n_frames is None else n_frames
    if t_end == 0:
        return np.array([0.0])
    if math.isfinite(horizon):
        grid = np.linspace(0.0, horizon, m)
        if run.absorbed and t_end < horizon:
            grid = np.append(grid[grid < t_end], t_end)
        return grid
    return np.linspace(0.0, t_end, m)


def _replay(state0: np.ndarray, run: _Run, grid: np.ndarray, nodes: list, track: bool):
    """States at each grid time (events with time ≤ grid point applied) and the
    transmissions that happened since the previous grid point."""
    t_len = grid.size
    values = np.empty((t_len, state0.size), dtype=state0.dtype)
    values[0] = state0
    activity: list[list[tuple]] = [[] for _ in range(t_len)]
    bounds = np.searchsorted(run.times, grid, side="right")
    state = state0.copy()
    for k in range(1, t_len):
        a, b = int(bounds[k - 1]), int(bounds[k])
        if b > a:
            nk, vk = run.node[a:b], run.new[a:b]
            # the last event of each node within the slice wins
            uniq, first = np.unique(nk[::-1], return_index=True)
            state[uniq] = vk[::-1][first]
            if track:
                sk = run.src[a:b]
                sel = sk >= 0
                activity[k] = [(nodes[s], nodes[d]) for s, d in zip(sk[sel].tolist(), nk[sel].tolist())]
        values[k] = state
    return values, (activity if track else None)


__all__ = [
    "CompartmentalModel",
    "Transition",
    "DEFAULT_T_MAX",
    "SI",
    "SIS",
    "SIR",
    "SEIR",
    "SIRS",
    "SEIRD",
    "si",
    "sis",
    "sir",
    "seir",
    "sirs",
    "seird",
]
