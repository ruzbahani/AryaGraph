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

"""Simulation results and the machinery the simulators share.

Every simulator in :mod:`aryagraph.sim` returns a :class:`SimulationResult`: a
``(T, N)`` array holding the value of each of ``N`` nodes in each of ``T``
frames, plus what the renderer and the charts need to draw it (state names,
their semantic colour roles, and the edges that fired in each frame).
Categorical models (epidemics, cascades, walks, voters, schedules) store small
integer state codes; continuous models (opinions, heat, phases) store floats.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Hashable

import numpy as np

from ..core.exceptions import DependencyError, NodeNotFound
from ..core.results import NodeMap
from ..core.utils import WeightSpec, weight_fn

if TYPE_CHECKING:
    from ..core.graph import Graph

Node = Hashable

#: Semantic colour roles a state can take; the renderer maps them to theme colours.
ROLES: tuple[str, ...] = ("neutral", "accent", "muted", "good", "warning", "serious", "critical")

_ROLE_OF_STATE: dict[str, str] = {
    # epidemics
    "S": "neutral", "susceptible": "neutral",
    "E": "warning", "exposed": "warning",
    "I": "critical", "infected": "critical", "infectious": "critical",
    "R": "good", "recovered": "good", "removed": "good", "immune": "good",
    "V": "good", "vaccinated": "good",
    "H": "serious", "hospitalized": "serious",
    "D": "muted", "dead": "muted",
    # cascades
    "inactive": "neutral", "active": "accent",
    # walks
    "unvisited": "neutral", "visited": "good", "current": "accent",
    # scheduling
    "pending": "muted", "ready": "warning", "running": "accent", "done": "good", "failed": "critical",
}
_ROLE_CYCLE = ("accent", "warning", "good", "serious", "critical", "muted", "neutral")


def default_roles(states: Iterable[str]) -> dict[str, str]:
    """Colour role of each state: conventional names get their usual role
    (``S`` → neutral, ``I`` → critical, ``done`` → good, …); any other state takes
    the next role of a fixed cycle, so the mapping is deterministic."""
    out: dict[str, str] = {}
    k = 0
    for s in states:
        role = _ROLE_OF_STATE.get(s)
        if role is None:
            role = _ROLE_CYCLE[k % len(_ROLE_CYCLE)]
            k += 1
        out[s] = role
    return out


def cycle_roles(states: Iterable[str]) -> dict[str, str]:
    """Roles for states without semantic meaning (opinions): a fixed cycle."""
    return {s: _ROLE_CYCLE[k % len(_ROLE_CYCLE)] for k, s in enumerate(states)}


def _code_dtype(n_states: int) -> np.dtype:
    return np.dtype(np.int8) if n_states <= 127 else np.dtype(np.int32)


@dataclass(repr=False)
class SimulationResult:
    """Frames of a simulation on a graph.

    Attributes
    ----------
    graph:
        The simulated graph.
    nodes:
        Column order of :attr:`values` (graph order).
    times:
        ``(T,)`` strictly increasing frame times, ``times[0] == 0``.
    values:
        ``(T, N)`` array: integer state codes (index into :attr:`states`) when
        ``kind == "categorical"``, floats when ``kind == "continuous"``.
    kind:
        ``"categorical"`` or ``"continuous"``.
    states:
        Categorical: state names by code. Continuous: ``[]``.
    roles:
        State name → semantic colour role (one of :data:`ROLES`).
    model, params, seed:
        Provenance: model name, its parameters, and the integer seed (``None``
        when the run was seeded from entropy or an existing generator).
    edge_activity:
        Per frame, the edges ``(u, v)`` that fired since the previous frame
        (transmissions, walker moves, …) oriented from cause to effect; ``None``
        when the model has no notion of edges firing.
    meta:
        Model-specific extras. Continuous results always carry
        ``meta["domain"] = (lo, hi)`` and ``meta["diverging"]`` (bool).

    A frame holds the state *after* every event with time ≤ its time, so
    :meth:`at` is exact between frames for jump processes.
    """

    graph: Graph
    nodes: list
    times: np.ndarray
    values: np.ndarray
    kind: str
    states: list[str]
    roles: dict[str, str]
    model: str
    params: dict
    seed: int | None
    edge_activity: list[list[tuple]] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = list(self.nodes)
        n = len(self.nodes)
        times = np.asarray(self.times, dtype=float).reshape(-1)
        if times.size == 0:
            raise ValueError("a simulation result needs at least one frame")
        if times[0] != 0.0:
            raise ValueError(f"times[0] must be 0, got {times[0]!r}")
        if times.size > 1 and not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing")
        if self.kind not in ("categorical", "continuous"):
            raise ValueError(f"kind must be 'categorical' or 'continuous', got {self.kind!r}")
        values = np.asarray(self.values)
        if values.shape != (times.size, n):
            raise ValueError(f"values must have shape {(times.size, n)}, got {values.shape}")
        roles = dict(self.roles or {})
        if self.kind == "categorical":
            states = list(self.states)
            if not states or len(set(states)) != len(states):
                raise ValueError("a categorical result needs distinct state names")
            if values.size and (values.min() < 0 or values.max() >= len(states)):
                raise ValueError("state codes must index into states")
            values = values.astype(_code_dtype(len(states)), copy=False)
            full = default_roles(states)
            full.update(roles)
            unknown = {s for s in full if s not in states}
            if unknown:
                raise ValueError(f"roles given for unknown states {sorted(unknown)!r}")
            roles = {s: full[s] for s in states}
            self.states = states
        else:
            if self.states:
                raise ValueError("a continuous result has no states (use states=[])")
            self.states = []
            values = values.astype(float, copy=False)
            if "domain" not in self.meta:
                finite = values[np.isfinite(values)]
                self.meta["domain"] = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            self.meta.setdefault("diverging", False)
        bad = {r for r in roles.values() if r not in ROLES}
        if bad:
            raise ValueError(f"unknown roles {sorted(bad)!r}; use one of {ROLES}")
        if self.edge_activity is not None:
            if len(self.edge_activity) != times.size:
                raise ValueError("edge_activity needs one list per frame")
            self.edge_activity = [[tuple(e) for e in frame] for frame in self.edge_activity]
        self.roles = roles
        self.times = times
        self.values = values
        self.params = dict(self.params or {})
        self._column = {node: i for i, node in enumerate(self.nodes)}

    # ------------------------------------------------------------------ #
    # shape
    # ------------------------------------------------------------------ #
    @property
    def T(self) -> int:
        """Number of frames."""
        return int(self.values.shape[0])

    @property
    def N(self) -> int:
        """Number of nodes."""
        return int(self.values.shape[1])

    @property
    def categorical(self) -> bool:
        return self.kind == "categorical"

    # ------------------------------------------------------------------ #
    # frames
    # ------------------------------------------------------------------ #
    def frame(self, i: int) -> dict:
        """``{node: state name}`` (categorical) or ``{node: value}`` of frame *i* (negative ok)."""
        row = self.values[i].tolist()
        if self.categorical:
            names = self.states
            return {n: names[c] for n, c in zip(self.nodes, row)}
        return dict(zip(self.nodes, row))

    def index_at(self, time: float) -> int:
        """Index of the last frame with ``times <= time``."""
        i = int(np.searchsorted(self.times, time, side="right")) - 1
        if i < 0:
            raise ValueError(f"time {time!r} is before the start of the simulation")
        return i

    def at(self, time: float) -> dict:
        """The frame in force at *time* (the last one with ``times <= time``)."""
        return self.frame(self.index_at(time))

    def final(self) -> dict:
        """The last frame."""
        return self.frame(-1)

    def history(self, node: Node) -> list:
        """The state names (or values) of *node* over all frames."""
        try:
            col = self.values[:, self._column[node]].tolist()
        except (KeyError, TypeError):
            raise NodeNotFound(node) from None
        if self.categorical:
            return [self.states[c] for c in col]
        return col

    # ------------------------------------------------------------------ #
    # aggregates
    # ------------------------------------------------------------------ #
    def _require_categorical(self, what: str) -> None:
        if not self.categorical:
            raise ValueError(f"{what} needs a categorical simulation; {self.model!r} is continuous (see summary())")

    def _code(self, state: str | int) -> int:
        self._require_categorical("a state lookup")
        if isinstance(state, Integral) and not isinstance(state, bool) and 0 <= int(state) < len(self.states):
            return int(state)
        try:
            return self.states.index(state)  # type: ignore[arg-type]
        except ValueError:
            raise ValueError(f"unknown state {state!r}; states are {self.states!r}") from None

    def counts(self) -> dict[str, np.ndarray]:
        """Number of nodes in each state per frame, as ``{state: (T,) int array}``."""
        self._require_categorical("counts()")
        return {s: np.count_nonzero(self.values == k, axis=1) for k, s in enumerate(self.states)}

    def fractions(self) -> dict[str, np.ndarray]:
        """Share of nodes in each state per frame, as ``{state: (T,) float array}``."""
        n = self.N
        return {s: (c / n if n else np.zeros(self.T)) for s, c in self.counts().items()}

    def summary(self) -> dict:
        """Continuous: ``{"mean", "min", "max", "std"}`` arrays over frames.

        Categorical: ``{state: {"initial", "final", "peak", "peak_time"}}``.
        """
        if not self.categorical:
            v = self.values
            if self.N == 0:
                nan = np.full(self.T, np.nan)
                return {"mean": nan, "min": nan.copy(), "max": nan.copy(), "std": nan.copy()}
            return {"mean": v.mean(axis=1), "min": v.min(axis=1), "max": v.max(axis=1), "std": v.std(axis=1)}
        out = {}
        for s, c in self.counts().items():
            i = int(np.argmax(c))
            out[s] = {"initial": int(c[0]), "final": int(c[-1]), "peak": int(c[i]), "peak_time": float(self.times[i])}
        return out

    def first_time(self, state: str) -> NodeMap:
        """Time each node first enters *state* (``nan`` if never)."""
        hit = self.values == self._code(state)
        ever = hit.any(axis=0)
        first = np.argmax(hit, axis=0)
        t = np.where(ever, self.times[first], np.nan)
        return NodeMap(zip(self.nodes, t.tolist()), name=f"first_time({state})")

    def peak(self, state: str) -> tuple[float, int]:
        """``(time, count)`` of the largest number of nodes in *state* (earliest on ties)."""
        c = np.count_nonzero(self.values == self._code(state), axis=1)
        i = int(np.argmax(c))
        return float(self.times[i]), int(c[i])

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #
    def to_records(self) -> list[dict]:
        """Long format: one ``{"frame", "time", "node", "state"|"value"}`` dict per node and frame."""
        key = "state" if self.categorical else "value"
        names = self.states
        out = []
        for i, t in enumerate(self.times.tolist()):
            row = self.values[i].tolist()
            for n, v in zip(self.nodes, row):
                out.append({"frame": i, "time": t, "node": n, key: names[v] if self.categorical else v})
        return out

    def to_pandas(self):
        """:meth:`to_records` as a ``pandas.DataFrame`` (pandas is optional)."""
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - exercised only without pandas
            raise DependencyError("pandas", "SimulationResult.to_pandas()", extra="interop") from None
        cols = ["frame", "time", "node", "state" if self.categorical else "value"]
        return pd.DataFrame(self.to_records(), columns=cols)

    # ------------------------------------------------------------------ #
    # rendering hooks (implemented in aryagraph.render / aryagraph.charts)
    # ------------------------------------------------------------------ #
    def animate(self, layout=None, **kwargs: Any):
        """Animate the frames over a drawing of the graph; see :func:`aryagraph.render.animate.animate`."""
        from ..render.animate import animate

        return animate(self, layout=layout, **kwargs)

    def plot(self, **kwargs: Any):
        """Line chart of state counts (or value summary) over time; see :func:`aryagraph.charts.sim.plot_simulation`."""
        from ..charts.sim import plot_simulation

        return plot_simulation(self, **kwargs)

    # ------------------------------------------------------------------ #
    def _final_text(self) -> str:
        if self.N == 0:
            return "no nodes"
        if self.categorical:
            last = np.bincount(self.values[-1].astype(np.intp), minlength=len(self.states))
            return "final " + ", ".join(f"{s}={int(c)}" for s, c in zip(self.states, last))
        v = self.values[-1]
        return f"final mean={v.mean():.4g} (min {v.min():.4g}, max {v.max():.4g})"

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.model}: {self.N} nodes, {self.T} frames, "
            f"t=0…{self.times[-1]:.4g}; {self._final_text()}>"
        )


# ---------------------------------------------------------------------- #
# shared private helpers
# ---------------------------------------------------------------------- #
def _seed_record(seed: Any) -> int | None:
    """The integer to store as ``result.seed`` (None for entropy / generators)."""
    if isinstance(seed, Integral) and not isinstance(seed, bool):
        return int(seed)
    return None


def _is_node(x: Any, index: Mapping) -> bool:
    try:
        return x in index
    except TypeError:
        return False


def _node_indices(spec: Any, index: Mapping, what: str = "nodes") -> list[int]:
    """Indices of a single node or an iterable of nodes (duplicates dropped, order kept)."""
    if _is_node(spec, index):
        return [index[spec]]
    if isinstance(spec, (str, bytes)) or not isinstance(spec, Iterable):
        raise NodeNotFound(spec, f"{what}: {spec!r} is not a node of the graph (nor an iterable of nodes)")
    out: list[int] = []
    seen: set[int] = set()
    for n in spec:
        if not _is_node(n, index):
            raise NodeNotFound(n)
        i = index[n]
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _node_values(
    spec: Any,
    nodes: list,
    index: Mapping,
    what: str,
    *,
    missing: float | None = None,
    scalar: bool = True,
) -> np.ndarray:
    """A float per node from a mapping, a scalar or an array aligned with *nodes*.

    Mapping entries absent from *spec* take *missing* (an error when None).
    """
    n = len(nodes)
    if isinstance(spec, Mapping):
        out = np.empty(n)
        given = np.zeros(n, dtype=bool)
        for k, v in spec.items():
            if not _is_node(k, index):
                raise NodeNotFound(k)
            out[index[k]] = float(v)
            given[index[k]] = True
        if not given.all():
            if missing is None:
                first = nodes[int(np.flatnonzero(~given)[0])]
                raise ValueError(f"{what}: no value for node {first!r}")
            out[~given] = missing
    elif scalar and isinstance(spec, Real) and not isinstance(spec, bool):
        out = np.full(n, float(spec))
    else:
        out = np.array(spec, dtype=float)
        if out.shape != (n,):
            raise ValueError(f"{what}: expected a mapping or {n} values in graph order, got shape {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{what} must be finite")
    return out


@dataclass
class _Arcs:
    """Influence arcs ``src[k] → dst[k]`` with weights ``w[k]`` (node indices).

    Undirected edges appear once per direction. Arcs are sorted by source in
    graph order (neighbour order within a source), which makes :meth:`out_ptr`
    a CSR row pointer.
    """

    nodes: list
    index: dict
    src: np.ndarray
    dst: np.ndarray
    w: np.ndarray

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def m(self) -> int:
        return int(self.src.size)

    def out_ptr(self) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(np.bincount(self.src, minlength=self.n)))).astype(np.intp)

    def in_order(self) -> tuple[np.ndarray, np.ndarray]:
        """``(order, ptr)``: arcs sorted by head (stable) and the matching row pointer."""
        order = np.argsort(self.dst, kind="stable")
        ptr = np.concatenate(([0], np.cumsum(np.bincount(self.dst, minlength=self.n)))).astype(np.intp)
        return order, ptr


def _arcs(
    g: Any,
    weight: WeightSpec = None,
    *,
    self_loops: bool = False,
    keep_zero: bool = False,
    what: str = "weight",
) -> _Arcs:
    """Collect the influence arcs of *g*.

    ``u → v`` means *u* acts on *v*: arcs of a directed graph as they are, both
    directions of an undirected edge. Weights must be finite and non-negative;
    zero-weight arcs are dropped unless *keep_zero* (they can never fire), and a
    weight function returning ``None`` hides the edge.
    """
    nodes = list(g._node)
    index = {n: i for i, n in enumerate(nodes)}
    wf = weight_fn(weight)
    src: list[int] = []
    dst: list[int] = []
    val: list[float] = []
    for i, u in enumerate(nodes):
        for v, d in g._succ[u].items():
            if v == u and not self_loops:
                continue
            x = wf(u, v, d)
            if x is None:
                continue
            x = float(x)
            if not (math.isfinite(x) and x >= 0.0):
                raise ValueError(f"{what} of edge ({u!r}, {v!r}) must be finite and non-negative, got {x!r}")
            if x == 0.0 and not keep_zero:
                continue
            src.append(i)
            dst.append(index[v])
            val.append(x)
    return _Arcs(
        nodes,
        index,
        np.asarray(src, dtype=np.intp),
        np.asarray(dst, dtype=np.intp),
        np.asarray(val, dtype=float),
    )


def _sccs(succ: list[list[int]]) -> list[list[int]]:
    """Strongly connected components (iterative Tarjan), each in discovery order."""
    n = len(succ)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    comps: list[list[int]] = []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work = [(root, 0)]
        while work:
            v, k = work[-1]
            nbrs = succ[v]
            if k < len(nbrs):
                work[-1] = (v, k + 1)
                w = nbrs[k]
                if index[w] == -1:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, 0))
                elif on_stack[w] and index[w] < low[v]:
                    low[v] = index[w]
                continue
            work.pop()
            if work:
                u = work[-1][0]
                if low[v] < low[u]:
                    low[u] = low[v]
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                comps.append(sorted(comp))
    return comps


def _closed_classes(succ: list[list[int]]) -> list[list[int]]:
    """The closed (recurrent) classes of a Markov chain given by its support graph."""
    comps = _sccs(succ)
    comp_of = [0] * len(succ)
    for c, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = c
    closed = [comp for c, comp in enumerate(comps) if all(comp_of[w] == c for v in comp for w in succ[v])]
    return sorted(closed, key=lambda comp: comp[0])


def _period(cls: list[int], succ: list[list[int]]) -> int:
    """Period of a strongly connected class (gcd of its cycle lengths)."""
    members = set(cls)
    level = {cls[0]: 0}
    queue = deque([cls[0]])
    while queue:
        u = queue.popleft()
        for v in succ[u]:
            if v in members and v not in level:
                level[v] = level[u] + 1
                queue.append(v)
    g = 0
    for u in cls:
        for v in succ[u]:
            if v in members:
                g = math.gcd(g, level[u] + 1 - level[v])
    return abs(g)


def _stationary(p: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic matrix with one closed class.

    Solves ``π (P - I) = 0, Σπ = 1`` directly; the system is non-singular exactly
    when the chain has a single closed class.
    """
    n = p.shape[0]
    a = p.T - np.eye(n)
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(a, b)
    pi = np.where(pi > 0, pi, 0.0)
    return pi / pi.sum()


__all__ = ["SimulationResult", "ROLES", "default_roles"]
