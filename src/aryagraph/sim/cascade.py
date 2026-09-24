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

"""Information cascades: independent cascade, linear threshold, influence maximisation.

Both models are *progressive*: nodes go from ``"inactive"`` to ``"active"`` and
never back. Influence travels along arcs ``u → v`` (both directions of an
undirected edge).

Independent cascade (IC)
    A node activated at step ``t`` gets one chance to activate each inactive
    out-neighbour ``v`` at step ``t + 1``, succeeding with probability ``p(u, v)``.
Linear threshold (LT)
    Arc weights are normalised by the in-weight of their head, ``b(u, v) =
    w(u, v) / Σ_x w(x, v)``; ``v`` activates once the normalised weight of its
    active in-neighbours reaches its threshold ``θ_v`` (uniform on (0, 1] unless
    given).

Monte-Carlo estimates use the *live-edge* representation (Kempe, Kleinberg &
Tardos 2003): IC keeps each arc independently with probability ``p``; LT lets
every node keep at most one incoming arc, arc ``(u, v)`` with probability
``b(u, v)``. The set reachable from the seeds in a random live-edge graph has
exactly the distribution of the final active set.
"""

from __future__ import annotations

import heapq
import math
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from ..core.exceptions import NodeNotFound
from ..core.results import NodeMap
from ..core.utils import WeightSpec, make_rng
from .base import SimulationResult, _arcs, _Arcs, _is_node, _node_indices, _seed_record

_STATES = ["inactive", "active"]
_REL_TOL = 1e-12


def _prob_arcs(g: Any, p: Any) -> tuple[_Arcs, Any]:
    """Arcs with their activation probability, and a printable description of *p*."""
    if isinstance(p, Real) and not isinstance(p, bool):
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must lie in [0, 1], got {p!r}")
        arcs = _arcs(g, None)
        keep = np.full(arcs.m, p > 0.0)
        arcs = _Arcs(arcs.nodes, arcs.index, arcs.src[keep], arcs.dst[keep], np.full(int(keep.sum()), p))
        return arcs, p
    if isinstance(p, str):
        name = p

        def lookup(u: Any, v: Any, d: dict) -> float:
            if name not in d:
                raise ValueError(f"edge ({u!r}, {v!r}) has no {name!r} attribute")
            return d[name]

        arcs, desc = _arcs(g, lookup, what=f"probability {name!r}"), name
    elif callable(p):
        arcs, desc = _arcs(g, p, what="probability"), "callable"
    else:
        raise TypeError(f"p must be a probability, an edge-attribute name or a callable, got {p!r}")
    if arcs.m and arcs.w.max() > 1.0:
        k = int(np.argmax(arcs.w))
        raise ValueError(
            f"probability of edge ({arcs.nodes[arcs.src[k]]!r}, {arcs.nodes[arcs.dst[k]]!r}) exceeds 1: {arcs.w[k]!r}"
        )
    return arcs, desc


def _seed_list(seeds: Any, index: dict) -> list[int]:
    return sorted(_node_indices(seeds, index, "seeds"))


def _cascade_result(
    g: Any, arcs: _Arcs, frames: list, newly: list, activity: list, model: str, params: dict, seed: Any, meta: dict
) -> SimulationResult:
    values = np.array(frames, dtype=np.int8).reshape(len(frames), arcs.n)
    meta = {"newly_active": [[arcs.nodes[i] for i in step] for step in newly], "spread": int(values[-1].sum()), **meta}
    return SimulationResult(
        graph=g,
        nodes=arcs.nodes,
        times=np.arange(len(frames), dtype=float),
        values=values,
        kind="categorical",
        states=list(_STATES),
        roles={"inactive": "neutral", "active": "accent"},
        model=model,
        params=params,
        seed=_seed_record(seed),
        edge_activity=activity,
        meta=meta,
    )


def _check_steps(max_steps: int | None) -> None:
    if max_steps is not None and (isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps < 0):
        raise ValueError(f"max_steps must be a non-negative integer, got {max_steps!r}")


def independent_cascade(
    g: Any,
    seeds: Any,
    p: float | str | Callable = 0.1,
    *,
    seed: int | np.random.Generator | None = None,
    max_steps: int | None = None,
) -> SimulationResult:
    """Simulate one independent cascade from *seeds*.

    Parameters
    ----------
    seeds:
        A node or an iterable of nodes, active at step 0.
    p:
        Activation probability: a number for every arc, an edge-attribute name
        (every edge must carry it) or a callable ``f(u, v, attrs)``.
    max_steps:
        Stop after this many steps (default: until no new activation).

    Returns
    -------
    SimulationResult
        One frame per step; ``edge_activity[t]`` holds the arc that activated
        each node newly active at step ``t`` (when several attempts succeed, the
        first newly-active parent in graph order gets the credit).
        ``meta["newly_active"]`` lists the nodes activated at each step and
        ``meta["spread"]`` the final number of active nodes.

    Notes
    -----
    Every arc's coin is flipped up front from *seed* (the live-edge
    representation), so for a fixed seed the active set grows monotonically
    with the seed set.
    """
    _check_steps(max_steps)
    arcs, desc = _prob_arcs(g, p)
    n = arcs.n
    rng = make_rng(seed)
    start = _seed_list(seeds, arcs.index)
    live = rng.random(arcs.m) < arcs.w
    ls, ld = arcs.src[live], arcs.dst[live]
    ptr = np.concatenate(([0], np.cumsum(np.bincount(ls, minlength=n)))).tolist()
    adj = ld.tolist()
    active = bytearray(n)
    for s in start:
        active[s] = 1
    frames = [bytes(active)]
    newly = [start]
    activity: list[list[tuple]] = [[]]
    frontier = start
    step = 0
    while frontier and (max_steps is None or step < max_steps):
        credit: dict[int, int] = {}
        for u in frontier:
            for j in range(ptr[u], ptr[u + 1]):
                v = adj[j]
                if not active[v]:
                    active[v] = 1
                    credit[v] = u
        if not credit:
            break
        step += 1
        frontier = sorted(credit)
        frames.append(bytes(active))
        newly.append(frontier)
        activity.append([(arcs.nodes[credit[v]], arcs.nodes[v]) for v in frontier])
    params = {"p": desc, "seeds": [arcs.nodes[i] for i in start], "max_steps": max_steps}
    frames_arr = [np.frombuffer(f, dtype=np.uint8) for f in frames]
    return _cascade_result(g, arcs, frames_arr, newly, activity, "IC", params, seed, {})


def _thresholds(thresholds: Any, g: Any, arcs: _Arcs, rng: np.random.Generator) -> np.ndarray:
    n = arcs.n
    if thresholds is None:
        return 1.0 - rng.random(n)  # uniform on (0, 1]
    if isinstance(thresholds, str):
        out = np.empty(n)
        for i, node in enumerate(arcs.nodes):
            attrs = g._node[node]
            if thresholds not in attrs:
                raise ValueError(f"node {node!r} has no {thresholds!r} attribute")
            out[i] = float(attrs[thresholds])
    elif isinstance(thresholds, Mapping):
        out = np.empty(n)
        for i, node in enumerate(arcs.nodes):
            if node not in thresholds:
                raise ValueError(f"no threshold for node {node!r}")
            out[i] = float(thresholds[node])
        extra = [k for k in thresholds if not _is_node(k, arcs.index)]
        if extra:
            raise NodeNotFound(extra[0])
    elif isinstance(thresholds, Real) and not isinstance(thresholds, bool):
        out = np.full(n, float(thresholds))
    else:
        raise TypeError("thresholds must be None, a number, a node-attribute name or a {node: threshold} mapping")
    if not np.all(np.isfinite(out)):
        raise ValueError("thresholds must be finite")
    return out


def linear_threshold(
    g: Any,
    seeds: Any,
    thresholds: Any = None,
    *,
    weight: WeightSpec = None,
    seed: int | np.random.Generator | None = None,
    max_steps: int | None = None,
) -> SimulationResult:
    """Simulate the linear threshold model from *seeds*.

    Parameters
    ----------
    thresholds:
        ``None`` (independent uniform thresholds on (0, 1], drawn from *seed* for
        every node), a number for all nodes, a node-attribute name, or a
        ``{node: threshold}`` mapping.
    weight:
        Arc influence before normalisation by in-weight (``None``: all 1).

    Returns
    -------
    SimulationResult
        One frame per step; ``edge_activity[t]`` holds, for each node activated
        at step ``t``, the arcs from the previous step's newly active nodes whose
        influence tipped it over. ``meta["thresholds"]`` is a :class:`NodeMap`.

    Notes
    -----
    Node ``v`` activates when the normalised weight of its active in-neighbours
    is at least ``θ_v`` (compared with a relative tolerance of 1e-12 so that
    ``θ = 1`` is reachable despite rounding) and at least one in-neighbour is
    active; a threshold above 1 can never be reached.
    """
    _check_steps(max_steps)
    arcs = _arcs(g, weight)
    n = arcs.n
    rng = make_rng(seed)
    theta = _thresholds(thresholds, g, arcs, rng)
    start = _seed_list(seeds, arcs.index)
    in_weight = np.bincount(arcs.dst, weights=arcs.w, minlength=n)
    need = (theta * in_weight * (1.0 - _REL_TOL)).tolist()
    ptr = arcs.out_ptr().tolist()
    adj, wts = arcs.dst.tolist(), arcs.w.tolist()
    acc = [0.0] * n
    active = bytearray(n)
    for s in start:
        active[s] = 1
    frames = [bytes(active)]
    newly = [start]
    activity: list[list[tuple]] = [[]]
    frontier = start
    step = 0
    while frontier and (max_steps is None or step < max_steps):
        pushed: dict[int, list[int]] = {}
        for u in frontier:
            for j in range(ptr[u], ptr[u + 1]):
                v = adj[j]
                if not active[v]:
                    acc[v] += wts[j]
                    pushed.setdefault(v, []).append(u)
        new = sorted(v for v in pushed if acc[v] > 0.0 and acc[v] >= need[v])
        if not new:
            break
        for v in new:
            active[v] = 1
        step += 1
        frontier = new
        frames.append(bytes(active))
        newly.append(new)
        activity.append([(arcs.nodes[u], arcs.nodes[v]) for v in new for u in pushed[v]])
    params = {
        "thresholds": thresholds if thresholds is None or isinstance(thresholds, (str, Real)) else "mapping",
        "seeds": [arcs.nodes[i] for i in start],
        "weight": weight if weight is None or isinstance(weight, str) else "callable",
        "max_steps": max_steps,
    }
    meta = {"thresholds": NodeMap(zip(arcs.nodes, theta.tolist()), name="threshold")}
    frames_arr = [np.frombuffer(f, dtype=np.uint8) for f in frames]
    return _cascade_result(g, arcs, frames_arr, newly, activity, "LT", params, seed, meta)


# ---------------------------------------------------------------------- #
# live-edge Monte Carlo
# ---------------------------------------------------------------------- #
class _LiveEdges:
    """Sampler of live-edge graphs for IC or LT, as CSR ``(ptr, adj)`` lists."""

    def __init__(self, g: Any, model: str, p: Any, weight: WeightSpec) -> None:
        model = str(model).lower()
        if model not in ("ic", "lt"):
            raise ValueError(f"model must be 'ic' or 'lt', got {model!r}")
        self.model = model
        if model == "ic":
            self.arcs, self.desc = _prob_arcs(g, p)
            return
        arcs = self.arcs = _arcs(g, weight)
        self.desc = None
        n = arcs.n
        order, ptr = arcs.in_order()
        self.in_src = arcs.src[order]
        self.in_dst = arcs.dst[order]
        in_weight = np.bincount(arcs.dst, weights=arcs.w, minlength=n)
        b = arcs.w[order] / in_weight[self.in_dst]
        within = np.empty_like(b)
        for v in np.flatnonzero(ptr[1:] > ptr[:-1]).tolist():
            a, e = ptr[v], ptr[v + 1]
            within[a:e] = np.cumsum(b[a:e])
            within[e - 1] = 1.0  # exactly one arc is kept: the probabilities sum to 1
        self.within = within
        self.start = ptr[:-1]
        self.has_in = ptr[1:] > ptr[:-1]

    @property
    def n(self) -> int:
        return self.arcs.n

    def sample(self, rng: np.random.Generator) -> tuple[list[int], list[int]]:
        n = self.n
        if self.model == "ic":
            live = rng.random(self.arcs.m) < self.arcs.w
            s, d = self.arcs.src[live], self.arcs.dst[live]
        else:
            u = rng.random(n)
            below = self.within <= u[self.in_dst]
            offset = np.bincount(self.in_dst, weights=below, minlength=n).astype(np.intp)
            chosen = self.start[self.has_in] + offset[self.has_in]
            s, d = self.in_src[chosen], self.in_dst[chosen]
            order = np.argsort(s, kind="stable")
            s, d = s[order], d[order]
        ptr = np.concatenate(([0], np.cumsum(np.bincount(s, minlength=n))))
        return ptr.tolist(), d.tolist()


def _reach_count(ptr: list[int], adj: list[int], sources: list[int], n: int) -> int:
    seen = bytearray(n)
    stack = []
    for s in sources:
        if not seen[s]:
            seen[s] = 1
            stack.append(s)
    count = len(stack)
    while stack:
        u = stack.pop()
        for j in range(ptr[u], ptr[u + 1]):
            v = adj[j]
            if not seen[v]:
                seen[v] = 1
                stack.append(v)
                count += 1
    return count


@dataclass(repr=False)
class SpreadEstimate:
    """Monte-Carlo estimate of the expected number of finally active nodes.

    ``sizes`` holds the final active count of every run (seeds included),
    ``mean`` their average and ``stderr`` its standard error
    ``std(sizes, ddof=1) / √runs``.
    """

    mean: float
    stderr: float
    runs: int
    sizes: np.ndarray
    model: str
    seeds: list

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Normal-approximation interval ``mean ± z·stderr``."""
        z = statistics.NormalDist().inv_cdf(0.5 + level / 2.0)
        return self.mean - z * self.stderr, self.mean + z * self.stderr

    def __float__(self) -> float:
        return self.mean

    def __repr__(self) -> str:
        return f"SpreadEstimate({self.model.upper()}: {self.mean:.4g} ± {self.stderr:.2g}, runs={self.runs})"


def influence_spread(
    g: Any,
    seeds: Any,
    model: str = "ic",
    runs: int = 1000,
    *,
    p: float | str | Callable = 0.1,
    weight: WeightSpec = None,
    seed: int | np.random.Generator | None = None,
) -> SpreadEstimate:
    """Expected final number of active nodes from *seeds*, by Monte Carlo.

    *model* is ``"ic"`` (activation probability *p*) or ``"lt"`` (uniform random
    thresholds, influence *weight*). Each run samples a live-edge graph and
    counts the nodes reachable from the seeds, which yields the same
    distribution as :func:`independent_cascade` / :func:`linear_threshold`, at a
    fraction of the cost. Complexity O(runs · (m + reached)).
    """
    runs = int(runs)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    sampler = _LiveEdges(g, model, p, weight)
    start = _seed_list(seeds, sampler.arcs.index)
    rng = make_rng(seed)
    sizes = np.empty(runs, dtype=np.int64)
    for r in range(runs):
        ptr, adj = sampler.sample(rng)
        sizes[r] = _reach_count(ptr, adj, start, sampler.n)
    mean = float(sizes.mean())
    stderr = float(sizes.std(ddof=1) / math.sqrt(runs)) if runs > 1 else math.nan
    return SpreadEstimate(mean, stderr, runs, sizes, sampler.model, [sampler.arcs.nodes[i] for i in start])


@dataclass(repr=False)
class InfluenceMaximization:
    """Result of :func:`greedy_influence_maximization`.

    ``seeds[:i]`` is the greedy seed set of size ``i``; ``spread[i-1]`` its
    estimated expected spread and ``gains[i-1]`` the marginal gain of
    ``seeds[i-1]``. ``evaluations`` counts marginal-gain evaluations (CELF
    saves most of the ``k·n`` a plain greedy needs).
    """

    seeds: list
    spread: list[float]
    gains: list[float]
    model: str
    runs: int
    evaluations: int

    def __repr__(self) -> str:
        total = self.spread[-1] if self.spread else 0.0
        return (
            f"InfluenceMaximization({self.model.upper()}: seeds={self.seeds!r}, spread≈{total:.4g}, "
            f"runs={self.runs}, evaluations={self.evaluations})"
        )


def greedy_influence_maximization(
    g: Any,
    k: int,
    model: str = "ic",
    runs: int = 200,
    *,
    p: float | str | Callable = 0.1,
    weight: WeightSpec = None,
    seed: int | np.random.Generator | None = None,
    candidates: Any = None,
) -> InfluenceMaximization:
    """Pick *k* seeds greedily by marginal gain in expected spread (CELF).

    The expected spread is estimated on *runs* live-edge graphs sampled once
    and shared by every evaluation (common random numbers). That estimate is a
    monotone submodular function in its own right, so the lazy CELF evaluation
    returns exactly the plain greedy choice and keeps the ``1 - 1/e``
    guarantee relative to it. Gains are compared as exact integer totals; ties
    go to the node that comes first in graph order.

    The reported ``spread`` is measured on the samples used for selection and is
    therefore optimistically biased; re-estimate with :func:`influence_spread`
    under another seed for an unbiased figure.

    *candidates* restricts the nodes that may be chosen (default: all).
    """
    runs = int(runs)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    sampler = _LiveEdges(g, model, p, weight)
    n = sampler.n
    index = sampler.arcs.index
    pool = list(range(n)) if candidates is None else sorted(_node_indices(candidates, index, "candidates"))
    if isinstance(k, bool) or int(k) != k or not 0 <= k <= len(pool):
        raise ValueError(f"k must be an integer between 0 and {len(pool)}, got {k!r}")
    k = int(k)
    rng = make_rng(seed)
    samples = [sampler.sample(rng) for _ in range(runs)]
    covered = [bytearray(n) for _ in range(runs)]

    def gain(v: int) -> int:
        total = 0
        for (ptr, adj), cov in zip(samples, covered):
            if cov[v]:
                continue
            seen = {v}
            stack = [v]
            while stack:
                u = stack.pop()
                for j in range(ptr[u], ptr[u + 1]):
                    x = adj[j]
                    if not cov[x] and x not in seen:
                        seen.add(x)
                        stack.append(x)
            total += len(seen)
        return total

    def commit(v: int) -> None:
        for (ptr, adj), cov in zip(samples, covered):
            if cov[v]:
                continue
            cov[v] = 1
            stack = [v]
            while stack:
                u = stack.pop()
                for j in range(ptr[u], ptr[u + 1]):
                    x = adj[j]
                    if not cov[x]:
                        cov[x] = 1
                        stack.append(x)

    chosen: list[int] = []
    spread: list[float] = []
    gains: list[float] = []
    evaluations = 0
    heap: list[tuple[int, int, int]] = []
    if k:
        for v in pool:
            heap.append((-gain(v), v, 0))
        evaluations = len(pool)
        heapq.heapify(heap)
    total = 0
    while len(chosen) < k:
        neg, v, fresh_at = heapq.heappop(heap)
        if fresh_at == len(chosen):
            commit(v)
            chosen.append(v)
            total -= neg
            gains.append(-neg / runs)
            spread.append(total / runs)
        else:
            heapq.heappush(heap, (-gain(v), v, len(chosen)))
            evaluations += 1
    nodes = sampler.arcs.nodes
    return InfluenceMaximization([nodes[i] for i in chosen], spread, gains, sampler.model, runs, evaluations)


__all__ = [
    "independent_cascade",
    "linear_threshold",
    "influence_spread",
    "greedy_influence_maximization",
    "SpreadEstimate",
    "InfluenceMaximization",
]
