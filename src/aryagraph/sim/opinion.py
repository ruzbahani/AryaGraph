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

"""Opinion dynamics: voter model, majority rule, DeGroot averaging, Deffuant bounded confidence.

Influence travels along arcs ``u → v`` (*v* listens to *u*); undirected edges
act both ways. Discrete opinions give categorical results, real-valued ones
continuous results.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from ..core.exceptions import NodeNotFound
from ..core.results import NodeMap
from ..core.utils import WeightSpec, make_rng
from .base import (
    SimulationResult,
    _Arcs,
    _arcs,
    _closed_classes,
    _code_dtype,
    _is_node,
    _node_values,
    _period,
    _seed_record,
    _stationary,
    cycle_roles,
)

#: Largest closed class for which :func:`degroot` computes the exact consensus.
CONSENSUS_MAX_NODES = 2000


def _initial_opinions(opinions: Any, arcs: _Arcs, rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """State names and initial codes from a number of opinions or a ``{node: opinion}`` mapping."""
    n = arcs.n
    if isinstance(opinions, Integral) and not isinstance(opinions, bool):
        k = int(opinions)
        if k < 1:
            raise ValueError("opinions must be at least 1")
        states = [str(i) for i in range(k)]
        return states, rng.integers(k, size=n).astype(_code_dtype(k))
    if not isinstance(opinions, Mapping):
        raise TypeError("opinions must be a number of opinions or a {node: opinion} mapping")
    for key in opinions:
        if not _is_node(key, arcs.index):
            raise NodeNotFound(key)
    missing = [v for v in arcs.nodes if v not in opinions]
    if missing:
        raise ValueError(f"no initial opinion for node {missing[0]!r}")
    labels = list(dict.fromkeys(opinions[v] for v in arcs.nodes))
    try:
        labels = sorted(labels)
    except TypeError:
        pass  # unorderable labels keep first-appearance order
    states = [str(x) for x in labels]
    if len(set(states)) != len(states):
        raise ValueError("opinion labels must stay distinct when converted to strings")
    code = {x: i for i, x in enumerate(labels)}
    values = np.array([code[opinions[v]] for v in arcs.nodes], dtype=_code_dtype(len(states)))
    return states, values


def _steps(steps: Any) -> int:
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError(f"steps must be a non-negative integer, got {steps!r}")
    return int(steps)


def _weight_desc(weight: Any) -> Any:
    return weight if weight is None or isinstance(weight, str) else "callable"


def voter_model(
    g: Any,
    *,
    opinions: int | Mapping = 2,
    steps: int = 100,
    weight: WeightSpec = None,
    seed: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Classic voter model with random-sequential updates.

    Each step is a sweep of ``n`` updates; an update picks a uniformly random
    node, which copies the opinion of one of its in-neighbours chosen in
    proportion to weight (nodes without in-neighbours never change). The run
    stops early at consensus, which is absorbing.

    Parameters
    ----------
    opinions:
        Number of opinions (uniform random start; states ``"0"``, ``"1"``, …) or
        a ``{node: opinion}`` mapping covering every node.

    Returns
    -------
    SimulationResult
        One frame per sweep; ``edge_activity`` lists the copies that changed an
        opinion (``u → v``: *v* adopted *u*'s view). ``meta["consensus"]`` is the
        winning opinion (or ``None``) and ``meta["consensus_step"]`` the sweep at
        which it was reached.
    """
    steps = _steps(steps)
    arcs = _arcs(g, weight)
    n = arcs.n
    rng = make_rng(seed)
    states, state = _initial_opinions(opinions, arcs, rng)
    order, ptr = arcs.in_order()
    in_src = arcs.src[order]
    in_w = arcs.w[order]
    cums = [np.cumsum(in_w[ptr[v] : ptr[v + 1]]).tolist() for v in range(n)]
    srcs = [in_src[ptr[v] : ptr[v + 1]].tolist() for v in range(n)]
    nodes = arcs.nodes
    current = state.tolist()
    frames = [state.copy()]
    activity: list[list[tuple]] = [[]]
    done_at = 0 if len(set(current)) <= 1 else None
    for step in range(1, steps + 1):
        if done_at is not None:
            break
        changes = []
        for v, x in zip(rng.integers(n, size=n).tolist(), rng.random(n).tolist()):
            c = cums[v]
            if not c:
                continue
            j = min(bisect_right(c, x * c[-1]), len(c) - 1)
            u = srcs[v][j]
            if current[u] != current[v]:
                current[v] = current[u]
                changes.append((nodes[u], nodes[v]))
        frames.append(np.array(current, dtype=state.dtype))
        activity.append(changes)
        if len(set(current)) == 1:
            done_at = step
    consensus = states[current[0]] if done_at is not None and n else None
    return SimulationResult(
        graph=g,
        nodes=nodes,
        times=np.arange(len(frames), dtype=float),
        values=np.array(frames).reshape(len(frames), n),
        kind="categorical",
        states=states,
        roles=cycle_roles(states),
        model="voter",
        params={"opinions": len(states), "steps": steps, "weight": _weight_desc(weight)},
        seed=_seed_record(seed),
        edge_activity=activity,
        meta={"consensus": consensus, "consensus_step": done_at},
    )


def majority_rule(
    g: Any,
    *,
    opinions: int | Mapping = 2,
    steps: int = 50,
    weight: WeightSpec = None,
    include_self: bool = False,
    seed: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Synchronous majority rule.

    Every step, each node adopts the opinion carrying the largest total weight
    among its in-neighbours (plus itself, with weight 1, when *include_self*).
    On a tie the node keeps its opinion if it is among the leaders, otherwise
    picks a leader uniformly at random. Nodes without in-neighbours keep their
    opinion. The run stops at a fixed point (synchronous dynamics may also end
    in a 2-cycle, which runs until *steps*).

    ``edge_activity`` lists, for each node that switched, the arcs from its
    in-neighbours holding the adopted opinion.
    """
    steps = _steps(steps)
    arcs = _arcs(g, weight)
    n = arcs.n
    rng = make_rng(seed)
    states, state = _initial_opinions(opinions, arcs, rng)
    k = len(states)
    src, dst, w = arcs.src, arcs.dst, arcs.w
    has_in = np.bincount(dst, minlength=n) > 0
    nodes = arcs.nodes
    rows = np.arange(n)
    frames = [state.copy()]
    activity: list[list[tuple]] = [[]]
    fixed = False
    for _ in range(steps):
        votes = np.bincount(dst * k + state[src], weights=w, minlength=n * k).reshape(n, k)
        if include_self:
            votes[rows, state] += 1.0
        best = votes.max(axis=1)
        leaders = votes == best[:, None]
        change = has_in & ~leaders[rows, state]
        new = state.copy()
        idx = np.flatnonzero(change)
        if idx.size:
            lead = leaders[idx]
            pick = np.floor(rng.random(idx.size) * lead.sum(axis=1)).astype(np.intp)
            new[idx] = np.argmax(np.cumsum(lead, axis=1) > pick[:, None], axis=1)
        if np.array_equal(new, state):
            fixed = True
            break
        moved = new != state
        sel = moved[dst] & (state[src] == new[dst])
        activity.append([(nodes[a], nodes[b]) for a, b in zip(src[sel].tolist(), dst[sel].tolist())])
        frames.append(new)
        state = new
    return SimulationResult(
        graph=g,
        nodes=nodes,
        times=np.arange(len(frames), dtype=float),
        values=np.array(frames).reshape(len(frames), n),
        kind="categorical",
        states=states,
        roles=cycle_roles(states),
        model="majority",
        params={"opinions": k, "steps": steps, "weight": _weight_desc(weight), "include_self": include_self},
        seed=_seed_record(seed),
        edge_activity=activity,
        meta={"fixed_point": fixed},
    )


def degroot(
    g: Any,
    initial: Any,
    *,
    steps: int = 50,
    weight: WeightSpec = None,
    self_weight: float = 1.0,
    tol: float | None = None,
) -> SimulationResult:
    """DeGroot averaging ``x(t+1) = W x(t)``.

    ``W`` is row-stochastic: node *v* averages itself (weight ``self_weight``
    plus any self-loop weight) and its in-neighbours ``u`` (weight ``w(u, v)``),
    normalised to sum to one. A node with nothing to listen to keeps its value.

    Parameters
    ----------
    initial:
        ``{node: value}`` for every node, or values in graph order.
    tol:
        Stop once no value moves by more than *tol* in a step.

    Returns
    -------
    SimulationResult
        Continuous. When the listening structure has a single closed class
        that is aperiodic (always true with ``self_weight > 0``), the values
        converge to the consensus ``π · x(0)``, where ``π`` is the left
        eigenvector of ``W`` for eigenvalue 1 (each node's social power);
        ``meta["consensus"]`` and ``meta["influence"]`` then hold them, computed
        exactly for classes up to :data:`CONSENSUS_MAX_NODES` nodes (else, and
        when there is no consensus, ``None``).
    """
    steps = _steps(steps)
    self_weight = float(self_weight)
    if not (math.isfinite(self_weight) and self_weight >= 0):
        raise ValueError(f"self_weight must be finite and non-negative, got {self_weight!r}")
    arcs = _arcs(g, weight, self_loops=True)
    n = arcs.n
    x = _node_values(initial, arcs.nodes, arcs.index, "initial", scalar=False)
    loop = arcs.src == arcs.dst
    own = np.full(n, self_weight) + np.bincount(arcs.dst[loop], weights=arcs.w[loop], minlength=n)
    s, d, w = arcs.src[~loop], arcs.dst[~loop], arcs.w[~loop]
    denom = own + np.bincount(d, weights=w, minlength=n)
    stuck = denom <= 0.0
    own[stuck] = 1.0
    denom[stuck] = 1.0
    frames = [x.copy()]
    converged = False
    for _ in range(steps):
        nxt = (own * x + np.bincount(d, weights=w * x[s], minlength=n)) / denom
        frames.append(nxt)
        moved = float(np.max(np.abs(nxt - x))) if n else 0.0
        x = nxt
        if tol is not None and moved <= tol:
            converged = True
            break

    consensus = influence = None
    if n:
        listens: list[list[int]] = [[] for _ in range(n)]
        for a, b in zip(s.tolist(), d.tolist()):
            listens[b].append(a)
        for i in np.flatnonzero(own > 0).tolist():
            listens[i].append(i)
        closed = _closed_classes(listens)
        if len(closed) == 1 and len(closed[0]) <= CONSENSUS_MAX_NODES and _period(closed[0], listens) == 1:
            cls = np.array(closed[0])
            pos = {v: k for k, v in enumerate(closed[0])}
            wm = np.zeros((cls.size, cls.size))
            wm[np.arange(cls.size), np.arange(cls.size)] = own[cls] / denom[cls]
            for a, b, wt in zip(s.tolist(), d.tolist(), w.tolist()):
                if b in pos:
                    wm[pos[b], pos[a]] += wt / denom[b]
            pi_c = _stationary(wm)
            pi = np.zeros(n)
            pi[cls] = pi_c
            consensus = float(pi_c @ frames[0][cls])
            influence = NodeMap(zip(arcs.nodes, pi.tolist()), name="influence")
    lo, hi = (float(frames[0].min()), float(frames[0].max())) if n else (0.0, 1.0)
    return SimulationResult(
        graph=g,
        nodes=arcs.nodes,
        times=np.arange(len(frames), dtype=float),
        values=np.array(frames).reshape(len(frames), n),
        kind="continuous",
        states=[],
        roles={},
        model="DeGroot",
        params={"steps": steps, "weight": _weight_desc(weight), "self_weight": self_weight, "tol": tol},
        seed=None,
        edge_activity=None,
        meta={
            "domain": (lo, hi),
            "diverging": lo < 0.0 < hi,
            "consensus": consensus,
            "influence": influence,
            "converged": converged,
        },
    )


def bounded_confidence(
    g: Any,
    initial: Any = None,
    *,
    epsilon: float = 0.2,
    mu: float = 0.5,
    steps: int = 100,
    seed: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Deffuant–Weisbuch bounded-confidence model on opinions in [0, 1].

    Each step performs ``m`` pairwise interactions (``m`` = number of edges,
    self-loops excluded), each on an edge drawn uniformly at random. If the two
    opinions differ by less than *epsilon* they move towards each other by
    *mu* times their difference; on a directed graph only the head of the arc
    moves. The run stops early when no edge can change anything.

    Parameters
    ----------
    initial:
        ``None`` (uniform on [0, 1]), ``{node: opinion}`` or values in graph order.
    epsilon:
        Confidence bound, > 0.
    mu:
        Convergence parameter in (0, 0.5].

    Returns
    -------
    SimulationResult
        Continuous on ``meta["domain"] == (0, 1)``; ``edge_activity`` lists the
        interactions that moved an opinion. ``meta["clusters"]`` gives the mean
        final opinion of each group of sorted final opinions separated by gaps
        of at least *epsilon* (groups that can no longer interact).
    """
    steps = _steps(steps)
    epsilon, mu = float(epsilon), float(mu)
    if not epsilon > 0:
        raise ValueError(f"epsilon must be positive, got {epsilon!r}")
    if not 0 < mu <= 0.5:
        raise ValueError(f"mu must lie in (0, 0.5], got {mu!r}")
    arcs = _arcs(g, None, keep_zero=True)
    n = arcs.n
    rng = make_rng(seed)
    if initial is None:
        x0 = rng.random(n)
    else:
        x0 = _node_values(initial, arcs.nodes, arcs.index, "initial", scalar=False)
        if n and (x0.min() < 0.0 or x0.max() > 1.0):
            raise ValueError("bounded-confidence opinions must lie in [0, 1]")
    if g.directed:
        pu, pv = arcs.src, arcs.dst
    else:
        keep = arcs.src < arcs.dst
        pu, pv = arcs.src[keep], arcs.dst[keep]
    m = int(pu.size)
    both = not g.directed
    nodes = arcs.nodes
    x = x0.tolist()
    frames = [x0.copy()]
    activity: list[list[tuple]] = [[]]
    settled = False
    for _ in range(steps):
        xa = np.asarray(x)
        gap = np.abs(xa[pu] - xa[pv])
        if not np.any((gap < epsilon) & (gap > 0)):
            settled = True
            break
        moved = []
        for e in rng.integers(m, size=m).tolist():
            a, b = int(pu[e]), int(pv[e])
            diff = x[a] - x[b]
            if diff != 0.0 and abs(diff) < epsilon:
                x[b] += mu * diff
                if both:
                    x[a] -= mu * diff
                moved.append((nodes[a], nodes[b]))
        frames.append(np.array(x))
        activity.append(moved)
    final = np.sort(np.asarray(x))
    clusters: list[float] = []
    if final.size:
        cuts = np.flatnonzero(np.diff(final) >= epsilon) + 1
        clusters = [float(part.mean()) for part in np.split(final, cuts)]
    return SimulationResult(
        graph=g,
        nodes=nodes,
        times=np.arange(len(frames), dtype=float),
        values=np.array(frames).reshape(len(frames), n),
        kind="continuous",
        states=[],
        roles={},
        model="Deffuant",
        params={"epsilon": epsilon, "mu": mu, "steps": steps},
        seed=_seed_record(seed),
        edge_activity=activity,
        meta={"domain": (0.0, 1.0), "diverging": False, "clusters": clusters, "settled": settled},
    )


__all__ = ["voter_model", "majority_rule", "degroot", "bounded_confidence", "CONSENSUS_MAX_NODES"]
