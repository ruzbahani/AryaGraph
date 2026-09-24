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

"""Random walks and their stationary distribution.

A walker at ``u`` moves to an out-neighbour ``v`` with probability
``w(u, v) / Σ_x w(u, x)`` (undirected edges can be walked both ways; a self-loop
counts once and lets the walker stay). A node without outgoing weight is
*dangling*: the walker teleports to a uniformly random node, as in PageRank.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..core.exceptions import NodeNotFound, NotConnected
from ..core.results import NodeMap
from ..core.utils import WeightSpec, make_rng
from .base import SimulationResult, _arcs, _closed_classes, _is_node, _seed_record, _stationary

_STATES = ["unvisited", "visited", "current"]


def random_walk(
    g: Any,
    *,
    walkers: int = 1,
    steps: int = 100,
    start: Any = None,
    restart: float = 0.0,
    weight: WeightSpec = None,
    seed: int | np.random.Generator | None = None,
    n_frames: int | None = None,
) -> SimulationResult:
    """Simulate independent random walkers.

    Parameters
    ----------
    walkers, steps:
        Number of walkers and of steps each takes.
    start:
        ``None`` (each walker starts at a uniformly random node), one node for
        all walkers, or a sequence of ``walkers`` nodes.
    restart:
        Probability, at each step, of jumping back to the walker's own start
        node instead of moving (random walk with restart).
    weight:
        Edge weights for the transition probabilities (``None``: all 1).
    n_frames:
        Keep only about this many evenly spaced frames (default: every step);
        trajectories are always complete.

    Returns
    -------
    SimulationResult
        Categorical states ``unvisited`` / ``visited`` / ``current`` (a node
        holding at least one walker). ``edge_activity`` lists the moves along
        edges (teleports and restarts are not edges). ``meta["trajectories"]``
        is a ``(walkers, steps + 1)`` array of node indices into ``result.nodes``;
        ``meta["visits"]`` the share of all walker positions (start included)
        spent at each node, which converges to :func:`stationary_distribution`
        when ``restart == 0`` (for an undirected connected graph, ``deg / 2m``).
    """
    arcs = _arcs(g, weight, self_loops=True)
    n = arcs.n
    if n == 0:
        raise ValueError("cannot walk on an empty graph")
    walkers, steps = int(walkers), int(steps)
    if walkers < 1 or steps < 0:
        raise ValueError("walkers must be >= 1 and steps >= 0")
    restart = float(restart)
    if not 0.0 <= restart <= 1.0:
        raise ValueError(f"restart must lie in [0, 1], got {restart!r}")
    rng = make_rng(seed)
    if start is None:
        origin = rng.integers(n, size=walkers)
    elif _is_node(start, arcs.index):
        origin = np.full(walkers, arcs.index[start], dtype=np.intp)
    else:
        if isinstance(start, (str, bytes)) or not isinstance(start, Iterable):
            raise NodeNotFound(start)
        items = list(start)
        for x in items:
            if not _is_node(x, arcs.index):
                raise NodeNotFound(x)
        if len(items) != walkers:
            raise ValueError(f"start lists {len(items)} nodes for {walkers} walkers")
        origin = np.array([arcs.index[x] for x in items], dtype=np.intp)
    origin = origin.astype(np.intp)

    ptr = arcs.out_ptr()
    cum = np.cumsum(arcs.w)
    has_out = ptr[1:] > ptr[:-1]
    base = np.zeros(n)
    first = ptr[:-1]
    base[first > 0] = cum[first[first > 0] - 1]
    out_weight = np.zeros(n)
    out_weight[has_out] = cum[ptr[1:][has_out] - 1] - base[has_out]
    dangling = out_weight <= 0.0

    traj = np.empty((walkers, steps + 1), dtype=np.intp)
    traj[:, 0] = origin
    pos = origin.copy()
    moves: list[tuple[np.ndarray, np.ndarray]] = []
    teleports = restarts = 0
    for t in range(1, steps + 1):
        back = rng.random(walkers) < restart if restart > 0 else np.zeros(walkers, dtype=bool)
        x = rng.random(walkers)
        nxt = np.empty_like(pos)
        dang = dangling[pos] & ~back
        move = ~dangling[pos] & ~back
        nxt[back] = origin[back]
        nxt[dang] = np.minimum((x[dang] * n).astype(np.intp), n - 1)
        u = pos[move]
        if u.size:
            j = np.searchsorted(cum, base[u] + x[move] * out_weight[u], side="right")
            j = np.clip(j, ptr[u], ptr[u + 1] - 1)
            nxt[move] = arcs.dst[j]
            moves.append((u, arcs.dst[j]))
        else:
            moves.append((u, u))
        restarts += int(back.sum())
        teleports += int(dang.sum())
        pos = nxt
        traj[:, t] = pos

    # frames
    total = steps + 1
    if n_frames is None or n_frames >= total:
        keep = np.arange(total)
    else:
        if n_frames < 2:
            raise ValueError("n_frames must be at least 2")
        keep = np.unique(np.rint(np.linspace(0, steps, n_frames)).astype(np.intp))
    first_visit = np.full(n, total, dtype=np.intp)
    np.minimum.at(first_visit, traj.T.ravel(), np.repeat(np.arange(total), walkers))
    values = (first_visit[None, :] <= keep[:, None]).astype(np.int8)
    for r, t in enumerate(keep.tolist()):
        values[r, traj[:, t]] = 2
    nodes = arcs.nodes
    activity: list[list[tuple]] = [[] for _ in keep]
    for r in range(1, keep.size):
        acts = activity[r]
        for t in range(int(keep[r - 1]) + 1, int(keep[r]) + 1):
            a, b = moves[t - 1]
            acts.extend((nodes[s], nodes[d]) for s, d in zip(a.tolist(), b.tolist()))

    visits = np.bincount(traj.ravel(), minlength=n) / traj.size
    params = {
        "walkers": walkers,
        "steps": steps,
        "restart": restart,
        "weight": weight if weight is None or isinstance(weight, str) else "callable",
    }
    meta = {
        "trajectories": traj,
        "visits": NodeMap(zip(nodes, visits.tolist()), name="visits"),
        "start": [nodes[i] for i in origin.tolist()],
        "teleports": teleports,
        "restarts": restarts,
    }
    return SimulationResult(
        graph=g,
        nodes=nodes,
        times=keep.astype(float),
        values=values,
        kind="categorical",
        states=list(_STATES),
        roles={"unvisited": "neutral", "visited": "good", "current": "accent"},
        model="random walk",
        params=params,
        seed=_seed_record(seed),
        edge_activity=activity,
        meta=meta,
    )


def transition_matrix(g: Any, weight: WeightSpec = None) -> np.ndarray:
    """Row-stochastic transition matrix of the walk (graph order; dangling rows uniform)."""
    arcs = _arcs(g, weight, self_loops=True)
    n = arcs.n
    p = np.zeros((n, n))
    np.add.at(p, (arcs.src, arcs.dst), arcs.w)
    out = p.sum(axis=1)
    dangling = out <= 0.0
    p[~dangling] /= out[~dangling, None]
    if n:
        p[dangling] = 1.0 / n
    return p


def stationary_distribution(g: Any, weight: WeightSpec = None) -> NodeMap:
    """Exact stationary distribution ``π = πP`` of the random walk on *g*.

    For an undirected graph without dangling nodes the closed form
    ``π(v) = deg_w(v) / Σ deg_w`` is used (self-loops counted once, matching the
    transition probabilities). Otherwise the linear system ``π(P - I) = 0,
    Σπ = 1`` is solved directly (O(n³)); it has a unique solution exactly when
    the walk has a single closed class, as in a strongly connected graph
    (periodicity does not matter). Nodes outside that class get 0.

    Raises
    ------
    NotConnected
        When the walk has several closed classes (the distribution is not unique).
    """
    arcs = _arcs(g, weight, self_loops=True)
    n = arcs.n
    if n == 0:
        return NodeMap({}, name="stationary")
    out_weight = np.bincount(arcs.src, weights=arcs.w, minlength=n)
    dangling = out_weight <= 0.0
    # support graph; dangling nodes reach everything through one virtual hub
    succ: list[list[int]] = [[] for _ in range(n)]
    for s, d in zip(arcs.src.tolist(), arcs.dst.tolist()):
        succ[s].append(d)
    if dangling.any():
        for v in np.flatnonzero(dangling).tolist():
            succ[v].append(n)
        succ.append(list(range(n)))
    closed = _closed_classes(succ)
    if len(closed) != 1:
        raise NotConnected(
            f"the random walk has {len(closed)} closed classes, so its stationary distribution is not unique"
        )
    if not g.directed and not dangling.any():
        pi = out_weight / out_weight.sum()
    else:
        pi = _stationary(transition_matrix(g, weight))
    return NodeMap(zip(arcs.nodes, pi.tolist()), name="stationary")


__all__ = ["random_walk", "stationary_distribution", "transition_matrix"]
