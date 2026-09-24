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

"""Continuous dynamics on graphs: heat diffusion and Kuramoto oscillators."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.results import NodeMap
from ..core.utils import WeightSpec, make_rng, weight_fn
from .base import SimulationResult, _arcs, _is_node, _node_values, _seed_record

TAU = 2.0 * math.pi


def _symmetric_adjacency(g: Any, weight: WeightSpec, nodes: list, index: dict) -> np.ndarray:
    """Dense symmetric weights; an arc in either direction links the pair (the
    larger weight wins when both exist) and self-loops are ignored, following
    the same convention as :func:`aryagraph.algorithms.laplacian_matrix`."""
    wf = weight_fn(weight)
    a = np.zeros((len(nodes), len(nodes)))
    for u, nbrs in g._succ.items():
        i = index[u]
        for v, d in nbrs.items():
            if v == u:
                continue
            x = wf(u, v, d)
            if x is None:
                continue
            x = float(x)
            if not (math.isfinite(x) and x >= 0):
                raise ValueError(f"weight of edge ({u!r}, {v!r}) must be finite and non-negative, got {x!r}")
            j = index[v]
            if x > a[i, j]:
                a[i, j] = a[j, i] = x
    return a


def _components(a: np.ndarray) -> list[np.ndarray]:
    """Connected components of the positive entries of a symmetric matrix."""
    n = a.shape[0]
    label = np.full(n, -1)
    comps = []
    for s in range(n):
        if label[s] >= 0:
            continue
        label[s] = len(comps)
        members = [s]
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in np.flatnonzero((a[u] > 0) & (label < 0)).tolist():
                label[v] = len(comps)
                members.append(v)
                queue.append(v)
        comps.append(np.array(sorted(members)))
    return comps


def _heat_initial(initial: Any, nodes: list, index: dict) -> np.ndarray:
    if isinstance(initial, Mapping):
        return _node_values(initial, nodes, index, "initial", missing=0.0)
    if _is_node(initial, index):
        x = np.zeros(len(nodes))
        x[index[initial]] = 1.0
        return x
    return _node_values(initial, nodes, index, "initial", scalar=False)


def heat_diffusion(
    g: Any,
    initial: Any,
    *,
    rate: float = 1.0,
    t_max: float = 5.0,
    n_frames: int = 101,
    weight: WeightSpec = None,
    normalized: bool = False,
) -> SimulationResult:
    """Exact heat flow ``dx/dt = -rate · L x`` on the graph.

    The solution ``x(t) = exp(-rate·t·L) x(0)`` is evaluated in closed form from
    one symmetric eigendecomposition of ``L`` (O(n³) once, then O(n²) per
    frame), so there is no time-stepping error.

    Parameters
    ----------
    initial:
        ``{node: heat}`` (missing nodes start at 0), a single node (one unit of
        heat there), or values in graph order.
    rate:
        Diffusion coefficient, ≥ 0.
    t_max, n_frames:
        Frames at ``linspace(0, t_max, n_frames)``.
    weight:
        Edge conductance (``None``: all 1). Directed graphs are symmetrised:
        heat flows both ways along an arc (the larger weight wins when both
        directions exist). Self-loops carry no heat.
    normalized:
        ``False``: combinatorial Laplacian ``L = D - A``; heat spreads to the
        component average. ``True``: random-walk Laplacian ``L D⁻¹``; a node
        sheds heat at a rate independent of its degree and the steady state is
        proportional to degree (computed through the symmetric normalised
        Laplacian, a similarity transform). Isolated nodes keep their heat.
        Both conserve the total heat.

    Returns
    -------
    SimulationResult
        Continuous; ``meta["total"]`` is the total heat per frame (constant up
        to rounding) and ``meta["steady_state"]`` the exact ``t → ∞`` limit.
    """
    rate, t_max = float(rate), float(t_max)
    if not (math.isfinite(rate) and rate >= 0):
        raise ValueError(f"rate must be finite and non-negative, got {rate!r}")
    if not (math.isfinite(t_max) and t_max > 0):
        raise ValueError(f"t_max must be positive, got {t_max!r}")
    n_frames = int(n_frames)
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2")
    nodes = list(g._node)
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    x0 = _heat_initial(initial, nodes, index)
    a = _symmetric_adjacency(g, weight, nodes, index)
    deg = a.sum(axis=1)
    times = np.linspace(0.0, t_max, n_frames)
    values = np.tile(x0, (n_frames, 1))
    steady = x0.copy()
    if not normalized:
        lap = np.diag(deg) - a
        lam, vec = np.linalg.eigh(lap)
        lam = np.clip(lam, 0.0, None)
        coef = vec.T @ x0
        values = (np.exp(-rate * np.outer(times, lam)) * coef) @ vec.T
        for comp in _components(a):
            steady[comp] = x0[comp].mean()
    else:
        pos = deg > 0
        if pos.any():
            root = np.sqrt(deg[pos])
            sym = np.eye(int(pos.sum())) - a[np.ix_(pos, pos)] / np.outer(root, root)
            lam, vec = np.linalg.eigh(sym)
            lam = np.clip(lam, 0.0, 2.0)
            coef = vec.T @ (x0[pos] / root)
            values[:, pos] = ((np.exp(-rate * np.outer(times, lam)) * coef) @ vec.T) * root
        for comp in _components(a):
            if deg[comp].sum() > 0:
                steady[comp] = deg[comp] * x0[comp].sum() / deg[comp].sum()
    values[0] = x0
    lo, hi = (float(values.min()), float(values.max())) if n else (0.0, 1.0)
    return SimulationResult(
        graph=g,
        nodes=nodes,
        times=times,
        values=values.reshape(n_frames, n),
        kind="continuous",
        states=[],
        roles={},
        model="heat",
        params={
            "rate": rate,
            "t_max": t_max,
            "normalized": bool(normalized),
            "weight": weight if weight is None or isinstance(weight, str) else "callable",
        },
        seed=None,
        edge_activity=None,
        meta={
            "domain": (lo, hi),
            "diverging": lo < 0.0 < hi,
            "total": values.sum(axis=1),
            "steady_state": NodeMap(zip(nodes, steady.tolist()), name="steady_state"),
            "laplacian": "random-walk" if normalized else "combinatorial",
        },
    )


def kuramoto(
    g: Any,
    *,
    coupling: float = 1.0,
    frequencies: Any = None,
    initial: Any = None,
    t_max: float = 20.0,
    dt: float = 0.05,
    weight: WeightSpec = None,
    normalize: bool = False,
    seed: int | np.random.Generator | None = None,
    n_frames: int | None = None,
) -> SimulationResult:
    """Kuramoto phase oscillators coupled along the graph.

    ``dθ_v/dt = ω_v + K · Σ_u w(u, v) sin(θ_u − θ_v)`` over the in-neighbours
    ``u`` of ``v`` (both directions for undirected edges), divided by the
    weighted in-degree of ``v`` when *normalize*. Integrated with classical
    fourth-order Runge–Kutta (global error O(dt⁴)) on unwrapped phases.

    Parameters
    ----------
    coupling:
        ``K``.
    frequencies:
        Natural frequencies ``ω``: ``None`` (standard normal, from *seed*), one
        number for all, ``{node: ω}`` or values in graph order.
    initial:
        Initial phases: ``None`` (uniform on [0, 2π), drawn after the
        frequencies), ``{node: θ}`` or values in graph order.
    t_max, dt:
        ``floor(t_max / dt)`` steps of size *dt*.
    n_frames:
        Keep about this many evenly spaced frames (default: every step).

    Returns
    -------
    SimulationResult
        Continuous phases in ``[0, 2π)`` (``meta["domain"] == (0, 2π)``,
        ``meta["cyclic"]``); ``meta["order_parameter"]`` is ``r(t) = |⟨e^{iθ}⟩|``
        per frame (1 = full synchrony) and ``meta["mean_phase"]`` its angle.
    """
    t_max, dt, coupling = float(t_max), float(dt), float(coupling)
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be positive, got {dt!r}")
    if not (math.isfinite(t_max) and t_max >= 0):
        raise ValueError(f"t_max must be non-negative, got {t_max!r}")
    arcs = _arcs(g, weight)
    n = arcs.n
    rng = make_rng(seed)
    if frequencies is None:
        omega = rng.standard_normal(n)
    else:
        omega = _node_values(frequencies, arcs.nodes, arcs.index, "frequencies")
    if initial is None:
        theta = rng.uniform(0.0, TAU, n)
    else:
        theta = _node_values(initial, arcs.nodes, arcs.index, "initial")
    src, dst, w = arcs.src, arcs.dst, arcs.w
    scale = np.ones(n)
    if normalize:
        indeg = np.bincount(dst, weights=w, minlength=n)
        scale[indeg > 0] = indeg[indeg > 0]
    gain = coupling / scale

    def velocity(th: np.ndarray) -> np.ndarray:
        return omega + gain * np.bincount(dst, weights=w * np.sin(th[src] - th[dst]), minlength=n)

    n_steps = int(math.floor(t_max / dt + 1e-9))
    if n_frames is None or n_frames >= n_steps + 1:
        keep = np.arange(n_steps + 1)
    else:
        if n_frames < 2:
            raise ValueError("n_frames must be at least 2")
        keep = np.unique(np.rint(np.linspace(0, n_steps, n_frames)).astype(np.intp))
    wanted = np.zeros(n_steps + 1, dtype=bool)
    wanted[keep] = True
    frames = [theta.copy()]
    th = theta.copy()
    for step in range(1, n_steps + 1):
        k1 = velocity(th)
        k2 = velocity(th + 0.5 * dt * k1)
        k3 = velocity(th + 0.5 * dt * k2)
        k4 = velocity(th + dt * k3)
        th = th + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if wanted[step]:
            frames.append(th.copy())
    raw = np.array(frames).reshape(len(frames), n)
    z = np.exp(1j * raw).mean(axis=1) if n else np.zeros(len(frames), dtype=complex)
    values = np.mod(raw, TAU)
    values[values >= TAU] = 0.0  # mod can round up to exactly 2π
    return SimulationResult(
        graph=g,
        nodes=arcs.nodes,
        times=keep * dt,
        values=values,
        kind="continuous",
        states=[],
        roles={},
        model="Kuramoto",
        params={"coupling": coupling, "t_max": t_max, "dt": dt, "normalize": bool(normalize),
                "weight": weight if weight is None or isinstance(weight, str) else "callable"},
        seed=_seed_record(seed),
        edge_activity=None,
        meta={
            "domain": (0.0, TAU),
            "diverging": False,
            "cyclic": True,
            "order_parameter": np.abs(z),
            "mean_phase": np.mod(np.angle(z), TAU),
            "frequencies": NodeMap(zip(arcs.nodes, omega.tolist()), name="frequency"),
        },
    )


__all__ = ["heat_diffusion", "kuramoto"]
