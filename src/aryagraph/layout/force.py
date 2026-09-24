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

"""Force-directed layouts: Fruchterman–Reingold and ForceAtlas2.

Both are fully vectorised. Exact all-pairs repulsion uses the Gram-matrix
identity ``Σ_j f_ij (x_i − x_j) = x_i Σ_j f_ij − (F x)_i`` so one iteration is a
few dense ``(n, n)`` passes plus two BLAS products. ForceAtlas2 switches to a
Barnes–Hut quadtree for large graphs; the tree is built level by level from
Morton codes and traversed breadth-first over ``(node, cell)`` pairs, so no
per-node Python recursion is involved.
"""

from __future__ import annotations

import math
from typing import Any, Hashable, Iterable, Mapping

import numpy as np

from ..core.exceptions import NodeNotFound
from ..core.utils import WeightSpec, make_rng, weight_fn
from .base import Layout
from .geometric import _as_positions, _edge_arrays, _empty, _sizes_array
from .stress import _principal_axes

Node = Hashable


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _initial_positions(
    nodes: list[Node], init: Any, rng: np.random.Generator, scale: float
) -> tuple[np.ndarray, bool]:
    """Start positions: *init* where given, seeded uniform noise elsewhere."""
    n = len(nodes)
    x = rng.random((n, 2)) * scale
    if init is None:
        return x, False
    given = _as_positions(init)
    known = [i for i, v in enumerate(nodes) if v in given]
    if not known:
        return x, False
    pts = np.array([np.asarray(given[nodes[i]], dtype=float)[:2] for i in known])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, max(float(np.max(hi - lo)), 1.0))
    x = lo + rng.random((n, 2)) * span
    x[known] = pts
    return x, True


def _exact_repulsion(
    x: np.ndarray,
    coef: float,
    mass: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    eps2: float = 1e-12,
    table: np.ndarray | None = None,
) -> np.ndarray:
    """Exact all-pairs repulsion ``Σ_j f_ij (x_i − x_j)``, O(n²) time, bounded memory.

    ``f_ij = coef·m_i·m_j / d_ij²`` (masses default to 1), i.e. a force of
    magnitude ``coef·m_i·m_j / d``. With *radii*, ForceAtlas2's anti-collision
    variant uses the border distance ``d' = d − r_i − r_j``: ``coef·m_i·m_j / d'²``
    when apart, ``100·coef·m_i·m_j`` when overlapping. Rows are processed in
    blocks through the Gram identity, so no ``(n, n, 2)`` array is formed.
    *table* optionally caches ``coef·m_i·m_j`` for repeated calls.
    """
    n = x.shape[0]
    xc = x - x.mean(axis=0)
    sq = np.einsum("ij,ij->i", xc, xc)
    out = np.empty_like(xc)
    step = max(1, min(n, 2_000_000 // max(n, 1)))
    for s in range(0, n, step):
        e = min(n, s + step)
        q = xc[s:e] @ xc.T
        q *= -2.0
        q += sq[s:e, None]
        q += sq[None, :]
        np.maximum(q, eps2, out=q)  # squared distances
        if table is not None:
            num = table[s:e]
        else:
            num = coef if mass is None else coef * np.outer(mass[s:e], mass)
        if radii is None:
            f = np.divide(num, q, out=q)
        else:
            dd = np.sqrt(q) - radii[s:e, None] - radii[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(dd > 0, num / np.maximum(dd * dd, eps2), np.where(dd < 0, 100.0 * num, 0.0))
        f[np.arange(e - s), np.arange(s, e)] = 0.0
        out[s:e] = xc[s:e] * f.sum(axis=1)[:, None] - f @ xc
    return out


# ---------------------------------------------------------------------- #
# Fruchterman–Reingold
# ---------------------------------------------------------------------- #
def fruchterman_reingold(
    g: Any,
    *,
    seed: int = 0,
    iterations: int = 300,
    k: float | None = None,
    weight: WeightSpec = "weight",
    init: Layout | Mapping[Node, Any] | None = None,
    fixed: Iterable[Node] | None = None,
    gravity: float = 0.0,
) -> Layout:
    """Fruchterman–Reingold spring embedder with simulated-annealing cooling.

    Parameters
    ----------
    k:
        Ideal edge length (default ``√(area / n)`` in the working frame).
    weight:
        Edge attribute / callable scaling the attraction (missing ⇒ 1).
    init:
        Start positions (:class:`Layout` or mapping); missing nodes start at
        random inside the given positions' bounding box.
    fixed:
        Nodes that never move (they keep their *init* coordinates, and the
        result stays in that coordinate frame).
    gravity:
        Pull towards the centroid (0 = classic FR); useful for disconnected graphs.

    Returns
    -------
    Layout
        Abstract layout. Without *fixed*, it is rescaled so the ideal edge
        length is 1 and rotated to its principal axes.

    Each step moves a node by at most the current temperature, which cools
    linearly to zero; the loop stops early once the mean displacement is tiny.
    """
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("fruchterman_reingold")
    index = {v: i for i, v in enumerate(nodes)}
    rng = make_rng(seed)
    x, has_init = _initial_positions(nodes, init, rng, 1.0)
    fixed_mask = np.zeros(n, dtype=bool)
    if fixed is not None:
        for v in fixed:
            if v not in index:
                raise NodeNotFound(v)
            fixed_mask[index[v]] = True
    if n == 1:
        return Layout(nodes, x if has_init else np.zeros((1, 2)), method="fruchterman_reingold")
    ei, ej, w = _edge_arrays(g, index, weight)
    frame = float(np.ptp(x, axis=0).max()) if has_init else 1.0
    frame = frame if frame > 0 else 1.0
    kk = float(k) if k else frame * math.sqrt(1.0 / n)
    t = 0.1 * max(frame, float(np.ptp(x, axis=0).max()))
    dt = t / (iterations + 1)
    eps2 = (1e-3 * kk) ** 2
    k2 = kk * kk
    for _ in range(int(iterations)):
        disp = _exact_repulsion(x, k2, eps2=eps2)  # k²/d along each pair
        if ei.size:
            delta = x[ei] - x[ej]
            dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            f = (w * dist / kk)[:, None] * delta
            disp[:, 0] -= np.bincount(ei, f[:, 0], minlength=n) - np.bincount(ej, f[:, 0], minlength=n)
            disp[:, 1] -= np.bincount(ei, f[:, 1], minlength=n) - np.bincount(ej, f[:, 1], minlength=n)
        if gravity:
            # constant pull of `gravity · k` towards the centroid (the same
            # magnitude as one neighbour's repulsion at the ideal distance)
            rel = x - x.mean(axis=0)
            r = np.sqrt(np.einsum("ij,ij->i", rel, rel))
            disp -= (gravity * kk / np.maximum(r, 1e-12))[:, None] * rel
        length = np.sqrt(np.einsum("ij,ij->i", disp, disp))
        step = disp * (np.minimum(length, t) / np.maximum(length, 1e-12))[:, None]
        step[fixed_mask] = 0.0
        x += step
        t -= dt
        if t <= 0 or float(np.sqrt(np.einsum("ij,ij->i", step, step)).mean()) < 1e-5 * kk:
            break
    meta: dict[str, Any] = {"k": kk}
    if fixed_mask.any():
        return Layout(nodes, x, method="fruchterman_reingold", meta=meta)
    x = x / kk
    meta["k"] = 1.0
    if not has_init:
        x = _principal_axes(x)
    return Layout(nodes, x, method="fruchterman_reingold", meta=meta)


# ---------------------------------------------------------------------- #
# Barnes–Hut quadtree (vectorised)
# ---------------------------------------------------------------------- #
_MAX_DEPTH = 16


def _spread_bits(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64) & np.uint64(0xFFFF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF)
    v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F)
    v = (v | (v << np.uint64(2))) & np.uint64(0x33333333)
    v = (v | (v << np.uint64(1))) & np.uint64(0x55555555)
    return v


class _QuadTree:
    """Level-by-level quadtree over Morton codes, rebuilt every iteration."""

    def __init__(self, x: np.ndarray, mass: np.ndarray) -> None:
        n = x.shape[0]
        lo = x.min(axis=0)
        span = float(np.ptp(x, axis=0).max()) * (1 + 1e-9) or 1.0
        self.span = span
        cells = 1 << _MAX_DEPTH
        q = np.clip(((x - lo) / span * cells).astype(np.int64), 0, cells - 1)
        self.code = _spread_bits(q[:, 0]) | (_spread_bits(q[:, 1]) << np.uint64(1))
        self.levels = []
        for lev in range(1, _MAX_DEPTH + 1):
            key = self.code >> np.uint64(2 * (_MAX_DEPTH - lev))
            uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            m = np.bincount(inv, mass, minlength=uk.size)
            cx = np.bincount(inv, mass * x[:, 0], minlength=uk.size) / m
            cy = np.bincount(inv, mass * x[:, 1], minlength=uk.size) / m
            first = np.full(uk.size, n, dtype=np.int64)
            np.minimum.at(first, inv, np.arange(n))
            self.levels.append({"key": uk, "mass": m, "cx": cx, "cy": cy, "count": cnt, "first": first, "inv": inv})
            if cnt.max() == 1:
                break
        # child ranges: children of cell c at level l are the level-(l+1) cells whose key>>2 == key_c
        for li in range(len(self.levels) - 1):
            parent_keys = self.levels[li + 1]["key"] >> np.uint64(2)
            keys = self.levels[li]["key"]
            self.levels[li]["cstart"] = np.searchsorted(parent_keys, keys, side="left")
            self.levels[li]["cend"] = np.searchsorted(parent_keys, keys, side="right")

    def repulsion(
        self, x: np.ndarray, mass: np.ndarray, kr: float, theta: float, radii: np.ndarray | None
    ) -> np.ndarray:
        n = x.shape[0]
        out = np.zeros_like(x)
        lv0 = self.levels[0]
        nc = lv0["key"].size
        node = np.repeat(np.arange(n), nc)
        cell = np.tile(np.arange(nc), n)
        last = len(self.levels) - 1
        for li, lv in enumerate(self.levels):
            if node.size == 0:
                break
            size = self.span / (1 << (li + 1))
            cnt = lv["count"][cell]
            own = lv["inv"][node] == cell
            dx = x[node, 0] - lv["cx"][cell]
            dy = x[node, 1] - lv["cy"][cell]
            d2 = dx * dx + dy * dy
            single = (cnt == 1) & ~own
            far = ~own & ~single & (size * size < theta * theta * d2)
            leaf_multi = (li == last) & ~single & ~far
            # 1. single-node cells: exact interaction (anti-collision aware)
            if single.any():
                i = node[single]
                j = lv["first"][cell[single]]
                self._pair(out, x, mass, kr, i, j, dx[single], dy[single], d2[single], radii)
            # 2. well-separated aggregated cells
            if far.any():
                i = node[far]
                f = kr * mass[i] * lv["mass"][cell[far]] / np.maximum(d2[far], 1e-18)
                out[:, 0] += np.bincount(i, f * dx[far], minlength=n)
                out[:, 1] += np.bincount(i, f * dy[far], minlength=n)
            # 3. deepest level: coincident clusters, interact with their centre of mass
            if leaf_multi.any():
                i = node[leaf_multi]
                c = cell[leaf_multi]
                m = lv["mass"][c].copy()
                cx = lv["cx"][c] * m
                cy = lv["cy"][c] * m
                mine = own[leaf_multi]
                m[mine] -= mass[i[mine]]
                cx[mine] -= mass[i[mine]] * x[i[mine], 0]
                cy[mine] -= mass[i[mine]] * x[i[mine], 1]
                ok = m > 1e-12
                i, m, cx, cy = i[ok], m[ok], cx[ok] / m[ok], cy[ok] / m[ok]
                ddx = x[i, 0] - cx
                ddy = x[i, 1] - cy
                dd2 = np.maximum(ddx * ddx + ddy * ddy, 1e-12)
                f = kr * mass[i] * m / dd2
                out[:, 0] += np.bincount(i, f * ddx, minlength=n)
                out[:, 1] += np.bincount(i, f * ddy, minlength=n)
            if li == last:
                break
            # 4. open the remaining cells
            opn = ~single & ~far & ~leaf_multi
            node, cell = node[opn], cell[opn]
            cs, ce = lv["cstart"][cell], lv["cend"][cell]
            k = ce - cs
            node = np.repeat(node, k)
            offs = np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)
            cell = np.repeat(cs, k) + offs
        return out

    @staticmethod
    def _pair(out, x, mass, kr, i, j, dx, dy, d2, radii) -> None:
        n = x.shape[0]
        mm = kr * mass[i] * mass[j]
        if radii is None:
            f = mm / np.maximum(d2, 1e-18)
        else:
            d = np.sqrt(d2)
            dd = d - radii[i] - radii[j]
            f = np.where(dd > 0, mm / np.maximum(dd * dd, 1e-18), np.where(dd < 0, 100.0 * mm, 0.0))
        out[:, 0] += np.bincount(i, f * dx, minlength=n)
        out[:, 1] += np.bincount(i, f * dy, minlength=n)


# ---------------------------------------------------------------------- #
# ForceAtlas2
# ---------------------------------------------------------------------- #
def force_atlas2(
    g: Any,
    *,
    seed: int = 0,
    iterations: int = 500,
    scaling: float = 2.0,
    gravity: float = 1.0,
    strong_gravity: bool = False,
    lin_log: bool = False,
    dissuade_hubs: bool = False,
    prevent_overlap: bool = False,
    sizes: Mapping[Node, Any] | None = None,
    barnes_hut: bool | None = None,
    theta: float = 1.2,
    weight: WeightSpec = "weight",
    init: Layout | Mapping[Node, Any] | None = None,
) -> Layout:
    """ForceAtlas2 (Jacomy, Venturini, Heymann & Bastian, 2014).

    Degree-weighted repulsion ``k_r (deg_i+1)(deg_j+1)/d``, linear attraction
    along edges (``log(1+d)`` with *lin_log*), gravity towards the origin, and
    the paper's adaptive speed: global speed from the ratio of *traction* to
    *swinging*, per-node speed damped by the node's own swinging.

    Parameters
    ----------
    scaling:
        Repulsion strength ``k_r``; larger values spread the graph.
    gravity, strong_gravity:
        Pull towards the centre (constant magnitude, or proportional to
        distance with *strong_gravity*).
    lin_log:
        Logarithmic attraction, which gives tighter, more readable clusters.
    dissuade_hubs:
        Distribute attraction by the source's mass, pushing hubs to the periphery.
    prevent_overlap:
        Anti-collision forces using the node radii from *sizes* (``max(w, h)/2``,
        default 18); best combined with a few hundred iterations.
    barnes_hut:
        Approximate repulsion with a quadtree (``theta`` = opening criterion);
        ``None`` enables it above 1500 nodes.
    weight:
        Edge attribute / callable multiplying the attraction.

    Returns
    -------
    Layout
        Abstract layout (ForceAtlas2's own units), rotated to principal axes
        unless *init* was given.
    """
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("forceatlas2")
    index = {v: i for i, v in enumerate(nodes)}
    rng = make_rng(seed)
    x, has_init = _initial_positions(nodes, init, rng, 10.0 * math.sqrt(n))
    if n == 1:
        return Layout(nodes, x if has_init else np.zeros((1, 2)), method="forceatlas2")
    wf = weight_fn(weight)
    src, dst, ew = [], [], []
    for u, v, d in g._iter_edges():
        if u == v:
            continue
        val = wf(u, v, d)
        src.append(index[u])
        dst.append(index[v])
        ew.append(1.0 if val is None else float(val))
    src_a = np.array(src, dtype=np.int64)
    dst_a = np.array(dst, dtype=np.int64)
    ew_a = np.array(ew, dtype=float)
    deg = np.bincount(src_a, minlength=n) + np.bincount(dst_a, minlength=n)
    mass = deg.astype(float) + 1.0
    radii = None
    if prevent_overlap:
        radii = _sizes_array(nodes, sizes).max(axis=1) / 2.0
    use_bh = (n > 1500) if barnes_hut is None else bool(barnes_hut)
    att_coef = float(mass.mean()) if dissuade_hubs else 1.0
    table = scaling * np.outer(mass, mass) if (not use_bh and n <= 3000) else None
    speed, speed_eff, jitter_tol = 1.0, 1.0, 1.0
    old = np.zeros_like(x)
    for _ in range(int(iterations)):
        # repulsion
        if use_bh:
            forces = _QuadTree(x, mass).repulsion(x, mass, scaling, theta, radii)
        else:
            forces = _exact_repulsion(x, scaling, mass, radii, table=table)
        # gravity (towards the origin, as in Gephi)
        dist0 = np.sqrt(np.einsum("ij,ij->i", x, x))
        if strong_gravity:
            forces -= (gravity * mass)[:, None] * x
        else:
            gf = np.where(dist0 > 0, gravity * mass / np.maximum(dist0, 1e-12), 0.0)
            forces -= gf[:, None] * x
        # attraction
        if src_a.size:
            delta = x[src_a] - x[dst_a]
            dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            if radii is not None:
                eff = dist - radii[src_a] - radii[dst_a]
            else:
                eff = dist
            if lin_log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    fac = np.where(eff > 0, np.log1p(np.maximum(eff, 0)) / np.maximum(eff, 1e-12), 0.0)
            else:
                fac = np.where(eff > 0, 1.0, 0.0) if radii is not None else np.ones_like(dist)
            fac = att_coef * ew_a * fac
            if dissuade_hubs:
                fac = fac / mass[src_a]
            fv = fac[:, None] * delta
            forces[:, 0] -= np.bincount(src_a, fv[:, 0], minlength=n) - np.bincount(dst_a, fv[:, 0], minlength=n)
            forces[:, 1] -= np.bincount(src_a, fv[:, 1], minlength=n) - np.bincount(dst_a, fv[:, 1], minlength=n)
        # adaptive speed (Gephi's implementation of the paper)
        sw = mass * np.linalg.norm(old - forces, axis=1)
        tr = mass * 0.5 * np.linalg.norm(old + forces, axis=1)
        total_sw, total_tr = float(sw.sum()), float(tr.sum())
        est_jt = 0.05 * math.sqrt(n)
        jt = jitter_tol * max(math.sqrt(est_jt), min(10.0, est_jt * total_tr / (n * n)))
        if total_tr > 0 and total_sw / total_tr > 2.0:
            if speed_eff > 0.05:
                speed_eff *= 0.5
            jt = max(jt, jitter_tol)
        target = jt * speed_eff * total_tr / max(total_sw, 1e-12)
        if total_sw > jt * total_tr:
            if speed_eff > 0.05:
                speed_eff *= 0.7
        elif speed < 1000:
            speed_eff *= 1.3
        speed = speed + min(target - speed, 0.5 * speed)
        if radii is not None:
            factor = 0.1 * speed / (1.0 + np.sqrt(speed * sw))
            df = np.linalg.norm(forces, axis=1)
            factor = np.minimum(factor * df, 10.0) / np.maximum(df, 1e-12)
        else:
            factor = speed / (1.0 + np.sqrt(speed * sw))
        x = x + forces * factor[:, None]
        old = forces
    if not has_init:
        x = _principal_axes(x)
    return Layout(nodes, x, method="forceatlas2", meta={"barnes_hut": use_bh, "scaling": scaling})


__all__ = ["fruchterman_reingold", "force_atlas2"]
