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

"""Geometric layouts: circle, shells, grid, random, spiral, bipartite and arc.

These place nodes on simple curves or lattices. They are *abstract* layouts
(``metric=False``): only relative positions matter and the renderer scales
them to the canvas. Spacing is chosen so that neighbouring slots are about one
unit apart, the same scale the force and stress layouts use.

The module also hosts the small private helpers the other layout engines share
(node sizes, edge index arrays, component packing).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np

from ..algorithms._core import components
from ..core.exceptions import NodeNotFound
from ..core.utils import make_rng, weight_fn
from .base import Layout, pack_components

Node = Hashable

DEFAULT_SIZE = (36.0, 36.0)


# ---------------------------------------------------------------------- #
# shared private helpers
# ---------------------------------------------------------------------- #
def _empty(method: str, metric: bool = False, **meta: Any) -> Layout:
    return Layout([], np.zeros((0, 2)), method=method, metric=metric, meta=dict(meta))


def _sizes_array(nodes: Sequence[Node], sizes: Any, default: Sequence[float] = DEFAULT_SIZE) -> np.ndarray:
    """``(n, 2)`` array of node box ``(w, h)``.

    *sizes* is ``None``, a mapping ``node -> (w, h)`` (missing nodes get
    *default*; a bare number means a square), or a single ``(w, h)`` pair
    applied to every node.
    """
    out = np.empty((len(nodes), 2), dtype=float)
    out[:] = default
    if sizes is None:
        return out
    if isinstance(sizes, Mapping):
        for i, n in enumerate(nodes):
            s = sizes.get(n)
            if s is None:
                continue
            if isinstance(s, (int, float)):
                out[i] = (float(s), float(s))
            else:
                out[i] = (float(s[0]), float(s[1]))
    else:
        arr = np.asarray(sizes, dtype=float).ravel()
        if arr.size == 1:
            out[:] = arr[0]
        elif arr.size == 2:
            out[:] = arr
        else:
            raise TypeError("sizes must be a mapping node -> (w, h) or a single (w, h) pair")
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise ValueError("node sizes must be finite and non-negative")
    return out


def _edge_arrays(g: Any, index: Mapping[Node, int], weight: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected, loop-free edges as index arrays ``(i, j, w)`` with ``i < j``.

    For directed graphs ``u→v`` and ``v→u`` merge into one pair (weights add).
    *weight* follows the library convention (None / attribute name / callable).
    """
    wf = weight_fn(weight)
    acc: dict[tuple[int, int], float] = {}
    for u, v, d in g._iter_edges():
        if u == v:
            continue
        i, j = index[u], index[v]
        if i > j:
            i, j = j, i
        w = wf(u, v, d)
        w = 1.0 if w is None else float(w)
        acc[(i, j)] = acc.get((i, j), 0.0) + w
    if not acc:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), np.zeros(0)
    keys = np.array(list(acc.keys()), dtype=np.int64)
    return keys[:, 0], keys[:, 1], np.fromiter(acc.values(), dtype=float, count=len(acc))


def _undirected_adjacency(g: Any, nodes: Sequence[Node], index: Mapping[Node, int]) -> list[list[int]]:
    """Loop-free, duplicate-free neighbour lists (both directions for digraphs), graph order."""
    adj: list[list[int]] = [[] for _ in nodes]
    seen: list[set[int]] = [set() for _ in nodes]
    succ, pred = g._succ, g._pred
    for u in nodes:
        i = index[u]
        for nbrs in ((succ[u], pred[u]) if g.directed else (succ[u],)):
            for v in nbrs:
                if v == u:
                    continue
                j = index[v]
                if j not in seen[i]:
                    seen[i].add(j)
                    adj[i].append(j)
    return adj


def _as_positions(pos: Any) -> Mapping[Node, Sequence[float]]:
    """Accept a :class:`Layout` or a ``{node: (x, y)}`` mapping."""
    if isinstance(pos, Layout):
        return {n: pos.xy[i] for i, n in enumerate(pos.nodes)}
    if isinstance(pos, Mapping):
        return pos
    raise TypeError(f"expected a Layout or a {{node: (x, y)}} mapping, got {type(pos).__name__}")


def _median_edge_length(lay: Layout, g: Any) -> float:
    idx = lay._index
    xy = lay.xy
    ls = []
    for u, v, _ in g._iter_edges():
        if u != v and u in idx and v in idx:
            d = float(np.hypot(*(xy[idx[u]] - xy[idx[v]])))
            if d > 1e-12:
                ls.append(d)
    return float(np.median(ls)) if ls else 0.0


def _reorder(lay: Layout, nodes: Sequence[Node]) -> Layout:
    """Same layout with rows in *nodes* order."""
    if list(lay.nodes) == list(nodes):
        return lay
    xy = lay.xy[[lay._index[n] for n in nodes]] if len(nodes) else np.zeros((0, 2))
    return Layout(list(nodes), xy, routes=lay.routes, method=lay.method, metric=lay.metric, meta=dict(lay.meta))


def _pack_by_components(g: Any, fn: Callable[..., Layout], gap: float = 1.0, **kwargs: Any) -> Layout | None:
    """Lay out each weakly connected component with *fn* and pack them.

    Returns None when the graph has a single component (the caller then lays
    out the whole graph). Every component is rescaled to a median edge length
    of 1 so the pieces share one scale; *gap* (in edge lengths) separates them.
    """
    comps = components(g)
    if len(comps) <= 1:
        return None
    parts = []
    for comp in comps:
        sub = g.subgraph(comp)
        lay = fn(sub, **kwargs)
        med = _median_edge_length(lay, sub)
        if med > 0:
            c = lay.centroid()
            lay = lay.translated(-c[0], -c[1]).scaled(1.0 / med)
        elif len(comp) == 1:
            lay = Layout(lay.nodes, np.zeros((1, 2)), method=lay.method, meta=lay.meta)
        parts.append(lay)
    packed = pack_components(parts, gap=gap)
    out = _reorder(packed, list(g._node))
    meta = dict(parts[0].meta)
    meta["components"] = len(comps)
    return Layout(out.nodes, out.xy, routes=out.routes, method=parts[0].method, metric=False, meta=meta)


def _peripheral(adj: Sequence[Sequence[int]], start: int) -> tuple[int, list[int]]:
    """A far-away node from *start* (double sweep) and the BFS distances from it."""
    dist = _bfs(adj, start)
    far = max(dist, key=lambda k: (dist[k], -k))
    return far, _bfs(adj, far)


def _bfs(adj: Sequence[Sequence[int]], s: int) -> dict[int, int]:
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        du = dist[u] + 1
        for v in adj[u]:
            if v not in dist:
                dist[v] = du
                q.append(v)
    return dist


# ---------------------------------------------------------------------- #
# circular / arc ordering
# ---------------------------------------------------------------------- #
def _chord_crossings(order_pos: np.ndarray, ei: np.ndarray, ej: np.ndarray, limit: int = 6000) -> int | None:
    """Crossings of chords drawn on a circle (or arcs on one side of a line)."""
    m = ei.size
    if m < 2:
        return 0
    if m > limit:
        return None
    a = order_pos[ei]
    b = order_pos[ej]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    total = 0
    step = max(1, 4_000_000 // m)
    for s in range(0, m, step):
        L, H = lo[s : s + step, None], hi[s : s + step, None]
        total += int(np.count_nonzero((L < lo[None, :]) & (lo[None, :] < H) & (H < hi[None, :])))
    return total


def _auto_cycle_order(g: Any, nodes: list[Node]) -> list[int]:
    """Node order (indices) for a circle that keeps chords short and uncrossed.

    Each component gets a contiguous arc. Within a component, two candidate
    orders are built: a DFS preorder from a peripheral node (optimal for
    trees and cycles) and a greedy "most placed neighbours" order (Baur &
    Brandes). The one with fewer chord crossings wins.
    """
    index = {n: i for i, n in enumerate(nodes)}
    adj = _undirected_adjacency(g, nodes, index)
    ei, ej, _ = _edge_arrays(g, index)
    comps = components(g)
    comp_id = np.zeros(len(nodes), dtype=np.int64)
    for k, comp in enumerate(comps):
        comp_id[[index[n] for n in comp]] = k
    out: list[int] = []
    for k, comp in enumerate(comps):
        ids = [index[n] for n in comp]
        if len(ids) <= 3:
            out.extend(ids)
            continue
        start, _ = _peripheral(adj, ids[0])
        cands = [_dfs_order(adj, start), _greedy_order(adj, ids, start)]
        if len(cands[0]) != len(ids):  # pragma: no cover - defensive
            out.extend(ids)
            continue
        mask = comp_id[ei] == k
        ci, cj = ei[mask], ej[mask]
        best, best_c = cands[0], None
        for cand in cands:
            pos = np.zeros(len(nodes), dtype=np.int64)
            pos[cand] = np.arange(len(cand))
            c = _chord_crossings(pos, ci, cj)
            if c is None:
                break
            if best_c is None or c < best_c:
                best, best_c = cand, c
        out.extend(best)
    return out


def _dfs_order(adj: Sequence[Sequence[int]], start: int) -> list[int]:
    seen = {start}
    order = [start]
    stack = [(start, iter(adj[start]))]
    while stack:
        _, it = stack[-1]
        for v in it:
            if v not in seen:
                seen.add(v)
                order.append(v)
                stack.append((v, iter(adj[v])))
                break
        else:
            stack.pop()
    return order


def _greedy_order(adj: Sequence[Sequence[int]], ids: Sequence[int], start: int) -> list[int]:
    import heapq

    placed_nbrs = {i: 0 for i in ids}
    unplaced_nbrs = {i: len(adj[i]) for i in ids}
    done: set[int] = set()
    order: list[int] = []
    heap = [(0, 0, start)]
    while heap:
        _, _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        order.append(u)
        for v in adj[u]:
            if v in done:
                continue
            placed_nbrs[v] += 1
            unplaced_nbrs[v] -= 1
            heapq.heappush(heap, (-placed_nbrs[v], unplaced_nbrs[v], v))
    for i in ids:  # unreachable only if ids is not a component
        if i not in done:
            order.append(i)
    return order


def _resolve_order(g: Any, nodes: list[Node], order: Any) -> list[int]:
    if order is None:
        return list(range(len(nodes)))
    if isinstance(order, str):
        if order != "auto":
            raise ValueError(f"order must be None, 'auto' or a node sequence, got {order!r}")
        return _auto_cycle_order(g, nodes)
    index = {n: i for i, n in enumerate(nodes)}
    seq = []
    seen = set()
    for n in order:
        if n not in index:
            raise NodeNotFound(n)
        if n not in seen:
            seen.add(n)
            seq.append(index[n])
    seq.extend(i for i, n in enumerate(nodes) if n not in seen)
    return seq


# ---------------------------------------------------------------------- #
# public layouts
# ---------------------------------------------------------------------- #
def circular(g: Any, *, order: Sequence[Node] | str | None = None, start_angle: float = -90.0, seed: int = 0) -> Layout:
    """Nodes evenly spaced on a circle, clockwise on screen from *start_angle*.

    Parameters
    ----------
    order:
        ``None`` keeps graph order; a node sequence fixes the order (unlisted
        nodes follow in graph order); ``"auto"`` picks an order that reduces
        chord crossings (DFS / greedy-adjacency heuristics, best one kept).
    start_angle:
        Angle of the first node in degrees; ``-90`` is 12 o'clock (y down).

    Notes
    -----
    The radius makes neighbouring nodes about one unit apart.
    """
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("circular")
    seq = _resolve_order(g, nodes, order)
    xy = np.zeros((n, 2))
    if n > 1:
        radius = max(0.5, n / (2 * math.pi))
        theta = math.radians(start_angle) + 2 * math.pi * np.arange(n) / n
        xy[seq, 0] = radius * np.cos(theta)
        xy[seq, 1] = radius * np.sin(theta)
    return Layout(nodes, xy, method="circular", meta={"order": [nodes[i] for i in seq]})


def arc(g: Any, *, order: Sequence[Node] | str | None = None, seed: int = 0) -> Layout:
    """All nodes on one horizontal line, one unit apart: the basis of an arc diagram.

    *order* works as in :func:`circular` (``"auto"`` minimises arc crossings,
    which for one-sided arcs is the same problem as chords on a circle).
    ``meta["arc"]`` is True so renderers draw edges as arcs.
    """
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("arc", arc=True)
    seq = _resolve_order(g, nodes, order)
    xy = np.zeros((n, 2))
    xy[seq, 0] = np.arange(n, dtype=float)
    return Layout(nodes, xy, method="arc", meta={"arc": True, "order": [nodes[i] for i in seq]})


def shell(g: Any, *, shells: Sequence[Sequence[Node]] | None = None, seed: int = 0) -> Layout:
    """Concentric circles.

    Parameters
    ----------
    shells:
        List of node lists, innermost first. ``None`` builds shells from BFS
        distance to each component's centre (minimum eccentricity, ties to the
        higher degree). Nodes left out of the given shells form an extra
        outer shell.

    Notes
    -----
    Within every shell after the first, nodes are ordered by the mean angle of
    their neighbours in the previous shell, which avoids most crossings.
    """
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("shell")
    index = {v: i for i, v in enumerate(nodes)}
    adj = _undirected_adjacency(g, nodes, index)
    if shells is None:
        from .tree import _center  # local import: tree builds on this module

        groups: list[list[int]] = []
        for comp in components(g):
            ids = [index[v] for v in comp]
            center = _center(adj, ids, sum(len(adj[i]) for i in ids) // 2)
            dist = _bfs(adj, center)
            for i in ids:
                d = dist.get(i, 0)
                while len(groups) <= d:
                    groups.append([])
                groups[d].append(i)
    else:
        groups = []
        seen: set[int] = set()
        for sh in shells:
            grp = []
            for v in sh:
                if v not in index:
                    raise NodeNotFound(v)
                if index[v] not in seen:
                    seen.add(index[v])
                    grp.append(index[v])
            if grp:
                groups.append(grp)
        rest = [i for i in range(n) if i not in seen]
        if rest:
            groups.append(rest)
    xy = np.zeros((n, 2))
    angle = np.zeros(n)
    placed = np.zeros(n, dtype=bool)
    base = -math.pi / 2
    radius = 0.0
    for k, grp in enumerate(groups):
        if k == 0 and len(grp) == 1:
            xy[grp[0]] = 0.0
            angle[grp[0]] = base
            placed[grp[0]] = True
            continue
        # ring radius: at least one unit further out and room for one unit per node
        radius = max(radius + 1.0, len(grp) / (2 * math.pi))
        if k > 0:
            keyed = []
            for pos_in, i in enumerate(grp):
                prev = [j for j in adj[i] if placed[j]]
                if prev:
                    s = sum(math.sin(angle[j]) for j in prev)
                    c = sum(math.cos(angle[j]) for j in prev)
                    a = math.atan2(s, c) if (abs(s) + abs(c)) > 1e-12 else angle[prev[0]]
                    a = (a - base) % (2 * math.pi)
                else:
                    a = 2 * math.pi * pos_in / max(len(grp), 1)
                keyed.append((a, pos_in, i))
            keyed.sort()
            grp = [i for _, _, i in keyed]
            # rotate the ring so the slots best match the preferred angles
            # (circular mean of preferred − slot): children centre on parents
            m = len(grp)
            s = sum(math.sin(a - 2 * math.pi * t / m) for t, (a, _, _) in enumerate(keyed))
            c = sum(math.cos(a - 2 * math.pi * t / m) for t, (a, _, _) in enumerate(keyed))
            offset = math.atan2(s, c) if (abs(s) + abs(c)) > 1e-12 else 0.0
        else:
            offset = 0.0
        m = len(grp)
        for t, i in enumerate(grp):
            a = base + offset + 2 * math.pi * t / m
            angle[i] = a
            xy[i] = (radius * math.cos(a), radius * math.sin(a))
            placed[i] = True
    return Layout(nodes, xy, method="shell", meta={"shells": [[nodes[i] for i in grp] for grp in groups]})


def grid(g: Any, *, columns: int | None = None, seed: int = 0) -> Layout:
    """Nodes on a square lattice in graph order, row by row (unit spacing)."""
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("grid")
    cols = int(columns) if columns else max(1, math.ceil(math.sqrt(n)))
    if cols < 1:
        raise ValueError("columns must be positive")
    k = np.arange(n)
    xy = np.column_stack([k % cols, k // cols]).astype(float)
    return Layout(nodes, xy, method="grid", meta={"columns": cols})


def random(g: Any, *, seed: int = 0) -> Layout:
    """Uniform random positions in the unit square (reproducible for a given *seed*)."""
    nodes = list(g._node)
    if not nodes:
        return _empty("random")
    xy = make_rng(seed).random((len(nodes), 2))
    return Layout(nodes, xy, method="random")


def spiral(g: Any, *, turns: float | None = None, seed: int = 0) -> Layout:
    """Nodes along an Archimedean spiral, in graph order from the centre outwards.

    Consecutive nodes are equally spaced along the curve. The default number of
    *turns* makes the gap between arms equal to the gap between nodes.
    """
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("spiral")
    xy = np.zeros((n, 2))
    if n > 1:
        t = float(turns) if turns else max(1.0, math.sqrt((n - 1) / math.pi))
        if t <= 0:
            raise ValueError("turns must be positive")
        theta_max = 2 * math.pi * t
        theta = theta_max * np.sqrt(np.arange(n) / (n - 1))
        # r = c·θ; arc length ≈ c·θ²/2, so spacing between nodes is ≈ c·θmax²/(2(n-1)) = 1
        c = 2.0 * (n - 1) / theta_max**2
        xy[:, 0] = c * theta * np.cos(theta - math.pi / 2)
        xy[:, 1] = c * theta * np.sin(theta - math.pi / 2)
    return Layout(nodes, xy, method="spiral", meta={"turns": turns})


def _two_coloring(adj: Sequence[Sequence[int]], n: int) -> list[int]:
    color = [-1] * n
    for s in range(n):
        if color[s] >= 0:
            continue
        color[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if color[v] < 0:
                    color[v] = 1 - color[u]
                    q.append(v)
    return color


def _bipartite_crossings(pa: np.ndarray, pb: np.ndarray) -> int:
    m = pa.size
    if m < 2:
        return 0
    total = 0
    step = max(1, 4_000_000 // m)
    for s in range(0, m, step):
        a, b = pa[s : s + step, None], pb[s : s + step, None]
        total += int(np.count_nonzero((a < pa[None, :]) & (b > pb[None, :])))
    return total


def bipartite(g: Any, *, top: Iterable[Node] | None = None, align: str = "vertical", seed: int = 0) -> Layout:
    """Two parallel lines of nodes, edges running between them.

    Parameters
    ----------
    top:
        Nodes of the first side. ``None`` 2-colours each component by BFS
        (for a non-bipartite graph this is a best-effort split).
    align:
        ``"vertical"`` gives two columns side by side (first side on the left);
        ``"horizontal"`` gives two rows (first side on top).

    Notes
    -----
    Both sides are ordered by alternating barycenter sweeps (the best ordering
    seen is kept), which removes most crossings.
    """
    if align not in ("vertical", "horizontal"):
        raise ValueError("align must be 'vertical' or 'horizontal'")
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("bipartite")
    index = {v: i for i, v in enumerate(nodes)}
    adj = _undirected_adjacency(g, nodes, index)
    if top is None:
        side = _two_coloring(adj, n)
    else:
        side = [1] * n
        for v in top:
            if v not in index:
                raise NodeNotFound(v)
            side[index[v]] = 0
    # initial order: DFS from a peripheral node of each component (keeps
    # components together and walks paths end to end), each side separately
    bfs_rank = {}
    for comp in components(g):
        start, _ = _peripheral(adj, index[comp[0]])
        for i in _dfs_order(adj, start):
            bfs_rank.setdefault(i, len(bfs_rank))
    sides: list[list[int]] = [[], []]
    for i in sorted(range(n), key=lambda i: bfs_rank.get(i, i)):
        sides[side[i]].append(i)
    cross = [(i, j) for i in range(n) for j in adj[i] if side[i] == 0 and side[j] == 1]
    ea = np.array([c[0] for c in cross], dtype=np.int64)
    eb = np.array([c[1] for c in cross], dtype=np.int64)

    def positions(order: list[list[int]]) -> np.ndarray:
        p = np.zeros(n)
        for s in (0, 1):
            L = len(order[s])
            if L:
                p[order[s]] = np.arange(L) / max(L - 1, 1)
        return p

    counting = ea.size <= 6000

    def count(order: list[list[int]]) -> int:
        if not counting:  # too many edges to count cheaply: trust the sweeps
            return -1
        p = positions(order)
        return _bipartite_crossings(p[ea], p[eb])

    best = [list(sides[0]), list(sides[1])]
    best_c = count(best)
    cur = [list(sides[0]), list(sides[1])]
    for it in range(12 if counting else 4):
        if best_c == 0:
            break
        s = 1 if it % 2 == 0 else 0
        p = positions(cur)
        keyed = []
        for k, i in enumerate(cur[s]):
            other = [p[j] for j in adj[i] if side[j] != s]
            key = sum(other) / len(other) if other else p[i]
            keyed.append((key, k, i))
        keyed.sort()
        cur[s] = [i for _, _, i in keyed]
        c = count(cur)
        if c < best_c or not counting:
            best_c, best = c, [list(cur[0]), list(cur[1])]
    L = max(len(best[0]), len(best[1]), 1)
    gap = max(1.0, 0.6 * (L - 1))
    xy = np.zeros((n, 2))
    for s in (0, 1):
        seq = best[s]
        along = np.arange(len(seq)) - (len(seq) - 1) / 2.0
        if align == "vertical":
            xy[seq, 0] = s * gap
            xy[seq, 1] = along
        else:
            xy[seq, 0] = along
            xy[seq, 1] = s * gap
    return Layout(
        nodes,
        xy,
        method="bipartite",
        meta={"top": [nodes[i] for i in best[0]], "bottom": [nodes[i] for i in best[1]], "align": align},
    )


__all__ = ["circular", "shell", "grid", "random", "spiral", "bipartite", "arc"]
