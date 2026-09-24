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

"""Node-overlap removal with minimal displacement (VPSC scan-line method).

Dwyer, Marriott & Stuckey (2006): first a horizontal pass separates the
overlapping pairs that are cheaper to separate sideways, then a vertical pass
separates everything that still overlaps. Each pass solves the 1-D quadratic
program ::

    minimise Σ (x_i − x_i⁰)²   subject to   x_r − x_l ≥ gap_lr

by incremental block merging ("satisfy"), which keeps the relative order of
constrained pairs. Constraints come from a scan line: the vertical pass links
every pair of boxes that share an x-range through a chain of neighbours, which
guarantees that no overlap survives. Deterministic, O(n log n + k) typical.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from typing import Any, Hashable, Mapping

import numpy as np

from .base import Layout
from .geometric import _sizes_array

Node = Hashable


def _solve(desired: np.ndarray, cons: list[tuple[int, int, float]]) -> np.ndarray:
    """``argmin Σ (x − desired)²`` s.t. ``x[r] ≥ x[l] + gap`` (VPSC *satisfy*)."""
    n = desired.size
    if not cons:
        return desired.copy()
    incoming: list[list[int]] = [[] for _ in range(n)]
    succ: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for k, (l, r, _) in enumerate(cons):
        incoming[r].append(k)
        succ[l].append(r)
        indeg[r] += 1
    # topological order of the constraint DAG, left-most desired position first
    import heapq

    d = desired.tolist()
    heap = [(d[v], v) for v in range(n) if indeg[v] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, v = heapq.heappop(heap)
        order.append(v)
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(heap, (d[w], w))
    if len(order) < n:  # cyclic constraints cannot come from a scan line; be safe anyway
        seen = set(order)
        order.extend(v for v in sorted(range(n), key=d.__getitem__) if v not in seen)
    block = list(range(n))
    offset = [0.0] * n
    members: dict[int, list[int]] = {v: [v] for v in range(n)}
    wsum: dict[int, float] = {v: d[v] for v in range(n)}  # Σ (desired − offset)
    posn: dict[int, float] = {v: d[v] for v in range(n)}
    bin_: dict[int, list[int]] = {v: list(incoming[v]) for v in range(n)}

    def pos(v: int) -> float:
        return posn[block[v]] + offset[v]

    for v in order:
        b = block[v]
        while True:
            best, best_viol = -1, 1e-9
            keep = []
            for k in bin_[b]:
                l, r, gap = cons[k]
                if block[l] == b:
                    continue  # became internal
                keep.append(k)
                viol = pos(l) + gap - pos(r)
                if viol > best_viol:
                    best, best_viol = k, viol
            bin_[b] = keep
            if best < 0:
                break
            l, r, gap = cons[best]
            bl = block[l]
            # merge the smaller block into the larger one, making `best` tight
            if len(members[bl]) > len(members[b]):
                keep_b, drop_b = bl, b
                shift = offset[l] + gap - offset[r]  # move the right block's offsets
            else:
                keep_b, drop_b = b, bl
                shift = offset[r] - gap - offset[l]  # move the left block's offsets
            dropped = members.pop(drop_b)
            for u in dropped:
                offset[u] += shift
                block[u] = keep_b
            members[keep_b].extend(dropped)
            wsum[keep_b] += wsum.pop(drop_b) - shift * len(dropped)
            posn[keep_b] = wsum[keep_b] / len(members[keep_b])
            posn.pop(drop_b)
            bin_[keep_b] = bin_[keep_b] + bin_.pop(drop_b)
            b = keep_b
    return np.array([pos(v) for v in range(n)])


def _x_constraints(x: np.ndarray, y: np.ndarray, hw: np.ndarray, hh: np.ndarray) -> list[tuple[int, int, float]]:
    """Horizontal constraints for pairs that are cheaper to separate sideways."""
    n = x.size
    events = [(y[i] - hh[i], 1, i) for i in range(n)] + [(y[i] + hh[i], 0, i) for i in range(n)]
    events.sort()
    line: list[tuple[float, int]] = []
    left_n: list[set[int]] = [set() for _ in range(n)]
    right_n: list[set[int]] = [set() for _ in range(n)]
    cons: set[tuple[int, int]] = set()
    for _, is_open, v in events:
        key = (x[v], v)
        if is_open:
            insort(line, key)
            k = bisect_left(line, key)
            for step, mine, theirs in ((-1, left_n, right_n), (1, right_n, left_n)):
                j = k + step
                # the walk is capped: in dense clumps the vertical pass (which is
                # exhaustive) takes care of pairs not linked here
                while 0 <= j < len(line) and abs(j - k) <= 64:
                    u = line[j][1]
                    ox = hw[u] + hw[v] - abs(x[u] - x[v])
                    oy = hh[u] + hh[v] - abs(y[u] - y[v])
                    if ox <= 0:
                        mine[v].add(u)
                        theirs[u].add(v)
                        break
                    if ox <= oy:
                        mine[v].add(u)
                        theirs[u].add(v)
                    j += step
        else:
            for u in left_n[v]:
                cons.add((u, v))
                right_n[u].discard(v)
            for u in right_n[v]:
                cons.add((v, u))
                left_n[u].discard(v)
            line.pop(bisect_left(line, key))
    return [(l, r, float(hw[l] + hw[r])) for l, r in sorted(cons)]


def _y_constraints(x: np.ndarray, y: np.ndarray, hw: np.ndarray, hh: np.ndarray) -> list[tuple[int, int, float]]:
    """Vertical constraints chaining every pair of boxes that share an x-range."""
    n = x.size
    events = [(x[i] - hw[i], 1, i) for i in range(n)] + [(x[i] + hw[i], 0, i) for i in range(n)]
    events.sort()
    line: list[tuple[float, int]] = []
    cons: set[tuple[int, int]] = set()
    for _, is_open, v in events:
        key = (y[v], v)
        if is_open:
            insort(line, key)
            k = bisect_left(line, key)
            if k > 0:
                cons.add((line[k - 1][1], v))
            if k + 1 < len(line):
                cons.add((v, line[k + 1][1]))
        else:
            k = bisect_left(line, key)
            line.pop(k)
            if 0 < k < len(line):
                cons.add((line[k - 1][1], line[k][1]))
    return [(a, b, float(hh[a] + hh[b])) for a, b in sorted(cons)]


def _overlapping_pairs(xy: np.ndarray, hw: np.ndarray, hh: np.ndarray, tol: float = 1e-7) -> int:
    n = xy.shape[0]
    count = 0
    step = max(1, 2_000_000 // max(n, 1))
    for s in range(0, n, step):
        dx = np.abs(xy[s : s + step, None, 0] - xy[None, :, 0]) < (hw[s : s + step, None] + hw[None, :] - tol)
        dy = np.abs(xy[s : s + step, None, 1] - xy[None, :, 1]) < (hh[s : s + step, None] + hh[None, :] - tol)
        both = dx & dy
        rows = np.arange(s, min(n, s + step))
        both[rows - s, rows] = False
        count += int(np.count_nonzero(both))
    return count // 2


def remove_overlaps(layout: Layout, sizes: Mapping[Node, Any] | Any = None, *, padding: float = 4.0, max_iter: int = 200) -> Layout:
    """Move nodes as little as possible so that no two node boxes overlap.

    Parameters
    ----------
    layout:
        Any layout; coordinates are taken as they are (same units as *sizes*).
    sizes:
        ``{node: (w, h)}``, a single ``(w, h)`` for every node, or None (36×36).
    padding:
        Extra space between boxes (each box grows by *padding*).
    max_iter:
        Upper bound on repeated horizontal/vertical rounds (one round always
        suffices in exact arithmetic; more guard against round-off).

    Returns
    -------
    Layout
        A new layout (same method, metric flag and meta) whose boxes are
        pairwise disjoint. Routes of edges touching a moved node are dropped;
        ``meta["overlap_displacement"]`` is the total distance moved.
    """
    n = len(layout)
    if n < 2:
        return Layout(layout.nodes, layout.xy.copy(), routes=layout.routes, method=layout.method, metric=layout.metric, meta=dict(layout.meta))
    size = _sizes_array(layout.nodes, sizes)
    hw = (size[:, 0] + padding) / 2.0
    hh = (size[:, 1] + padding) / 2.0
    xy0 = layout.xy.astype(float)
    xy = xy0.copy()
    for _ in range(max(1, int(max_iter))):
        if _overlapping_pairs(xy, hw, hh) == 0:
            break
        # jitter-free tie breaking: identical centres are ordered by index in the scan lines
        cx = _x_constraints(xy[:, 0], xy[:, 1], hw, hh)
        xy[:, 0] = _solve(xy[:, 0], cx)
        cy = _y_constraints(xy[:, 0], xy[:, 1], hw, hh)
        xy[:, 1] = _solve(xy[:, 1], cy)
    moved = np.hypot(*(xy - xy0).T) > 1e-9
    moved_nodes = {layout.nodes[i] for i in np.flatnonzero(moved)}
    routes = {e: p for e, p in layout.routes.items() if e[0] not in moved_nodes and e[1] not in moved_nodes}
    meta = dict(layout.meta)
    meta["overlap_displacement"] = float(np.hypot(*(xy - xy0).T).sum())
    return Layout(list(layout.nodes), xy, routes=routes, method=layout.method, metric=layout.metric, meta=meta)


__all__ = ["remove_overlaps"]
