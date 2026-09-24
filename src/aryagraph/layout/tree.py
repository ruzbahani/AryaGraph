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

"""Tidy tree layouts: layered (Walker / Buchheim–Jünger–Leipert) and radial.

:func:`tree` implements Walker's algorithm in the linear-time formulation of
Buchheim, Jünger & Leipert (2002) with variable node sizes: siblings and
cousins are packed as closely as their widths allow, every parent is centred
over its first and last child, and isomorphic subtrees are drawn identically.
Both walks are iterative, so arbitrarily deep trees (long paths) are fine.

Graphs that are not trees are drawn through a BFS spanning tree; the other
edges get no route and are drawn straight by the renderer.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Hashable, Iterable, Sequence

import numpy as np

from ..algorithms._core import components
from ..core.exceptions import NodeNotFound
from ..core.results import NodeMap
from .base import Layout, pack_components
from .geometric import _bfs, _empty, _sizes_array, _undirected_adjacency

Node = Hashable

_ORIENTATIONS = ("TB", "BT", "LR", "RL")


def _check_orientation(orientation: str) -> str:
    o = str(orientation).upper()
    if o not in _ORIENTATIONS:
        raise ValueError(f"orientation must be one of {_ORIENTATIONS}, got {orientation!r}")
    return o


# ---------------------------------------------------------------------- #
# spanning forest
# ---------------------------------------------------------------------- #
def _center(adj: Sequence[Sequence[int]], ids: list[int], n_edges: int) -> int:
    """Node of minimum eccentricity (exact for trees and small graphs)."""
    if len(ids) <= 2:
        return ids[0]
    if n_edges == len(ids) - 1 or len(ids) > 1500:
        # tree (or large graph): middle of a double-sweep longest path
        d0 = _bfs(adj, ids[0])
        a = max(d0, key=lambda k: (d0[k], -k))
        parent = {a: a}
        q = deque([a])
        last = a
        while q:
            u = q.popleft()
            last = u
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        path = [last]
        while path[-1] != a:
            path.append(parent[path[-1]])
        cands = {path[(len(path) - 1) // 2], path[len(path) // 2]}
        return min(cands, key=lambda k: (-len(adj[k]), k))
    from .stress import _hop_distances

    local = {v: i for i, v in enumerate(ids)}
    sub = [[local[w] for w in adj[v]] for v in ids]
    ecc = _hop_distances(sub).max(axis=1)
    best = min(range(len(ids)), key=lambda i: (int(ecc[i]), -len(sub[i]), i))
    return ids[best]


def _spanning_forest(
    g: Any, nodes: list[Node], index: dict[Node, int], root: Any
) -> tuple[list[int], list[list[int]], list[int], list[int]]:
    """BFS spanning forest: ``(roots, children, parent, depth)`` over node indices.

    For directed graphs the BFS first follows arcs forward from the root, then
    attaches whatever is left of the component through arcs in any direction.
    """
    n = len(nodes)
    adj = _undirected_adjacency(g, nodes, index)
    succ = [[index[v] for v in g._succ[u] if v != u] for u in nodes] if g.directed else adj
    if root is None:
        wanted: list[int] = []
    elif not isinstance(root, (list, tuple, set, frozenset)) or (isinstance(root, tuple) and root in index):
        if root not in index:
            raise NodeNotFound(root)
        wanted = [index[root]]
    else:
        wanted = []
        for r in root:
            if r not in index:
                raise NodeNotFound(r)
            wanted.append(index[r])
    parent = [-1] * n
    depth = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for comp in components(g):
        ids = [index[v] for v in comp]
        idset = set(ids)
        chosen = next((r for r in wanted if r in idset), None)
        if chosen is None:
            if g.directed:
                sources = [i for i in ids if not any(p != nodes[i] for p in g._pred[nodes[i]])]
                chosen = sources[0] if len(sources) == 1 else None
            if chosen is None:
                n_edges = sum(len(adj[i]) for i in ids) // 2
                chosen = _center(adj, ids, n_edges)
        roots.append(chosen)
        depth[chosen] = 0
        order = [chosen]
        q = deque([chosen])
        while q:  # phase 1: along the arcs
            u = q.popleft()
            for v in succ[u]:
                if depth[v] < 0:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    children[u].append(v)
                    order.append(v)
                    q.append(v)
        if len(order) < len(ids):  # phase 2: attach the rest through any edge
            q = deque(order)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if depth[v] < 0:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        children[u].append(v)
                        q.append(v)
    return roots, children, parent, depth


# ---------------------------------------------------------------------- #
# Buchheim–Jünger–Leipert
# ---------------------------------------------------------------------- #
def _walker(roots: list[int], children: list[list[int]], parent: list[int], widths: np.ndarray, sep: float) -> np.ndarray:
    """Relative x of every node (each tree's root at 0)."""
    n = len(children)
    wd = widths.tolist()
    prelim = [0.0] * n
    mod = [0.0] * n
    shift = [0.0] * n
    change = [0.0] * n
    thread = [-1] * n
    ancestor = list(range(n))
    number = [0] * n
    for v in range(n):
        for k, c in enumerate(children[v]):
            number[c] = k

    def next_left(v: int) -> int:
        ch = children[v]
        return ch[0] if ch else thread[v]

    def next_right(v: int) -> int:
        ch = children[v]
        return ch[-1] if ch else thread[v]

    def apportion(v: int, default: int) -> int:
        k = number[v]
        if k == 0:
            return default
        sibs = children[parent[v]]
        vip = vop = v
        vim = sibs[k - 1]
        vom = sibs[0]
        sip, sop, sim, som = mod[vip], mod[vop], mod[vim], mod[vom]
        nr, nl = next_right(vim), next_left(vip)
        while nr != -1 and nl != -1:
            vim, vip = nr, nl
            vom, vop = next_left(vom), next_right(vop)
            ancestor[vop] = v
            sh = (prelim[vim] + sim) - (prelim[vip] + sip) + (wd[vim] + wd[vip]) / 2.0 + sep
            if sh > 0:
                a = ancestor[vim]
                wm = a if parent[a] == parent[v] else default
                subtrees = number[v] - number[wm]
                change[v] -= sh / subtrees
                shift[v] += sh
                change[wm] += sh / subtrees
                prelim[v] += sh
                mod[v] += sh
                sip += sh
                sop += sh
            sim += mod[vim]
            sip += mod[vip]
            som += mod[vom]
            sop += mod[vop]
            nr, nl = next_right(vim), next_left(vip)
        if nr != -1 and next_right(vop) == -1:
            thread[vop] = nr
            mod[vop] += sim - sop
        if nl != -1 and next_left(vom) == -1:
            thread[vom] = nl
            mod[vom] += sip - som
            default = v
        return default

    x = np.zeros(n)
    for r in roots:
        default: dict[int, int] = {}
        stack = [(r, 0)]
        while stack:
            v, i = stack[-1]
            ch = children[v]
            if i < len(ch):
                if i == 0:
                    default[v] = ch[0]
                else:
                    default[v] = apportion(ch[i - 1], default[v])
                stack[-1] = (v, i + 1)
                stack.append((ch[i], 0))
                continue
            stack.pop()
            k = number[v]
            left = children[parent[v]][k - 1] if (k > 0 and parent[v] >= 0) else -1
            if ch:
                default[v] = apportion(ch[-1], default[v])
                # execute shifts
                s = c = 0.0
                for w in reversed(ch):
                    prelim[w] += s
                    mod[w] += s
                    c += change[w]
                    s += shift[w] + c
                mid = (prelim[ch[0]] + prelim[ch[-1]]) / 2.0
                if left >= 0:
                    prelim[v] = prelim[left] + (wd[left] + wd[v]) / 2.0 + sep
                    mod[v] = prelim[v] - mid
                else:
                    prelim[v] = mid
            else:
                prelim[v] = prelim[left] + (wd[left] + wd[v]) / 2.0 + sep if left >= 0 else 0.0
        # second walk
        stack2 = [(r, -prelim[r])]
        while stack2:
            v, m = stack2.pop()
            x[v] = prelim[v] + m
            for w in children[v]:
                stack2.append((w, m + mod[v]))
    return x


def _level_positions(depth: Sequence[int], extent: np.ndarray, rank_sep: float) -> list[float]:
    """Centre coordinate of every depth level: tallest box per level, *rank_sep* between boxes."""
    top = max(depth) if len(depth) else 0
    hmax = [0.0] * (top + 1)
    for v, d in enumerate(depth):
        hmax[d] = max(hmax[d], float(extent[v]))
    pos = [hmax[0] / 2.0]
    for d in range(1, top + 1):
        pos.append(pos[-1] + hmax[d - 1] / 2.0 + rank_sep + hmax[d] / 2.0)
    return pos


def _orient(s: np.ndarray, t: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "TB":
        return np.column_stack([s, t])
    if orientation == "BT":
        return np.column_stack([s, -t])
    if orientation == "LR":
        return np.column_stack([t, s])
    return np.column_stack([-t, s])


def _to_origin(xy: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Translate so the union of node boxes starts at (0, 0)."""
    if xy.shape[0] == 0:
        return xy
    lo = (xy - size / 2.0).min(axis=0)
    return xy - lo


def tree(
    g: Any,
    *,
    root: Node | Iterable[Node] | None = None,
    orientation: str = "TB",
    sizes: Any = None,
    node_sep: float = 24.0,
    rank_sep: float = 64.0,
    seed: int = 0,
) -> Layout:
    """Tidy layered tree drawing (Walker's algorithm, linear time).

    Parameters
    ----------
    root:
        Root node (or one root per component). Default: the unique source of
        a directed graph if there is one, else the node of minimum eccentricity.
    orientation:
        ``"TB"`` (root on top), ``"BT"``, ``"LR"`` (root on the left) or ``"RL"``.
    sizes:
        ``{node: (w, h)}`` in pixels (default 36×36). Neighbours on a level are
        separated by their half-widths plus *node_sep*; levels by the tallest
        box of each level plus *rank_sep*.

    Returns
    -------
    Layout
        Metric layout (pixels, boxes start at (0, 0)). ``meta`` holds
        ``orientation``, ``roots``, ``depth`` and ``parent`` (spanning tree),
        ``rank_positions``, ``node_sep`` and ``rank_sep``. Forests are drawn
        tree by tree, side by side, sharing level coordinates.

    Complexity: O(n + m).
    """
    orientation = _check_orientation(orientation)
    nodes = list(g._node)
    n = len(nodes)
    meta_base = {"orientation": orientation, "node_sep": float(node_sep), "rank_sep": float(rank_sep)}
    if n == 0:
        return _empty("tree", metric=True, roots=[], depth=NodeMap({}, name="depth"), parent={}, rank_positions=[], **meta_base)
    index = {v: i for i, v in enumerate(nodes)}
    size = _sizes_array(nodes, sizes)
    across = size[:, 1] if orientation in ("LR", "RL") else size[:, 0]  # extent along a level
    along = size[:, 0] if orientation in ("LR", "RL") else size[:, 1]  # extent across levels
    roots, children, parent, depth = _spanning_forest(g, nodes, index, root)
    s = _walker(roots, children, parent, across, float(node_sep))
    # place the trees side by side (each tree's own extent), sharing level coordinates
    tree_of = [0] * n
    for k, r in enumerate(roots):
        stack = [r]
        while stack:
            v = stack.pop()
            tree_of[v] = k
            stack.extend(children[v])
    lo = np.full(len(roots), np.inf)
    hi = np.full(len(roots), -np.inf)
    for v in range(n):
        k = tree_of[v]
        lo[k] = min(lo[k], s[v] - across[v] / 2.0)
        hi[k] = max(hi[k], s[v] + across[v] / 2.0)
    offset = np.zeros(len(roots))
    cursor = 0.0
    gap = 2.0 * float(node_sep)
    for k in range(len(roots)):
        offset[k] = cursor - lo[k]
        cursor += (hi[k] - lo[k]) + gap
    s = s + offset[tree_of]
    levels = _level_positions(depth, along, float(rank_sep))
    t = np.array([levels[d] for d in depth])
    xy = _orient(s, t, orientation)
    shift = (xy - size / 2.0).min(axis=0)
    xy = xy - shift
    if orientation == "TB":
        rank_pos = [p - shift[1] for p in levels]
    elif orientation == "BT":
        rank_pos = [-p - shift[1] for p in levels]
    elif orientation == "LR":
        rank_pos = [p - shift[0] for p in levels]
    else:
        rank_pos = [-p - shift[0] for p in levels]
    meta = dict(meta_base)
    meta.update(
        roots=[nodes[r] for r in roots],
        depth=NodeMap({nodes[v]: depth[v] for v in range(n)}, name="depth"),
        parent={nodes[v]: (nodes[parent[v]] if parent[v] >= 0 else None) for v in range(n)},
        rank_positions=[float(p) for p in rank_pos],
    )
    return Layout(nodes, xy, method="tree", metric=True, meta=meta)


def radial(
    g: Any,
    *,
    root: Node | Iterable[Node] | None = None,
    rank_sep: float = 80.0,
    sizes: Any = None,
    seed: int = 0,
    node_sep: float = 16.0,
    uniform: bool = True,
) -> Layout:
    """Radial tidy tree: the root in the centre, depth levels on concentric rings.

    Every node owns an angular wedge proportional to its number of leaves and
    sits in the middle of it, so subtrees never interleave. Ring radii grow
    with *rank_sep* (gap between the boxes of consecutive rings) and are
    enlarged where needed so neighbours on a ring keep *node_sep* between
    their boxes. With *uniform* (default) the rings are then evenly spaced
    (``r_d = d · max_k(r_k / k)``), which keeps every ring's minimum while
    avoiding long spokes to a crowded outer ring. Roots are chosen as in
    :func:`tree`; forests are packed.

    Returns a metric layout; ``meta`` has ``roots``, ``depth``, ``parent`` and
    ``radii`` (ring radius per depth, per tree).
    """
    nodes = list(g._node)
    n = len(nodes)
    if n == 0:
        return _empty("radial", metric=True, roots=[], depth=NodeMap({}, name="depth"), parent={}, radii=[])
    index = {v: i for i, v in enumerate(nodes)}
    size = _sizes_array(nodes, sizes)
    ext = size.max(axis=1)
    roots, children, parent, depth = _spanning_forest(g, nodes, index, root)
    xy = np.zeros((n, 2))
    parts = []
    all_radii = []
    for r in roots:
        members = []
        stack = [r]
        while stack:
            v = stack.pop()
            members.append(v)
            stack.extend(children[v])
        # leaf counts (post-order over the reversed pre-order)
        leaves = {}
        for v in reversed(members):
            leaves[v] = sum(leaves[c] for c in children[v]) if children[v] else 1
        angle = {r: 0.0}
        wedge_lo = {r: -math.pi / 2}
        stack = [r]
        while stack:
            v = stack.pop()
            lo = wedge_lo[v]
            total = leaves[v]
            span = 2 * math.pi * total / leaves[r]
            for c in children[v]:
                share = span * leaves[c] / total
                wedge_lo[c] = lo
                angle[c] = lo + share / 2.0
                lo += share
                stack.append(c)
        top = max(depth[v] for v in members)
        by_depth: list[list[int]] = [[] for _ in range(top + 1)]
        for v in members:
            by_depth[depth[v]].append(v)
        emax = [max(ext[v] for v in grp) for grp in by_depth]
        radii = [0.0]
        for d in range(1, top + 1):
            rad = radii[-1] + (emax[d - 1] + emax[d]) / 2.0 + rank_sep
            grp = sorted(by_depth[d], key=lambda v: angle[v])
            if len(grp) > 1:
                for a, b in zip(grp, grp[1:] + grp[:1]):  # neighbours on the ring, wrapping around
                    dth = (angle[b] - angle[a]) % (2 * math.pi)
                    need = (ext[a] + ext[b]) / 2.0 + node_sep
                    if dth < math.pi:
                        rad = max(rad, need / (2 * math.sin(max(dth, 1e-9) / 2)))
                    else:
                        rad = max(rad, need / 2.0)
            radii.append(rad)
        if uniform and top > 0:
            step = max(radii[d] / d for d in range(1, top + 1))
            radii = [d * step for d in range(top + 1)]
        for v in members:
            if v == r:
                xy[v] = 0.0
            else:
                rad = radii[depth[v]]
                xy[v] = (rad * math.cos(angle[v]), rad * math.sin(angle[v]))
        all_radii.append([float(x) for x in radii])
        parts.append(members)
    if len(roots) > 1:
        lays = []
        for members in parts:
            lays.append(Layout([nodes[v] for v in members], xy[members], method="radial", metric=True))
        packed = pack_components(lays, gap=float(ext.max()) + node_sep)
        pos = {v: packed.xy[i] for i, v in enumerate(packed.nodes)}
        xy = np.array([pos[v] for v in nodes])
    xy = _to_origin(xy, size)
    meta = {
        "roots": [nodes[r] for r in roots],
        "depth": NodeMap({nodes[v]: depth[v] for v in range(n)}, name="depth"),
        "parent": {nodes[v]: (nodes[parent[v]] if parent[v] >= 0 else None) for v in range(n)},
        "radii": all_radii,
        "rank_sep": float(rank_sep),
        "node_sep": float(node_sep),
    }
    return Layout(nodes, xy, method="radial", metric=True, meta=meta)


__all__ = ["tree", "radial"]
