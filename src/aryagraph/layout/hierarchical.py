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

"""Layered (Sugiyama) drawing of directed graphs: AryaGraph's DAG layout.

The classic four phases, each with the algorithm that ``dot`` popularised or
its best-known successor:

1. **Cycle removal**: only arcs inside strongly connected components can lie
   on cycles, so each non-trivial SCC is linearised with the Eades–Lin–Smyth
   greedy heuristic and its backward arcs are reversed (drawn pointing up).
   Undirected graphs are oriented by BFS depth from a peripheral node.
2. **Layering**: network simplex (Gansner, Koutsofios, North & Vo, 1993)
   finds the ranking of minimum total edge length, which is then balanced so
   nodes with equal in/out weight move to the least crowded feasible rank.
   Longest-path and Coffman–Graham layerings are available too, and
   ``ranks`` overrides it all.
3. **Crossing minimisation**: long edges get dummy nodes; DFS and BFS
   initial orders from both ends (plus seeded random ones for small graphs),
   then alternating down/up sweeps of the weighted median heuristic
   (barycenter breaks ties), layer sifting and dot's *transpose* step, keeping
   the best ordering seen. Crossings are counted with the Barth–Jünger–Mutzel
   accumulator tree. Among equally good orders, graph order wins.
4. **Coordinates**: the Brandes–Köpf method uses four vertical alignments
   with type-1 conflicts marked so inner segments of long edges stay vertical,
   block compaction with variable node widths, and the balanced median of the
   four.
   With ``compact=True`` (default) the result then warm-starts dot's
   auxiliary-graph network simplex, which minimises the weighted horizontal
   edge length exactly (long edges weigh 8×) and centres parents over their
   children. Layers are spaced by their tallest node plus ``rank_sep``.

Everything is laid out top-to-bottom internally; LR/RL swap node width and
height first and rotate at the end.
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left, bisect_right
from collections import deque
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from ..algorithms._core import components
from ..core.results import NodeMap
from ..core.utils import make_rng
from .base import Layout
from .geometric import _empty, _peripheral, _sizes_array, _undirected_adjacency
from .tree import _check_orientation, _orient

Node = Hashable

_LAYERINGS = ("network_simplex", "longest_path", "coffman_graham")


# ====================================================================== #
# 1. cycle removal / orientation
# ====================================================================== #
def _scc(n: int, succ: Sequence[Sequence[int]]) -> list[list[int]]:
    """Strongly connected components (iterative Tarjan)."""
    index = [-1] * n
    low = [0] * n
    on = [False] * n
    stack: list[int] = []
    out: list[list[int]] = []
    counter = 0
    for s in range(n):
        if index[s] >= 0:
            continue
        index[s] = low[s] = counter
        counter += 1
        stack.append(s)
        on[s] = True
        work = [(s, 0)]
        while work:
            v, i = work[-1]
            nb = succ[v]
            if i < len(nb):
                work[-1] = (v, i + 1)
                w = nb[i]
                if index[w] < 0:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on[w] = True
                    work.append((w, 0))
                elif on[w] and index[w] < low[v]:
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
                    on[w] = False
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
    return out


def _eades_lin_smyth(nodes: list[int], succ: Sequence[Sequence[int]], pred: Sequence[Sequence[int]]) -> dict[int, int]:
    """Position of every node in the Eades–Lin–Smyth sequence of the subgraph on *nodes*.

    Sinks are peeled to the back, sources to the front, and otherwise the node
    maximising ``outdeg − indeg`` goes to the front. Arcs pointing backwards in
    the sequence form a small feedback arc set. O(m log n) with a lazy heap.
    """
    inside = set(nodes)
    outd = {v: sum(1 for w in succ[v] if w in inside) for v in nodes}
    ind = {v: sum(1 for u in pred[v] if u in inside) for v in nodes}
    order_key = {v: k for k, v in enumerate(nodes)}
    removed: set[int] = set()
    s1: list[int] = []
    s2: list[int] = []
    sinks = deque(v for v in nodes if outd[v] == 0)
    sources = deque(v for v in nodes if ind[v] == 0 and outd[v] > 0)
    heap = [(-(outd[v] - ind[v]), order_key[v], v) for v in nodes]
    heapq.heapify(heap)

    def remove(v: int) -> None:
        removed.add(v)
        for w in succ[v]:
            if w in inside and w not in removed:
                ind[w] -= 1
                if ind[w] == 0:
                    sources.append(w)
                heapq.heappush(heap, (-(outd[w] - ind[w]), order_key[w], w))
        for u in pred[v]:
            if u in inside and u not in removed:
                outd[u] -= 1
                if outd[u] == 0:
                    sinks.append(u)
                heapq.heappush(heap, (-(outd[u] - ind[u]), order_key[u], u))

    left = len(nodes)
    while left:
        progress = True
        while progress:
            progress = False
            while sinks:
                v = sinks.popleft()
                if v in removed:
                    continue
                s2.append(v)
                remove(v)
                left -= 1
                progress = True
            while sources:
                v = sources.popleft()
                if v in removed or ind[v] != 0:
                    continue
                s1.append(v)
                remove(v)
                left -= 1
                progress = True
                break  # new sinks may have appeared: peel them first
        if not left:
            break
        while heap:
            d, _, v = heapq.heappop(heap)
            if v not in removed and -d == outd[v] - ind[v]:
                s1.append(v)
                remove(v)
                left -= 1
                break
    seq = s1 + s2[::-1]
    return {v: k for k, v in enumerate(seq)}


class _Edge:
    """An original edge as the layering sees it."""

    __slots__ = ("key", "tail", "head", "flip", "reversed")

    def __init__(self, key: tuple, tail: int, head: int, flip: bool, reversed_: bool) -> None:
        self.key = key  # (u, v) as the graph reports it
        self.tail = tail  # upper endpoint (index)
        self.head = head  # lower endpoint (index)
        self.flip = flip  # tail/head are (v, u) rather than (u, v)
        self.reversed = reversed_  # counts as a reversed arc of a directed graph


def _orient_edges(g: Any, nodes: list[Node], index: dict[Node, int]) -> list[_Edge]:
    """Acyclic orientation of all non-loop edges."""
    raw = [(index[u], index[v], (u, v)) for u, v, _ in g._iter_edges() if u != v]
    n = len(nodes)
    if not g.directed:
        adj = _undirected_adjacency(g, nodes, index)
        depth: dict[int, int] = {}
        for comp in components(g):
            _, dist = _peripheral(adj, index[comp[0]])
            depth.update(dist)
        out = []
        for a, b, key in raw:
            if (depth[a], a) <= (depth[b], b):
                out.append(_Edge(key, a, b, False, False))
            else:
                out.append(_Edge(key, b, a, True, False))
        return out
    succ: list[list[int]] = [[] for _ in range(n)]
    pred: list[list[int]] = [[] for _ in range(n)]
    for a, b, _ in raw:
        succ[a].append(b)
        pred[b].append(a)
    comp_of = [0] * n
    rank_in = [0] * n
    for k, comp in enumerate(_scc(n, succ)):
        for v in comp:
            comp_of[v] = k
        if len(comp) > 1:
            comp.sort()  # graph order for deterministic tie-breaking
            for v, p in _eades_lin_smyth(comp, succ, pred).items():
                rank_in[v] = p
    out = []
    for a, b, key in raw:
        if comp_of[a] == comp_of[b] and rank_in[a] > rank_in[b]:
            out.append(_Edge(key, b, a, True, True))
        else:
            out.append(_Edge(key, a, b, False, False))
    return out


# ====================================================================== #
# 2. layering
# ====================================================================== #
def _longest_path(n: int, tails: Sequence[int], heads: Sequence[int]) -> list[int]:
    """As-soon-as-possible ranks: sources at 0, every edge spans ≥ 1."""
    succ: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for t, h in zip(tails, heads):
        succ[t].append(h)
        indeg[h] += 1
    rank = [0] * n
    q = deque(v for v in range(n) if indeg[v] == 0)
    while q:
        v = q.popleft()
        r = rank[v] + 1
        for w in succ[v]:
            if rank[w] < r:
                rank[w] = r
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return rank


def _coffman_graham(n: int, tails: Sequence[int], heads: Sequence[int], width: int) -> list[int]:
    """Coffman–Graham layering with at most *width* nodes per layer."""
    succ: list[list[int]] = [[] for _ in range(n)]
    pred: list[list[int]] = [[] for _ in range(n)]
    for t, h in zip(tails, heads):
        if h not in succ[t]:
            succ[t].append(h)
            pred[h].append(t)
    # 1. labels: repeatedly the node whose (decreasing) predecessor-label list is smallest
    label = [0] * n
    missing = [len(pred[v]) for v in range(n)]
    heap = [((), v) for v in range(n) if missing[v] == 0]
    heapq.heapify(heap)
    nxt = 1
    while heap:
        _, v = heapq.heappop(heap)
        label[v] = nxt
        nxt += 1
        for w in succ[v]:
            missing[w] -= 1
            if missing[w] == 0:
                heapq.heappush(heap, (tuple(sorted((label[u] for u in pred[w]), reverse=True)), w))
    # 2. fill layers bottom-up, highest label first
    layer = [-1] * n
    remaining = [len(succ[v]) for v in range(n)]
    now = [(-label[v], v) for v in range(n) if remaining[v] == 0]
    heapq.heapify(now)
    later: list[tuple[int, int]] = []
    k = 0
    filled = 0
    placed = 0
    while placed < n:
        if not now or filled >= width:
            k += 1
            filled = 0
            for item in later:
                heapq.heappush(now, item)
            later = []
            if not now:  # pragma: no cover - cannot happen on a DAG
                break
            continue
        _, v = heapq.heappop(now)
        layer[v] = k
        filled += 1
        placed += 1
        for u in pred[v]:
            remaining[u] -= 1
            if remaining[u] == 0:
                later.append((-label[u], u))  # a successor sits in layer k: u goes above
    top = max(layer) if n else 0
    return [top - x for x in layer]


class _Disconnected(Exception):
    """Raised when network simplex is given a disconnected graph."""


class _NetworkSimplex:
    """Network simplex for ``min Σ w_e (r[h] − r[t])`` s.t. ``r[h] − r[t] ≥ minlen_e``.

    Gansner et al. (1993): a feasible *tight* spanning tree, cut values, and
    leave/enter exchanges until every tree edge has a non-negative cut value.
    Cut values and the low/lim postorder numbering are updated incrementally
    (only the fundamental cycle and the subtree under its lowest common
    ancestor change), and edge scans are vectorised. The graph must be
    connected (as an undirected graph) and *rank0* must be feasible.
    """

    SEARCH_SIZE = 30

    def __init__(self, n: int, tails: Sequence[int], heads: Sequence[int], minlen: Sequence[int], weight: Sequence[float], rank0: Sequence[int]) -> None:
        self.n = n
        self.m = m = len(tails)
        self.T = np.asarray(tails, dtype=np.int64)
        self.H = np.asarray(heads, dtype=np.int64)
        self.ML = np.asarray(minlen, dtype=np.int64)
        self.W = np.asarray(weight, dtype=float)
        self.Tl, self.Hl, self.MLl = self.T.tolist(), self.H.tolist(), self.ML.tolist()
        self.out_e: list[list[int]] = [[] for _ in range(n)]
        self.in_e: list[list[int]] = [[] for _ in range(n)]
        for e in range(m):
            self.out_e[self.Tl[e]].append(e)
            self.in_e[self.Hl[e]].append(e)
        self.rank = np.array(rank0, dtype=np.int64)
        self.tree_e = [False] * m
        self.tadj: list[list[int]] = [[] for _ in range(n)]
        self.par = [-1] * n
        self.low = [0] * n
        self.lim_l = [0] * n
        self.lim = np.zeros(n, dtype=np.int64)
        self._feasible_tree()
        post = self._dfs_range(0, -1, 1)
        self._init_cut_values(post)

    # -- feasible tight tree ---------------------------------------------
    def _feasible_tree(self) -> None:
        n, T, H = self.n, self.T, self.H
        Tl, Hl, MLl = self.Tl, self.Hl, self.MLl
        out_e, in_e, tadj, tree_e = self.out_e, self.in_e, self.tadj, self.tree_e
        in_tree = [False] * n

        def grow(v0: int, rk: list[int]) -> int:
            added = 0
            stack = [v0]
            while stack:
                v = stack.pop()
                rv = rk[v]
                for e in out_e[v]:
                    w = Hl[e]
                    if not in_tree[w] and rk[w] - rv == MLl[e]:
                        in_tree[w] = True
                        tree_e[e] = True
                        tadj[v].append(e)
                        tadj[w].append(e)
                        stack.append(w)
                        added += 1
                for e in in_e[v]:
                    u = Tl[e]
                    if not in_tree[u] and rv - rk[u] == MLl[e]:
                        in_tree[u] = True
                        tree_e[e] = True
                        tadj[v].append(e)
                        tadj[u].append(e)
                        stack.append(u)
                        added += 1
            return added

        rank = self.rank
        in_tree[0] = True
        size = 1 + grow(0, rank.tolist())
        while size < n:
            it = np.array(in_tree)
            tin = it[T]
            cand = np.flatnonzero(tin != it[H])
            if cand.size == 0:
                raise _Disconnected
            sl = rank[H[cand]] - rank[T[cand]] - self.ML[cand]
            k = int(np.argmin(sl))
            e = int(cand[k])
            delta = int(sl[k])
            if tin[e]:
                rank[it] += delta
                new = Hl[e]
            else:
                rank[it] -= delta
                new = Tl[e]
            in_tree[new] = True
            tree_e[e] = True
            tadj[Tl[e]].append(e)
            tadj[Hl[e]].append(e)
            size += 1 + grow(new, rank.tolist())

    # -- postorder numbering ------------------------------------------------
    def _dfs_range(self, root: int, pe: int, start: int) -> list[int]:
        par, low, lim_l, lim, tadj = self.par, self.low, self.lim_l, self.lim, self.tadj
        Tl, Hl = self.Tl, self.Hl
        post: list[int] = []
        counter = start
        par[root] = pe
        low[root] = counter
        stack = [(root, pe, 0)]
        while stack:
            v, p, i = stack[-1]
            te = tadj[v]
            if i < len(te):
                stack[-1] = (v, p, i + 1)
                e = te[i]
                if e == p:
                    continue
                w = Hl[e] if Tl[e] == v else Tl[e]
                par[w] = e
                low[w] = counter
                stack.append((w, e, 0))
            else:
                lim_l[v] = counter
                counter += 1
                post.append(v)
                stack.pop()
        lim[post] = [lim_l[v] for v in post]
        return post

    def _init_cut_values(self, post: list[int]) -> None:
        Tl, W, par, tree_e = self.Tl, self.W.tolist(), self.par, self.tree_e
        cut = [0.0] * self.m
        for v in post:
            pe = par[v]
            if pe < 0:
                continue
            child_is_tail = Tl[pe] == v
            cv = W[pe]
            for e in self.out_e[v]:
                if e == pe:
                    continue
                # an out-edge of the child points the same way as pe iff the child is its tail
                if child_is_tail:
                    cv += W[e] - (cut[e] if tree_e[e] else 0.0)
                else:
                    cv += -W[e] + (cut[e] if tree_e[e] else 0.0)
            for e in self.in_e[v]:
                if e == pe:
                    continue
                if not child_is_tail:
                    cv += W[e] - (cut[e] if tree_e[e] else 0.0)
                else:
                    cv += -W[e] + (cut[e] if tree_e[e] else 0.0)
            cut[pe] = cv
        self.cut = np.array(cut)
        self.tree_mask = np.array(tree_e, dtype=bool)

    # -- pivoting -------------------------------------------------------------
    def _enter(self, e: int) -> tuple[int, int, int, np.ndarray] | None:
        """Min-slack non-tree edge from the head component of *e* to its tail component."""
        te, he = self.Tl[e], self.Hl[e]
        c = te if self.par[te] == e else he
        lim = self.lim
        insub = (lim >= self.low[c]) & (lim <= self.lim_l[c])
        tin, hin = insub[self.T], insub[self.H]
        cand = np.flatnonzero((hin & ~tin) if c == te else (tin & ~hin))
        if cand.size == 0:
            return None
        sl = self.rank[self.H[cand]] - self.rank[self.T[cand]] - self.ML[cand]
        k = int(np.argmin(sl))
        return int(cand[k]), int(sl[k]), c, insub

    def _tree_update(self, v: int, w: int, cv: float, direction: bool) -> int:
        low, lim_l, par, Tl, Hl, cut = self.low, self.lim_l, self.par, self.Tl, self.Hl, self.cut
        lw = lim_l[w]
        while not (low[v] <= lw <= lim_l[v]):
            pe = par[v]
            if (v == Tl[pe]) == direction:
                cut[pe] += cv
            else:
                cut[pe] -= cv
            a, b = Tl[pe], Hl[pe]
            v = a if lim_l[a] > lim_l[b] else b
        return v

    def solve(self, max_iter: int) -> int:
        """Pivot until optimal (or *max_iter* exchanges); returns the number of pivots."""
        cut, tree_mask, tadj = self.cut, self.tree_mask, self.tadj
        s_i = 0
        it = 0
        for it in range(max_iter):
            neg = np.flatnonzero(tree_mask & (cut < -1e-9))
            if neg.size == 0:
                return it
            start = int(np.searchsorted(neg, s_i))
            window = np.concatenate([neg[start:], neg[:start]])[: self.SEARCH_SIZE]
            e = int(window[int(np.argmin(cut[window]))])
            s_i = e + 1
            found = self._enter(e)
            if found is None:  # pragma: no cover - impossible for a negative cut value
                cut[e] = 0.0
                continue
            f, delta, c, insub = found
            te, he = self.Tl[e], self.Hl[e]
            if delta:
                if c == te:
                    self.rank[insub] -= delta
                else:
                    self.rank[insub] += delta
            cv = float(cut[e])
            tf, hf = self.Tl[f], self.Hl[f]
            lca = self._tree_update(tf, hf, cv, True)
            self._tree_update(hf, tf, cv, False)
            cut[f] = -cv
            cut[e] = 0.0
            tree_mask[e] = False
            tree_mask[f] = True
            self.tree_e[e] = False
            self.tree_e[f] = True
            tadj[te].remove(e)
            tadj[he].remove(e)
            tadj[tf].append(f)
            tadj[hf].append(f)
            self._dfs_range(lca, self.par[lca], self.low[lca])
        return it + 1

    def lr_balance(self) -> None:
        """Centre free subtrees: for zero-cut tree edges, split the slack evenly (dot's LR_balance)."""
        for e in np.flatnonzero(self.tree_mask & (np.abs(self.cut) <= 1e-9)).tolist():
            found = self._enter(e)
            if found is None:
                continue
            f, delta, c, insub = found
            if delta <= 1:
                continue
            if c == self.Tl[e]:
                self.rank[insub] -= delta // 2
            else:
                self.rank[insub] += delta // 2


def _network_simplex(n: int, tails: Sequence[int], heads: Sequence[int], weights: Sequence[float]) -> list[int]:
    """Optimal layering of a connected DAG: min Σ w·(rank(h) − rank(t)) with spans ≥ 1,
    normalised to start at 0 and balanced (dot's TB_balance)."""
    if n <= 1 or len(tails) == 0:
        return [0] * n
    try:
        ns = _NetworkSimplex(n, tails, heads, [1] * len(tails), weights, _longest_path(n, tails, heads))
    except _Disconnected:  # pragma: no cover - callers pass connected components
        return _longest_path(n, tails, heads)
    ns.solve(50 * n + 1000)
    rank = ns.rank - ns.rank.min()
    return _balance(n, list(tails), list(heads), list(map(float, weights)), rank.tolist())


def _balance(n: int, tails: list[int], heads: list[int], weights: list[float], rank: list[int]) -> list[int]:
    """Move nodes with equal in/out weight to the least crowded feasible rank (dot's TB_balance)."""
    top = max(rank) if n else 0
    count = [0] * (top + 1)
    for r in rank:
        count[r] += 1
    win = [0.0] * n
    wout = [0.0] * n
    for t, h, w in zip(tails, heads, weights):
        wout[t] += w
        win[h] += w
    ins: list[list[int]] = [[] for _ in range(n)]
    outs: list[list[int]] = [[] for _ in range(n)]
    for t, h in zip(tails, heads):
        outs[t].append(h)
        ins[h].append(t)
    for v in range(n):
        if win[v] != wout[v]:
            continue
        low_v = max((rank[u] + 1 for u in ins[v]), default=0)
        high_v = min((rank[w] - 1 for w in outs[v]), default=top)
        if high_v <= low_v:
            continue
        cur = rank[v]
        best, best_c = cur, count[cur] - 1
        for r in range(low_v, high_v + 1):
            if r != cur and count[r] < best_c:
                best, best_c = r, count[r]
        if best != cur:
            count[cur] -= 1
            count[best] += 1
            rank[v] = best
    return rank


def _layer_component(n: int, tails: list[int], heads: list[int], method: str, max_width: int | None) -> list[int]:
    if n <= 1 or not tails:
        return [0] * n
    if method == "longest_path":
        return _longest_path(n, tails, heads)
    if method == "coffman_graham":
        width = int(max_width) if max_width else max(1, math.ceil(math.sqrt(n)))
        return _coffman_graham(n, tails, heads, max(1, width))
    # merge parallel arcs (weights add) for network simplex
    acc: dict[tuple[int, int], float] = {}
    for t, h in zip(tails, heads):
        acc[(t, h)] = acc.get((t, h), 0.0) + 1.0
    ts = [k[0] for k in acc]
    hs = [k[1] for k in acc]
    return _network_simplex(n, ts, hs, list(acc.values()))


# ====================================================================== #
# 3. crossing minimisation
# ====================================================================== #
def _bjm(south: list[int], q: int) -> int:
    """Inversions of *south* (values < q) with the Barth–Jünger–Mutzel accumulator tree."""
    first = 1
    while first < q:
        first <<= 1
    tree = [0] * (2 * first - 1)
    first -= 1
    cross = 0
    for k in south:
        index = k + first
        tree[index] += 1
        while index > 0:
            if index & 1:
                cross += tree[index + 1]
            index = (index - 1) >> 1
            tree[index] += 1
    return cross


def _count_crossings(layers: list[list[int]], down: list[list[int]], pos: list[int]) -> int:
    total = 0
    for r in range(len(layers) - 1):
        south: list[int] = []
        for v in layers[r]:
            d = down[v]
            if len(d) == 1:
                south.append(pos[d[0]])
            elif d:
                south.extend(sorted(pos[w] for w in d))
        if len(south) > 1:
            total += _bjm(south, len(layers[r + 1]))
    return total


def _pair_cross2(a: list[int], b: list[int], pos: list[int]) -> tuple[int, int]:
    """Crossings among the edges to *a* and to *b*: ``(owner of a left, owner of b left)``."""
    la, lb = len(a), len(b)
    if la == 1 and lb == 1:
        pa, pb = pos[a[0]], pos[b[0]]
        return (1, 0) if pa > pb else ((0, 1) if pa < pb else (0, 0))
    c0 = c1 = 0
    if la * lb <= 64:
        pbs = [pos[y] for y in b]
        for x in a:
            px = pos[x]
            for py in pbs:
                if px > py:
                    c0 += 1
                elif px < py:
                    c1 += 1
        return c0, c1
    pbs = sorted(pos[y] for y in b)
    for x in a:
        px = pos[x]
        c0 += bisect_left(pbs, px)
        c1 += lb - bisect_right(pbs, px)
    return c0, c1


def _median_value(ps: list[int]) -> float:
    m = len(ps)
    h = m // 2
    if m % 2 == 1:
        return float(ps[h])
    if m == 2:
        return (ps[0] + ps[1]) / 2.0
    left = ps[h - 1] - ps[0]
    right = ps[-1] - ps[h]
    if left + right == 0:
        return (ps[h - 1] + ps[h]) / 2.0
    return (ps[h - 1] * right + ps[h] * left) / (left + right)


class _Ordering:
    """Mutable layer orders of the proper layered graph plus the sweep operators."""

    def __init__(self, layers: list[list[int]], up: list[list[int]], down: list[list[int]]) -> None:
        self.layers = [list(L) for L in layers]
        self.up = up
        self.down = down
        self.pos = [0] * len(up)
        for L in self.layers:
            for i, v in enumerate(L):
                self.pos[v] = i

    def snapshot(self) -> list[list[int]]:
        return [list(L) for L in self.layers]

    def restore(self, layers: list[list[int]]) -> None:
        self.layers = [list(L) for L in layers]
        pos = self.pos
        for L in self.layers:
            for i, v in enumerate(L):
                pos[v] = i

    def crossings(self) -> int:
        return _count_crossings(self.layers, self.down, self.pos)

    def reorder(self, r: int, nb: list[list[int]], reverse: bool) -> None:
        """Sort layer *r* by weighted median (then barycenter) of the neighbours in *nb*.

        Nodes without such neighbours keep their slot; exact ties keep (or,
        with *reverse*, flip) their current relative order.
        """
        pos = self.pos
        L = self.layers[r]
        movable = []
        fixed = [False] * len(L)
        for i, v in enumerate(L):
            ns = nb[v]
            if not ns:
                fixed[i] = True
                continue
            ps = sorted(pos[u] for u in ns)
            movable.append((_median_value(ps), sum(ps) / len(ps), -i if reverse else i, v))
        if not movable:
            return
        movable.sort()
        it = iter(movable)
        new = [L[i] if fixed[i] else next(it)[3] for i in range(len(L))]
        self.layers[r] = new
        for i, v in enumerate(new):
            pos[v] = i

    def sweep(self, iteration: int, sort_reverse: bool = True) -> None:
        """One down (even) or up (odd) pass: weighted median, then sifting.

        As in dot, pairs of iterations alternate between flipping and keeping
        the relative order of equally ranked nodes, which lets the search walk
        across plateaus (*sort_reverse* = False keeps the sort stable). On the
        non-flipping iterations a transpose pass that also swaps equal pairs
        follows; sifting already covers the strictly improving swaps.
        """
        reverse = (iteration % 4) < 2
        R = len(self.layers)
        rows = range(1, R) if iteration % 2 == 0 else range(R - 2, -1, -1)
        nb = self.up if iteration % 2 == 0 else self.down
        for r in rows:
            self.reorder(r, nb, reverse and sort_reverse)
        for r in (range(R) if iteration % 2 == 0 else range(R - 1, -1, -1)):
            self.sift(r)
        if not reverse:
            self.transpose(True)

    def canonicalize(self, best: int, n_real: int, budget: int) -> int:
        """Bubble adjacent real nodes into graph order where it costs no crossings.

        A trial swap is propagated downwards by a *stable* median re-sort of
        the layers below (so the subtrees follow their roots) and kept only if
        the total crossing count does not grow. At most *budget* trials.
        """
        R = len(self.layers)
        trials = 0
        for r in range(R):
            changed = True
            while changed and trials < budget:
                changed = False
                L = self.layers[r]
                for i in range(len(L) - 1):
                    if trials >= budget:
                        break
                    a, b = L[i], L[i + 1]
                    if not (a < n_real and b < n_real and a > b):
                        continue
                    trials += 1
                    saved = [list(x) for x in self.layers[r:]]
                    L[i], L[i + 1] = b, a
                    self.pos[a], self.pos[b] = i + 1, i
                    for rr in range(r + 1, R):
                        self.reorder(rr, self.up, False)
                    cur = self.crossings()
                    if cur <= best:
                        best = cur
                        changed = True
                    else:
                        for k, x in enumerate(saved):
                            self.layers[r + k] = x
                            for j, v in enumerate(x):
                                self.pos[v] = j
                    L = self.layers[r]
        return best

    def pair_matrix(self, r: int) -> np.ndarray | None:
        """``C[i, j]`` = crossings between the edges of ``L[i]`` and ``L[j]`` if i is left of j.

        With ``A`` the node × adjacent-position incidence counts and ``S`` its
        exclusive prefix sums, ``C = A Sᵀ`` summed over both adjacent layers.
        """
        L = self.layers[r]
        k = len(L)
        out = None
        R = len(self.layers)
        pos = self.pos
        for nb, rr in ((self.up, r - 1), (self.down, r + 1)):
            if rr < 0 or rr >= R or not self.layers[rr]:
                continue
            m = len(self.layers[rr])
            if k * m > 4_000_000:
                return None
            rows: list[int] = []
            cols: list[int] = []
            for i, v in enumerate(L):
                for u in nb[v]:
                    rows.append(i)
                    cols.append(pos[u])
            if not rows:
                continue
            a = np.zeros((k, m))
            np.add.at(a, (np.array(rows), np.array(cols)), 1.0)
            s = np.cumsum(a, axis=1)
            s -= a
            part = a @ s.T
            out = part if out is None else out + part
        return out

    def sift(self, r: int) -> None:
        """Move every node of layer *r* to its best position given the other nodes (sifting)."""
        L = self.layers[r]
        k = len(L)
        if k < 3:
            return
        c = self.pair_matrix(r)
        if c is None:
            return
        # extra[v, w] = additional crossings when v sits right of w instead of
        # left of it; putting v into gap g (after the first g nodes) costs the
        # prefix sum of extra[v, ·] over the current order, relative to the far
        # left (extra[v, v] = 0, so v's own slot does not disturb the sums)
        extra = c.T - c
        order = np.arange(k)
        where = np.arange(k)
        moved = False
        for v in range(k):
            cum = np.cumsum(extra[v, order])
            m = int(np.argmin(cum))
            best, g = (float(cum[m]), m + 1) if cum[m] < 0 else (0.0, 0)
            p = int(where[v])
            cur = float(cum[p - 1]) if p > 0 else 0.0
            if best < cur - 0.5:
                if g <= p:
                    order = np.concatenate((order[:g], [v], order[g:p], order[p + 1 :]))
                else:
                    order = np.concatenate((order[:p], order[p + 1 : g], [v], order[g:]))
                where[order] = np.arange(k)
                moved = True
        if moved:
            new = [L[i] for i in order.tolist()]
            self.layers[r] = new
            pos = self.pos
            for i, v in enumerate(new):
                pos[v] = i

    def transpose(self, reverse: bool) -> None:
        up, down, pos = self.up, self.down, self.pos
        R = len(self.layers)
        cand = [True] * R
        rounds = 0
        while True:
            delta = 0
            rounds += 1
            for r in range(R):
                if not cand[r]:
                    continue
                cand[r] = False
                L = self.layers[r]
                for i in range(len(L) - 1):
                    v, w = L[i], L[i + 1]
                    uv, uw, dv, dw = up[v], up[w], down[v], down[w]
                    c0 = c1 = 0
                    if uv and uw:
                        c0, c1 = _pair_cross2(uv, uw, pos)
                    if dv and dw:
                        a0, a1 = _pair_cross2(dv, dw, pos)
                        c0 += a0
                        c1 += a1
                    if c1 < c0 or (reverse and c0 > 0 and c1 == c0):
                        L[i], L[i + 1] = w, v
                        pos[v], pos[w] = i + 1, i
                        delta += c0 - c1
                        cand[r] = True
                        if r > 0:
                            cand[r - 1] = True
                        if r + 1 < R:
                            cand[r + 1] = True
            if delta < 1 or rounds > 50:
                break


def _dfs_order(n: int, R: int, layer: list[int], starts: list[int], nb: list[list[int]]) -> list[list[int]]:
    """Layers filled in DFS preorder from *starts* following *nb* (no crossings on trees)."""
    seen = [False] * n
    layers: list[list[int]] = [[] for _ in range(R)]
    for s in list(starts) + list(range(n)):
        if seen[s]:
            continue
        seen[s] = True
        layers[layer[s]].append(s)
        stack = [iter(nb[s])]
        while stack:
            for w in stack[-1]:
                if not seen[w]:
                    seen[w] = True
                    layers[layer[w]].append(w)
                    stack.append(iter(nb[w]))
                    break
            else:
                stack.pop()
    return layers


def _bfs_order(n: int, R: int, layer: list[int], starts: list[int], first: list[list[int]], second: list[list[int]]) -> list[list[int]]:
    """dot's ``build_ranks``: BFS over both directions, preferring *first*."""
    seen = [False] * n
    layers: list[list[int]] = [[] for _ in range(R)]
    for s in list(starts) + list(range(n)):
        if seen[s]:
            continue
        seen[s] = True
        q = deque([s])
        while q:
            v = q.popleft()
            layers[layer[v]].append(v)
            for w in first[v]:
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
            for w in second[v]:
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
    return layers


def _minimise_crossings(
    n: int, R: int, layer: list[int], up: list[list[int]], down: list[list[int]], passes: int, rng: np.random.Generator, n_real: int
) -> tuple[list[list[int]], int]:
    """Order the layers of a proper layered graph; returns ``(layers, crossings)``.

    dot's scheme, strengthened: four initial orders (DFS and BFS from the
    sources and from the sinks) plus a few seeded random ones for small
    graphs, each improved by a short run of sweeps in two tie-handling
    variants; the two best candidates then run up to *passes* further sweeps
    (stopping after 8 without a 0.5 % gain). Every sweep is weighted median →
    sifting → transpose. A final sifting polish and canonical tie-breaking
    (equal positions follow graph order) never add crossings.
    """
    sources = [v for v in range(n) if not up[v]]
    sinks = [v for v in range(n) if not down[v]]
    inits = [
        _dfs_order(n, R, layer, sources, down),
        _dfs_order(n, R, layer, sinks, up),
        _bfs_order(n, R, layer, sources, down, up),
        _bfs_order(n, R, layer, sinks, up, down),
    ]
    size = n + sum(len(d) for d in down)
    for _ in range(min(6, 10000 // max(size, 1))):  # cheap for small graphs, and they pay off
        rand = [list(L) for L in inits[0]]
        for L in rand:
            rng.shuffle(L)
        inits.append(rand)
    large = size > 20000
    variants = (True,) if large else (True, False)
    # phase 1: a few sweeps from every initial order (dot's passes 0 and 1)
    cands: list[tuple[int, int, list[list[int]], bool]] = []
    for init in inits:
        for var in variants:
            od = _Ordering(init, up, down)
            od.transpose(False)
            lb, ll = od.crossings(), od.snapshot()
            for it in range(4):
                if lb == 0:
                    break
                od.sweep(it, var)
                cur = od.crossings()
                if cur < lb:
                    lb, ll = cur, od.snapshot()
            cands.append((lb, len(cands), ll, var))
            if lb == 0:
                break
        if min(cands)[0] == 0:
            break
    cands.sort(key=lambda c: (c[0], c[1]))
    best, best_layers = cands[0][0], cands[0][2]
    # phase 2: continue from the best candidates until sweeps stop paying off
    for lb, _, ll, var in cands[: 1 if large else 2]:
        if best == 0:
            break
        od = _Ordering(ll, up, down)
        trying = 0
        for it in range(passes):
            if lb == 0 or trying >= 8:
                break
            trying += 1
            od.sweep(it, var)
            cur = od.crossings()
            if cur <= lb:
                if cur < 0.995 * lb:
                    trying = 0
                lb, ll = cur, od.snapshot()
        if lb < best:
            best, best_layers = lb, ll
    # phase 3: sifting polish, then a plain transpose
    od = _Ordering(best_layers, up, down)
    for _ in range(10):
        if best == 0:
            break
        for r in range(R):
            od.sift(r)
        for r in range(R - 1, -1, -1):
            od.sift(r)
        od.transpose(False)
        cur = od.crossings()
        if cur >= best:
            break
        best, best_layers = cur, od.snapshot()
    # phase 4: canonical order. Equally good orders are resolved towards graph
    # order, so siblings keep their natural order ("clean0, clean1, …")
    od.restore(best_layers)
    best = od.canonicalize(best, n_real, budget=max(0, 4000 - size // 2))
    return od.snapshot(), int(best)


# ======================================================================
# 4. Brandes-Koepf coordinates
# ======================================================================

def _type1_conflicts(layers: list[list[int]], up: list[list[int]], pos: list[int], dummy: list[bool]) -> set[tuple[int, int]]:
    """Segments (upper, lower) that cross an inner segment (dummy–dummy)."""
    marked: set[tuple[int, int]] = set()
    for i in range(len(layers) - 1):
        upper, lower = layers[i], layers[i + 1]
        if not upper or not lower:
            continue
        k0 = 0
        l = 0
        last = len(lower) - 1
        for l1, v in enumerate(lower):
            inner = up[v][0] if (dummy[v] and up[v] and dummy[up[v][0]]) else -1
            if l1 == last or inner >= 0:
                k1 = pos[inner] if inner >= 0 else len(upper) - 1
                while l <= l1:
                    w = lower[l]
                    for u in up[w]:
                        k = pos[u]
                        if k < k0 or k > k1:
                            if not (dummy[w] and dummy[u]):
                                marked.add((u, w))
                    l += 1
                k0 = k1
    return marked


def _brandes_koepf(
    layers: list[list[int]], up: list[list[int]], down: list[list[int]], width: list[float], gap: list[float], dummy: list[bool]
) -> list[float]:
    n = len(width)
    pos = [0] * n
    for L in layers:
        for i, v in enumerate(L):
            pos[v] = i
    marked = _type1_conflicts(layers, up, pos, dummy)
    results: list[list[float]] = []
    for vdir in (0, 1):
        for hdir in (0, 1):
            seq = layers if vdir == 0 else layers[::-1]
            if hdir == 1:
                seq = [L[::-1] for L in seq]
            nb = up if vdir == 0 else down
            p = [0] * n
            for L in seq:
                for i, v in enumerate(L):
                    p[v] = i
            # vertical alignment
            root = list(range(n))
            align = list(range(n))
            for L in seq[1:]:
                r = -1
                for v in L:
                    ns = nb[v]
                    if not ns:
                        continue
                    if len(ns) > 1:
                        ns = sorted(ns, key=p.__getitem__)
                    d = len(ns)
                    lo_m, hi_m = (d - 1) // 2, d // 2
                    for m in ((lo_m,) if lo_m == hi_m else (lo_m, hi_m)):
                        if align[v] != v:
                            break
                        u = ns[m]
                        key = (u, v) if vdir == 0 else (v, u)
                        if key not in marked and r < p[u]:
                            align[u] = v
                            root[v] = root[u]
                            align[v] = root[v]
                            r = p[u]
            # horizontal compaction on the block graph (two passes, as in dagre)
            out_b: dict[int, dict[int, float]] = {}
            indeg: dict[int, int] = {}
            for L in seq:
                for a, b in zip(L, L[1:]):
                    ra, rb = root[a], root[b]
                    sep = (width[a] + width[b]) / 2.0 + (gap[a] + gap[b]) / 2.0
                    d = out_b.setdefault(ra, {})
                    if rb in d:
                        if sep > d[rb]:
                            d[rb] = sep
                    else:
                        d[rb] = sep
                        indeg[rb] = indeg.get(rb, 0) + 1
            blocks = [v for L in seq for v in L if root[v] == v]
            q = deque(b for b in blocks if indeg.get(b, 0) == 0)
            order = []
            while q:
                b = q.popleft()
                order.append(b)
                for c in out_b.get(b, ()):
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        q.append(c)
            xs = {b: 0.0 for b in order}
            for b in order:
                xb = xs[b]
                for c, s in out_b.get(b, {}).items():
                    if xs[c] < xb + s:
                        xs[c] = xb + s
            for b in reversed(order):
                d = out_b.get(b)
                if d:
                    m = min(xs[c] - s for c, s in d.items())
                    if m > xs[b]:
                        xs[b] = m
            x = [xs[root[v]] for v in range(n)]
            if hdir == 1:
                x = [-xi for xi in x]
            results.append(x)
    arr = np.array(results)
    w = np.asarray(width)
    spans = (arr + w / 2).max(axis=1) - (arr - w / 2).min(axis=1)
    k = int(np.argmin(spans))
    for i in range(4):
        if i % 2 == 0:  # left alignments share the left border
            arr[i] += (arr[k] - w / 2).min() - (arr[i] - w / 2).min()
        else:
            arr[i] += (arr[k] + w / 2).max() - (arr[i] + w / 2).max()
    arr.sort(axis=0)
    return ((arr[1] + arr[2]) / 2.0).tolist()


def _refine_coordinates(
    layers: list[list[int]],
    down: list[list[int]],
    width: list[float],
    gap: list[float],
    dummy: list[bool],
    x0: list[float],
    max_aux: int = 40000,
) -> list[float] | None:
    """dot's x-coordinate LP, warm-started from Brandes–Köpf.

    Minimises ``Σ Ω(e)·|x_u − x_v|`` over the layered edges subject to the
    in-layer separations, with Ω = 1 / 2 / 8 for real–real / real–dummy /
    dummy–dummy segments, so long edges stay straight and short edges
    vertical wherever the separations allow. Solved exactly by network
    simplex on the auxiliary graph of Gansner et al. (one extra node per
    edge), then zero-cost freedom is split evenly (LR balance) so parents
    sit centred over their children. Returns None when the auxiliary graph
    would exceed *max_aux* nodes.
    """
    n = len(width)
    edges = [(v, w) for v in range(n) for w in down[v]]
    n_aux = n + len(edges)
    if n_aux > max_aux or not edges:
        return None
    tails: list[int] = []
    heads: list[int] = []
    minlen: list[int] = []
    weight: list[float] = []
    for k, (u, v) in enumerate(edges):
        a = n + k
        omega = 8.0 if (dummy[u] and dummy[v]) else 2.0 if (dummy[u] or dummy[v]) else 1.0
        tails += (a, a)
        heads += (u, v)
        minlen += (0, 0)
        weight += (omega, omega)
    xi = [0] * n
    for L in layers:
        prev = -1
        for v in L:
            xv = int(round(x0[v]))
            if prev >= 0:
                sep = int(math.ceil((width[prev] + width[v]) / 2.0 + (gap[prev] + gap[v]) / 2.0 - 1e-9))
                tails.append(prev)
                heads.append(v)
                minlen.append(sep)
                weight.append(0.0)
                xv = max(xv, xi[prev] + sep)
            xi[v] = xv
            prev = v
    rank0 = xi + [min(xi[u], xi[v]) for u, v in edges]
    try:
        ns = _NetworkSimplex(n_aux, tails, heads, minlen, weight, rank0)
    except _Disconnected:
        return None
    ns.solve(20 * n_aux + 1000)
    ns.lr_balance()
    return ns.rank[:n].astype(float).tolist()


# ====================================================================== #
# 5. driver
# ====================================================================== #
class _Component:
    """Proper layered graph of one connected component."""

    def __init__(self) -> None:
        self.real: list[int] = []  # global indices of the real nodes (local id = position)
        self.layer: list[int] = []  # local layer of every local node
        self.up: list[list[int]] = []
        self.down: list[list[int]] = []
        self.width: list[float] = []
        self.gap: list[float] = []
        self.dummy: list[bool] = []
        self.chains: list[tuple[_Edge, list[int]]] = []
        self.order: list[list[int]] = []
        self.x: list[float] = []
        self.crossings = 0
        self.base = 0  # global layer of local layer 0


def _build_component(
    real: list[int], layer_of: dict[int, int], edges: list[_Edge], across: np.ndarray, node_sep: float
) -> _Component:
    comp = _Component()
    comp.real = list(real)
    local = {v: i for i, v in enumerate(real)}
    base = min(layer_of[v] for v in real)
    comp.base = base
    comp.layer = [layer_of[v] - base for v in real]
    comp.width = [float(across[v]) for v in real]
    comp.gap = [float(node_sep)] * len(real)
    comp.dummy = [False] * len(real)
    comp.up = [[] for _ in real]
    comp.down = [[] for _ in real]
    for e in edges:
        t, h = local[e.tail], local[e.head]
        lt, lh = comp.layer[t], comp.layer[h]
        span = lh - lt
        if span <= 0:  # flat edge (only possible with user ranks)
            continue
        prev = t
        chain = []
        for k in range(1, span):
            d = len(comp.layer)
            comp.layer.append(lt + k)
            comp.width.append(0.0)
            comp.gap.append(node_sep / 2.0)
            comp.dummy.append(True)
            comp.up.append([prev])
            comp.down.append([])
            comp.down[prev].append(d)
            chain.append(d)
            prev = d
        comp.down[prev].append(h)
        comp.up[h].append(prev)
        if chain:
            comp.chains.append((e, chain))
    return comp


def hierarchical(
    g: Any,
    *,
    orientation: str = "TB",
    sizes: Any = None,
    node_sep: float = 28.0,
    rank_sep: float = 72.0,
    layering: str = "network_simplex",
    ranks: Mapping[Node, int] | None = None,
    crossing_passes: int = 24,
    seed: int = 0,
    compact: bool = True,
    max_width: int | None = None,
) -> Layout:
    """Layered (Sugiyama) layout: the right picture for DAGs, pipelines and flows.

    Parameters
    ----------
    orientation:
        ``"TB"`` (sources on top, the default), ``"BT"``, ``"LR"`` or ``"RL"``.
    sizes:
        ``{node: (w, h)}`` rendered box in pixels (default 36×36). Neighbours
        in a layer keep ``half-widths + node_sep`` apart; consecutive layers
        keep *rank_sep* between their tallest boxes.
    layering:
        ``"network_simplex"`` (minimum total edge length, balanced; the default),
        ``"longest_path"`` or ``"coffman_graham"`` (at most *max_width* real
        nodes per layer, default ⌈√n⌉).
    ranks:
        ``{node: int}`` layer override for every node. Edges that then point
        upwards are treated as reversed; edges inside a layer are drawn flat.
    crossing_passes:
        Maximum number of sweeps (median, sifting, transpose) in the main phase.
    seed:
        Seeds the extra random initial orders tried on small graphs.
    compact:
        Refine the Brandes–Köpf coordinates with dot's network-simplex
        positioning (straighter, narrower drawings; skipped for very large
        layered graphs) and pack isolated nodes into a block of rows.
        ``False`` keeps the plain balanced Brandes–Köpf result.
    max_width:
        Width bound for ``layering="coffman_graham"``.

    Returns
    -------
    Layout
        Metric layout (pixels; node boxes and bend points start at (0, 0)). ``routes[(u, v)]``
        holds the bend points of every edge spanning more than one layer,
        ordered from u to v (also for reversed edges). ``meta``:
        ``orientation``, ``layers`` (NodeMap node → layer), ``crossings``,
        ``reversed_edges``, ``flat_edges``, ``rank_sep``, ``node_sep`` and
        ``rank_positions`` (coordinate of each layer along the rank axis).

    Examples
    --------
    >>> lay = hierarchical(DAG([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]))
    >>> lay.meta["layers"]["d"], lay.meta["crossings"]
    (2, 0)
    """
    orientation = _check_orientation(orientation)
    if layering not in _LAYERINGS:
        raise ValueError(f"layering must be one of {_LAYERINGS}, got {layering!r}")
    node_sep = float(node_sep)
    rank_sep = float(rank_sep)
    nodes = list(g._node)
    n = len(nodes)
    meta_base: dict[str, Any] = {"orientation": orientation, "rank_sep": rank_sep, "node_sep": node_sep}
    if n == 0:
        return _empty(
            "hierarchical", metric=True, layers=NodeMap({}, name="layer"), crossings=0,
            reversed_edges=[], flat_edges=[], rank_positions=[], **meta_base,
        )
    index = {v: i for i, v in enumerate(nodes)}
    size = _sizes_array(nodes, sizes)
    swap = orientation in ("LR", "RL")
    across = size[:, 1] if swap else size[:, 0]  # extent along a layer
    along = size[:, 0] if swap else size[:, 1]  # extent across layers
    rng = make_rng(seed)

    # ---- 1. orientation & layering --------------------------------------
    if ranks is not None:
        missing = [v for v in nodes if v not in ranks]
        if missing:
            raise ValueError(f"ranks has no layer for {len(missing)} node(s), e.g. {missing[:3]!r}")
        layer_of = {index[v]: int(ranks[v]) for v in nodes}
        lo = min(layer_of.values())
        layer_of = {k: r - lo for k, r in layer_of.items()}
        edges = []
        for u, v, _ in g._iter_edges():
            if u == v:
                continue
            a, b = index[u], index[v]
            if layer_of[a] <= layer_of[b]:
                edges.append(_Edge((u, v), a, b, False, False))
            else:
                edges.append(_Edge((u, v), b, a, True, bool(g.directed)))
    else:
        edges = _orient_edges(g, nodes, index)
        layer_of = {}
    comp_lists = components(g)
    edges_of: list[list[_Edge]] = [[] for _ in comp_lists]
    comp_id = [0] * n
    for k, comp in enumerate(comp_lists):
        for v in comp:
            comp_id[index[v]] = k
    for e in edges:
        edges_of[comp_id[e.tail]].append(e)
    isolated = [k for k, comp in enumerate(comp_lists) if not edges_of[k]]
    connected = [k for k, comp in enumerate(comp_lists) if edges_of[k]]
    if ranks is None:
        for k in connected:
            real = [index[v] for v in comp_lists[k]]
            local = {v: i for i, v in enumerate(real)}
            ts = [local[e.tail] for e in edges_of[k]]
            hs = [local[e.head] for e in edges_of[k]]
            lay = _layer_component(len(real), ts, hs, layering, max_width)
            for v, r in zip(real, lay):
                layer_of[v] = r
        # isolated nodes: one row, or a compact block of rows
        iso = [index[comp_lists[k][0]] for k in isolated]
        if iso:
            depth = 1 + max((layer_of[v] for v in layer_of), default=-1) if layer_of else 0
            if compact:
                rows = max(1, min(depth if depth > 0 else len(iso), math.ceil(math.sqrt(len(iso)))))
            else:
                rows = 1
            cols = math.ceil(len(iso) / rows)
            for j, v in enumerate(iso):
                layer_of[v] = j // cols
    flat_edges = [e.key for e in edges if layer_of[e.tail] == layer_of[e.head]]

    # ---- 2. per component: dummies, ordering, coordinates ------------------
    groups: list[list[int]] = [[index[v] for v in comp_lists[k]] for k in connected]
    group_edges: list[list[_Edge]] = [edges_of[k] for k in connected]
    iso_nodes = [index[comp_lists[k][0]] for k in isolated]
    if iso_nodes:
        groups.append(iso_nodes)
        group_edges.append([])
    comps: list[_Component] = []
    total_cross = 0
    for real, es in zip(groups, group_edges):
        comp = _build_component(real, layer_of, es, across, node_sep)
        nl = len(comp.layer)
        R = max(comp.layer) + 1
        if es:
            order, cross = _minimise_crossings(nl, R, comp.layer, comp.up, comp.down, int(crossing_passes), rng, len(real))
        else:
            order = [[] for _ in range(R)]
            for v in range(nl):
                order[comp.layer[v]].append(v)
            cross = 0
        comp.order = order
        comp.crossings = cross
        total_cross += cross
        comp.x = _brandes_koepf(order, comp.up, comp.down, comp.width, comp.gap, comp.dummy)
        if compact and es:
            refined = _refine_coordinates(order, comp.down, comp.width, comp.gap, comp.dummy, comp.x)
            if refined is not None:
                comp.x = refined
        comps.append(comp)

    # ---- 3. assemble: layer coordinates shared by all components ----------
    n_layers = 1 + max(layer_of.values())
    hmax = [0.0] * n_layers
    for v in range(n):
        r = layer_of[v]
        if along[v] > hmax[r]:
            hmax[r] = float(along[v])
    level = [hmax[0] / 2.0]
    for r in range(1, n_layers):
        level.append(level[-1] + hmax[r - 1] / 2.0 + rank_sep + hmax[r] / 2.0)
    s = np.zeros(n)
    t = np.zeros(n)
    routes_st: dict[tuple, np.ndarray] = {}
    cursor = 0.0
    comp_gap = 2.0 * node_sep
    for comp in comps:
        xs = np.asarray(comp.x)
        wd = np.asarray(comp.width)
        lo = float((xs - wd / 2).min())
        hi = float((xs + wd / 2).max())
        shift = cursor - lo
        for i, v in enumerate(comp.real):
            s[v] = xs[i] + shift
            t[v] = level[comp.layer[i] + comp.base]
        for e, chain in comp.chains:
            pts = np.array([[xs[d] + shift, level[comp.layer[d] + comp.base]] for d in chain])
            if e.flip:
                pts = pts[::-1]
            routes_st[e.key] = pts
        cursor += (hi - lo) + comp_gap
    xy = _orient(s, t, orientation)
    routes = {k: _orient(p[:, 0], p[:, 1], orientation) for k, p in routes_st.items()}
    corner = (xy - size / 2.0).min(axis=0)
    if routes:  # bend points may lie outside every node box
        corner = np.minimum(corner, np.vstack(list(routes.values())).min(axis=0))
    xy = xy - corner
    routes = {k: p - corner for k, p in routes.items()}
    if orientation == "TB":
        rank_pos = [p - corner[1] for p in level]
    elif orientation == "BT":
        rank_pos = [-p - corner[1] for p in level]
    elif orientation == "LR":
        rank_pos = [p - corner[0] for p in level]
    else:
        rank_pos = [-p - corner[0] for p in level]
    meta = dict(meta_base)
    meta.update(
        layers=NodeMap({nodes[v]: int(layer_of[v]) for v in range(n)}, name="layer"),
        crossings=int(total_cross),
        reversed_edges=[e.key for e in edges if e.reversed],
        flat_edges=flat_edges,
        rank_positions=[float(p) for p in rank_pos],
        components=len(comps),
    )
    return Layout(nodes, xy, routes=routes, method="hierarchical", metric=True, meta=meta)


__all__ = ["hierarchical"]
