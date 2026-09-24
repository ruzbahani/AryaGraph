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

"""Community detection and partition quality.

Communities are returned as a list of sets, largest first (ties: the
community whose first node comes earlier in graph order goes first), so
``communities[0]`` is always the biggest group. :func:`community_labels`
turns such a list into a node → index :class:`NodeMap` for colouring.

Modularity follows networkx's definition, including its directed form
(Leicht–Newman) and the *resolution* parameter γ:

``Q = Σ_c [ L_c / m  −  γ · K_c^out · K_c^in / m² ]``

where ``L_c`` is the edge weight inside community ``c``, ``m`` the total
edge weight and ``K_c`` the summed (out/in) strengths; undirected graphs use
``K_c^out = K_c^in = K_c / 2``.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Hashable

import numpy as np

from ..core.exceptions import NotConnected
from ..core.results import NodeMap
from ..core.utils import WeightSpec, make_rng, require_undirected, weight_fn
from ._core import components
from .centrality import edge_betweenness_centrality

Node = Hashable
Partition = list[set]


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _check_partition(g: Any, communities: Iterable[Iterable[Node]]) -> tuple[list[set], dict[Node, int]]:
    """Materialise *communities* and verify they partition the nodes of *g*."""
    comms = [set(c) for c in communities]
    owner: dict[Node, int] = {}
    for i, c in enumerate(comms):
        for v in c:
            if v not in g._node:
                raise ValueError(f"not a partition: community {i} contains {v!r}, which is not in the graph")
            if v in owner:
                raise ValueError(f"not a partition: node {v!r} is in communities {owner[v]} and {i}")
            owner[v] = i
    if len(owner) != len(g):
        missing = next(v for v in g._node if v not in owner)
        raise ValueError(f"not a partition: node {missing!r} is in no community")
    return comms, owner


def _ordered(groups: Iterable[Iterable[Node]], g: Any) -> Partition:
    """Groups as sets, largest first; ties broken by each group's first node in graph order."""
    index = {v: i for i, v in enumerate(g._node)}
    keyed = [(-len(s), min(index[v] for v in s), s) for s in (set(x) for x in groups) if s]
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [s for _, _, s in keyed]


def _groups_from_labels(nodes: list[Node], labels: list[int]) -> list[list[Node]]:
    groups: dict[int, list[Node]] = {}
    for v, lab in zip(nodes, labels):
        groups.setdefault(lab, []).append(v)
    return list(groups.values())


# ---------------------------------------------------------------------- #
# quality
# ---------------------------------------------------------------------- #
def modularity(
    g: Any,
    communities: Iterable[Iterable[Node]],
    weight: WeightSpec = "weight",
    resolution: float = 1.0,
) -> float:
    """Modularity of a partition (networkx's definition, directed graphs included).

    Parameters
    ----------
    communities:
        Iterable of node collections that must partition the nodes (``ValueError`` otherwise).
    weight:
        Edge weight (missing attribute ⇒ 1); ``None`` treats the graph as unweighted.
    resolution:
        γ; below 1 favours larger communities, above 1 smaller ones.

    Raises
    ------
    ValueError
        If *communities* is not a partition of the nodes, or for a graph whose
        total edge weight is 0 (modularity is undefined).

    Examples
    --------
    >>> modularity(Graph([(0, 1), (2, 3)]), [{0, 1}, {2, 3}])
    0.5
    """
    comms, owner = _check_partition(g, communities)
    wf = weight_fn(weight)
    k = len(comms)
    inside = [0.0] * k
    k_out = [0.0] * k
    k_in = [0.0] * k
    total = 0.0
    directed = g.directed
    for u, v, d in g._iter_edges():
        w = wf(u, v, d)
        total += w
        cu, cv = owner[u], owner[v]
        if cu == cv:
            inside[cu] += w
        k_out[cu] += w
        if directed:
            k_in[cv] += w
        else:
            k_out[cv] += w  # an undirected edge adds to both endpoints' strength
    if total == 0:
        raise ValueError("modularity is undefined for a graph whose total edge weight is 0")
    if directed:
        norm = 1.0 / total**2
        return sum(inside[c] / total - resolution * k_out[c] * k_in[c] * norm for c in range(k))
    norm = 1.0 / (2 * total) ** 2
    return sum(inside[c] / total - resolution * k_out[c] * k_out[c] * norm for c in range(k))


def partition_quality(
    g: Any,
    communities: Iterable[Iterable[Node]],
    weight: WeightSpec = "weight",
    resolution: float = 1.0,
) -> dict[str, float]:
    """``{"coverage", "performance", "modularity"}`` of a partition.

    *coverage* is the share of edges inside communities; *performance* the
    share of node pairs classified correctly (intra-community edges plus
    inter-community non-edges). Both are unweighted, as in
    ``networkx.community.partition_quality``. *weight* and *resolution* apply
    to the modularity only. Raises ``ValueError`` when the graph has no edges
    or fewer than two nodes.
    """
    comms, owner = _check_partition(g, communities)
    n, m = len(g), g.num_edges
    if m == 0 or n < 2:
        raise ValueError("partition quality is undefined for a graph without edges or with fewer than 2 nodes")
    sizes = [len(c) for c in comms]
    inter_pairs = (n * n - sum(s * s for s in sizes)) // 2
    total_pairs = n * (n - 1)
    if g.directed:
        inter_pairs *= 2
    else:
        total_pairs //= 2
    intra = 0
    inter_non_edges = inter_pairs
    for u, v, _ in g._iter_edges():
        if owner[u] == owner[v]:
            intra += 1
        else:
            inter_non_edges -= 1
    return {
        "coverage": intra / m,
        "performance": (intra + inter_non_edges) / total_pairs,
        "modularity": modularity(g, comms, weight=weight, resolution=resolution),
    }


def community_labels(communities: Iterable[Iterable[Node]]) -> NodeMap:
    """``NodeMap`` node → community index, where 0 is the largest community.

    Equal-sized communities keep their given order. A node listed twice raises ``ValueError``.

    >>> community_labels([{"a"}, {"b", "c"}])
    NodeMap('community', 3 nodes; top: {'a': 1, 'b': 0, 'c': 0})
    """
    comms = [list(dict.fromkeys(c)) for c in communities]
    order = sorted(range(len(comms)), key=lambda i: -len(comms[i]))
    labels: dict[Node, int] = {}
    for rank, i in enumerate(order):
        for v in comms[i]:
            if v in labels:
                raise ValueError(f"node {v!r} appears in more than one community")
            labels[v] = rank
    return NodeMap(labels, name="community")


# ---------------------------------------------------------------------- #
# Louvain
# ---------------------------------------------------------------------- #
class _Level:
    """Weighted (multi)graph of one Louvain level on nodes ``0 … k-1``.

    Self-loops live in ``loops``; ``nbrs`` merges both arc directions for
    directed graphs (the modularity gain only needs their sum).
    """

    __slots__ = ("k", "directed", "out", "inn", "nbrs", "loops", "deg_out", "deg_in")

    def __init__(self, k: int, directed: bool) -> None:
        self.k = k
        self.directed = directed
        self.out: list[dict[int, float]] = [{} for _ in range(k)]
        self.inn: list[dict[int, float]] = [{} for _ in range(k)] if directed else self.out
        self.loops = [0.0] * k

    def add(self, i: int, j: int, w: float) -> None:
        if i == j:
            self.loops[i] += w
            return
        self.out[i][j] = self.out[i].get(j, 0.0) + w
        if self.directed:
            self.inn[j][i] = self.inn[j].get(i, 0.0) + w
        else:
            self.out[j][i] = self.out[j].get(i, 0.0) + w

    def finish(self) -> "_Level":
        if self.directed:
            self.deg_out = [sum(self.out[i].values()) + self.loops[i] for i in range(self.k)]
            self.deg_in = [sum(self.inn[i].values()) + self.loops[i] for i in range(self.k)]
            nbrs = []
            for i in range(self.k):
                merged = dict(self.out[i])
                for j, w in self.inn[i].items():
                    merged[j] = merged.get(j, 0.0) + w
                nbrs.append(merged)
            self.nbrs = nbrs
        else:
            self.deg_out = self.deg_in = [sum(self.out[i].values()) + 2 * self.loops[i] for i in range(self.k)]
            self.nbrs = self.out
        return self

    def modularity(self, com: list[int], m: float, gamma: float) -> float:
        inside: dict[int, float] = {}
        k_out: dict[int, float] = {}
        k_in: dict[int, float] = {}
        for i in range(self.k):
            c = com[i]
            inside[c] = inside.get(c, 0.0) + self.loops[i]
            k_out[c] = k_out.get(c, 0.0) + self.deg_out[i]
            k_in[c] = k_in.get(c, 0.0) + self.deg_in[i]
            for j, w in self.out[i].items():
                if com[j] == c and (self.directed or i < j):
                    inside[c] += w
        if self.directed:
            return sum(inside[c] / m - gamma * k_out[c] * k_in[c] / m**2 for c in inside)
        return sum(inside[c] / m - gamma * (k_out[c] / (2 * m)) ** 2 for c in inside)

    def aggregate(self, com: list[int]) -> "_Level":
        """Collapse each community into one node (community ids must be ``0 … c-1``)."""
        nxt = _Level(max(com) + 1, self.directed)
        for i in range(self.k):
            ci = com[i]
            nxt.loops[ci] += self.loops[i]
            for j, w in self.out[i].items():
                if self.directed or i < j:
                    nxt.add(ci, com[j], w)
        return nxt.finish()


def _one_level(
    level: _Level, m: float, gamma: float, rng: np.random.Generator, start: list[int] | None = None
) -> tuple[list[int], bool]:
    """Local moving phase: greedily move nodes to the neighbouring community with the best gain.

    Starts from singletons, or from the community labels *start* (each < k).
    Returns labels renumbered ``0 … c-1`` (keeping their relative order) and
    whether any node moved.
    """
    k = level.k
    deg_out, deg_in, nbrs, directed = level.deg_out, level.deg_in, level.nbrs, level.directed
    if start is None:
        com = list(range(k))
        tot_out = list(deg_out)
        tot_in = list(deg_in) if directed else tot_out
    else:
        com = list(start)
        tot_out = [0.0] * k
        tot_in = [0.0] * k if directed else tot_out
        for i, c in enumerate(com):
            tot_out[c] += deg_out[i]
            if directed:
                tot_in[c] += deg_in[i]
    m2 = m * m
    order = rng.permutation(k).tolist()
    improved = False
    moves = 1
    while moves:
        moves = 0
        for u in order:
            own = com[u]
            w2c: dict[int, float] = {}
            for v, w in nbrs[u].items():
                c = com[v]
                w2c[c] = w2c.get(c, 0.0) + w
            w_own = w2c.setdefault(own, 0.0)
            du_out, du_in = deg_out[u], deg_in[u]
            if directed:
                tot_out[own] -= du_out
                tot_in[own] -= du_in
                remove = -w_own / m + gamma * (du_out * tot_in[own] + du_in * tot_out[own]) / m2
            else:
                tot_out[own] -= du_out
                remove = -w_own / m + gamma * tot_out[own] * du_out / (2 * m2)
            best, best_gain = own, 0.0
            for c, w in w2c.items():
                if directed:
                    gain = remove + w / m - gamma * (du_out * tot_in[c] + du_in * tot_out[c]) / m2
                else:
                    gain = remove + w / m - gamma * tot_out[c] * du_out / (2 * m2)
                if gain > best_gain:
                    best, best_gain = c, gain
            tot_out[best] += du_out
            if directed:
                tot_in[best] += du_in
            if best != own:
                com[u] = best
                moves += 1
                improved = True
    # renumber communities 0 … c-1 in order of their original label
    relabel = {c: i for i, c in enumerate(sorted(set(com)))}
    return [relabel[c] for c in com], improved


class _LouvainRun:
    """One Louvain run: the levels, with the labels found on each (networkx's schedule).

    ``steps[t] = (level, com, members)``: the graph of level *t*, the community
    of each of its nodes, and the original node indices inside each of its nodes.
    """

    def __init__(
        self, g: Any, weight: WeightSpec, resolution: float, threshold: float, seed: Any, max_level: int | None
    ) -> None:
        if max_level is not None and max_level <= 0:
            raise ValueError("max_level must be a positive integer")
        self.nodes = nodes = list(g._node)
        index = {v: i for i, v in enumerate(nodes)}
        wf = weight_fn(weight)
        level = _Level(len(nodes), g.directed)
        m = 0.0
        for u, v, d in g._iter_edges():
            w = wf(u, v, d)
            if w < 0:
                raise ValueError(f"Louvain needs non-negative weights; edge ({u!r}, {v!r}) has {w}")
            m += w
            level.add(index[u], index[v], w)
        level.finish()
        self.m, self.gamma = m, resolution
        self.rng = make_rng(seed)
        self.steps: list[tuple[_Level, list[int], list[list[int]]]] = []
        if m == 0:
            return
        members: list[list[int]] = [[i] for i in range(len(nodes))]
        mod = level.modularity(list(range(level.k)), m, resolution)
        com, _ = _one_level(level, m, resolution, self.rng)
        improved = True  # the first level is always reported, as in networkx
        while improved:
            self.steps.append((level, com, members))
            if max_level is not None and len(self.steps) >= max_level:
                return
            new_mod = level.modularity(com, m, resolution)
            if new_mod - mod <= threshold:
                return
            mod = new_mod
            members = _merge_members(members, com)
            level = level.aggregate(com)
            com, improved = _one_level(level, m, resolution, self.rng)

    def partitions(self) -> list[list[list[Node]]]:
        if not self.steps:
            return [[[v] for v in self.nodes]]
        return [self._named(_merge_members(members, com)) for _, com, members in self.steps]

    def refined_labels(self) -> list[int]:
        """Community label of every original node after multilevel refinement (Rotta & Noack 2011).

        The top-level labels are projected down one level at a time and the
        local-moving phase is re-run there, starting from them, down to the
        original nodes. Every move has positive gain, so modularity never drops.
        """
        if not self.steps:
            return list(range(len(self.nodes)))
        _, top_com, top_members = self.steps[-1]
        label = [0] * len(self.nodes)
        for i, c in enumerate(top_com):
            for o in top_members[i]:
                label[o] = c
        for level, _, members in reversed(self.steps[:-1]):
            start = [label[grp[0]] for grp in members]
            com, _ = _one_level(level, self.m, self.gamma, self.rng, start=start)
            for i, c in enumerate(com):
                for o in members[i]:
                    label[o] = c
        return label

    def _named(self, groups: list[list[int]]) -> list[list[Node]]:
        return [[self.nodes[i] for i in grp] for grp in groups]


def _merge_members(members: list[list[int]], com: list[int]) -> list[list[int]]:
    grouped: list[list[int]] = [[] for _ in range(max(com) + 1)]
    for i, c in enumerate(com):
        grouped[c].extend(members[i])
    return grouped


def louvain_communities(
    g: Any,
    weight: WeightSpec = "weight",
    resolution: float = 1.0,
    threshold: float = 1e-7,
    seed: Any = None,
    max_level: int | None = None,
    refine: bool = True,
) -> Partition:
    """Louvain community detection (Blondel et al. 2008) with node aggregation between levels.

    Each level moves nodes, in a random order drawn from *seed*, to the
    neighbouring community with the largest modularity gain until no move
    helps, then collapses communities into super-nodes; the run stops when a
    level gains at most *threshold* modularity (or after *max_level* levels).
    This is networkx's schedule, directed graphs included. With *refine*
    (default), the final partition is then polished by multilevel refinement:
    the local-moving phase is re-run on every finer level, starting from the
    final communities, so modularity can only go up. ``refine=False`` returns
    the plain Louvain result (``louvain_hierarchy(...)[-1]``).

    Weights must be non-negative. Roughly ``O(m)`` per sweep.
    Returns the communities as a list of sets, largest first.

    >>> comms = louvain_communities(g, seed=1)
    >>> labels = community_labels(comms)  # node -> community index for colouring
    """
    run = _LouvainRun(g, weight, resolution, threshold, seed, max_level)
    if not refine:
        return _ordered(run.partitions()[-1], g)
    return _ordered(_groups_from_labels(run.nodes, run.refined_labels()), g)


def louvain_hierarchy(
    g: Any,
    weight: WeightSpec = "weight",
    resolution: float = 1.0,
    threshold: float = 1e-7,
    seed: Any = None,
) -> list[Partition]:
    """The Louvain dendrogram: the partition reached after every level, finest first.

    Levels are unrefined; the last entry equals
    ``louvain_communities(..., refine=False)`` for the same arguments.
    """
    run = _LouvainRun(g, weight, resolution, threshold, seed, None)
    return [_ordered(p, g) for p in run.partitions()]


# ---------------------------------------------------------------------- #
# Clauset–Newman–Moore greedy modularity
# ---------------------------------------------------------------------- #
def greedy_modularity_communities(
    g: Any,
    weight: WeightSpec = None,
    resolution: float = 1.0,
    cutoff: int = 1,
    best_n: int | None = None,
) -> Partition:
    """Greedy agglomerative modularity maximisation (Clauset–Newman–Moore).

    Starting from singletons, repeatedly merge the pair of adjacent
    communities with the largest modularity gain ``ΔQ`` while ``ΔQ ≥ 0``
    (``O(m d log n)`` with a lazily-updated max-heap). Ties go to the pair
    whose first community comes earliest in graph order, which reproduces
    networkx's merges exactly whenever node labels sort in insertion order.

    Parameters
    ----------
    cutoff:
        Stop once only this many communities are left.
    best_n:
        Keep merging (even with ``ΔQ < 0``) until at most this many remain.
    """
    n = len(g)
    if g.num_edges == 0:
        return _ordered(([v] for v in g._node), g)
    if not 1 <= cutoff <= n:
        raise ValueError(f"cutoff must be between 1 and {n}, got {cutoff}")
    if best_n is not None:
        if not 1 <= best_n <= n:
            raise ValueError(f"best_n must be between 1 and {n}, got {best_n}")
        if best_n < cutoff:
            raise ValueError(f"best_n must be >= cutoff, got {best_n} < {cutoff}")
        if best_n == 1:
            return [set(g._node)]
    else:
        best_n = n

    nodes = list(g._node)
    index = {v: i for i, v in enumerate(nodes)}
    directed = g.directed
    wf = weight_fn(weight)

    # Strengths accumulated exactly as networkx does, so every ΔQ is bit-identical.
    if directed:
        k_out = [sum(wf(u, v, d) for v, d in g._succ[u].items()) for u in nodes]
        k_in = [sum(wf(v, u, d) for v, d in g._pred[u].items()) for u in nodes]
        deg_total = sum(ko + ki for ko, ki in zip(k_out, k_in))
    else:
        k_out = []
        for u in nodes:
            nb = g._succ[u]
            s = sum(wf(u, v, d) for v, d in nb.items())
            k_out.append(s + wf(u, u, nb[u]) if u in nb else s)
        deg_total = sum(k_out)
    m = deg_total // 2 if weight is None else deg_total / 2
    if m == 0:
        return _ordered(([v] for v in nodes), g)
    q0 = 1 / m
    if directed:
        a = [d * q0 for d in k_out]
        b = [d * q0 for d in k_in]
    else:
        a = b = [d * q0 * 0.5 for d in k_out]

    dq: list[dict[int, float]] = [{} for _ in range(n)]
    for u, v, d in g._iter_edges():
        if u == v:
            continue
        i, j = index[u], index[v]
        w = wf(u, v, d)
        dq[i][j] = dq[i].get(j, 0.0) + w
        dq[j][i] = dq[j].get(i, 0.0) + w
    heap: list[tuple[float, int, int]] = []
    for i in range(n):
        row = dq[i]
        for j, w in row.items():
            val = q0 * w - resolution * (a[i] * b[j] + b[i] * a[j])
            row[j] = val
            heap.append((-val, i, j))
    heapq.heapify(heap)

    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    while len(members) > cutoff:
        # pop the best live entry; stale ones no longer match the current ΔQ
        while heap:
            neg, u, v = heapq.heappop(heap)
            if dq[u].get(v) == -neg:
                break
        else:
            # no adjacent pair left: communities are the components
            ordered = sorted(members.values(), key=len, reverse=True)
            while len(ordered) > best_n:
                first, second, *rest = ordered
                ordered = [first + second, *rest]
            return _ordered(([nodes[i] for i in grp] for grp in ordered), g)
        if -neg < 0 and len(members) <= best_n:
            break
        # merge community u into v and update ΔQ of their neighbours
        du, dv = dq[u], dq[v]
        for w in (set(du) | set(dv)) - {u, v}:
            if w in du and w in dv:
                val = dv[w] + du[w]
            elif w in dv:
                val = dv[w] - resolution * (a[u] * b[w] + a[w] * b[u])
            else:
                val = du[w] - resolution * (a[v] * b[w] + a[w] * b[v])
            dv[w] = val
            dq[w][v] = val
            heapq.heappush(heap, (-val, v, w))
            heapq.heappush(heap, (-val, w, v))
        for w in du:
            del dq[w][u]
        dq[u] = {}
        a[v] += a[u]
        a[u] = 0
        if directed:
            b[v] += b[u]
            b[u] = 0
        members[v] = members[u] + members[v]
        del members[u]
    return _ordered(([nodes[i] for i in grp] for grp in members.values()), g)


# ---------------------------------------------------------------------- #
# label propagation and fluid communities
# ---------------------------------------------------------------------- #
def _weighted_neighbors(g: Any, weight: WeightSpec) -> tuple[list[Node], list[list[tuple[int, float]]]]:
    """Symmetric weighted neighbour lists (reciprocal arcs add up; self-loops kept once)."""
    nodes = list(g._node)
    index = {v: i for i, v in enumerate(nodes)}
    wf = weight_fn(weight)
    acc: list[dict[int, float]] = [{} for _ in nodes]
    for u, v, d in g._iter_edges():
        i, j = index[u], index[v]
        w = wf(u, v, d)
        acc[i][j] = acc[i].get(j, 0.0) + w
        if i != j:
            acc[j][i] = acc[j].get(i, 0.0) + w
    return nodes, [list(row.items()) for row in acc]


def label_propagation_communities(
    g: Any,
    seed: Any = None,
    weight: WeightSpec = None,
    max_iter: int | None = None,
) -> Partition:
    """Asynchronous label propagation (Raghavan, Albert & Kumara 2007).

    Every node starts with its own label; in each sweep (random order from
    *seed*) a node adopts the label carrying the largest (weighted) share of
    its neighbours, keeping its current label when that is among the best and
    otherwise picking one of the best at random. The run ends when no node
    changes (or after *max_iter* sweeps). Directed graphs are read as
    undirected; networkx follows successors only, which can oscillate.
    Linear time per sweep.
    """
    nodes, nbrs = _weighted_neighbors(g, weight)
    n = len(nodes)
    rng = make_rng(seed)
    labels = list(range(n))
    sweeps = 0
    changed = True
    while changed and (max_iter is None or sweeps < max_iter):
        changed = False
        for i in rng.permutation(n).tolist():
            row = nbrs[i]
            if not row:
                continue
            freq: dict[int, float] = {}
            for j, w in row:
                lab = labels[j]
                freq[lab] = freq.get(lab, 0.0) + w
            top = max(freq.values())
            best = [lab for lab, f in freq.items() if f == top]
            if labels[i] not in best:
                labels[i] = best[int(rng.integers(len(best)))] if len(best) > 1 else best[0]
                changed = True
        sweeps += 1
    return _ordered(_groups_from_labels(nodes, labels), g)


def asyn_fluid_communities(g: Any, k: int, max_iter: int = 100, seed: Any = None) -> Partition:
    """Fluid communities (Parés et al. 2017): *k* communities compete by density.

    Needs a connected undirected graph and ``1 ≤ k ≤ n``. Each community has
    total density 1 spread over its members; nodes (visited in random order)
    join the community with the largest summed density among themselves and
    their neighbours. Stops when stable or after *max_iter* sweeps.
    """
    require_undirected(g, "asyn_fluid_communities")
    n = len(g)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and the number of nodes ({n}), got {k}")
    if len(components(g)) != 1:
        raise NotConnected("asyn_fluid_communities needs a connected graph")
    nodes = list(g._node)
    index = {v: i for i, v in enumerate(nodes)}
    nbrs = [[index[u] for u in g._succ[v]] for v in nodes]
    rng = make_rng(seed)
    com = [-1] * n
    size = [0] * k
    density = [1.0] * k
    for c, i in enumerate(rng.permutation(n)[:k].tolist()):
        com[i] = c
        size[c] = 1
    for _ in range(max_iter + 1):
        changed = False
        for i in rng.permutation(n).tolist():
            score: dict[int, float] = {}
            if com[i] >= 0:
                score[com[i]] = density[com[i]]
            for j in nbrs[i]:
                c = com[j]
                if c >= 0:
                    score[c] = score.get(c, 0.0) + density[c]
            if not score:
                continue
            top = max(score.values())
            best = [c for c, s in score.items() if top - s < 1e-4]
            if com[i] in best:
                continue
            changed = True
            new = best[int(rng.integers(len(best)))] if len(best) > 1 else best[0]
            old = com[i]
            if old >= 0:
                size[old] -= 1
                density[old] = 1.0 / size[old]
            com[i] = new
            size[new] += 1
            density[new] = 1.0 / size[new]
        if not changed:
            break
    return _ordered(_groups_from_labels(nodes, com), g)


# ---------------------------------------------------------------------- #
# Girvan–Newman
# ---------------------------------------------------------------------- #
def _most_central_edge(g: Any) -> tuple[Node, Node]:
    eb = edge_betweenness_centrality(g)
    return max(eb, key=eb.__getitem__)


def girvan_newman(
    g: Any,
    most_valuable_edge: Callable[[Any], tuple[Node, Node]] | None = None,
) -> Iterator[Partition]:
    """Divisive communities (Girvan & Newman 2002): yield a finer partition at each split.

    Works on an undirected copy without self-loops, repeatedly deleting the
    edge of highest betweenness (first in edge order on ties, as networkx)
    until the number of components grows, then yielding the components.
    *most_valuable_edge(graph) -> (u, v)* replaces the betweenness rule.
    Each split costs ``O(n·m)`` per removed edge, so this suits small graphs.

    >>> from itertools import islice
    >>> first_split = next(girvan_newman(g))
    """
    if g.num_edges == 0:
        yield _ordered(components(g), g)
        return
    pick = most_valuable_edge or _most_central_edge
    work = g.to_undirected() if g.directed else g.copy()
    for u, v in work.selfloops():
        work.remove_edge(u, v)
    while work.num_edges > 0:
        before = len(components(work))
        comps = None
        while comps is None or len(comps) <= before:
            u, v = pick(work)
            work.remove_edge(u, v)
            comps = components(work)
        yield _ordered(comps, g)


__all__ = [
    "modularity",
    "partition_quality",
    "community_labels",
    "louvain_communities",
    "louvain_hierarchy",
    "greedy_modularity_communities",
    "label_propagation_communities",
    "asyn_fluid_communities",
    "girvan_newman",
]
