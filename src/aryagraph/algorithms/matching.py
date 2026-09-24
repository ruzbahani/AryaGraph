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

"""Matchings: bipartite (Hopcroft–Karp), greedy maximal, and maximum-weight (Edmonds' blossoms).

A matching is a set of edges no two of which share a node. Matchings are
returned as sets of edges ``(u, v)`` oriented as :attr:`Graph.edges` reports
them, except :func:`hopcroft_karp`, which (like networkx) returns a dict
listing every matched pair in both directions. All functions require an
undirected graph; self-loops never take part in a matching.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Hashable, Iterable

from ..core.exceptions import GraphTypeError, NodeNotFound
from ..core.graph import Graph
from ..core.utils import WeightSpec, require_undirected, weight_fn

Node = Hashable


# ---------------------------------------------------------------------- #
# bipartite: Hopcroft–Karp
# ---------------------------------------------------------------------- #
def _hopcroft_karp_core(adj: list[list[int]], n_right: int) -> tuple[list[int], list[int]]:
    """Maximum bipartite matching on integer vertices.

    Left vertices are ``0 … len(adj)-1`` with ``adj[u]`` their right
    neighbors in ``0 … n_right-1``. Returns ``(mate_left, mate_right)`` with
    -1 for unmatched vertices. Each phase layers the graph by BFS from the free
    left vertices and augments along a maximal set of vertex-disjoint shortest
    paths (iterative DFS with current-arc pointers): O(m·√n).
    """
    n_left = len(adj)
    mate_l = [-1] * n_left
    mate_r = [-1] * n_right
    unreached = n_left + 1
    dist = [0] * n_left
    while True:
        queue: deque[int] = deque()
        for u in range(n_left):
            if mate_l[u] < 0:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = unreached
        shortest = unreached  # length (in left layers) of the shortest augmenting paths
        while queue:
            u = queue.popleft()
            if dist[u] >= shortest:
                continue
            for v in adj[u]:
                w = mate_r[v]
                if w < 0:
                    if shortest == unreached:
                        shortest = dist[u] + 1
                elif dist[w] == unreached:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        if shortest == unreached:
            return mate_l, mate_r
        ptr = [0] * n_left
        for root in range(n_left):
            if mate_l[root] >= 0:
                continue
            stack = [root]
            via: list[int] = []  # via[i]: right vertex used to leave stack[i]
            while stack:
                u = stack[-1]
                nbrs = adj[u]
                moved = False
                while ptr[u] < len(nbrs):
                    v = nbrs[ptr[u]]
                    ptr[u] += 1
                    w = mate_r[v]
                    if w < 0:
                        if dist[u] + 1 == shortest:
                            via.append(v)
                            for x, y in zip(stack, via):
                                mate_l[x] = y
                                mate_r[y] = x
                            stack = []
                            moved = True
                            break
                    elif dist[w] == dist[u] + 1:
                        via.append(v)
                        stack.append(w)
                        moved = True
                        break
                if not moved:  # dead end: drop u from this phase
                    dist[u] = unreached
                    stack.pop()
                    if via:
                        via.pop()


def hopcroft_karp(g: Graph, top_nodes: Iterable[Node] | None = None) -> dict[Node, Node]:
    """Maximum-cardinality matching of a bipartite graph (Hopcroft–Karp, O(m·√n)).

    Parameters
    ----------
    top_nodes:
        The nodes of one side. Every edge must join a top node to a non-top
        node (:class:`GraphTypeError` otherwise). If omitted, sides come from
        :func:`aryagraph.algorithms.coloring.bipartite_sets`; the matching size
        does not depend on which side is called "top".

    Returns
    -------
    ``{u: v, v: u, …}``: each matched pair in both directions, in graph order.
    """
    require_undirected(g, "hopcroft_karp")
    if top_nodes is None:
        from .coloring import bipartite_sets

        top = bipartite_sets(g)[0]
    else:
        top = set()
        for n in top_nodes:
            if n not in g:
                raise NodeNotFound(n)
            top.add(n)
    left = [n for n in g._node if n in top]
    right = [n for n in g._node if n not in top]
    right_index = {n: i for i, n in enumerate(right)}
    succ = g._succ
    adj: list[list[int]] = []
    for u in left:
        row = []
        for v in succ[u]:
            if v in top:
                raise GraphTypeError(f"edge ({u!r}, {v!r}) joins two top nodes; the graph is not bipartite with these sides")
            row.append(right_index[v])
        adj.append(row)
    for v in right:
        for w in succ[v]:
            if w not in top:
                raise GraphTypeError(f"edge ({v!r}, {w!r}) joins two bottom nodes; the graph is not bipartite with these sides")
    mate_l, mate_r = _hopcroft_karp_core(adj, len(right))
    left_index = {n: i for i, n in enumerate(left)}
    out: dict[Node, Node] = {}
    for n in g._node:
        if n in top:
            j = mate_l[left_index[n]]
            if j >= 0:
                out[n] = right[j]
        else:
            i = mate_r[right_index[n]]
            if i >= 0:
                out[n] = left[i]
    return out


bipartite_maximum_matching = hopcroft_karp


# ---------------------------------------------------------------------- #
# general graphs
# ---------------------------------------------------------------------- #
def maximal_matching(g: Graph) -> set[tuple[Node, Node]]:
    """A maximal matching (no edge can be added), greedily in edge order. O(m).

    Its size is at least half the maximum; use :func:`max_weight_matching`
    with ``maxcardinality=True`` for a maximum one.
    """
    require_undirected(g, "maximal_matching")
    matched: set[Node] = set()
    out: set[tuple[Node, Node]] = set()
    for u, v in g.edges:
        if u != v and u not in matched and v not in matched:
            out.add((u, v))
            matched.add(u)
            matched.add(v)
    return out


def _matching_pairs(g: Graph, matching: Mapping[Node, Node] | Iterable[tuple[Node, Node]]) -> list[tuple[Node, Node]] | None:
    """Normalise a dict or edge collection into a pair list (None if a dict is inconsistent)."""
    pairs: list[tuple[Node, Node]] = []
    if isinstance(matching, Mapping):
        done: set[Node] = set()
        for u, v in matching.items():
            if matching.get(v, u) != u:
                return None
            if u not in done:
                pairs.append((u, v))
                done.update((u, v))
    else:
        for e in matching:
            u, v = e
            pairs.append((u, v))
    for u, v in pairs:
        for n in (u, v):
            if n not in g:
                raise NodeNotFound(n)
    return pairs


def is_matching(g: Graph, matching: Mapping[Node, Node] | Iterable[tuple[Node, Node]]) -> bool:
    """True when *matching* (a dict ``{u: v}`` or a collection of edges) is a matching of *g*.

    Every pair must be an edge of *g* (no self-loop) and no node may appear
    twice. Nodes missing from *g* raise :class:`NodeNotFound`.
    """
    require_undirected(g, "is_matching")
    pairs = _matching_pairs(g, matching)
    if pairs is None:
        return False
    used: set[Node] = set()
    for u, v in pairs:
        if u == v or not g.has_edge(u, v) or u in used or v in used:
            return False
        used.add(u)
        used.add(v)
    return True


def _matched_nodes(g: Graph, matching: Mapping[Node, Node] | Iterable[tuple[Node, Node]]) -> set[Node] | None:
    """The nodes covered by *matching*, or None if it is not a matching of *g*."""
    if not isinstance(matching, Mapping):
        matching = list(matching)  # may be a one-shot iterator
    if not is_matching(g, matching):
        return None
    return {n for pair in _matching_pairs(g, matching) or () for n in pair}


def is_maximal_matching(g: Graph, matching: Mapping[Node, Node] | Iterable[tuple[Node, Node]]) -> bool:
    """True when *matching* is a matching that no edge of *g* can extend."""
    used = _matched_nodes(g, matching)
    if used is None:
        return False
    return not any(u != v and u not in used and v not in used for u, v in g.edges)


def is_perfect_matching(g: Graph, matching: Mapping[Node, Node] | Iterable[tuple[Node, Node]]) -> bool:
    """True when *matching* is a matching covering every node of *g*."""
    used = _matched_nodes(g, matching)
    return used is not None and len(used) == len(g)


def max_weight_matching(g: Graph, weight: WeightSpec = "weight", maxcardinality: bool = False) -> set[tuple[Node, Node]]:
    """Maximum-weight matching of a general graph (Edmonds' blossom algorithm, exact).

    Parameters
    ----------
    weight:
        Edge-attribute name (missing ⇒ 1), ``None`` (all 1, giving a maximum
        cardinality matching), or ``f(u, v, attrs)`` (``None`` hides the edge).
    maxcardinality:
        If True, the heaviest among the matchings of maximum cardinality.

    Primal–dual method with blossom shrinking (Galil 1986, after van
    Rantwijk's formulation), O(n³). With integer weights all arithmetic is
    integral and the optimum is exact; with floats it is exact up to
    round-off. Edges of weight ≤ 0 are never needed unless *maxcardinality*.

    >>> g = Graph([(1, 2, 6), (1, 3, 2), (2, 3, 1), (2, 4, 7), (3, 5, 9), (4, 5, 3)])
    >>> sorted(max_weight_matching(g))
    [(2, 4), (3, 5)]
    """
    require_undirected(g, "max_weight_matching")
    wf = weight_fn(weight)
    nodes = list(g._node)
    index = {n: i for i, n in enumerate(nodes)}
    edges: list[tuple[int, int, Any]] = []
    for u, v, d in g._iter_edges():
        if u == v:
            continue
        w = wf(u, v, d)
        if w is None:
            continue
        edges.append((index[u], index[v], w))
    mate = _blossom(len(nodes), edges, maxcardinality)
    return {(u, v) for u, v in g.edges if u != v and mate[index[u]] == index[v]}


def min_weight_matching(g: Graph, weight: WeightSpec = "weight") -> set[tuple[Node, Node]]:
    """Lightest matching among those of maximum cardinality.

    Runs :func:`max_weight_matching` with ``maxcardinality=True`` on the
    weights ``(max_w + 1) - w`` (the networkx definition): every maximum
    cardinality matching has the same size, so maximising the transformed
    total minimises the original one.
    """
    require_undirected(g, "min_weight_matching")
    wf = weight_fn(weight)
    ws = [w for u, v, d in g._iter_edges() if u != v and (w := wf(u, v, d)) is not None]
    if not ws:
        return set()
    top = max(ws) + 1
    return max_weight_matching(
        g,
        weight=lambda u, v, d: None if (w := wf(u, v, d)) is None else top - w,
        maxcardinality=True,
    )


def _blossom(n: int, edges: list[tuple[int, int, Any]], maxcardinality: bool) -> list[int]:
    """Edmonds' maximum-weight matching on vertices ``0 … n-1``; returns ``mate`` (-1 = single).

    Vertices are ``0 … n-1`` and non-trivial blossoms ``n … 2n-1``. Edge k has
    endpoints ``2k`` (its first vertex) and ``2k+1`` (its second); ``p ^ 1`` is
    the other end of endpoint ``p``. Dual variables are stored doubled so
    that integer weights keep every quantity integral.
    """
    if not edges:
        return [-1] * n
    integral = all(isinstance(w, Integral) for _, _, w in edges)
    weights = [int(w) if integral else float(w) for _, _, w in edges]
    eu = [i for i, _, _ in edges]
    ev = [j for _, j, _ in edges]
    nedge = len(edges)
    endpoint = [ev[p >> 1] if p & 1 else eu[p >> 1] for p in range(2 * nedge)]
    neighbend: list[list[int]] = [[] for _ in range(n)]
    for k in range(nedge):
        neighbend[eu[k]].append(2 * k + 1)
        neighbend[ev[k]].append(2 * k)
    maxweight = max(0, max(weights))

    mate = [-1] * n  # remote endpoint of v's matched edge
    label = [0] * (2 * n)  # 0 free, 1 S, 2 T (5 = S with breadcrumb)
    labelend = [-1] * (2 * n)  # endpoint through which the (sub)blossom got its label
    inblossom = list(range(n))  # top-level blossom containing each vertex
    blossomparent = [-1] * (2 * n)
    blossomchilds: list[list[int] | None] = [None] * (2 * n)
    blossombase = list(range(n)) + [-1] * n
    blossomendps: list[list[int] | None] = [None] * (2 * n)
    bestedge = [-1] * (2 * n)  # least-slack edge used by delta2 / delta3
    blossombestedges: list[list[int] | None] = [None] * (2 * n)
    unusedblossoms = list(range(n, 2 * n))
    dualvar: list[Any] = [maxweight] * n + [0] * n
    allowedge = [False] * nedge
    queue: list[int] = []

    def slack(k: int) -> Any:
        return dualvar[eu[k]] + dualvar[ev[k]] - 2 * weights[k]

    def leaves(b: int) -> list[int]:
        if b < n:
            return [b]
        out = []
        stack = [b]
        while stack:
            t = stack.pop()
            if t < n:
                out.append(t)
            else:
                stack.extend(reversed(blossomchilds[t]))  # type: ignore[arg-type]
        return out

    def assign_label(w: int, t: int, p: int) -> None:
        while True:
            b = inblossom[w]
            label[w] = label[b] = t
            labelend[w] = labelend[b] = p
            bestedge[w] = bestedge[b] = -1
            if t == 1:
                queue.extend(leaves(b))
                return
            # b became T: its base's mate becomes S (a T-blossom has one external mate).
            mb = mate[blossombase[b]]
            w, t, p = endpoint[mb], 1, mb ^ 1

    def scan_blossom(v: int, w: int) -> int:
        """Trace back from S-vertices v and w; a new blossom's base, or -1 for an augmenting path."""
        path = []
        base = -1
        while v != -1 or w != -1:
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            path.append(b)
            label[b] = 5
            if labelend[b] == -1:
                v = -1  # reached a single vertex: this side is done
            else:
                v = endpoint[labelend[b]]
                b = inblossom[v]
                v = endpoint[labelend[b]]  # one more step, through the T-blossom
            if w != -1:
                v, w = w, v
        for b in path:
            label[b] = 1
        return base

    def add_blossom(base: int, k: int) -> None:
        v, w = eu[k], ev[k]
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        b = unusedblossoms.pop()
        blossombase[b] = base
        blossomparent[b] = -1
        blossomparent[bb] = b
        path: list[int] = []
        endps: list[int] = []
        while bv != bb:
            blossomparent[bv] = b
            path.append(bv)
            endps.append(labelend[bv])
            v = endpoint[labelend[bv]]
            bv = inblossom[v]
        path.append(bb)
        path.reverse()
        endps.reverse()
        endps.append(2 * k)
        while bw != bb:
            blossomparent[bw] = b
            path.append(bw)
            endps.append(labelend[bw] ^ 1)
            w = endpoint[labelend[bw]]
            bw = inblossom[w]
        blossomchilds[b] = path
        blossomendps[b] = endps
        label[b] = 1
        labelend[b] = labelend[bb]
        dualvar[b] = 0
        for x in leaves(b):
            if label[inblossom[x]] == 2:  # former T-vertices become S: scan them
                queue.append(x)
            inblossom[x] = b
        # Least-slack edges from the new blossom to every other S-blossom.
        bestedgeto = [-1] * (2 * n)
        for sub in path:
            if blossombestedges[sub] is None:
                nblists = [[p >> 1 for p in neighbend[x]] for x in leaves(sub)]
            else:
                nblists = [blossombestedges[sub]]  # type: ignore[list-item]
            for nblist in nblists:
                for k2 in nblist:
                    i, j = eu[k2], ev[k2]
                    if inblossom[j] == b:
                        i, j = j, i
                    bj = inblossom[j]
                    if bj != b and label[bj] == 1 and (bestedgeto[bj] == -1 or slack(k2) < slack(bestedgeto[bj])):
                        bestedgeto[bj] = k2
            blossombestedges[sub] = None
            bestedge[sub] = -1
        mine = [k2 for k2 in bestedgeto if k2 != -1]
        blossombestedges[b] = mine
        best = -1
        for k2 in mine:
            if best == -1 or slack(k2) < slack(best):
                best = k2
        bestedge[b] = best

    def expand_blossom(b0: int, endstage: bool) -> None:
        todo = [b0]
        while todo:
            b = todo.pop()
            childs: list[int] = blossomchilds[b]  # type: ignore[assignment]
            for s in childs:
                blossomparent[s] = -1
                if s < n:
                    inblossom[s] = s
                elif endstage and dualvar[s] == 0:
                    todo.append(s)  # expanded later; nothing below depends on the order
                else:
                    for x in leaves(s):
                        inblossom[x] = s
            if not endstage and label[b] == 2:
                # Mid-stage expansion of a T-blossom: relabel the sub-blossoms on the
                # even-length path from the entry child to the base.
                endps: list[int] = blossomendps[b]  # type: ignore[assignment]
                entrychild = inblossom[endpoint[labelend[b] ^ 1]]
                j = childs.index(entrychild)
                if j & 1:
                    j -= len(childs)
                    jstep, endptrick = 1, 0
                else:
                    jstep, endptrick = -1, 1
                p = labelend[b]
                while j != 0:
                    label[endpoint[p ^ 1]] = 0
                    label[endpoint[endps[j - endptrick] ^ endptrick ^ 1]] = 0
                    assign_label(endpoint[p ^ 1], 2, p)
                    allowedge[endps[j - endptrick] >> 1] = True
                    j += jstep
                    p = endps[j - endptrick] ^ endptrick
                    allowedge[p >> 1] = True
                    j += jstep
                bv = childs[j]
                label[endpoint[p ^ 1]] = label[bv] = 2
                labelend[endpoint[p ^ 1]] = labelend[bv] = p
                bestedge[bv] = -1
                j += jstep
                while childs[j] != entrychild:
                    bv = childs[j]
                    if label[bv] == 1:  # got S through a neighbor meanwhile
                        j += jstep
                        continue
                    reached = next((x for x in leaves(bv) if label[x] != 0), -1)
                    if reached != -1:
                        label[reached] = 0
                        label[endpoint[mate[blossombase[bv]]]] = 0
                        assign_label(reached, 2, labelend[reached])
                    j += jstep
            label[b] = labelend[b] = -1
            blossomchilds[b] = blossomendps[b] = None
            blossombase[b] = -1
            blossombestedges[b] = None
            bestedge[b] = -1
            unusedblossoms.append(b)

    def augment_blossom(b0: int, v0: int) -> None:
        """Swap matched/unmatched edges on the alternating path from v0 to the base of b0."""
        todo = [(b0, v0)]
        while todo:
            b, v = todo.pop()
            t = v
            while blossomparent[t] != b:
                t = blossomparent[t]
            if t >= n:
                todo.append((t, v))
            childs: list[int] = blossomchilds[b]  # type: ignore[assignment]
            endps: list[int] = blossomendps[b]  # type: ignore[assignment]
            i = j = childs.index(t)
            if i & 1:
                j -= len(childs)
                jstep, endptrick = 1, 0
            else:
                jstep, endptrick = -1, 1
            while j != 0:
                j += jstep
                t = childs[j]
                p = endps[j - endptrick] ^ endptrick
                if t >= n:
                    todo.append((t, endpoint[p]))
                j += jstep
                t = childs[j]
                if t >= n:
                    todo.append((t, endpoint[p ^ 1]))
                mate[endpoint[p]] = p ^ 1
                mate[endpoint[p ^ 1]] = p
            # Rotate so the sub-blossom holding v comes first: v is the new base.
            blossomchilds[b] = childs[i:] + childs[:i]
            blossomendps[b] = endps[i:] + endps[:i]
            blossombase[b] = v

    def augment_matching(k: int) -> None:
        v, w = eu[k], ev[k]
        for s, p in ((v, 2 * k + 1), (w, 2 * k)):
            while True:
                bs = inblossom[s]
                if bs >= n:
                    augment_blossom(bs, s)
                mate[s] = p
                if labelend[bs] == -1:
                    break  # reached a single vertex
                t = endpoint[labelend[bs]]
                bt = inblossom[t]
                s = endpoint[labelend[bt]]
                j = endpoint[labelend[bt] ^ 1]
                if bt >= n:
                    augment_blossom(bt, j)
                mate[j] = labelend[bt]
                p = labelend[bt] ^ 1

    while True:  # one stage per augmentation
        label[:] = [0] * (2 * n)
        bestedge[:] = [-1] * (2 * n)
        blossombestedges[n:] = [None] * n
        allowedge[:] = [False] * nedge
        queue.clear()
        for v in range(n):
            if mate[v] == -1 and label[inblossom[v]] == 0:
                assign_label(v, 1, -1)
        augmented = False
        while True:  # substages: grow the forest, then adjust duals
            while queue and not augmented:
                v = queue.pop()
                for p in neighbend[v]:
                    k = p >> 1
                    w = endpoint[p]
                    if inblossom[v] == inblossom[w]:
                        continue
                    if not allowedge[k]:
                        kslack = slack(k)
                        if kslack <= 0:
                            allowedge[k] = True
                    if allowedge[k]:
                        if label[inblossom[w]] == 0:
                            assign_label(w, 2, p ^ 1)
                        elif label[inblossom[w]] == 1:
                            base = scan_blossom(v, w)
                            if base >= 0:
                                add_blossom(base, k)
                            else:
                                augment_matching(k)
                                augmented = True
                                break
                        elif label[w] == 0:
                            # w sits in a T-blossom but was not reached yet; remember how.
                            label[w] = 2
                            labelend[w] = p ^ 1
                    elif label[inblossom[w]] == 1:
                        b = inblossom[v]
                        if bestedge[b] == -1 or kslack < slack(bestedge[b]):
                            bestedge[b] = k
                    elif label[w] == 0:
                        if bestedge[w] == -1 or kslack < slack(bestedge[w]):
                            bestedge[w] = k
            if augmented:
                break
            deltatype = -1
            delta: Any = None
            deltaedge = deltablossom = -1
            if not maxcardinality:  # delta1: smallest vertex dual
                deltatype = 1
                delta = min(dualvar[:n])
            for v in range(n):  # delta2: S-vertex to free vertex
                if label[inblossom[v]] == 0 and bestedge[v] != -1:
                    d = slack(bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta, deltatype, deltaedge = d, 2, bestedge[v]
            for b in range(2 * n):  # delta3: half the slack between two S-blossoms
                if blossomparent[b] == -1 and label[b] == 1 and bestedge[b] != -1:
                    kslack = slack(bestedge[b])
                    d = kslack // 2 if integral else kslack / 2
                    if deltatype == -1 or d < delta:
                        delta, deltatype, deltaedge = d, 3, bestedge[b]
            for b in range(n, 2 * n):  # delta4: smallest dual of a T-blossom
                if blossombase[b] >= 0 and blossomparent[b] == -1 and label[b] == 2 and (deltatype == -1 or dualvar[b] < delta):
                    delta, deltatype, deltablossom = dualvar[b], 4, b
            if deltatype == -1:
                # Max-cardinality optimum reached; a final shift keeps duals valid.
                deltatype = 1
                delta = max(0, min(dualvar[:n]))
            for v in range(n):
                lb = label[inblossom[v]]
                if lb == 1:
                    dualvar[v] -= delta
                elif lb == 2:
                    dualvar[v] += delta
            for b in range(n, 2 * n):
                if blossombase[b] >= 0 and blossomparent[b] == -1:
                    if label[b] == 1:
                        dualvar[b] += delta
                    elif label[b] == 2:
                        dualvar[b] -= delta
            if deltatype == 1:
                break
            if deltatype == 2:
                allowedge[deltaedge] = True
                i, j = eu[deltaedge], ev[deltaedge]
                if label[inblossom[i]] == 0:
                    i = j
                queue.append(i)
            elif deltatype == 3:
                allowedge[deltaedge] = True
                queue.append(eu[deltaedge])
            else:
                expand_blossom(deltablossom, False)
        if not augmented:
            break
        # End of stage: dissolve S-blossoms whose dual dropped to zero.
        for b in range(n, 2 * n):
            if blossomparent[b] == -1 and blossombase[b] >= 0 and label[b] == 1 and dualvar[b] == 0:
                expand_blossom(b, True)
    return [endpoint[m] if m >= 0 else -1 for m in mate]


__all__ = [
    "hopcroft_karp",
    "bipartite_maximum_matching",
    "maximal_matching",
    "is_matching",
    "is_maximal_matching",
    "is_perfect_matching",
    "max_weight_matching",
    "min_weight_matching",
]
