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

"""Structural statistics: degrees, clustering, assortativity, distances, cores.

Definitions and normalisations follow networkx; the few deliberate
differences (mostly: sensible values instead of exceptions on the null graph)
are noted per function. Distance measures (eccentricity, diameter, average
path length, …) raise :class:`~aryagraph.core.exceptions.NotConnected` when some
pair of nodes has no path, since the value would be infinite.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from itertools import accumulate, chain
from typing import Any, Hashable

import numpy as np

from ..core.exceptions import ConvergenceError, GraphTypeError, NotConnected
from ..core.results import NodeMap
from ..core.utils import WeightSpec, make_rng, require_directed, require_undirected, weight_fn
from ._core import bfs_distances, components, dijkstra_distances

Node = Hashable


# ---------------------------------------------------------------------- #
# small helpers
# ---------------------------------------------------------------------- #
def _nbr_sets(adj: dict) -> dict[Node, set]:
    """``{v: neighbors of v without v}`` from an adjacency dict."""
    return {v: set(nbrs) - {v} for v, nbrs in adj.items()}


def _undirected_nbr_sets(g: Any) -> dict[Node, set]:
    """Neighbor sets of the undirected view (self-loops dropped)."""
    if not g.directed:
        return _nbr_sets(g._succ)
    return {v: (set(g._succ[v]) | set(g._pred[v])) - {v} for v in g._node}


def _degrees(g: Any, weight: WeightSpec, which: str = "total") -> dict[Node, float]:
    """(Weighted) degrees as networkx counts them: an undirected self-loop counts twice.

    *which* ∈ {"in", "out", "total"} matters only for directed graphs.
    """
    wf = weight_fn(weight)
    out: dict[Node, float] = {}
    for v in g._node:
        succ = g._succ[v]
        if not g.directed:
            s = sum(wf(v, u, d) for u, d in succ.items())
            out[v] = s + wf(v, v, succ[v]) if v in succ else s
            continue
        s_in = sum(wf(u, v, d) for u, d in g._pred[v].items()) if which != "out" else 0
        s_out = sum(wf(v, u, d) for u, d in succ.items()) if which != "in" else 0
        out[v] = s_in + s_out
    return out


def _distances(g: Any, source: Node, weight: WeightSpec) -> dict[Node, float]:
    if weight is None:
        return bfs_distances(g, source)
    return dijkstra_distances(g, source, weight)


def _not_connected(g: Any, what: str) -> NotConnected:
    kind = "strongly connected" if g.directed else "connected"
    return NotConnected(f"{what} is undefined: the graph is not {kind} (some distances are infinite)")


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of paired samples; NaN when either variance is 0 or there are no pairs."""
    if not xs:
        return math.nan
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    dx = x - x.mean()
    dy = y - y.mean()
    denom = math.sqrt(float((dx * dx).mean()) * float((dy * dy).mean()))
    if denom == 0:
        return math.nan
    return float((dx * dy).mean()) / denom


# ---------------------------------------------------------------------- #
# density and degrees
# ---------------------------------------------------------------------- #
def density(g: Any) -> float:
    """Edges present over edges possible: ``2m / n(n-1)`` (undirected) or ``m / n(n-1)``.

    Self-loops count in ``m``, as in networkx, so the value can exceed 1.
    """
    n, m = len(g), g.num_edges
    if m == 0 or n <= 1:
        return 0.0
    d = m / (n * (n - 1))
    return 2 * d if not g.directed else d


def degree_histogram(g: Any) -> list[int]:
    """``hist[k]`` = number of nodes of degree ``k``, for ``k = 0 … max degree``."""
    counts = Counter(g.degree().values())
    return [counts.get(k, 0) for k in range(max(counts) + 1 if counts else 0)]


def degree_distribution(g: Any) -> dict[int, int]:
    """``{degree: number of nodes}`` for the degrees that occur, ascending."""
    counts = Counter(g.degree().values())
    return {k: counts[k] for k in sorted(counts)}


def average_degree(g: Any) -> float:
    """Mean of :meth:`Graph.degree` (``2m / n``; in + out for directed graphs), 0 for the null graph."""
    n = len(g)
    return float(sum(g.degree().values())) / n if n else 0.0


def reciprocity(g: Any) -> float:
    """Share of arcs whose reverse arc also exists (directed graphs; self-loops never count).

    Raises ``ValueError`` for a graph without arcs, where the ratio is undefined.
    """
    require_directed(g, "reciprocity")
    m = g.num_edges
    if m == 0:
        raise ValueError("reciprocity is undefined for a graph without edges")
    mutual = sum(1 for u, nbrs in g._succ.items() for v in nbrs if u != v and u in g._succ[v])
    return mutual / m


# ---------------------------------------------------------------------- #
# triangles and clustering
# ---------------------------------------------------------------------- #
def _triangle_counts(nbrs: dict[Node, set]) -> dict[Node, int]:
    """Triangles through each node, each triangle counted once per corner (``O(m·d_max)``)."""
    later: dict[Node, set] = {}
    for u, ns in nbrs.items():
        later[u] = {v for v in ns if v not in later}
    counts = dict.fromkeys(nbrs, 0)
    for u, ns in later.items():
        for v in ns:
            third = ns & later[v]
            k = len(third)
            if k:
                counts[u] += k
                counts[v] += k
                for w in third:
                    counts[w] += 1
    return counts


def triangles(g: Any) -> NodeMap:
    """Number of triangles through each node (undirected graphs; self-loops ignored)."""
    require_undirected(g, "triangles")
    return NodeMap(_triangle_counts(_nbr_sets(g._succ)), name="triangles")


def _unweighted_clustering(nbrs: dict[Node, set]) -> dict[Node, float]:
    tri = _triangle_counts(nbrs)
    out: dict[Node, float] = {}
    for v, ns in nbrs.items():
        d = len(ns)
        out[v] = 2 * tri[v] / (d * (d - 1)) if tri[v] else 0.0
    return out


def clustering(g: Any, weight: WeightSpec = None) -> NodeMap:
    """Local clustering coefficient of every node.

    * undirected, unweighted: ``2T(v) / d(d-1)``;
    * undirected, weighted: geometric mean of the (max-normalised) triangle
      edge weights (Onnela et al.);
    * directed: Fagiolo's count of directed triangles over
      ``2·(d_tot(d_tot-1) - 2 d_↔)``, weighted likewise when *weight* is given.

    Self-loops are ignored, exactly as in networkx.
    """
    if g.directed:
        values = _directed_clustering(g, weight)
    elif weight is None:
        values = _unweighted_clustering(_nbr_sets(g._succ))
    else:
        values = _weighted_clustering(g, weight)
    return NodeMap(values, name="clustering")


def _max_weight(g: Any, wf: Any) -> float:
    ws = [wf(u, v, d) for u, v, d in g._iter_edges()]
    return max(ws) if ws else 1


def _weighted_clustering(g: Any, weight: WeightSpec) -> dict[Node, float]:
    wf = weight_fn(weight)
    wmax = _max_weight(g, wf)
    succ = g._succ

    def wt(u: Node, v: Node) -> float:
        return wf(u, v, succ[u][v]) / wmax

    nbrs = _nbr_sets(succ)
    out: dict[Node, float] = {}
    for i, inbrs in nbrs.items():
        prods: list[float] = []
        seen: set = set()
        for j in inbrs:
            seen.add(j)
            wij = wt(i, j)
            prods.extend(wij * wt(j, k) * wt(k, i) for k in inbrs & (nbrs[j] - seen))
        total = 2 * float(np.cbrt(prods).sum()) if prods else 0.0
        d = len(inbrs)
        out[i] = total / (d * (d - 1)) if total != 0 else 0.0
    return out


def _directed_clustering(g: Any, weight: WeightSpec) -> dict[Node, float]:
    preds = _nbr_sets(g._pred)
    succs = _nbr_sets(g._succ)
    wt = None
    if weight is not None:
        wf = weight_fn(weight)
        wmax = _max_weight(g, wf)
        succ = g._succ

        def wt(u: Node, v: Node) -> float:
            return wf(u, v, succ[u][v]) / wmax

    out: dict[Node, float] = {}
    for i in g._node:
        ip, is_ = preds[i], succs[i]
        if wt is None:
            t: float = 0
            for j in chain(ip, is_):
                jp, js = preds[j], succs[j]
                t += len(ip & jp) + len(ip & js) + len(is_ & jp) + len(is_ & js)
        else:
            # Fagiolo's weighted count: cube roots of the three arc weights of each
            # directed triangle, over the eight arc orientations.
            prods: list[float] = []
            for j in ip:
                jp, js = preds[j], succs[j]
                wji = wt(j, i)
                prods.extend(wji * wt(k, i) * wt(k, j) for k in ip & jp)
                prods.extend(wji * wt(k, i) * wt(j, k) for k in ip & js)
                prods.extend(wji * wt(i, k) * wt(k, j) for k in is_ & jp)
                prods.extend(wji * wt(i, k) * wt(j, k) for k in is_ & js)
            for j in is_:
                jp, js = preds[j], succs[j]
                wij = wt(i, j)
                prods.extend(wij * wt(k, i) * wt(k, j) for k in ip & jp)
                prods.extend(wij * wt(k, i) * wt(j, k) for k in ip & js)
                prods.extend(wij * wt(i, k) * wt(k, j) for k in is_ & jp)
                prods.extend(wij * wt(i, k) * wt(j, k) for k in is_ & js)
            t = float(np.cbrt(prods).sum()) if prods else 0.0
        dtot = len(ip) + len(is_)
        dbi = len(ip & is_)
        out[i] = t / ((dtot * (dtot - 1) - 2 * dbi) * 2) if t != 0 else 0.0
    return out


def average_clustering(g: Any, weight: WeightSpec = None, count_zeros: bool = True) -> float:
    """Mean of :func:`clustering`; with ``count_zeros=False`` zero-valued nodes are skipped.

    Returns 0.0 when there is nothing to average (networkx raises ``ZeroDivisionError``).
    """
    vals = list(clustering(g, weight).values())
    if not count_zeros:
        vals = [v for v in vals if abs(v) > 0]
    return sum(vals) / len(vals) if vals else 0.0


def transitivity(g: Any) -> float:
    """Global clustering ``3 × triangles / connected triples`` (undirected; self-loops ignored)."""
    require_undirected(g, "transitivity")
    nbrs = _nbr_sets(g._succ)
    tri = _triangle_counts(nbrs)
    closed = 2 * sum(tri.values())
    triads = sum(len(ns) * (len(ns) - 1) for ns in nbrs.values())
    return closed / triads if closed else 0.0


def square_clustering(g: Any) -> NodeMap:
    """Squares clustering ``C4(v)`` (Lind et al.): share of possible squares through *v*.

    Uses the networkx formulation (successor sets for directed graphs; self-loops ignored).
    """
    adj = _nbr_sets(g._succ)
    out: dict[Node, float] = {}
    for v, vn in adj.items():
        dm1 = len(vn) - 1
        if dm1 <= 0:
            out[v] = 0.0
            continue
        uw_degrees = 0
        uw_count = len(vn) * dm1
        tri = 0
        squares = 0
        for u in vn:
            un = adj[u]
            uw_degrees += len(un) * dm1
            p2 = len(un & vn)
            tri += p2
            squares += p2 * (p2 - 1)
        two_hop = set().union(*(adj[u] for u in vn)) - vn
        two_hop.discard(v)
        for x in two_hop:
            p2 = len(vn & adj[x])
            squares += p2 * (p2 - 1)
        squares //= 2
        potential = uw_degrees - uw_count - tri - squares
        out[v] = squares / potential if potential > 0 else 0.0
    return NodeMap(out, name="square_clustering")


# ---------------------------------------------------------------------- #
# assortativity
# ---------------------------------------------------------------------- #
def degree_assortativity(g: Any, x: str = "out", y: str = "in", weight: WeightSpec = None) -> float:
    """Degree assortativity: Pearson correlation of the degrees at either end of each edge.

    For directed graphs *x* ∈ {"in", "out"} picks the degree of the tail and *y*
    that of the head (networkx's ``degree_assortativity_coefficient``);
    undirected graphs use each edge in both directions. *weight* uses weighted
    degrees. Returns NaN when a degree variance is 0 (e.g. regular graphs) or
    there are no edges.
    """
    if g.directed:
        for side in (x, y):
            if side not in ("in", "out"):
                raise ValueError(f"x and y must be 'in' or 'out', got {side!r}")
        xdeg = _degrees(g, weight, x)
        ydeg = _degrees(g, weight, y)
    else:
        xdeg = ydeg = _degrees(g, weight)
    xs: list[float] = []
    ys: list[float] = []
    for u, nbrs in g._succ.items():
        du = xdeg[u]
        for v in nbrs:
            xs.append(du)
            ys.append(ydeg[v])
    return _pearson(xs, ys)


def _attribute_pairs(g: Any, attribute: str, *, required: bool) -> tuple[list, list]:
    node_attr = g._node
    if required:
        for n, d in node_attr.items():
            if attribute not in d:
                raise ValueError(f"node {n!r} has no attribute {attribute!r}")
    xs: list = []
    ys: list = []
    for u, nbrs in g._succ.items():
        au = node_attr[u].get(attribute)
        for v in nbrs:
            xs.append(au)
            ys.append(node_attr[v].get(attribute))
    return xs, ys


def attribute_assortativity(g: Any, attribute: str) -> float:
    """Newman's assortativity for a categorical node attribute (missing values form a category).

    ``r = (Σ e_ii − Σ a_i b_i) / (1 − Σ a_i b_i)`` over the mixing matrix of
    edge-end attribute pairs; NaN when undefined (no edges, or a single category).
    """
    xs, ys = _attribute_pairs(g, attribute, required=False)
    if not xs:
        return math.nan
    total = len(xs)
    rows = Counter(xs)
    cols = Counter(ys)
    diag = sum(1 for a, b in zip(xs, ys) if a == b) / total
    s = sum(rows[k] * cols.get(k, 0) for k in rows) / (total * total)
    if s == 1:
        return math.nan
    return (diag - s) / (1 - s)


def numeric_assortativity(g: Any, attribute: str) -> float:
    """Pearson correlation of a numeric node attribute across edges (NaN when undefined).

    Every node must carry *attribute* (``ValueError`` otherwise).
    """
    xs, ys = _attribute_pairs(g, attribute, required=True)
    return _pearson(xs, ys)


# ---------------------------------------------------------------------- #
# distances
# ---------------------------------------------------------------------- #
def eccentricity(g: Any, weight: WeightSpec = None) -> NodeMap:
    """Greatest distance from each node (outgoing distances for directed graphs).

    Raises
    ------
    NotConnected
        If some node cannot reach every other node.
    """
    n = len(g)
    out: dict[Node, float] = {}
    for v in g._node:
        dist = _distances(g, v, weight)
        if len(dist) != n:
            raise _not_connected(g, "eccentricity")
        out[v] = max(dist.values())
    return NodeMap(out, name="eccentricity")


def _checked_eccentricity(g: Any, weight: WeightSpec, what: str) -> NodeMap:
    if len(g) == 0:
        raise NotConnected(f"{what} is undefined for the null graph")
    return eccentricity(g, weight)


def diameter(g: Any, weight: WeightSpec = None) -> float:
    """Largest eccentricity (:class:`NotConnected` if the graph is not (strongly) connected)."""
    return max(_checked_eccentricity(g, weight, "diameter").values())


def radius(g: Any, weight: WeightSpec = None) -> float:
    """Smallest eccentricity (:class:`NotConnected` if the graph is not (strongly) connected)."""
    return min(_checked_eccentricity(g, weight, "radius").values())


def center(g: Any, weight: WeightSpec = None) -> list[Node]:
    """Nodes whose eccentricity equals the radius, in graph order."""
    ecc = _checked_eccentricity(g, weight, "center")
    r = min(ecc.values())
    return [v for v, e in ecc.items() if e == r]


def periphery(g: Any, weight: WeightSpec = None) -> list[Node]:
    """Nodes whose eccentricity equals the diameter, in graph order."""
    ecc = _checked_eccentricity(g, weight, "periphery")
    d = max(ecc.values())
    return [v for v, e in ecc.items() if e == d]


def average_shortest_path_length(g: Any, weight: WeightSpec = None) -> float:
    """Mean distance over all ordered pairs of distinct nodes (0.0 for a single node).

    Raises
    ------
    NotConnected
        For the null graph and for graphs that are not (strongly) connected.
    """
    n = len(g)
    if n == 0:
        raise NotConnected("average_shortest_path_length is undefined for the null graph")
    if n == 1:
        return 0.0
    total = 0.0
    for v in g._node:
        dist = _distances(g, v, weight)
        if len(dist) != n:
            raise _not_connected(g, "average_shortest_path_length")
        total += sum(dist.values())
    return total / (n * (n - 1))


def wiener_index(g: Any, weight: WeightSpec = None) -> float:
    """Sum of distances over all (unordered, or ordered if directed) node pairs.

    Like networkx, a graph that is not (strongly) connected has index ``inf``;
    the null graph has index 0.
    """
    n = len(g)
    total = 0.0
    for v in g._node:
        dist = _distances(g, v, weight)
        if len(dist) != n:
            return math.inf
        total += sum(dist.values())
    return total if g.directed else total / 2


def global_efficiency(g: Any) -> float:
    """Average inverse distance ``Σ 1/d(u, v) / n(n-1)`` (undirected; 0 for n < 2)."""
    require_undirected(g, "global_efficiency")
    return _efficiency(_nbr_sets(g._succ))


def _efficiency(nbrs: dict[Node, set]) -> float:
    n = len(nbrs)
    if n < 2:
        return 0.0
    total = 0.0
    for s in nbrs:
        seen = {s}
        frontier = [s]
        level = 0
        while frontier:
            level += 1
            nxt = []
            for u in frontier:
                for w in nbrs[u]:
                    if w not in seen:
                        seen.add(w)
                        nxt.append(w)
            total += len(nxt) / level
            frontier = nxt
    return total / (n * (n - 1))


def local_efficiency(g: Any) -> float:
    """Mean over nodes of the global efficiency of the subgraph induced by their neighbors.

    Undirected graphs only. A node's own self-loop does not put it into its
    neighborhood (networkx includes it; the Latora–Marchiori definition does not).
    Returns 0.0 for the null graph.
    """
    require_undirected(g, "local_efficiency")
    nbrs = _nbr_sets(g._succ)
    if not nbrs:
        return 0.0
    total = 0.0
    for v, ns in nbrs.items():
        if len(ns) > 1:
            total += _efficiency({u: nbrs[u] & ns for u in ns})
    return total / len(nbrs)


# ---------------------------------------------------------------------- #
# cores and rich club
# ---------------------------------------------------------------------- #
def core_number(g: Any) -> NodeMap:
    """k-core number of every node (Batagelj–Zaversnik bucket algorithm, ``O(n + m)``).

    Directed graphs use in-degree + out-degree (a reciprocated pair counts
    twice), as networkx. Self-loops are ignored (they cannot hold a node in a
    core), whereas networkx refuses graphs that have them.
    """
    nodes = list(g._node)
    index = {v: i for i, v in enumerate(nodes)}
    succ, pred = g._succ, g._pred
    if g.directed:
        nbr_idx = [[index[u] for u in chain(pred[v], succ[v]) if u != v] for v in nodes]
    else:
        nbr_idx = [[index[u] for u in succ[v] if u != v] for v in nodes]
    deg = [len(ns) for ns in nbr_idx]
    n = len(nodes)
    max_deg = max(deg, default=0)
    # counting sort of nodes by degree; bin_start[d] = first slot of degree d
    bin_start = [0] * (max_deg + 2)
    for d in deg:
        bin_start[d + 1] += 1
    bin_start = list(accumulate(bin_start))
    fill = bin_start[:]
    vert = [0] * n
    pos = [0] * n
    for i, d in enumerate(deg):
        pos[i] = fill[d]
        vert[fill[d]] = i
        fill[d] += 1
    for k in range(n):
        v = vert[k]
        dv = deg[v]
        for u in nbr_idx[v]:
            du = deg[u]
            if du > dv:
                # move u to the front of its bucket, then shrink the bucket
                pu, pw = pos[u], bin_start[du]
                w = vert[pw]
                if u != w:
                    vert[pu], vert[pw] = w, u
                    pos[u], pos[w] = pw, pu
                bin_start[du] += 1
                deg[u] = du - 1
    return NodeMap(((v, deg[i]) for i, v in enumerate(nodes)), name="core_number")


def k_core(g: Any, k: int | None = None) -> Any:
    """Subgraph induced by the nodes of core number ≥ *k* (``None``: the innermost core)."""
    core = core_number(g)
    if k is None:
        k = max(core.values(), default=0)
    return g.subgraph(v for v, c in core.items() if c >= k)


def onion_layers(g: Any) -> NodeMap:
    """Onion decomposition layer of every node (Hébert-Dufresne et al.; undirected).

    Isolated nodes form layer 1. Self-loops are ignored (networkx refuses them).
    """
    require_undirected(g, "onion_layers")
    nbrs = _nbr_sets(g._succ)
    layer: dict[Node, int] = {}
    current_layer = 1
    isolated = [v for v, ns in nbrs.items() if not ns]
    if isolated:
        for v in isolated:
            layer[v] = 1
        current_layer = 2
    remaining = {v: len(ns) for v, ns in nbrs.items() if ns}
    current_core = 1
    while remaining:
        current_core = max(current_core, min(remaining.values()))
        this_layer = [v for v, d in remaining.items() if d <= current_core]
        for v in this_layer:
            layer[v] = current_layer
            for u in nbrs[v]:
                nbrs[u].discard(v)
                if u in remaining:
                    remaining[u] -= 1
            del remaining[v]
        current_layer += 1
    return NodeMap(((v, layer[v]) for v in g._node), name="onion_layers")


def rich_club_coefficient(g: Any, normalized: bool = False, Q: int = 100, seed: Any = None) -> dict[int, float]:
    """Rich-club coefficient ``φ(k) = 2 E_k / N_k (N_k - 1)`` among the nodes of degree > k.

    Undirected graphs without self-loops; keys run over ``k`` while ``N_k > 1``.
    With *normalized*, each value is divided by that of a degree-preserving
    randomisation made with ``Q · m`` double-edge swaps seeded by *seed* (the
    swap sequence differs from networkx's, so normalised values agree only
    statistically). A 0/0 ratio is NaN and ``x/0`` is ``inf``.
    """
    require_undirected(g, "rich_club_coefficient")
    if g.selfloops():
        raise GraphTypeError("rich_club_coefficient is not defined for graphs with self-loops")
    adj = {v: dict.fromkeys(nbrs) for v, nbrs in g._succ.items()}
    rc = _rich_club(adj)
    if normalized:
        m = g.num_edges
        rc_rand = _rich_club(_double_edge_swap(adj, Q * m, Q * m * 10, make_rng(seed)))
        rc = {k: _ratio(v, rc_rand.get(k, 0.0)) for k, v in rc.items()}
    return rc


def _ratio(a: float, b: float) -> float:
    if b:
        return a / b
    return math.nan if a == 0 else math.inf


def _rich_club(adj: dict[Node, dict]) -> dict[int, float]:
    """Unnormalised rich-club coefficients of a simple undirected adjacency."""
    deg = {v: len(ns) for v, ns in adj.items()}
    order = {v: i for i, v in enumerate(adj)}
    edge_min = sorted(min(deg[u], deg[v]) for u in adj for v in adj[u] if order[u] < order[v])
    if not edge_min:
        return {}
    hist = Counter(deg.values())
    total = len(adj)
    rc: dict[int, float] = {}
    below = 0
    for d in range(max(hist) + 1):
        below += hist.get(d, 0)
        nk = total - below  # nodes with degree > d
        if nk <= 1:
            break
        ek = len(edge_min) - bisect.bisect_right(edge_min, d)  # edges with both ends of degree > d
        rc[d] = 2 * ek / (nk * (nk - 1))
    return rc


def _double_edge_swap(adj: dict[Node, dict], nswap: int, max_tries: int, rng: np.random.Generator) -> dict[Node, dict]:
    """Degree-preserving randomisation of a copy of *adj* (networkx's double-edge-swap scheme).

    Pick two distinct nodes ``u, x`` with probability ∝ degree and random
    neighbors ``v`` of u and ``y`` of x; rewire ``u–v, x–y`` into ``u–x, v–y``
    unless that would create a self-loop or a parallel edge. Neighbor dicts
    keep insertion order, so a given seed always gives the same graph.
    """
    adj = {v: dict(ns) for v, ns in adj.items()}
    if len(adj) < 4:
        raise ValueError("a double-edge swap needs at least 4 nodes")
    if sum(len(ns) for ns in adj.values()) < 4:
        raise ValueError("a double-edge swap needs at least 2 edges")
    if nswap > max_tries:
        raise ValueError("number of swaps exceeds the number of tries")
    keys = list(adj)
    deg = np.array([len(adj[k]) for k in keys], dtype=float)
    p = deg / deg.sum()
    swaps = tries = 0
    while swaps < nswap:
        tries += 1
        if tries > max_tries:
            raise ConvergenceError(f"double-edge swap gave up after {max_tries} attempts ({swaps} of {nswap} swaps done)")
        ui, xi = rng.choice(len(keys), size=2, p=p)
        if ui == xi:
            continue
        u, x = keys[ui], keys[xi]
        un, xn = list(adj[u]), list(adj[x])
        v = un[int(rng.integers(len(un)))]
        y = xn[int(rng.integers(len(xn)))]
        if v == y or x in adj[u] or y in adj[v]:
            continue
        del adj[u][v], adj[v][u], adj[x][y], adj[y][x]
        adj[u][x] = adj[x][u] = adj[v][y] = adj[y][v] = None
        swaps += 1
    return adj


# ---------------------------------------------------------------------- #
# global shape
# ---------------------------------------------------------------------- #
def is_forest(g: Any) -> bool:
    """True when the (underlying undirected) graph has no cycle: every component has ``size - 1`` edges.

    Reciprocal arcs and self-loops are cycles. The null graph is a forest
    (networkx raises instead).
    """
    comps = components(g)
    comp_of = {v: i for i, c in enumerate(comps) for v in c}
    edges = [0] * len(comps)
    for u, v, _ in g._iter_edges():
        edges[comp_of[u]] += 1
    return all(len(c) - 1 == e for c, e in zip(comps, edges))


def is_tree(g: Any) -> bool:
    """True for a connected forest. The null graph is not a tree (networkx raises instead)."""
    n = len(g)
    return n > 0 and g.num_edges == n - 1 and len(components(g)) == 1


def is_regular(g: Any) -> bool:
    """True when all degrees agree (in- and out-degrees separately for directed graphs).

    The null graph is regular, vacuously (networkx raises instead).
    """
    if g.directed:
        return len({len(p) for p in g._pred.values()}) <= 1 and len({len(s) for s in g._succ.values()}) <= 1
    return len(set(g.degree().values())) <= 1


def s_metric(g: Any) -> float:
    """``Σ deg(u)·deg(v)`` over the edges (Li et al.)."""
    deg = g.degree()
    return float(sum(deg[u] * deg[v] for u, v, _ in g._iter_edges()))


def summary(g: Any) -> dict[str, Any]:
    """Cheap headline statistics; never raises.

    Keys: ``n``, ``m``, ``directed``, ``density``, ``self_loops``,
    ``avg_degree``, ``max_degree``, ``isolates``, ``components`` (weak, for
    directed graphs), ``largest_component``, ``avg_clustering`` (of the
    undirected view), ``assortativity`` (degree; NaN when undefined), and for
    directed graphs also ``reciprocity`` and ``is_dag``.
    """
    n, m = len(g), g.num_edges
    deg = g.degree()
    comps = components(g) if n else []
    out: dict[str, Any] = {
        "n": n,
        "m": m,
        "directed": bool(g.directed),
        "density": density(g),
        "self_loops": len(g.selfloops()),
        "avg_degree": average_degree(g),
        "max_degree": max(deg.values(), default=0),
        "isolates": sum(1 for v in g._node if not g._succ[v] and not g._pred[v]),
        "components": len(comps),
        "largest_component": max((len(c) for c in comps), default=0),
    }
    clust = _unweighted_clustering(_undirected_nbr_sets(g))
    out["avg_clustering"] = sum(clust.values()) / n if n else 0.0
    out["assortativity"] = degree_assortativity(g)
    if g.directed:
        out["reciprocity"] = reciprocity(g) if m else 0.0
        out["is_dag"] = bool(g.is_dag())
    return out


__all__ = [
    "density",
    "degree_histogram",
    "degree_distribution",
    "average_degree",
    "reciprocity",
    "triangles",
    "clustering",
    "average_clustering",
    "transitivity",
    "square_clustering",
    "degree_assortativity",
    "attribute_assortativity",
    "numeric_assortativity",
    "eccentricity",
    "diameter",
    "radius",
    "center",
    "periphery",
    "average_shortest_path_length",
    "wiener_index",
    "global_efficiency",
    "local_efficiency",
    "core_number",
    "k_core",
    "onion_layers",
    "rich_club_coefficient",
    "is_tree",
    "is_forest",
    "is_regular",
    "s_metric",
    "summary",
]
