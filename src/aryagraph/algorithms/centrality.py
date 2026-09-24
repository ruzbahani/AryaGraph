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

"""Node and edge centrality.

Every measure follows the networkx definition and normalisation so results can
be cross-checked; deliberate differences are spelled out in the docstrings.
Spectral measures (eigenvector, Katz, PageRank, HITS) run a vectorised power
iteration over the arc list (``O(m)`` per step, never an ``n × n`` matrix)
and raise :class:`~aryagraph.core.exceptions.ConvergenceError` when the budget runs out.
"""

from __future__ import annotations

import heapq
import inspect
import math
from collections import deque
from collections.abc import Mapping
from itertools import count
from typing import Any, Callable, Hashable, Iterable

import numpy as np

from ..core.exceptions import ConvergenceError, NegativeWeightError, NodeNotFound
from ..core.results import EdgeMap, NodeMap
from ..core.utils import WeightSpec, make_rng, require_directed, weight_fn
from ._core import bfs_distances, dijkstra_distances

Node = Hashable


# ---------------------------------------------------------------------- #
# degree
# ---------------------------------------------------------------------- #
def degree_centrality(g: Any) -> NodeMap:
    """Degree divided by ``n - 1`` (a self-loop adds 2, as in :meth:`Graph.degree`).

    Graphs with at most one node give every node 1.0, as networkx does.
    """
    n = len(g)
    if n <= 1:
        return NodeMap(((v, 1.0) for v in g._node), name="degree_centrality")
    s = 1.0 / (n - 1.0)
    return NodeMap(((v, d * s) for v, d in g.degree().items()), name="degree_centrality")


def in_degree_centrality(g: Any) -> NodeMap:
    """In-degree divided by ``n - 1`` (directed graphs only)."""
    require_directed(g, "in_degree_centrality")
    n = len(g)
    if n <= 1:
        return NodeMap(((v, 1.0) for v in g._node), name="in_degree_centrality")
    s = 1.0 / (n - 1.0)
    return NodeMap(((v, len(g._pred[v]) * s) for v in g._node), name="in_degree_centrality")


def out_degree_centrality(g: Any) -> NodeMap:
    """Out-degree divided by ``n - 1`` (directed graphs only)."""
    require_directed(g, "out_degree_centrality")
    n = len(g)
    if n <= 1:
        return NodeMap(((v, 1.0) for v in g._node), name="out_degree_centrality")
    s = 1.0 / (n - 1.0)
    return NodeMap(((v, len(g._succ[v]) * s) for v in g._node), name="out_degree_centrality")


# ---------------------------------------------------------------------- #
# distance based
# ---------------------------------------------------------------------- #
def _incoming_distances(g: Any, v: Node, weight: WeightSpec) -> dict[Node, float]:
    """Distances *to* v (for undirected graphs the direction is irrelevant)."""
    if weight is None:
        return bfs_distances(g, v, reverse=g.directed)
    return dijkstra_distances(g, v, weight, reverse=g.directed)


def closeness_centrality(g: Any, weight: WeightSpec = None, wf_improved: bool = True) -> NodeMap:
    """Closeness ``(r - 1) / Σ d(u, v)`` over the ``r`` nodes that reach *v*.

    For directed graphs the *incoming* distance is used, as in networkx. With
    *wf_improved* (Wasserman–Faust) the score is further scaled by
    ``(r - 1) / (n - 1)`` so nodes in small components are not overrated.
    *weight* is read as an edge length (Dijkstra). Complexity ``O(n·m)``.
    """
    n = len(g)
    out: dict[Node, float] = {}
    for v in g._node:
        dist = _incoming_distances(g, v, weight)
        total = sum(dist.values())
        c = 0.0
        if total > 0.0 and n > 1:
            reach = len(dist) - 1.0
            c = reach / total
            if wf_improved:
                c *= reach / (n - 1)
        out[v] = c
    return NodeMap(out, name="closeness_centrality")


def harmonic_centrality(g: Any, weight: WeightSpec = None) -> NodeMap:
    """Harmonic centrality ``Σ_{u ≠ v} 1 / d(u, v)`` (unnormalised, like networkx).

    Directed graphs use incoming distances; unreachable nodes contribute 0 and
    zero-length paths (zero-weight edges) are skipped.
    """
    out: dict[Node, float] = {}
    for v in g._node:
        dist = _incoming_distances(g, v, weight)
        out[v] = float(sum(1.0 / d for d in dist.values() if d != 0))
    return NodeMap(out, name="harmonic_centrality")


# ---------------------------------------------------------------------- #
# betweenness (Brandes)
# ---------------------------------------------------------------------- #
def _sssp_bfs(succ: dict, s: Node) -> tuple[list, dict, dict]:
    """Brandes' forward stage, unweighted: (S, P, sigma), S in BFS order."""
    S: list[Node] = []
    P: dict[Node, list[Node]] = {s: []}
    sigma: dict[Node, float] = {s: 1.0}
    D: dict[Node, int] = {s: 0}
    queue = deque([s])
    while queue:
        v = queue.popleft()
        S.append(v)
        dv = D[v] + 1
        sv = sigma[v]
        for w in succ[v]:
            if w not in D:
                queue.append(w)
                D[w] = dv
                sigma[w] = 0.0
                P[w] = []
            if D[w] == dv:
                sigma[w] += sv
                P[w].append(v)
    return S, P, sigma


def _sssp_dijkstra(succ: dict, s: Node, wf: Callable) -> tuple[list, dict, dict]:
    """Brandes' forward stage, weighted: (S, P, sigma), S by non-decreasing distance.

    Mirrors networkx step for step (predecessor carried in the heap, exact
    float equality for ties) so path counts agree bit for bit.
    """
    S: list[Node] = []
    P: dict[Node, list[Node]] = {s: []}
    sigma: dict[Node, float] = {s: 0.0}
    D: dict[Node, float] = {}
    seen: dict[Node, float] = {s: 0}
    tie = count()
    heap: list = [(0, next(tie), s, s)]
    root = True
    while heap:
        dist, _, pred, v = heapq.heappop(heap)
        if v in D:
            continue
        sigma[v] += 1.0 if root else sigma[pred]
        root = False
        S.append(v)
        D[v] = dist
        for w, attrs in succ[v].items():
            wt = wf(v, w, attrs)
            if wt is None:  # hidden edge
                continue
            if wt < 0:
                raise NegativeWeightError(v, w, wt, "shortest paths are undefined")
            vw = dist + wt
            if w not in D and (w not in seen or vw < seen[w]):
                seen[w] = vw
                heapq.heappush(heap, (vw, next(tie), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw == seen.get(w):
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma


def _sssp(g: Any, s: Node, wf: Callable | None) -> tuple[list, dict, dict]:
    return _sssp_bfs(g._succ, s) if wf is None else _sssp_dijkstra(g._succ, s, wf)


def _rescale(
    values: dict,
    n: int,
    *,
    normalized: bool,
    directed: bool,
    endpoints: bool = True,
    sampled: list | None = None,
) -> dict:
    """networkx's betweenness rescaling, including the ``k``-sampling correction."""
    k = None if sampled is None else len(sampled)
    N = n if endpoints else n - 1
    if N < 2:
        return values
    k_source = N if k is None else k
    if k is None or endpoints:
        if normalized:
            scale = 1 / (k_source * (N - 1))
        else:
            # Paths are counted for ordered (s, t) pairs; undirected graphs count
            # each unordered pair twice.
            scale = N / (k_source * (1 if directed else 2))
        if scale != 1:
            for key in values:
                values[key] *= scale
        return values
    # Sampling without endpoints: a sampled source can never be its own target.
    if normalized:
        scale_source = 1 / ((k_source - 1) * (N - 1)) if k_source > 1 else math.nan
        scale_other = 1 / (k_source * (N - 1))
    else:
        corr = 1 if directed else 2
        scale_source = N / ((k_source - 1) * corr) if k_source > 1 else math.nan
        scale_other = N / (k_source * corr)
    sampled_set = set(sampled)
    for key in values:
        values[key] *= scale_source if key in sampled_set else scale_other
    return values


def _sample_sources(g: Any, k: int | None, seed: Any) -> list[Node] | None:
    """The *k* sampled source nodes (``None`` = all), drawn with :func:`make_rng`."""
    n = len(g)
    if k is None or k == n:
        return None
    if not 0 < k <= n:
        raise ValueError(f"k must be between 1 and the number of nodes ({n}), got {k}")
    nodes = list(g._node)
    picks = make_rng(seed).choice(n, size=k, replace=False)
    return [nodes[i] for i in picks.tolist()]


def betweenness_centrality(
    g: Any,
    normalized: bool = True,
    weight: WeightSpec = None,
    endpoints: bool = False,
    k: int | None = None,
    seed: Any = None,
) -> NodeMap:
    """Shortest-path betweenness (Brandes' algorithm, ``O(n·m)``; ``O(n·m + n² log n)`` weighted).

    Parameters
    ----------
    normalized:
        Divide by the number of ordered source/target pairs that can pass
        through a node (``(n-1)(n-2)``, or ``n(n-1)`` with *endpoints*), exactly
        as networkx. Unnormalised undirected values count each pair once.
    weight:
        Edge length for Dijkstra; ``None`` counts hops.
    endpoints:
        Count the path endpoints as lying on the path.
    k, seed:
        Estimate from *k* sampled sources; the partial sums are rescaled like
        networkx (by ``n / k``, with the source/non-source correction when
        endpoints are excluded). Sources are drawn with ``make_rng(seed)``, so
        the sample itself differs from networkx's for the same seed.
    """
    wf = None if weight is None else weight_fn(weight)
    bc: dict[Node, float] = dict.fromkeys(g._node, 0.0)
    sampled = _sample_sources(g, k, seed)
    for s in g._node if sampled is None else sampled:
        S, P, sigma = _sssp(g, s, wf)
        if endpoints:
            bc[s] += len(S) - 1
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                bc[w] += delta[w] + 1 if endpoints else delta[w]
    _rescale(bc, len(g), normalized=normalized, directed=g.directed, endpoints=endpoints, sampled=sampled)
    return NodeMap(bc, name="betweenness_centrality")


def edge_betweenness_centrality(
    g: Any,
    normalized: bool = True,
    weight: WeightSpec = None,
    k: int | None = None,
    seed: Any = None,
) -> EdgeMap:
    """Betweenness of every edge: the share of shortest paths running through it.

    Normalisation divides by ``n(n-1)`` (directed) or ``n(n-1)/2`` (undirected)
    pairs, as networkx. Keys follow ``g.edges`` order and orientation.
    *k*/*seed* sample sources like :func:`betweenness_centrality`.
    """
    wf = None if weight is None else weight_fn(weight)
    eb: dict[tuple[Node, Node], float] = dict.fromkeys(g.edges, 0.0)
    sampled = _sample_sources(g, k, seed)
    for s in g._node if sampled is None else sampled:
        S, P, sigma = _sssp(g, s, wf)
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                c = sigma[v] * coeff
                if (v, w) in eb:
                    eb[(v, w)] += c
                else:
                    eb[(w, v)] += c
                delta[v] += c
    _rescale(eb, len(g), normalized=normalized, directed=g.directed, sampled=sampled)
    return EdgeMap(eb, name="edge_betweenness_centrality")


# ---------------------------------------------------------------------- #
# spectral measures
# ---------------------------------------------------------------------- #
def _arcs(g: Any, weight: WeightSpec) -> tuple[list[Node], np.ndarray, np.ndarray, np.ndarray]:
    """Arc list ``(nodes, src, dst, w)`` of the adjacency matrix in graph order.

    Undirected edges appear in both directions and self-loops once, matching
    the matrix networkx builds.
    """
    nodes = list(g._node)
    index = {v: i for i, v in enumerate(nodes)}
    wf = weight_fn(weight)
    src: list[int] = []
    dst: list[int] = []
    wts: list[float] = []
    for u, nbrs in g._succ.items():
        iu = index[u]
        for v, d in nbrs.items():
            src.append(iu)
            dst.append(index[v])
            wts.append(float(wf(u, v, d)))
    return (
        nodes,
        np.asarray(src, dtype=np.intp),
        np.asarray(dst, dtype=np.intp),
        np.asarray(wts, dtype=float),
    )


def _push(src: np.ndarray, dst: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """``x ↦ xᵀA`` as a scatter-add: out[dst] += vals (one entry per arc)."""
    return np.bincount(dst, weights=vals, minlength=n).astype(float, copy=False)


def eigenvector_centrality(
    g: Any,
    max_iter: int = 1000,
    tol: float = 1e-6,
    weight: WeightSpec = None,
) -> NodeMap:
    """Eigenvector centrality (left Perron vector, unit Euclidean norm).

    Power iteration on ``A + I`` from the uniform vector, stopping when the
    L1 change is below ``n·tol`` (the networkx scheme), so results agree to
    rounding. Directed graphs score a node by its predecessors.

    Raises
    ------
    ConvergenceError
        If *max_iter* iterations are not enough.
    """
    nodes, src, dst, w = _arcs(g, weight)
    n = len(nodes)
    if n == 0:
        return NodeMap({}, name="eigenvector_centrality")
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        xlast = x
        x = xlast + _push(src, dst, xlast[src] * w, n)
        norm = float(np.sqrt(x @ x)) or 1.0
        x = x / norm
        if float(np.abs(x - xlast).sum()) < n * tol:
            return NodeMap(zip(nodes, x.tolist()), name="eigenvector_centrality")
    raise ConvergenceError(f"eigenvector_centrality did not converge in {max_iter} iterations")


def katz_centrality(
    g: Any,
    alpha: float = 0.1,
    beta: float | Mapping[Node, float] = 1.0,
    weight: WeightSpec = None,
    normalized: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> NodeMap:
    """Katz centrality ``x = α Aᵀx + β`` by fixed-point iteration (as networkx).

    *beta* is a scalar or a per-node mapping covering every node. The iteration
    converges only for ``α < 1/λ_max``; otherwise :class:`ConvergenceError` is
    raised. *normalized* scales the result to unit Euclidean norm.
    """
    nodes, src, dst, w = _arcs(g, weight)
    n = len(nodes)
    if n == 0:
        return NodeMap({}, name="katz_centrality")
    if isinstance(beta, Mapping):
        missing = [v for v in nodes if v not in beta]
        if missing:
            raise ValueError(f"beta has no value for node {missing[0]!r}")
        b = np.array([float(beta[v]) for v in nodes])
    else:
        b = np.full(n, float(beta))
    x = np.zeros(n)
    for _ in range(max_iter):
        xlast = x
        with np.errstate(over="ignore", invalid="ignore"):
            x = alpha * _push(src, dst, xlast[src] * w, n) + b
            err = float(np.abs(x - xlast).sum())
        if not math.isfinite(err):
            break  # diverged: alpha ≥ 1/λ_max
        if err < n * tol:
            if normalized:
                norm = float(np.sqrt(x @ x))
                if norm:
                    x = x / norm
            return NodeMap(zip(nodes, x.tolist()), name="katz_centrality")
    raise ConvergenceError(f"katz_centrality did not converge in {max_iter} iterations (is alpha < 1/λ_max?)")


def _node_vector(values: Mapping[Node, float], nodes: list[Node], what: str) -> np.ndarray:
    """Non-negative, normalised-to-sum-1 vector from a node mapping (missing nodes are 0)."""
    index = {v: i for i, v in enumerate(nodes)}
    vec = np.zeros(len(nodes))
    for v, val in values.items():
        i = index.get(v)
        if i is None:
            raise NodeNotFound(v, f"{what} names node {v!r}, which is not in the graph")
        vec[i] = float(val)
    if (vec < 0).any():
        raise ValueError(f"{what} values must be non-negative")
    total = vec.sum()
    if total == 0:
        raise ValueError(f"{what} values sum to zero")
    return vec / total


def pagerank(
    g: Any,
    alpha: float = 0.85,
    personalization: Mapping[Node, float] | None = None,
    weight: WeightSpec = "weight",
    max_iter: int = 100,
    tol: float = 1e-6,
    dangling: Mapping[Node, float] | None = None,
    nstart: Mapping[Node, float] | None = None,
) -> NodeMap:
    """PageRank by vectorised power iteration (``O(m)`` per step).

    Undirected graphs are treated as bidirected. A node whose out-weight is 0
    is *dangling*: its mass is redistributed by the *dangling* vector, which
    defaults to the personalization vector (uniform unless given). This is
    exactly the networkx scheme. Stops when the L1 change is below ``n·tol``.

    Parameters
    ----------
    personalization, dangling, nstart:
        Node → non-negative value mappings, normalised to sum 1; nodes left out
        count as 0 (unknown nodes raise :class:`NodeNotFound`).

    Raises
    ------
    ConvergenceError
        If *max_iter* iterations are not enough.

    Examples
    --------
    >>> pr = pagerank(Graph([(1, 2), (2, 3)]))
    >>> round(sum(pr.values()), 12)
    1.0
    """
    nodes, src, dst, w = _arcs(g, weight)
    n = len(nodes)
    if n == 0:
        return NodeMap({}, name="pagerank")
    out_w = np.bincount(src, weights=w, minlength=n).astype(float, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        trans = np.where(out_w[src] != 0, w / out_w[src], 0.0)
    x = np.full(n, 1.0 / n) if nstart is None else _node_vector(nstart, nodes, "nstart")
    p = np.full(n, 1.0 / n) if personalization is None else _node_vector(personalization, nodes, "personalization")
    dw = p if dangling is None else _node_vector(dangling, nodes, "dangling")
    is_dangling = np.flatnonzero(out_w == 0)
    for _ in range(max_iter):
        xlast = x
        x = alpha * (_push(src, dst, xlast[src] * trans, n) + xlast[is_dangling].sum() * dw) + (1 - alpha) * p
        if float(np.abs(x - xlast).sum()) < n * tol:
            return NodeMap(zip(nodes, x.tolist()), name="pagerank")
    raise ConvergenceError(f"pagerank did not converge in {max_iter} iterations")


def hits(
    g: Any,
    max_iter: int = 100,
    tol: float = 1e-8,
    normalized: bool = True,
    weight: WeightSpec = "weight",
) -> tuple[NodeMap, NodeMap]:
    """HITS hub and authority scores ``(hubs, authorities)``.

    Power iteration ``a ← Aᵀh, h ← Aa`` from the uniform vector, rescaling by
    the maximum each step and stopping when the L1 change of *h* drops below
    *tol*. With *normalized* each vector sums to 1; otherwise authorities have
    unit Euclidean norm and ``hubs = A·authorities`` (networkx's convention).
    A graph without edges gives every node the same score.

    Raises
    ------
    ConvergenceError
        If *max_iter* iterations are not enough (e.g. a near-degenerate top
        singular value).
    """
    nodes, src, dst, w = _arcs(g, weight)
    n = len(nodes)
    if n == 0:
        return NodeMap({}, name="hubs"), NodeMap({}, name="authorities")
    if src.size == 0 or not np.any(w):
        v = 1.0 / n if normalized else 1.0 / math.sqrt(n)
        flat = dict.fromkeys(nodes, v)
        return NodeMap(flat, name="hubs"), NodeMap(flat, name="authorities")
    h = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        hlast = h
        a = _push(src, dst, hlast[src] * w, n)
        h = np.bincount(src, weights=a[dst] * w, minlength=n).astype(float, copy=False)
        h = h / h.max()
        a = a / a.max()
        if float(np.abs(h - hlast).sum()) < tol:
            break
    else:
        raise ConvergenceError(f"hits did not converge in {max_iter} iterations")
    if normalized:
        h = h / h.sum()
        a = a / a.sum()
    else:
        a = a / float(np.sqrt(a @ a))
        h = np.bincount(src, weights=a[dst] * w, minlength=n).astype(float, copy=False)
    return NodeMap(zip(nodes, h.tolist()), name="hubs"), NodeMap(zip(nodes, a.tolist()), name="authorities")


# ---------------------------------------------------------------------- #
# several at once
# ---------------------------------------------------------------------- #
_KINDS: dict[str, Callable[..., Any]] = {
    "degree": degree_centrality,
    "in_degree": in_degree_centrality,
    "out_degree": out_degree_centrality,
    "closeness": closeness_centrality,
    "harmonic": harmonic_centrality,
    "betweenness": betweenness_centrality,
    "eigenvector": eigenvector_centrality,
    "katz": katz_centrality,
    "pagerank": pagerank,
    "hubs": hits,
    "authorities": hits,
}


def centralities(
    g: Any,
    kinds: Iterable[str] = ("degree", "betweenness", "closeness", "pagerank", "eigenvector"),
    **options: Mapping[str, Any],
) -> dict[str, NodeMap]:
    """Several centralities in one call: ``{kind: NodeMap}``.

    Kinds: ``degree``, ``in_degree``, ``out_degree``, ``closeness``,
    ``harmonic``, ``betweenness``, ``eigenvector``, ``katz``, ``pagerank``,
    ``hubs``, ``authorities``. Keyword arguments named after a kind pass
    options to that measure, e.g. ``centralities(g, pagerank={"alpha": 0.9})``.
    Iterative measures that fail to converge are left out of the result
    instead of raising.

    >>> sorted(centralities(Graph([(1, 2), (2, 3)]), kinds=("degree", "pagerank")))
    ['degree', 'pagerank']
    """
    kinds = list(kinds)
    unknown = [k for k in (*kinds, *options) if k not in _KINDS]
    if unknown:
        raise ValueError(f"unknown centrality {unknown[0]!r}; choose from {sorted(_KINDS)}")
    for kind, opts in options.items():
        if not isinstance(opts, Mapping):
            raise TypeError(f"options for {kind!r} must be a mapping of keyword arguments")
        params = inspect.signature(_KINDS[kind]).parameters
        bad = [p for p in opts if p not in params or p == "g"]
        if bad:
            raise TypeError(f"{_KINDS[kind].__name__}() has no option {bad[0]!r}")
    # hubs and authorities come from one HITS run, configured by either key.
    hits_opts = {**options.get("hubs", {}), **options.get("authorities", {})}
    hits_result: tuple[NodeMap, NodeMap] | None | bool = None
    out: dict[str, NodeMap] = {}
    for kind in kinds:
        try:
            if kind in ("hubs", "authorities"):
                if hits_result is None:
                    hits_result = False  # marks a failed run until it succeeds
                    hits_result = hits(g, **hits_opts)
                if hits_result:
                    out[kind] = hits_result[0] if kind == "hubs" else hits_result[1]
            else:
                out[kind] = _KINDS[kind](g, **options.get(kind, {}))
        except ConvergenceError:
            continue
    return out


__all__ = [
    "degree_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
    "closeness_centrality",
    "harmonic_centrality",
    "betweenness_centrality",
    "edge_betweenness_centrality",
    "eigenvector_centrality",
    "katz_centrality",
    "pagerank",
    "hits",
    "centralities",
]
