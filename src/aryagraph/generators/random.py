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

"""Random graph models.

Every generator takes ``seed`` (an int, a ``numpy.random.Generator`` or
``None``) and draws all randomness through :func:`aryagraph.core.utils.make_rng`,
so a fixed seed always reproduces the same graph. Nodes are the integers
``0 … n-1`` and are inserted in that order; edges are inserted in a canonical
(sorted) order where the model allows it.

The models follow the networkx definitions (same parameters, same edge-count
formulas); the random streams differ, so equal seeds do not give equal graphs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.dag import DAG
from ..core.exceptions import ConvergenceError
from ..core.graph import Graph
from ..core.utils import make_rng
from .classic import _count, _new, complete_graph, empty_graph, star_graph

Seed = int | np.random.Generator | None


# ---------------------------------------------------------------------- #
# sampling primitives
# ---------------------------------------------------------------------- #
def _check_prob(p: Any, what: str = "p") -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{what} must be a probability in [0, 1], got {p}")
    return p


def _bernoulli_indices(rng: np.random.Generator, total: int, p: float) -> np.ndarray:
    """Sorted indices of ``range(total)``, each kept independently with probability *p*.

    Draws the geometric gaps between successes (Batagelj & Brandes 2005), so the
    cost is O(expected successes), not O(total).
    """
    if total <= 0 or p <= 0.0:
        return np.empty(0, dtype=np.int64)
    if p >= 1.0:
        return np.arange(total, dtype=np.int64)
    chunks: list[np.ndarray] = []
    last = -1
    while True:
        expected = (total - 1 - last) * p
        k = int(expected + 4.0 * math.sqrt(expected + 1.0) + 16)
        # Clipping keeps the cumulative sum far from int64 overflow for tiny p.
        gaps = np.minimum(rng.geometric(p, size=k), total + 1)
        pos = last + np.cumsum(gaps)
        if pos[-1] >= total:
            chunks.append(pos[pos < total])
            break
        chunks.append(pos)
        last = int(pos[-1])
    return np.concatenate(chunks)


def _pair_index(k: np.ndarray, loops: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Decode linear indices into unordered pairs ``(w, v)`` with ``w < v``.

    Pairs are numbered column by column: ``k = v(v-1)/2 + w``. With *loops*,
    pairs ``w <= v`` are numbered ``k = v(v+1)/2 + w``.
    """
    k = np.asarray(k, dtype=np.int64)
    v = ((1.0 + np.sqrt(1.0 + 8.0 * k.astype(float))) // 2).astype(np.int64)
    # Correct the float estimate (off by at most one for huge k).
    v -= (v * (v - 1) // 2 > k).astype(np.int64)
    v += ((v + 1) * v // 2 <= k).astype(np.int64)
    w = k - v * (v - 1) // 2
    if loops:
        v = v - 1
    return w, v


def _ordered_index(k: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode linear indices into ordered pairs ``(u, v)``, ``u ≠ v``, of ``n`` nodes (row-major)."""
    k = np.asarray(k, dtype=np.int64)
    u = k // (n - 1)
    j = k % (n - 1)
    return u, j + (j >= u)


def _edges_from(u: np.ndarray, v: np.ndarray, offset_u: int = 0, offset_v: int = 0) -> list[tuple[int, int]]:
    return list(zip((u + offset_u).tolist(), (v + offset_v).tolist()))


def _random_subset(rng: np.random.Generator, pool: list[int], m: int) -> list[int]:
    """*m* distinct elements of *pool*, each draw uniform over the pool (so ∝ multiplicity)."""
    chosen: dict[int, None] = {}
    while len(chosen) < m:
        for i in rng.integers(0, len(pool), size=2 * (m - len(chosen))).tolist():
            chosen.setdefault(pool[i])
            if len(chosen) == m:
                break
    return list(chosen)


# ---------------------------------------------------------------------- #
# Erdős–Rényi family
# ---------------------------------------------------------------------- #
def erdos_renyi(n: int, p: float, *, directed: bool = False, seed: Seed = None) -> Graph:
    """``G(n, p)``: every possible edge (arc, if *directed*) is present independently with probability *p*.

    Uses geometric skipping, so the cost is O(n + m) rather than O(n²) for
    sparse graphs. No self-loops. The expected edge count is ``p·n(n-1)/2``
    (``p·n(n-1)`` directed).

    >>> g = erdos_renyi(1000, 0.005, seed=1)
    """
    n, p = _count(n), _check_prob(p)
    rng = make_rng(seed)
    g = _new(directed, f"erdos_renyi({n}, {p:g})")
    g.add_nodes(range(n))
    if n < 2:
        return g
    total = n * (n - 1) if directed else n * (n - 1) // 2
    k = _bernoulli_indices(rng, total, p)
    u, v = _ordered_index(k, n) if directed else _pair_index(k)
    g.add_edges(_edges_from(u, v))
    return g


def gnm_random_graph(n: int, m: int, *, directed: bool = False, seed: Seed = None) -> Graph:
    """``G(n, m)``: a graph chosen uniformly among those with ``n`` nodes and exactly ``m`` edges.

    Raises ``ValueError`` when ``m`` exceeds the number of possible edges
    (``n(n-1)/2``, or ``n(n-1)`` directed). No self-loops.
    """
    n, m = _count(n), _count(m, "m")
    rng = make_rng(seed)
    total = n * (n - 1) if directed else n * (n - 1) // 2
    if m > total:
        raise ValueError(f"a {'directed ' if directed else ''}graph on {n} nodes has at most {total} edges, got m={m}")
    g = _new(directed, f"gnm_random_graph({n}, {m})")
    g.add_nodes(range(n))
    if m:
        k = np.sort(rng.choice(total, size=m, replace=False))
        u, v = _ordered_index(k, n) if directed else _pair_index(k)
        g.add_edges(_edges_from(u, v))
    return g


# ---------------------------------------------------------------------- #
# growth models
# ---------------------------------------------------------------------- #
def barabasi_albert(n: int, m: int, *, seed: Seed = None) -> Graph:
    """Barabási–Albert preferential attachment.

    Starts from ``star_graph(m)`` (``m + 1`` nodes, as networkx does); each new
    node links to ``m`` distinct existing nodes chosen with probability
    proportional to their degree. The result has ``n`` nodes and exactly
    ``m·(n - m)`` edges. Requires ``1 <= m < n``.
    """
    n, m = _count(n), _count(m, "m")
    if not 1 <= m < n:
        raise ValueError(f"barabasi_albert needs 1 <= m < n, got n={n}, m={m}")
    rng = make_rng(seed)
    g = star_graph(m)
    g.name = f"barabasi_albert({n}, {m})"
    g.add_nodes(range(m + 1, n))
    # Every node appears once per incident edge: uniform draws are degree-proportional.
    pool = [0] * m + list(range(1, m + 1))
    for source in range(m + 1, n):
        targets = _random_subset(rng, pool, m)
        g.add_edges((source, t) for t in targets)
        pool.extend(targets)
        pool.extend([source] * m)
    return g


def powerlaw_cluster(n: int, m: int, p: float, *, seed: Seed = None) -> Graph:
    """Holme–Kim model: preferential attachment plus triad formation.

    Like :func:`barabasi_albert`, but after each preferential link the new node
    closes a triangle with probability *p* (links to a random neighbor of the
    node it just attached to). Starts from ``m`` isolated nodes, as networkx
    does. Each new node gets exactly ``m`` edges, so there are ``m·(n - m)``
    edges (networkx occasionally adds fewer, when a triangle step picks a node
    it later draws again). Requires ``1 <= m < n`` and ``0 <= p <= 1``.
    """
    n, m, p = _count(n), _count(m, "m"), _check_prob(p)
    if not 1 <= m < n:
        raise ValueError(f"powerlaw_cluster needs 1 <= m < n, got n={n}, m={m}")
    rng = make_rng(seed)
    g = Graph(name=f"powerlaw_cluster({n}, {m}, {p:g})")
    g.add_nodes(range(n))
    adj = g._succ
    pool = list(range(m))
    for source in range(m, n):
        linked = adj[source]
        candidates = _random_subset(rng, pool, m)
        target = candidates.pop()
        g.add_edge(source, target)
        pool.append(target)
        while len(linked) < m:
            if rng.random() < p:
                nbrs = [x for x in adj[target] if x != source and x not in linked]
                if nbrs:
                    x = nbrs[int(rng.integers(len(nbrs)))]
                    g.add_edge(source, x)
                    pool.append(x)
                    continue
            while candidates and candidates[-1] in linked:
                candidates.pop()
            if candidates:
                target = candidates.pop()
            else:  # every pre-drawn target got linked by triangle steps: draw afresh
                target = pool[int(rng.integers(len(pool)))]
                while target in linked:
                    target = pool[int(rng.integers(len(pool)))]
            g.add_edge(source, target)
            pool.append(target)
        pool.extend([source] * m)
    return g


def watts_strogatz(n: int, k: int, p: float, *, seed: Seed = None) -> Graph:
    """Watts–Strogatz small world.

    Starts from a ring where each node links to its ``k // 2`` nearest
    neighbors on each side, then rewires the far end of every edge with
    probability *p* to a uniformly chosen node, avoiding self-loops and
    duplicates (networkx's procedure). The edge count ``n·(k // 2)`` is
    preserved. ``k == n`` gives the complete graph; ``k > n`` is an error.
    """
    n, k, p = _count(n), _count(k, "k"), _check_prob(p)
    if k > n:
        raise ValueError(f"watts_strogatz needs k <= n, got n={n}, k={k}")
    if k == n:
        g = complete_graph(n)
        g.name = f"watts_strogatz({n}, {k}, {p:g})"
        return g
    rng = make_rng(seed)
    g = Graph(name=f"watts_strogatz({n}, {k}, {p:g})")
    g.add_nodes(range(n))
    half = k // 2
    for j in range(1, half + 1):
        g.add_edges((u, (u + j) % n) for u in range(n))
    adj = g._succ
    for j in range(1, half + 1):
        for u in range(n):
            v = (u + j) % n
            if rng.random() >= p or len(adj[u]) >= n - 1:
                continue
            w = int(rng.integers(n))
            while w == u or w in adj[u]:
                w = int(rng.integers(n))
            g.remove_edge(u, v)
            g.add_edge(u, w)
    return g


# ---------------------------------------------------------------------- #
# block models
# ---------------------------------------------------------------------- #
def stochastic_block_model(
    sizes: Sequence[int],
    p: Sequence[Sequence[float]] | np.ndarray,
    *,
    directed: bool = False,
    selfloops: bool = False,
    seed: Seed = None,
) -> Graph:
    """Stochastic block model.

    Nodes are numbered block after block and carry ``block`` (the block
    index). A pair in blocks ``(i, j)`` is linked independently with
    probability ``p[i][j]``; *p* must be symmetric unless *directed*. Sampling
    uses geometric skipping per block pair, so sparse models cost O(n + m).

    >>> g = stochastic_block_model([20, 30], [[0.5, 0.02], [0.02, 0.4]], seed=7)
    """
    sizes = [_count(s, "block size") for s in sizes]
    probs = np.asarray(p, dtype=float)
    k = len(sizes)
    if probs.shape != (k, k):
        raise ValueError(f"p must be a {k}×{k} matrix for {k} blocks, got shape {probs.shape}")
    if np.any(probs < 0) or np.any(probs > 1) or np.any(np.isnan(probs)):
        raise ValueError("every entry of p must be a probability in [0, 1]")
    if not directed and not np.allclose(probs, probs.T):
        raise ValueError("p must be symmetric for an undirected block model (or pass directed=True)")
    rng = make_rng(seed)
    g = _new(directed, f"stochastic_block_model({sizes})")
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int).tolist()
    for b, size in enumerate(sizes):
        g.add_nodes(range(starts[b], starts[b] + size), block=b)
    for i in range(k):
        for j in range(k) if directed else range(i, k):
            ni, nj, pij = sizes[i], sizes[j], float(probs[i, j])
            if i == j:
                if directed:
                    total = ni * ni if selfloops else ni * (ni - 1)
                    idx = _bernoulli_indices(rng, total, pij)
                    if selfloops:
                        u, v = idx // max(ni, 1), idx % max(ni, 1)
                    else:
                        u, v = _ordered_index(idx, ni) if ni > 1 else (idx, idx)
                else:
                    total = ni * (ni + 1) // 2 if selfloops else ni * (ni - 1) // 2
                    u, v = _pair_index(_bernoulli_indices(rng, total, pij), loops=selfloops)
            else:
                idx = _bernoulli_indices(rng, ni * nj, pij)
                u, v = idx // nj, idx % nj
            g.add_edges(_edges_from(u, v, starts[i], starts[j]))
    return g


def planted_partition(
    l: int,
    k: int,
    p_in: float,
    p_out: float,
    seed: Seed = None,
    *,
    directed: bool = False,
) -> Graph:
    """Planted partition: ``l`` groups of ``k`` nodes, linked with ``p_in`` inside a group and ``p_out`` across.

    A :func:`stochastic_block_model` with a constant diagonal; nodes carry ``block``.
    """
    l, k = _count(l, "l"), _count(k, "k")
    p_in, p_out = _check_prob(p_in, "p_in"), _check_prob(p_out, "p_out")
    probs = np.full((l, l), p_out)
    np.fill_diagonal(probs, p_in)
    g = stochastic_block_model([k] * l, probs, directed=directed, seed=seed)
    g.name = f"planted_partition({l}, {k}, {p_in:g}, {p_out:g})"
    return g


# ---------------------------------------------------------------------- #
# spatial and degree-constrained models
# ---------------------------------------------------------------------- #
def random_geometric(n: int, radius: float, *, seed: Seed = None) -> Graph:
    """Random geometric graph in the unit square.

    Nodes get ``pos = (x, y)`` uniform in ``[0, 1)²``; two nodes are adjacent
    when their Euclidean distance is at most *radius* (``<=``, as in
    networkx). Candidate pairs come from a grid of ``radius``-sized cells, so
    the cost is O(n + m) for small radii.
    """
    n = _count(n)
    radius = float(radius)
    if radius < 0 or math.isnan(radius):
        raise ValueError(f"radius must be non-negative, got {radius}")
    rng = make_rng(seed)
    pts = rng.random((n, 2))
    g = Graph(name=f"random_geometric({n}, {radius:g})")
    for i, (x, y) in enumerate(pts.tolist()):
        g.add_node(i, pos=(x, y))
    if n < 2 or radius == 0.0:
        return g
    r2 = radius * radius
    cells = np.floor(pts / radius).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (cx, cy) in enumerate(cells.tolist()):
        buckets.setdefault((cx, cy), []).append(i)
    arrays = {c: np.asarray(ix, dtype=np.int64) for c, ix in buckets.items()}
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    for (cx, cy), a in arrays.items():
        # Half of the 3×3 neighborhood, so every cell pair is examined once.
        for dx, dy in ((0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            b = arrays.get((cx + dx, cy + dy))
            if b is None:
                continue
            diff = pts[a][:, None, :] - pts[b][None, :, :]
            ii, jj = np.nonzero(np.einsum("ijk,ijk->ij", diff, diff) <= r2)
            u, v = a[ii], b[jj]
            if dx == 0 and dy == 0:
                keep = u < v
                u, v = u[keep], v[keep]
            us.append(np.minimum(u, v))
            vs.append(np.maximum(u, v))
    if us:
        u, v = np.concatenate(us), np.concatenate(vs)
        order = np.lexsort((v, u))
        g.add_edges(_edges_from(u[order], v[order]))
    return g


def random_regular(d: int, n: int, *, seed: Seed = None, max_tries: int = 1000) -> Graph:
    """Random ``d``-regular simple graph on ``n`` nodes (argument order as in networkx).

    Pairs degree "stubs" at random and re-pairs only the stubs that would
    create a loop or duplicate, restarting when that gets stuck (the
    Steger–Wormald scheme networkx uses). For ``d > (n-1)/2`` the complement
    of a random ``(n-1-d)``-regular graph is returned, which is much faster
    and always succeeds. Requires ``0 <= d < n`` and ``n·d`` even.

    Raises :class:`ConvergenceError` if *max_tries* restarts all fail
    (practically never).
    """
    d, n = _count(d, "d"), _count(n)
    if n * d % 2:
        raise ValueError(f"n·d must be even, got n={n}, d={d}")
    if n and d >= n:
        raise ValueError(f"random_regular needs d < n, got d={d}, n={n}")
    rng = make_rng(seed)
    name = f"random_regular({d}, {n})"
    if n and d > (n - 1) / 2:
        comp = random_regular(n - 1 - d, n, seed=rng, max_tries=max_tries)
        g = complete_graph(n)
        g.remove_edges(list(comp.edges))
        g.name = name
        return g
    g = empty_graph(n)
    g.name = name
    if d == 0:
        return g
    for _ in range(max_tries):
        edges = _try_regular(rng, n, d)
        if edges is not None:
            g.add_edges(sorted(edges))
            return g
    raise ConvergenceError(f"random_regular({d}, {n}) failed after {max_tries} attempts")


def _try_regular(rng: np.random.Generator, n: int, d: int) -> set[tuple[int, int]] | None:
    edges: set[tuple[int, int]] = set()
    stubs = np.repeat(np.arange(n), d)
    while stubs.size:
        stubs = rng.permutation(stubs)
        leftover: dict[int, int] = {}
        for s1, s2 in zip(stubs[0::2].tolist(), stubs[1::2].tolist()):
            if s1 > s2:
                s1, s2 = s2, s1
            if s1 != s2 and (s1, s2) not in edges:
                edges.add((s1, s2))
            else:
                leftover[s1] = leftover.get(s1, 0) + 1
                leftover[s2] = leftover.get(s2, 0) + 1
        if leftover and not _suitable(edges, leftover):
            return None
        stubs = np.repeat(np.fromiter(leftover, dtype=np.int64), list(leftover.values()))
    return edges


def _suitable(edges: set[tuple[int, int]], leftover: dict[int, int]) -> bool:
    """True if some pair of leftover stubs could still form a new edge."""
    nodes = list(leftover)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if (min(a, b), max(a, b)) not in edges:
                return True
    return False


def random_tree(n: int, *, directed: bool = False, seed: Seed = None) -> Graph:
    """Uniformly random labeled tree on ``n >= 1`` nodes (decoded from a random Prüfer sequence).

    Every one of the ``n^(n-2)`` labeled trees is equally likely. With
    *directed*, arcs point away from node ``0`` (an arborescence rooted at 0).
    Linear-time decoding.
    """
    n = _count(n)
    if n < 1:
        raise ValueError("a tree needs at least one node")
    rng = make_rng(seed)
    g = _new(directed, f"random_tree({n})")
    g.add_nodes(range(n))
    if n == 1:
        return g
    code = rng.integers(0, n, size=n - 2).tolist()
    degree = [1] * n
    for x in code:
        degree[x] += 1
    ptr = degree.index(1)
    leaf = ptr
    edges: list[tuple[int, int]] = []
    for v in code:
        edges.append((leaf, v))
        degree[v] -= 1
        if degree[v] == 1 and v < ptr:
            leaf = v
        else:
            ptr += 1
            while degree[ptr] != 1:
                ptr += 1
            leaf = ptr
    edges.append((leaf, n - 1))
    if not directed:
        g.add_edges(edges)
        return g
    nbrs: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    seen = {0}
    frontier = [0]
    for u in frontier:  # breadth-first; the list grows while we walk it
        for v in sorted(nbrs[u]):
            if v not in seen:
                seen.add(v)
                frontier.append(v)
                g.add_edge(u, v)
    return g


def configuration_model(degrees: Sequence[int], *, seed: Seed = None) -> Graph:
    """Simple-graph configuration model for the given degree sequence.

    Stubs are paired uniformly at random, then self-loops and duplicate edges
    are **dropped**, so a node's realized degree can fall short of the
    requested one (rarely, for sparse sequences). networkx's
    ``configuration_model`` keeps them in a multigraph instead; compare
    ``g.degree()`` with *degrees* when exact degrees matter. The degree sum
    must be even.
    """
    degs = [_count(x, "degree") for x in degrees]
    if sum(degs) % 2:
        raise ValueError(f"the degree sum must be even, got {sum(degs)}")
    rng = make_rng(seed)
    g = Graph(name=f"configuration_model(n={len(degs)})")
    g.add_nodes(range(len(degs)))
    stubs = rng.permutation(np.repeat(np.arange(len(degs)), degs))
    for u, v in zip(stubs[0::2].tolist(), stubs[1::2].tolist()):
        if u != v:
            g.add_edge(u, v)
    return g


# ---------------------------------------------------------------------- #
# random DAGs
# ---------------------------------------------------------------------- #
def random_dag(n: int, p: float | None = None, m: int | None = None, *, seed: Seed = None) -> DAG:
    """Random DAG: a random topological order, then only forward arcs.

    Nodes ``0 … n-1`` are shuffled into a hidden order; each forward pair is
    an arc independently with probability *p*, or exactly *m* forward pairs
    are chosen uniformly. Give at most one of *p* and *m*; with neither,
    ``p = min(1, 3 / (n - 1))`` (about three arcs per node on average).
    """
    n = _count(n)
    if p is not None and m is not None:
        raise ValueError("give either p (arc probability) or m (arc count), not both")
    rng = make_rng(seed)
    total = n * (n - 1) // 2
    order = rng.permutation(n)
    if m is not None:
        m = _count(m, "m")
        if m > total:
            raise ValueError(f"a DAG on {n} nodes has at most {total} arcs, got m={m}")
        k = np.sort(rng.choice(total, size=m, replace=False)) if m else np.empty(0, dtype=np.int64)
        label = f"m={m}"
    else:
        p = _check_prob(p) if p is not None else min(1.0, 3.0 / max(n - 1, 1))
        k = _bernoulli_indices(rng, total, p)
        label = f"p={p:g}"
    w, v = _pair_index(k)
    dag = DAG(name=f"random_dag({n}, {label})")
    dag.add_nodes(range(n))
    dag.add_edges(zip(order[w].tolist(), order[v].tolist()))
    return dag


def layered_dag(
    layers: Sequence[int],
    p: float = 0.4,
    *,
    seed: Seed = None,
    connected: bool = True,
) -> DAG:
    """Random layered DAG: arcs only between consecutive layers.

    Nodes are integers numbered layer by layer and carry ``layer`` (0-based)
    and ``index`` (position within the layer). Each pair of nodes in layers
    ``k`` and ``k+1`` is linked with probability *p*.

    With *connected* (the default) the result is also repaired so that every
    node below the first layer has a parent in the previous layer, every node
    above the last layer has a child in the next one, and the DAG is weakly
    connected; ``layer`` then equals :meth:`DAG.levels`. With a single layer
    there are no arcs at all.

    >>> dag = layered_dag([3, 5, 5, 2], p=0.3, seed=4)
    """
    sizes = [_count(s, "layer size") for s in layers]
    if any(s == 0 for s in sizes):
        raise ValueError("every layer needs at least one node")
    p = _check_prob(p)
    rng = make_rng(seed)
    dag = DAG(name=f"layered_dag({sizes})")
    starts = [0]
    for s in sizes:
        starts.append(starts[-1] + s)
    for k, s in enumerate(sizes):
        for i in range(s):
            dag.add_node(starts[k] + i, layer=k, index=i)
    arcs: set[tuple[int, int]] = set()
    for k in range(len(sizes) - 1):
        a, b = sizes[k], sizes[k + 1]
        idx = _bernoulli_indices(rng, a * b, p)
        arcs.update(_edges_from(idx // b, idx % b, starts[k], starts[k + 1]))
    if connected and len(sizes) > 1:
        has_parent = {v for _, v in arcs}
        has_child = {u for u, _ in arcs}
        for k in range(1, len(sizes)):
            for v in range(starts[k], starts[k + 1]):
                if v not in has_parent:
                    u = starts[k - 1] + int(rng.integers(sizes[k - 1]))
                    arcs.add((u, v))
                    has_child.add(u)
        for k in range(len(sizes) - 1):
            for u in range(starts[k], starts[k + 1]):
                if u not in has_child:
                    arcs.add((u, starts[k + 1] + int(rng.integers(sizes[k + 1]))))
        arcs |= _bridge_components(rng, arcs, sizes, starts)
    dag.add_edges(sorted(arcs))
    return dag


def _bridge_components(
    rng: np.random.Generator,
    arcs: set[tuple[int, int]],
    sizes: list[int],
    starts: list[int],
) -> set[tuple[int, int]]:
    """Arcs joining the weak components of a layered DAG into one.

    Every component spans all layers (each node has a parent above and a child
    below), so any two components can be bridged between some layer ``k`` and
    ``k+1``.
    """
    n = starts[-1]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in arcs:
        parent[find(u)] = find(v)
    comps: dict[int, list[int]] = {}
    for x in range(n):
        comps.setdefault(find(x), []).append(x)
    groups = list(comps.values())
    extra: set[tuple[int, int]] = set()
    layer_of = [k for k, s in enumerate(sizes) for _ in range(s)]
    for left, right in zip(groups, groups[1:]):
        k = int(rng.integers(len(sizes) - 1))
        tails = [x for x in left if layer_of[x] == k]
        heads = [x for x in right if layer_of[x] == k + 1]
        extra.add((tails[int(rng.integers(len(tails)))], heads[int(rng.integers(len(heads)))]))
    return extra


def random_task_dag(n: int, *, seed: Seed = None) -> DAG:
    """Random project network of ``n`` tasks, ready for CPM/PERT and scheduling demos.

    Tasks ``0 … n-1`` are numbered in a topological order. Task ``0`` is the
    only source and task ``n-1`` the only sink; every other task has 1–3
    predecessors among the recent tasks, so the network is long and
    moderately parallel, like a real plan.

    Every task carries triangular duration estimates in hours, ``min <= mode
    <= max`` (log-normal most-likely time around 6 h, optimistic 50–90 % and
    pessimistic 120–250 % of it, rounded to half hours), plus ``duration``
    equal to ``mode``.
    """
    n = _count(n)
    if n < 1:
        raise ValueError("a task network needs at least one task")
    rng = make_rng(seed)
    dag = DAG(name=f"random_task_dag({n})")
    for i in range(n):
        mode = _half_hours(float(np.clip(rng.lognormal(math.log(6.0), 0.7), 0.5, 120.0)))
        low = min(mode, max(0.5, _half_hours(mode * rng.uniform(0.5, 0.9))))
        high = max(mode, _half_hours(mode * rng.uniform(1.2, 2.5)))
        dag.add_node(i, duration=mode, min=low, mode=mode, max=high)
    window = max(2, round(1.5 * math.sqrt(n)))
    arcs: set[tuple[int, int]] = set()
    for i in range(1, n):
        lo = max(0, i - window)
        k = min(i - lo, 1 + int(rng.binomial(2, 0.35)))
        for j in rng.choice(np.arange(lo, i), size=k, replace=False).tolist():
            arcs.add((j, i))
    has_child = {u for u, _ in arcs}
    for j in range(n - 1):
        if j not in has_child:
            arcs.add((j, int(rng.integers(j + 1, min(n - 1, j + window) + 1))))
    dag.add_edges(sorted(arcs))
    return dag


def _half_hours(x: float) -> float:
    return round(x * 2.0) / 2.0


__all__ = [
    "erdos_renyi",
    "gnm_random_graph",
    "barabasi_albert",
    "watts_strogatz",
    "stochastic_block_model",
    "planted_partition",
    "random_geometric",
    "random_regular",
    "random_tree",
    "configuration_model",
    "powerlaw_cluster",
    "random_dag",
    "layered_dag",
    "random_task_dag",
]
