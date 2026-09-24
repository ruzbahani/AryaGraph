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

"""Deterministic graph families: paths, cycles, complete graphs, lattices, trees, …

Node labels are the integers ``0 … n-1`` (grids use ``(row, col)`` pairs) and
follow networkx's numbering, so an undirected AryaGraph graph has exactly the edge
set of the networkx generator of the same name.

``directed=True`` orients each family in its natural way, documented per
function: usually from lower to higher labels (paths, trees, hypercubes,
multipartite graphs become DAG-shaped), and in both directions for complete
graphs, as networkx does.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

from ..core.graph import DiGraph, Graph


# ---------------------------------------------------------------------- #
# helpers shared with the other generator modules
# ---------------------------------------------------------------------- #
def _count(value: Any, what: str = "n") -> int:
    """Validate a non-negative integer size argument."""
    try:
        k = operator.index(value)
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {value!r}") from None
    if k < 0:
        raise ValueError(f"{what} must be non-negative, got {k}")
    return k


def _new(directed: bool, name: str) -> Graph:
    return DiGraph(name=name) if directed else Graph(name=name)


def _pairwise(nodes: list, cyclic: bool = False) -> list[tuple]:
    pairs = list(zip(nodes, nodes[1:]))
    if cyclic and len(nodes) > 1:
        pairs.append((nodes[-1], nodes[0]))
    return pairs


# ---------------------------------------------------------------------- #
# simple families
# ---------------------------------------------------------------------- #
def empty_graph(n: int, *, directed: bool = False) -> Graph:
    """``n`` nodes and no edges."""
    n = _count(n)
    g = _new(directed, f"empty_graph({n})")
    g.add_nodes(range(n))
    return g


def path_graph(n: int, *, directed: bool = False) -> Graph:
    """Path through ``0, 1, …, n-1`` in order (arcs ``i → i+1`` when *directed*)."""
    n = _count(n)
    g = _new(directed, f"path_graph({n})")
    g.add_nodes(range(n))
    g.add_edges(_pairwise(list(range(n))))
    return g


def cycle_graph(n: int, *, directed: bool = False) -> Graph:
    """Cycle through ``0, 1, …, n-1`` and back to ``0`` (arcs ``i → i+1 mod n`` when *directed*).

    As in networkx, ``n = 1`` gives a single self-loop and an undirected
    ``n = 2`` gives a single edge (a directed one gives two opposite arcs).
    """
    n = _count(n)
    g = _new(directed, f"cycle_graph({n})")
    g.add_nodes(range(n))
    if n == 1:
        g.add_edge(0, 0)
    else:
        g.add_edges(_pairwise(list(range(n)), cyclic=True))
    return g


def complete_graph(n: int, *, directed: bool = False) -> Graph:
    """Complete graph ``K_n``; directed, every ordered pair ``u ≠ v`` is an arc."""
    n = _count(n)
    g = _new(directed, f"complete_graph({n})")
    g.add_nodes(range(n))
    if directed:
        g.add_edges((u, v) for u in range(n) for v in range(n) if u != v)
    else:
        g.add_edges((u, v) for u in range(n) for v in range(u + 1, n))
    return g


def complete_bipartite_graph(n1: int, n2: int, *, directed: bool = False) -> Graph:
    """Complete bipartite graph ``K_{n1,n2}``.

    Nodes ``0 … n1-1`` carry ``bipartite=0`` and ``n1 … n1+n2-1`` carry
    ``bipartite=1``; directed arcs go from side 0 to side 1.
    """
    n1, n2 = _count(n1, "n1"), _count(n2, "n2")
    g = _new(directed, f"complete_bipartite_graph({n1}, {n2})")
    g.add_nodes(range(n1), bipartite=0)
    g.add_nodes(range(n1, n1 + n2), bipartite=1)
    g.add_edges((u, v) for u in range(n1) for v in range(n1, n1 + n2))
    return g


def star_graph(n: int, *, directed: bool = False) -> Graph:
    """Star with center ``0`` and ``n`` leaves ``1 … n`` (``n + 1`` nodes, like networkx).

    Directed arcs point from the center to the leaves.
    """
    n = _count(n)
    g = _new(directed, f"star_graph({n})")
    g.add_nodes(range(n + 1))
    g.add_edges((0, v) for v in range(1, n + 1))
    return g


def wheel_graph(n: int, *, directed: bool = False) -> Graph:
    """Wheel on ``n`` nodes: hub ``0`` joined to every node of the cycle ``1 … n-1``.

    Directed: hub → rim arcs and a directed rim cycle ``1 → 2 → … → n-1 → 1``.
    """
    n = _count(n)
    g = _new(directed, f"wheel_graph({n})")
    g.add_nodes(range(n))
    rim = list(range(1, n))
    g.add_edges((0, v) for v in rim)
    if len(rim) > 1:
        g.add_edges(_pairwise(rim, cyclic=True))
    return g


def grid_graph(
    rows: int,
    cols: int,
    periodic: bool | tuple[bool, bool] = False,
    *,
    directed: bool = False,
) -> Graph:
    """Two-dimensional ``rows × cols`` lattice with nodes ``(r, c)``.

    Every node carries ``pos = (c, r)`` (screen coordinates: x to the right,
    y downward), so the grid draws as itself without a layout algorithm.

    Parameters
    ----------
    periodic:
        Wrap around in both dimensions (a torus), or give ``(rows_wrap,
        cols_wrap)``. As in networkx, a dimension of length ≤ 2 never wraps
        (the wrap edge would duplicate an existing one).
    directed:
        Arcs point to increasing row / column (wrap arcs go from the last
        row/column to the first). networkx instead adds both directions; use
        ``grid_graph(...).to_directed()`` for that.

    Matches ``networkx.grid_2d_graph(rows, cols, periodic)`` when undirected.
    """
    rows, cols = _count(rows, "rows"), _count(cols, "cols")
    if isinstance(periodic, Iterable):
        wrap_r, wrap_c = (bool(x) for x in periodic)
    else:
        wrap_r = wrap_c = bool(periodic)
    g = _new(directed, f"grid_graph({rows}, {cols}{', periodic' if wrap_r or wrap_c else ''})")
    for r in range(rows):
        for c in range(cols):
            g.add_node((r, c), pos=(float(c), float(r)))
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                g.add_edge((r, c), (r, c + 1))
            if r + 1 < rows:
                g.add_edge((r, c), (r + 1, c))
    if wrap_c and cols > 2:
        g.add_edges(((r, cols - 1), (r, 0)) for r in range(rows))
    if wrap_r and rows > 2:
        g.add_edges(((rows - 1, c), (0, c)) for c in range(cols))
    return g


def hypercube_graph(d: int, *, directed: bool = False) -> Graph:
    """``d``-dimensional hypercube ``Q_d``: nodes ``0 … 2^d - 1``, edges between labels differing in one bit.

    Each node carries ``bits``, its label as a ``d``-character binary string
    (networkx uses tuples of bits as the node labels instead). Directed arcs
    set a bit (``i → i | 2^k``), which makes the Boolean lattice, a DAG.
    """
    d = _count(d, "d")
    g = _new(directed, f"hypercube_graph({d})")
    size = 1 << d
    for i in range(size):
        g.add_node(i, bits=format(i, "b").zfill(d) if d else "")
    g.add_edges((i, i | (1 << k)) for i in range(size) for k in range(d) if not i & (1 << k))
    return g


def balanced_tree(r: int, h: int, *, directed: bool = False) -> Graph:
    """Perfectly balanced ``r``-ary tree of height ``h``.

    Nodes are numbered breadth-first from the root ``0``; node ``i``'s children
    are ``r·i + 1 … r·i + r``. There are ``(r^(h+1) - 1) / (r - 1)`` nodes
    (``h + 1`` when ``r = 1``). Directed arcs point from parent to child.
    """
    r, h = _count(r, "r"), _count(h, "h")
    if r == 0:
        n = 1
    elif r == 1:
        n = h + 1
    else:
        n = (r ** (h + 1) - 1) // (r - 1)
    g = _new(directed, f"balanced_tree({r}, {h})")
    g.add_nodes(range(n))
    g.add_edges(((child - 1) // r, child) for child in range(1, n))
    return g


def binomial_tree(k: int, *, directed: bool = False) -> Graph:
    """Binomial tree of order ``k`` (``2^k`` nodes, root ``0``).

    ``B_k`` is two copies of ``B_{k-1}`` with the second root hung under the
    first; numbering follows networkx. Directed arcs point away from the root.
    """
    k = _count(k, "k")
    g = _new(directed, f"binomial_tree({k})")
    g.add_node(0)
    edges: list[tuple[int, int]] = []
    size = 1
    for _ in range(k):
        edges += [(u + size, v + size) for u, v in edges]
        edges.append((0, size))
        size *= 2
    g.add_nodes(range(size))
    g.add_edges(edges)
    return g


def ladder_graph(n: int, *, directed: bool = False) -> Graph:
    """Ladder with ``n`` rungs: rails ``0 … n-1`` and ``n … 2n-1``, rungs joining ``i`` and ``i+n``.

    Directed arcs run along the rails (``i → i+1``) and across the rungs (``i → i+n``).
    """
    n = _count(n)
    g = _new(directed, f"ladder_graph({n})")
    g.add_nodes(range(2 * n))
    g.add_edges(_pairwise(list(range(n))))
    g.add_edges(_pairwise(list(range(n, 2 * n))))
    g.add_edges((v, v + n) for v in range(n))
    return g


def circular_ladder_graph(n: int, *, directed: bool = False) -> Graph:
    """Ladder whose two rails are closed into cycles (the prism graph ``C_n × K_2``).

    Same numbering as networkx (so ``n = 1`` has self-loops, like networkx).
    Directed: both rails are directed cycles and rungs point outward ``i → i+n``.
    """
    n = _count(n)
    g = ladder_graph(n, directed=directed)
    g.name = f"circular_ladder_graph({n})"
    if n:
        if directed:
            g.add_edge(n - 1, 0)
            g.add_edge(2 * n - 1, n)
        else:
            g.add_edge(0, n - 1)
            g.add_edge(n, 2 * n - 1)
    return g


def lollipop_graph(m: int, n: int) -> Graph:
    """Complete graph ``K_m`` (nodes ``0 … m-1``) with a path of ``n`` nodes hanging from node ``m-1``."""
    m, n = _count(m, "m"), _count(n)
    if m < 2:
        raise ValueError(f"lollipop_graph needs m >= 2 nodes in the candy, got {m}")
    g = complete_graph(m)
    g.name = f"lollipop_graph({m}, {n})"
    g.add_nodes(range(m, m + n))
    g.add_edges(_pairwise(list(range(m - 1, m + n))))
    return g


def barbell_graph(m1: int, m2: int) -> Graph:
    """Two ``K_{m1}`` bells joined by a path of ``m2`` nodes (networkx numbering).

    Left bell ``0 … m1-1``, path ``m1 … m1+m2-1``, right bell ``m1+m2 … 2·m1+m2-1``.
    """
    m1, m2 = _count(m1, "m1"), _count(m2, "m2")
    if m1 < 2:
        raise ValueError(f"barbell_graph needs m1 >= 2, got {m1}")
    g = Graph(name=f"barbell_graph({m1}, {m2})")
    n = 2 * m1 + m2
    g.add_nodes(range(n))
    g.add_edges((u, v) for u in range(m1) for v in range(u + 1, m1))
    g.add_edges(_pairwise(list(range(m1 - 1, m1 + m2 + 1))))
    right = range(m1 + m2, n)
    g.add_edges((u, v) for u in right for v in range(u + 1, n))
    return g


_PETERSEN = {
    0: (1, 4, 5),
    1: (0, 2, 6),
    2: (1, 3, 7),
    3: (2, 4, 8),
    4: (3, 0, 9),
    5: (0, 7, 8),
    6: (1, 8, 9),
    7: (2, 5, 9),
    8: (3, 5, 6),
    9: (4, 6, 7),
}


def petersen_graph() -> Graph:
    """The Petersen graph: 10 nodes, 15 edges, 3-regular, girth 5 (outer cycle 0–4, inner star 5–9)."""
    g = Graph(name="petersen_graph()")
    g.add_nodes(range(10))
    g.add_edges((u, v) for u, nbrs in _PETERSEN.items() for v in nbrs if u < v)
    return g


def complete_multipartite_graph(*sizes: int, directed: bool = False) -> Graph:
    """Complete multipartite graph with parts of the given sizes.

    Nodes are numbered part after part and carry ``subset`` (the part index,
    as in networkx); nodes in different parts are adjacent. Directed arcs go
    from the earlier part to the later one.

    >>> complete_multipartite_graph(2, 3).num_edges
    6
    """
    parts = [_count(s, "part size") for s in sizes]
    g = _new(directed, f"complete_multipartite_graph({', '.join(map(str, parts))})")
    blocks: list[range] = []
    start = 0
    for i, size in enumerate(parts):
        block = range(start, start + size)
        g.add_nodes(block, subset=i)
        blocks.append(block)
        start += size
    for i, a in enumerate(blocks):
        for b in blocks[i + 1 :]:
            g.add_edges((u, v) for u in a for v in b)
    return g


def turan_graph(n: int, r: int, *, directed: bool = False) -> Graph:
    """Turán graph ``T(n, r)``: complete ``r``-partite graph with parts as equal as possible.

    The ``n mod r`` larger parts come last, as in networkx. It is the densest
    graph on ``n`` nodes without a ``K_{r+1}``.
    """
    n, r = _count(n), _count(r, "r")
    if not 1 <= r <= n:
        raise ValueError(f"turan_graph needs 1 <= r <= n, got n={n}, r={r}")
    q, extra = divmod(n, r)
    g = complete_multipartite_graph(*([q] * (r - extra) + [q + 1] * extra), directed=directed)
    g.name = f"turan_graph({n}, {r})"
    return g


__all__ = [
    "empty_graph",
    "path_graph",
    "cycle_graph",
    "complete_graph",
    "complete_bipartite_graph",
    "star_graph",
    "wheel_graph",
    "grid_graph",
    "hypercube_graph",
    "balanced_tree",
    "binomial_tree",
    "ladder_graph",
    "circular_ladder_graph",
    "lollipop_graph",
    "barbell_graph",
    "petersen_graph",
    "complete_multipartite_graph",
    "turan_graph",
]
