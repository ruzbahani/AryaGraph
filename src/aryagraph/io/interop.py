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

"""Bridges to networkx, pandas, numpy and scipy.

networkx, pandas and scipy are optional: they are imported only when one of
these functions runs, and a missing package raises :class:`DependencyError`
(``pip install 'aryagraph[interop]'``). numpy is a core dependency.

Matrix conventions follow networkx's ``to_numpy_array``: rows and columns
follow graph order (or *nodes*), an edge without the *weight* attribute
weighs 1, an undirected graph gives a symmetric matrix and a self-loop's
weight sits once on the diagonal.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..core.dag import DAG
from ..core.exceptions import DependencyError, GraphTypeError, NodeNotFound
from ..core.graph import DiGraph, Graph
from ..core.utils import WeightSpec, weight_fn


def _require(module: str, feature: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError:
        raise DependencyError(module.split(".")[0], feature, extra="interop") from None


def _new(directed: bool, dag: bool) -> Graph:
    return DAG() if dag else DiGraph() if directed else Graph()


def _py(value: Any) -> Any:
    """numpy scalars → Python scalars (so attributes serialize and compare plainly)."""
    return value.item() if isinstance(value, np.generic) else value


def _missing(value: Any, pd: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if np.ndim(value) != 0:
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------- #
# networkx
# ---------------------------------------------------------------------- #
def from_networkx(G: Any, *, as_dag: bool = False, collapse: bool = False) -> Graph:
    """AryaGraph copy of a networkx graph (node, edge and graph attributes are copied shallowly).

    ``nx.Graph`` → :class:`Graph`, ``nx.DiGraph`` → :class:`DiGraph`, or a
    :class:`DAG` with *as_dag* (raising :class:`CycleError` on a cycle).
    Multigraphs raise :class:`GraphTypeError` unless *collapse* is true, in
    which case parallel edges merge into one whose attributes are updated
    edge by edge (later edges win), the same rule as ``nx.Graph(M)``.
    """
    _require("networkx", "from_networkx()")
    if G.is_multigraph() and not collapse:
        raise GraphTypeError(
            f"{type(G).__name__} is a multigraph; pass collapse=True to merge parallel edges (later attributes win)"
        )
    directed = G.is_directed()
    if as_dag and not directed:
        raise GraphTypeError("as_dag=True needs a directed networkx graph")
    g = _new(directed, as_dag)
    g.attrs.update(G.graph)
    g.add_nodes((n, dict(d)) for n, d in G.nodes(data=True))
    batch = [(u, v, dict(d)) for u, v, d in G.edges(data=True)]
    if as_dag:
        g.add_edges(batch)
    else:
        for u, v, d in batch:
            g.add_edge(u, v, **d)
    return g


def to_networkx(g: Graph) -> Any:
    """``nx.Graph`` (undirected) or ``nx.DiGraph`` (directed, including DAGs) with every attribute copied."""
    nx = _require("networkx", "to_networkx()")
    G = nx.DiGraph() if g.directed else nx.Graph()
    G.graph.update(g.attrs)
    G.add_nodes_from((n, dict(d)) for n, d in g._node.items())
    G.add_edges_from((u, v, dict(d)) for u, v, d in g._iter_edges())
    return G


# ---------------------------------------------------------------------- #
# pandas
# ---------------------------------------------------------------------- #
def from_pandas_edgelist(
    df: Any,
    source: str = "source",
    target: str = "target",
    edge_attr: bool | str | Sequence[str] | None = None,
    directed: bool = False,
    *,
    dag: bool = False,
    graph: Graph | None = None,
) -> Graph:
    """Graph from a DataFrame with one row per edge.

    *edge_attr* is ``None``/``False`` (no attributes), ``True`` (every other
    column), a column name or a list of names. Missing cells (NaN, None,
    NA) leave the attribute out; numpy scalars become Python scalars. Pass
    *graph* to add the edges to an existing graph instead.
    """
    pd = _require("pandas", "from_pandas_edgelist()")
    columns = list(df.columns)
    for col in (source, target):
        if col not in columns:
            raise ValueError(f"column {col!r} is not in the DataFrame (columns: {columns})")
    if edge_attr is None or edge_attr is False:
        attr_cols: list[Any] = []
    elif edge_attr is True:
        attr_cols = [c for c in columns if c not in (source, target)]
    elif isinstance(edge_attr, str):
        attr_cols = [edge_attr]
    else:
        attr_cols = list(edge_attr)
    unknown = [c for c in attr_cols if c not in columns]
    if unknown:
        raise ValueError(f"edge_attr columns {unknown} are not in the DataFrame")
    g = graph if graph is not None else _new(directed, dag)
    sources, targets = df[source].tolist(), df[target].tolist()
    values = [df[c].tolist() for c in attr_cols]  # tolist() yields Python scalars
    batch = []
    for i, (u, v) in enumerate(zip(sources, targets)):
        u, v = _py(u), _py(v)
        if _missing(u, pd) or _missing(v, pd):
            raise ValueError(f"row {i}: missing source or target")
        attrs = {str(c): _py(col[i]) for c, col in zip(attr_cols, values) if not _missing(col[i], pd)}
        batch.append((u, v, attrs))
    if isinstance(g, DAG):
        g.add_edges(batch)
    else:
        for u, v, attrs in batch:
            g.add_edge(u, v, **attrs)
    return g


def from_pandas(
    nodes_df: Any = None,
    edges_df: Any = None,
    *,
    node_id: str = "id",
    source: str = "source",
    target: str = "target",
    directed: bool = False,
    dag: bool = False,
) -> Graph:
    """Graph from a node table and/or an edge table (the inverse of :func:`to_pandas`).

    Node ids come from the *node_id* column when it exists, else from the
    index; every other column is a node attribute. Every edge column besides
    *source* and *target* is an edge attribute. Missing cells are skipped.
    """
    pd = _require("pandas", "from_pandas()")
    g = _new(directed, dag)
    if nodes_df is not None:
        columns = list(nodes_df.columns)
        use_col = node_id in columns
        attr_cols = [c for c in columns if not (use_col and c == node_id)]
        ids = (nodes_df[node_id] if use_col else nodes_df.index).tolist()
        values = [nodes_df[c].tolist() for c in attr_cols]
        for i, n in enumerate(ids):
            n = _py(n)
            if _missing(n, pd):
                raise ValueError(f"row {i} of the node table has no id")
            g.add_node(n, **{str(c): _py(col[i]) for c, col in zip(attr_cols, values) if not _missing(col[i], pd)})
    if edges_df is not None:
        from_pandas_edgelist(edges_df, source, target, edge_attr=True, graph=g)
    return g


def to_pandas(g: Graph, *, node_id: str = "id", source: str = "source", target: str = "target") -> tuple[Any, Any]:
    """``(nodes_df, edges_df)``: one row per node (``id`` + attributes) and per edge (``source``, ``target`` + attributes).

    Attribute columns are the union of keys in first-seen order; missing
    values are NaN. Integer columns with gaps use pandas' nullable ``Int64``
    dtype so their values stay integers.
    """
    pd = _require("pandas", "to_pandas()")

    def frame(rows: list[dict], fixed: list[str]) -> Any:
        keys = list(dict.fromkeys(k for r in rows for k in r if k not in fixed))
        df = pd.DataFrame.from_records(rows, columns=[*fixed, *keys]) if rows else pd.DataFrame(columns=[*fixed, *keys])
        for k in keys:
            present = [r[k] for r in rows if k in r]
            if len(present) < len(rows) and present and all(
                isinstance(_py(x), int) and not isinstance(_py(x), bool) for x in present
            ):
                df[k] = pd.array([r.get(k) for r in rows], dtype="Int64")
        return df

    node_rows = []
    for n, d in g._node.items():
        if node_id in d:
            raise ValueError(f"node {n!r} has an attribute named {node_id!r}; pass node_id=...")
        node_rows.append({node_id: n, **d})
    edge_rows = []
    for u, v, d in g._iter_edges():
        for k in (source, target):
            if k in d:
                raise ValueError(f"edge ({u!r}, {v!r}) has an attribute named {k!r}; pass source=/target=")
        edge_rows.append({source: u, target: v, **d})
    return frame(node_rows, [node_id]), frame(edge_rows, [source, target])


# ---------------------------------------------------------------------- #
# numpy and scipy
# ---------------------------------------------------------------------- #
def _labels(n: int, nodes: Sequence[Any] | None) -> list[Any]:
    if nodes is None:
        return list(range(n))
    labels = list(nodes)
    if len(labels) != n:
        raise ValueError(f"{len(labels)} node labels for a {n}×{n} matrix")
    if len(set(labels)) != n:
        raise ValueError("node labels must be distinct")
    return labels


def _order(g: Graph, nodes: Iterable[Any] | None) -> list[Any]:
    order = list(g._node) if nodes is None else list(nodes)
    for n in order:
        if n not in g._node:
            raise NodeNotFound(n)
    if len(set(order)) != len(order):
        raise ValueError("nodes contains duplicates")
    return order


def from_numpy(
    a: Any,
    *,
    directed: bool | None = None,
    nodes: Sequence[Any] | None = None,
    weight: str | None = "weight",
) -> Graph:
    """Graph from a square adjacency matrix; every non-zero entry is an edge.

    ``directed=None`` decides by symmetry (a symmetric matrix gives a
    :class:`Graph` read from its upper triangle incl. the diagonal, else a
    :class:`DiGraph`); ``directed=False`` with an asymmetric matrix raises
    ``ValueError``. Entries become the *weight* attribute as Python numbers
    (``weight=None`` for none; boolean matrices are always unweighted).
    *nodes* labels the rows (default ``0 … n-1``).
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"an adjacency matrix must be square, got shape {a.shape}")
    labels = _labels(a.shape[0], nodes)
    symmetric = bool(np.array_equal(a, a.T))
    if directed is None:
        directed = not symmetric
    elif not directed and not symmetric:
        raise ValueError("the matrix is not symmetric; pass directed=True for a directed graph")
    g: Graph = DiGraph() if directed else Graph()
    g.add_nodes(labels)
    rows, cols = np.nonzero(a if directed else np.triu(a))
    if weight is None or a.dtype == bool:
        g.add_edges((labels[i], labels[j]) for i, j in zip(rows.tolist(), cols.tolist()))
    else:
        values = a[rows, cols].tolist()
        g.add_edges((labels[i], labels[j], {weight: w}) for i, j, w in zip(rows.tolist(), cols.tolist(), values))
    return g


def to_numpy(
    g: Graph,
    weight: WeightSpec = "weight",
    nodes: Iterable[Any] | None = None,
    *,
    dtype: Any = float,
    nonedge: float = 0.0,
) -> np.ndarray:
    """Dense adjacency matrix ``A[i, j]`` = weight of edge ``nodes[i] → nodes[j]`` (*nonedge* elsewhere).

    *weight* is an attribute name (missing ⇒ 1), a callable ``f(u, v, attrs)``
    or ``None`` (0/1 matrix). Edges leaving the *nodes* subset are ignored.
    """
    order = _order(g, nodes)
    index = {n: i for i, n in enumerate(order)}
    wf = weight_fn(weight)
    a = np.full((len(order), len(order)), nonedge, dtype=dtype)
    for u, v, d in g._iter_edges():
        i, j = index.get(u), index.get(v)
        if i is None or j is None:
            continue
        w = wf(u, v, d)
        a[i, j] = w
        if not g.directed:
            a[j, i] = w
    return a


def from_scipy_sparse(
    m: Any,
    *,
    directed: bool | None = None,
    nodes: Sequence[Any] | None = None,
    weight: str | None = "weight",
) -> Graph:
    """Graph from a scipy sparse adjacency matrix (any format); same rules as :func:`from_numpy`.

    Explicitly stored zeros are not edges; duplicate entries are summed first.
    """
    sparse = _require("scipy.sparse", "from_scipy_sparse()")
    if not sparse.issparse(m):
        raise TypeError(f"expected a scipy sparse matrix or array, got {type(m).__name__}")
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"an adjacency matrix must be square, got shape {m.shape}")
    csr = sparse.csr_array(m, copy=True)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    labels = _labels(csr.shape[0], nodes)
    symmetric = (csr != csr.T).nnz == 0
    if directed is None:
        directed = not symmetric
    elif not directed and not symmetric:
        raise ValueError("the matrix is not symmetric; pass directed=True for a directed graph")
    coo = (csr if directed else sparse.triu(csr, format="csr")).tocoo()
    order = np.lexsort((coo.col, coo.row))
    rows, cols = coo.row[order].tolist(), coo.col[order].tolist()
    g: Graph = DiGraph() if directed else Graph()
    g.add_nodes(labels)
    if weight is None or csr.dtype == bool:
        g.add_edges((labels[i], labels[j]) for i, j in zip(rows, cols))
    else:
        values = coo.data[order].tolist()
        g.add_edges((labels[i], labels[j], {weight: w}) for i, j, w in zip(rows, cols, values))
    return g


def to_scipy_sparse(
    g: Graph,
    weight: WeightSpec = "weight",
    format: str = "csr",
    nodes: Iterable[Any] | None = None,
    *,
    dtype: Any = float,
) -> Any:
    """Sparse adjacency matrix (a ``scipy.sparse`` array in *format*: ``csr``, ``csc``, ``coo``, …)."""
    sparse = _require("scipy.sparse", "to_scipy_sparse()")
    order = _order(g, nodes)
    index = {n: i for i, n in enumerate(order)}
    wf = weight_fn(weight)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u, v, d in g._iter_edges():
        i, j = index.get(u), index.get(v)
        if i is None or j is None:
            continue
        w = wf(u, v, d)
        rows.append(i)
        cols.append(j)
        data.append(w)
        if not g.directed and i != j:
            rows.append(j)
            cols.append(i)
            data.append(w)
    n = len(order)
    coo = sparse.coo_array((np.asarray(data, dtype=dtype), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))), shape=(n, n))
    return coo.asformat(format)


__all__ = [
    "from_networkx",
    "to_networkx",
    "from_pandas_edgelist",
    "from_pandas",
    "to_pandas",
    "from_numpy",
    "to_numpy",
    "from_scipy_sparse",
    "to_scipy_sparse",
]
