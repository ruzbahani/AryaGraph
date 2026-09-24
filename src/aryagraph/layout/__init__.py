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

"""Layout engines and the :func:`compute` dispatcher.

Every engine has the signature ``fn(g, *, seed=0, **options) -> Layout``, is
deterministic for a fixed seed, returns nodes in graph order and handles empty
graphs, single nodes, isolated nodes and self-loops.

========================  ===============================================
``hierarchical``          layered Sugiyama drawing for DAGs and flows (metric)
``tree`` / ``radial``     tidy trees, using a spanning tree for other graphs (metric)
``stress``                stress majorization, the default for general graphs
``fruchterman_reingold``  spring embedder
``force_atlas2``          ForceAtlas2 with Barnes–Hut for large graphs
``spectral``              Laplacian eigenvectors
geometric                 ``circular``, ``shell``, ``grid``, ``random``,
                          ``spiral``, ``bipartite``, ``arc``
========================  ===============================================

Coordinates follow the screen convention: x right, **y down**.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Hashable, Mapping

import numpy as np

from ..core.exceptions import AryaGraphError
from .base import Layout, pack_components
from .force import force_atlas2, fruchterman_reingold
from .geometric import _pack_by_components, arc, bipartite, circular, grid, random, shell, spiral
from .hierarchical import hierarchical
from .overlap import remove_overlaps
from .spectral import spectral
from .stress import stress
from .tree import radial, tree

Node = Hashable

METHODS: dict[str, Callable[..., Layout]] = {
    "stress": stress,
    "kamada_kawai": stress,
    "force": fruchterman_reingold,
    "fruchterman_reingold": fruchterman_reingold,
    "spring": fruchterman_reingold,
    "forceatlas2": force_atlas2,
    "force_atlas2": force_atlas2,
    "spectral": spectral,
    "circular": circular,
    "shell": shell,
    "grid": grid,
    "random": random,
    "spiral": spiral,
    "bipartite": bipartite,
    "arc": arc,
    "tree": tree,
    "radial": radial,
    "hierarchical": hierarchical,
    "layered": hierarchical,
    "sugiyama": hierarchical,
}

# engines whose drawing of a disconnected graph is improved by per-component packing
_PACKED = {stress, fruchterman_reingold, force_atlas2, spectral}


def _is_dag(g: Any) -> bool:
    """Kahn's algorithm on the arcs (self-loops count as cycles)."""
    if not g.directed:
        return False
    indeg = {v: len(g._pred[v]) for v in g._node}
    stack = [v for v, k in indeg.items() if k == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in g._succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return seen == len(indeg)


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return True
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return kwargs
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _passthrough(g: Any, given: Layout | Mapping[Node, Any]) -> Layout:
    nodes = list(g._node)
    if isinstance(given, Layout):
        missing = [v for v in nodes if v not in given]
        if missing:
            raise AryaGraphError(f"layout has no position for {len(missing)} node(s), e.g. {missing[:3]!r}")
        if list(given.nodes) == nodes:
            return given
        keep = set(nodes)
        xy = given.xy[[given.index(v) for v in nodes]] if nodes else np.zeros((0, 2))
        routes = {e: p for e, p in given.routes.items() if e[0] in keep and e[1] in keep}
        return Layout(nodes, xy, routes=routes, method=given.method, metric=given.metric, meta=dict(given.meta))
    missing = [v for v in nodes if v not in given]
    if missing:
        raise AryaGraphError(f"positions are missing for {len(missing)} node(s), e.g. {missing[:3]!r}")
    xy = np.array([np.asarray(given[v], dtype=float).reshape(-1)[:2] for v in nodes], dtype=float).reshape(-1, 2)
    return Layout(nodes, xy, method="custom")


def auto_method(g: Any) -> str:
    """The method :func:`compute` picks for ``method="auto"``.

    Directed acyclic graphs up to 2000 nodes → ``"hierarchical"``; otherwise
    up to 3000 nodes → ``"stress"``; larger graphs → ``"forceatlas2"``.
    """
    n = len(g)
    if g.directed and n <= 2000 and _is_dag(g):
        return "hierarchical"
    if n <= 3000:
        return "stress"
    return "forceatlas2"


def compute(
    g: Any,
    method: str | Layout | Mapping[Node, Any] | Callable[..., Any] = "auto",
    *,
    seed: int = 0,
    sizes: Mapping[Node, Any] | None = None,
    components: str | None = "pack",
    **kwargs: Any,
) -> Layout:
    """Compute a layout by name (the entry point used by :func:`aryagraph.draw`).

    Parameters
    ----------
    method:
        A name (see :data:`METHODS`; ``"auto"`` picks via :func:`auto_method`),
        a :class:`Layout` or ``{node: (x, y)}`` mapping (validated and returned
        in graph order), or a callable ``f(g, **kwargs)`` returning either.
    seed:
        Forwarded to the engine (every engine is deterministic per seed).
    sizes:
        ``{node: (w, h)}`` node boxes in pixels; forwarded to size-aware
        engines (hierarchical, tree, radial, forceatlas2) and ignored by others.
    components:
        ``"pack"`` (default): stress, force, forceatlas2 and spectral lay out
        each weakly connected component separately and pack the pieces with a
        gap of one typical edge length. ``None`` lays out the whole graph at
        once with the force-directed engines; stress and spectral need
        connected input, so they pack components in either case.
    **kwargs:
        Engine options. With ``"auto"``, options the chosen engine does not
        accept are ignored (so e.g. ``orientation`` is harmless for a cyclic graph).

    Raises
    ------
    ValueError
        For an unknown method name (the message lists the valid ones).
    """
    if isinstance(method, (Layout, Mapping)):
        return _passthrough(g, method)
    if callable(method) and not isinstance(method, str):
        res = method(g, **kwargs)
        return _passthrough(g, res) if not (isinstance(res, Layout) and list(res.nodes) == list(g._node)) else res
    if not isinstance(method, str):
        raise TypeError(f"method must be a name, a Layout, a mapping or a callable, got {type(method).__name__}")
    name = method.strip().lower().replace("-", "_")
    auto = name == "auto"
    if auto:
        name = auto_method(g)
    fn = METHODS.get(name)
    if fn is None:
        valid = ", ".join(["auto", *sorted(METHODS)])
        raise ValueError(f"unknown layout method {method!r}; valid names: {valid}")
    opts = dict(kwargs)
    opts["seed"] = seed
    if sizes is not None and _accepts(fn, "sizes"):
        opts["sizes"] = sizes
    if auto:
        opts = _filter_kwargs(fn, opts)
    if components not in ("pack", None, "none", False):
        raise ValueError("components must be 'pack' or None")
    if components == "pack" and fn in _PACKED and "fixed" not in opts and len(g) > 1:
        packed = _pack_by_components(g, fn, gap=1.0, **opts)
        if packed is not None:
            return packed
    return fn(g, **opts)


__all__ = [
    "Layout",
    "pack_components",
    "compute",
    "auto_method",
    "METHODS",
    "stress",
    "fruchterman_reingold",
    "force_atlas2",
    "spectral",
    "circular",
    "shell",
    "grid",
    "random",
    "spiral",
    "bipartite",
    "arc",
    "tree",
    "radial",
    "hierarchical",
    "remove_overlaps",
]
