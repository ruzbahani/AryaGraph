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

"""Small helpers shared by every subpackage."""

from __future__ import annotations

from typing import Any, Callable, Hashable

import numpy as np

from .exceptions import GraphTypeError, NodeNotFound

WeightSpec = str | Callable[[Hashable, Hashable, dict], float] | None
WeightFn = Callable[[Hashable, Hashable, dict], float]


def weight_fn(weight: WeightSpec, default: float = 1.0) -> WeightFn:
    """Normalise a *weight* argument into ``f(u, v, attrs) -> float``.

    ``None`` means every edge weighs *default*; a string names an edge
    attribute (edges without it weigh *default*); a callable is used as is.
    """
    if weight is None:
        return lambda u, v, d: default
    if callable(weight):
        return weight
    if isinstance(weight, str):
        return lambda u, v, d: d.get(weight, default)
    raise TypeError(f"weight must be None, an attribute name or a callable, got {weight!r}")


def make_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """A numpy ``Generator`` from an int seed, an existing generator, or fresh entropy."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def require_directed(g: Any, what: str) -> None:
    if not g.directed:
        raise GraphTypeError(f"{what} requires a directed graph")


def require_undirected(g: Any, what: str) -> None:
    if g.directed:
        raise GraphTypeError(f"{what} requires an undirected graph (use g.to_undirected())")


def require_node(g: Any, node: Hashable) -> None:
    if node not in g:
        raise NodeNotFound(node)


__all__ = ["weight_fn", "make_rng", "require_directed", "require_undirected", "require_node", "WeightSpec"]
