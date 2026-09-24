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

"""Result containers returned by analysis functions.

:class:`NodeMap` and :class:`EdgeMap` are plain ``dict`` subclasses (anything
that accepts a dict accepts them) with a few helpers for the questions people
ask of a metric: *who ranks highest*, *what does the distribution look like*,
*give me an array aligned with this node order*.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Hashable, Iterable, Sequence

import numpy as np


class _MetricMap(dict):
    """Shared behaviour of :class:`NodeMap` and :class:`EdgeMap`."""

    _kind = "items"

    def __init__(self, data: Any = (), name: str | None = None) -> None:
        super().__init__(data)
        self.name = name

    # -- ranking ---------------------------------------------------------
    def top(self, k: int = 10) -> list[tuple[Any, Any]]:
        """The *k* highest-valued ``(key, value)`` pairs, highest first (ties keep insertion order)."""
        return sorted(self.items(), key=lambda kv: kv[1], reverse=True)[: max(k, 0)]

    def bottom(self, k: int = 10) -> list[tuple[Any, Any]]:
        """The *k* lowest-valued ``(key, value)`` pairs, lowest first."""
        return sorted(self.items(), key=lambda kv: kv[1])[: max(k, 0)]

    def argmax(self) -> Any:
        """Key with the largest value (first one on ties)."""
        if not self:
            raise ValueError(f"argmax of an empty {type(self).__name__}")
        return max(self, key=self.__getitem__)

    def argmin(self) -> Any:
        """Key with the smallest value (first one on ties)."""
        if not self:
            raise ValueError(f"argmin of an empty {type(self).__name__}")
        return min(self, key=self.__getitem__)

    def rank(self, descending: bool = True) -> "_MetricMap":
        """Dense 1-based rank of every key (1 = highest when *descending*)."""
        ordered = sorted(set(self.values()), reverse=descending)
        position = {v: i + 1 for i, v in enumerate(ordered)}
        return type(self)({k: position[v] for k, v in self.items()}, name=f"rank({self.name})" if self.name else "rank")

    # -- numeric views ---------------------------------------------------
    def to_array(self, keys: Sequence[Any] | None = None, default: float = math.nan) -> np.ndarray:
        """Values as a float array, ordered by *keys* (insertion order if omitted)."""
        keys = list(self) if keys is None else keys
        return np.array([self.get(k, default) for k in keys], dtype=float)

    def normalized(self, method: str = "max") -> "_MetricMap":
        """Rescaled copy.

        ``"max"`` divides by the max absolute value, ``"sum"`` by the total,
        ``"minmax"`` maps onto [0, 1], ``"zscore"`` standardises.
        """
        vals = self.to_array()
        if vals.size == 0:
            return type(self)({}, name=self.name)
        if method == "max":
            scale = np.max(np.abs(vals))
            out = vals / scale if scale else np.zeros_like(vals)
        elif method == "sum":
            total = vals.sum()
            out = vals / total if total else np.zeros_like(vals)
        elif method == "minmax":
            lo, hi = vals.min(), vals.max()
            out = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
        elif method == "zscore":
            sd = vals.std()
            out = (vals - vals.mean()) / sd if sd else np.zeros_like(vals)
        else:
            raise ValueError(f"unknown normalization {method!r}; use 'max', 'sum', 'minmax' or 'zscore'")
        return type(self)(zip(self.keys(), out.tolist()), name=self.name)

    def map(self, fn: Callable[[Any], Any]) -> "_MetricMap":
        """Copy with *fn* applied to every value."""
        return type(self)({k: fn(v) for k, v in self.items()}, name=self.name)

    def filter(self, predicate: Callable[[Any, Any], bool]) -> "_MetricMap":
        """Copy keeping the ``(key, value)`` pairs for which *predicate* is true."""
        return type(self)({k: v for k, v in self.items() if predicate(k, v)}, name=self.name)

    def describe(self) -> dict[str, float]:
        """Summary statistics: count, mean, std, min, quartiles, max."""
        vals = self.to_array()
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return {"count": 0}
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        return {
            "count": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "25%": float(q1),
            "50%": float(med),
            "75%": float(q3),
            "max": float(vals.max()),
        }

    def to_pandas(self):
        """A ``pandas.Series`` named after the metric (pandas is optional)."""
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - exercised only without pandas
            from .exceptions import DependencyError

            raise DependencyError("pandas", "to_pandas()", extra="interop") from None
        return pd.Series(dict(self), name=self.name)

    # -- display ---------------------------------------------------------
    def __repr__(self) -> str:
        label = f"{self.name!r}, " if self.name else ""
        head = self.top(3)
        shown = ", ".join(f"{k!r}: {_fmt(v)}" for k, v in head)
        more = ", …" if len(self) > 3 else ""
        return f"{type(self).__name__}({label}{len(self)} {self._kind}; top: {{{shown}{more}}})"

    def copy(self) -> "_MetricMap":
        return type(self)(self, name=self.name)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return repr(v)


class NodeMap(_MetricMap):
    """``dict`` of node → value with ranking and statistics helpers.

    >>> pr = aryagraph.alg.pagerank(g)
    >>> pr.top(3)          # [(node, score), ...]
    >>> pr.describe()      # {'mean': ..., 'max': ...}
    """

    _kind = "nodes"


class EdgeMap(_MetricMap):
    """``dict`` of ``(u, v)`` edge → value, same helpers as :class:`NodeMap`."""

    _kind = "edges"


def as_nodemap(values: Iterable[tuple[Hashable, Any]] | dict, name: str | None = None) -> NodeMap:
    """Wrap a mapping or pair-iterable as a :class:`NodeMap`."""
    return NodeMap(values, name=name)


__all__ = ["NodeMap", "EdgeMap", "as_nodemap"]
