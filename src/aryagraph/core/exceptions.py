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

"""Exception hierarchy.

Every error raised on purpose by AryaGraph derives from :class:`AryaGraphError`, so
callers can catch the library's failures without swallowing unrelated ones.
Where a builtin exception is the natural fit (``KeyError`` for a missing node,
``ValueError`` for a cycle) the AryaGraph error also inherits from it.
"""

from __future__ import annotations

from typing import Any, Hashable, Sequence


class AryaGraphError(Exception):
    """Base class for all AryaGraph errors."""


class NodeNotFound(AryaGraphError, KeyError):
    """A node was requested that is not in the graph."""

    def __init__(self, node: Hashable, message: str | None = None) -> None:
        self.node = node
        super().__init__(message or f"node {node!r} is not in the graph")

    def __str__(self) -> str:  # KeyError would otherwise wrap the text in quotes
        return str(self.args[0])


class EdgeNotFound(AryaGraphError, KeyError):
    """An edge was requested that is not in the graph."""

    def __init__(self, u: Hashable, v: Hashable, message: str | None = None) -> None:
        self.edge = (u, v)
        super().__init__(message or f"edge ({u!r}, {v!r}) is not in the graph")

    def __str__(self) -> str:
        return str(self.args[0])


class CycleError(AryaGraphError, ValueError):
    """An operation needs an acyclic graph but found a cycle.

    ``cycle`` holds the offending cycle as a node list whose first and last
    elements are equal, e.g. ``['a', 'b', 'c', 'a']`` (empty if unknown).
    """

    def __init__(self, message: str, cycle: Sequence[Hashable] = ()) -> None:
        self.cycle = list(cycle)
        super().__init__(message)


class NegativeCycleError(AryaGraphError, ValueError):
    """A shortest-path computation met a negative-weight cycle."""

    def __init__(self, message: str, cycle: Sequence[Hashable] = ()) -> None:
        self.cycle = list(cycle)
        super().__init__(message)


class NegativeWeightError(AryaGraphError, ValueError):
    """A method that needs non-negative edge weights met a negative one."""

    def __init__(self, u: Hashable, v: Hashable, weight: float, hint: str = "") -> None:
        self.edge = (u, v)
        self.weight = weight
        msg = f"negative weight {weight} on edge ({u!r}, {v!r})"
        super().__init__(msg + (f"; {hint}" if hint else ""))


class UnboundedFlowError(AryaGraphError, ValueError):
    """A maximum flow is infinite (an all-infinite-capacity path joins source and sink)."""


class NoPath(AryaGraphError):
    """No path exists between the requested nodes."""

    def __init__(self, source: Any, target: Any, message: str | None = None) -> None:
        self.source = source
        self.target = target
        super().__init__(message or f"no path from {source!r} to {target!r}")


class NotConnected(AryaGraphError, ValueError):
    """The measure is only defined for a connected (or strongly connected) graph."""


class GraphTypeError(AryaGraphError, TypeError):
    """The algorithm does not support this kind of graph (e.g. needs directed)."""


class ConvergenceError(AryaGraphError, RuntimeError):
    """An iterative method did not converge within its iteration budget."""


class DependencyError(AryaGraphError, ImportError):
    """An optional dependency needed for this feature is not installed."""

    def __init__(self, package: str, feature: str, extra: str | None = None) -> None:
        super().__init__(f"{feature} requires the optional package {package!r} (pip install {package})")
        self.package = package
        self.extra = extra


__all__ = [
    "AryaGraphError",
    "NodeNotFound",
    "EdgeNotFound",
    "CycleError",
    "NegativeCycleError",
    "NegativeWeightError",
    "UnboundedFlowError",
    "NoPath",
    "NotConnected",
    "GraphTypeError",
    "ConvergenceError",
    "DependencyError",
]
