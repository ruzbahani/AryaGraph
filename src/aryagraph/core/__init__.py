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

"""Core data structures: :class:`Graph`, :class:`DiGraph`, :class:`DAG` and result maps."""

from .dag import DAG
from .exceptions import (
    ConvergenceError,
    CycleError,
    DependencyError,
    EdgeNotFound,
    GraphTypeError,
    AryaGraphError,
    NegativeCycleError,
    NegativeWeightError,
    NodeNotFound,
    NoPath,
    NotConnected,
    UnboundedFlowError,
)
from .graph import DiGraph, Graph
from .results import EdgeMap, NodeMap
from .utils import make_rng, weight_fn

__all__ = [
    "Graph",
    "DiGraph",
    "DAG",
    "NodeMap",
    "EdgeMap",
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
    "make_rng",
    "weight_fn",
]
