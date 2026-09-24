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

"""Rendering: :func:`draw` builds a :class:`Figure` (SVG, interactive HTML, PNG, PDF)."""

from __future__ import annotations

from typing import Any

from .figure import Figure
from .scene import Scene, build_scene


def draw(
    g: Any,
    *,
    layout: Any = "auto",
    seed: int = 0,
    layout_options: dict[str, Any] | None = None,
    # nodes
    node_color: Any = None,
    node_size: Any = None,
    node_shape: Any = None,
    node_opacity: Any = 1.0,
    labels: Any = "auto",
    label_position: str = "auto",
    label_size: float | None = None,
    label_max_width: float = 160,
    label_collisions: str = "hide",
    tooltip: Any = None,
    # edges
    edge_color: Any = None,
    edge_width: Any = None,
    edge_opacity: float | None = None,
    edge_style: str = "auto",
    edge_label: Any = None,
    arrows: bool | None = None,
    arrow_size: float | None = None,
    curvature: float = 0.16,
    # emphasis
    highlight: Any = None,
    highlight_path: Any = None,
    highlight_color: str | None = None,
    # canvas
    theme: Any = "light",
    width: float | None = None,
    height: float | None = None,
    scale: float | None = None,
    padding: float | None = None,
    background: Any = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    legend: Any = "auto",
    avoid_overlap: bool = True,
    base_size: float | None = None,
) -> Figure:
    """Draw a graph.

    Every visual channel accepts a constant, an attribute name, a
    ``{node: value}`` mapping (e.g. the result of an algorithm), a callable, a
    sequence, or a :func:`aryagraph.by` spec. Data-driven channels get a
    legend automatically.

    Parameters
    ----------
    layout:
        ``"auto"`` (hierarchical for DAGs, stress otherwise), any name accepted
        by :func:`aryagraph.layout.compute`, a :class:`~aryagraph.layout.Layout`, a
        ``{node: (x, y)}`` mapping, or a callable ``f(g) -> Layout``.
    node_color, node_size, node_shape, node_opacity:
        Node encodings. Categorical colors use the fixed-order palette and
        fold the tail into "Other"; numeric colors use a sequential scale;
        sizes map values to *area*.
    labels:
        ``"auto"`` (all labels for ≤ 80 nodes, else the 30 most prominent),
        ``True``/``False``, an int (top-k), or an attribute / mapping / callable
        giving the text. Labels never overlap nodes or each other; those that
        cannot be placed are hidden (``label_collisions="show"`` keeps them).
    label_position:
        ``"auto"``, ``"center"`` (inside the node), ``"right"``, ``"left"``,
        ``"above"``, ``"below"`` or a diagonal like ``"below-right"``.
    edge_style:
        ``"auto"``, ``"straight"``, ``"curved"``, ``"flow"`` (S-curves along the
        hierarchy), ``"orthogonal"``, ``"spline"``.
    highlight, highlight_path:
        Nodes/edges to emphasise (everything else is dimmed); a path is drawn
        along its consecutive edges.
    theme:
        ``"light"``, ``"dark"``, ``"paper"``, ``"blueprint"`` or a :class:`~aryagraph.style.Theme`.
    width, height:
        Display size; the drawing is fitted into it. Omit for natural size.
    """
    options = {k: v for k, v in locals().items() if k != "g"}
    scene = build_scene(g, **options)
    return Figure(scene, graph=g, extras={"options": options})


__all__ = ["draw", "Figure", "Scene", "build_scene"]
