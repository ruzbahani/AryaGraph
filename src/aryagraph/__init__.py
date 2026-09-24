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

"""AryaGraph: graph & DAG visualization, analysis and simulation.

>>> import aryagraph as ag
>>> g = ag.gen.ucalgary_campus()
>>> ag.draw(g, node_color="kind").save("campus.svg")
>>> print(ag.analyze(g))

Namespaces
----------
``ag.alg``      algorithms (paths, centrality, communities, DAG, flow, …)
``ag.layout``   layout engines and the :class:`Layout` type
``ag.sim``      simulations (epidemics, cascades, walks, opinions, scheduling, …)
``ag.gen``      generators, example DAGs and bundled datasets
``ag.io``       file formats and converters (also ``ag.read`` / ``ag.write``)
``ag.style``    themes, palettes, colormaps, shapes
``ag.charts``   small SVG charts
"""

from __future__ import annotations

__version__ = "0.1.1"

from . import algorithms, analysis, charts, generators, io, layout, render, sim, style
from . import algorithms as alg
from . import generators as gen
from .analysis import GraphReport, analyze
from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .io import read, write
from .layout.base import Layout
from .render import Figure, Scene, draw
from .render.animate import animate
from .style.scales import By, by
from .style.themes import Theme, get_theme, register_theme

__all__ = [
    *_core_all,
    "__version__",
    # namespaces
    "alg",
    "algorithms",
    "gen",
    "generators",
    "layout",
    "sim",
    "io",
    "style",
    "charts",
    "render",
    "analysis",
    # top-level API
    "draw",
    "animate",
    "analyze",
    "read",
    "write",
    "by",
    "By",
    "Figure",
    "Scene",
    "Layout",
    "GraphReport",
    "Theme",
    "get_theme",
    "register_theme",
]
