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

"""Visual vocabulary: colors, palettes, themes, scales, text metrics and node shapes."""

from .colors import Colormap, contrast_ratio, darken, lighten, mix, parse, readable_on, to_hex, with_alpha
from .palettes import CATEGORICAL_DARK, CATEGORICAL_LIGHT, STATUS, available_colormaps, colormap, diverging
from .scales import By, Legend, LegendEntry, by
from .shapes import SHAPES, Shape, get_shape
from .text import is_rtl, text_width, truncate, wrap
from .themes import BLUEPRINT, DARK, LIGHT, PAPER, THEMES, Theme, get_theme, register_theme

__all__ = [
    "Colormap",
    "colormap",
    "diverging",
    "available_colormaps",
    "parse",
    "to_hex",
    "mix",
    "lighten",
    "darken",
    "with_alpha",
    "contrast_ratio",
    "readable_on",
    "CATEGORICAL_LIGHT",
    "CATEGORICAL_DARK",
    "STATUS",
    "By",
    "by",
    "Legend",
    "LegendEntry",
    "Shape",
    "SHAPES",
    "get_shape",
    "text_width",
    "wrap",
    "truncate",
    "is_rtl",
    "Theme",
    "THEMES",
    "LIGHT",
    "DARK",
    "PAPER",
    "BLUEPRINT",
    "get_theme",
    "register_theme",
]
