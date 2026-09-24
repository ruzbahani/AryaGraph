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

"""Themes: every color, font and size a rendering needs, named by role.

Renderers never hard-code a color; they ask the theme for a role
(``surface``, ``ink``, ``edge``, ``series[i]`` …). A theme is therefore the
single place to restyle everything. Dark mode is its own tuned set of
steps, not an automatic inversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .colors import Colormap
from .palettes import (
    CATEGORICAL_DARK,
    CATEGORICAL_LIGHT,
    OTHER_DARK,
    OTHER_LIGHT,
    STATUS,
    colormap,
    diverging,
)

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Vazirmatn", sans-serif'
MONO_STACK = 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace'


@dataclass(frozen=True)
class Theme:
    """A complete visual vocabulary. Create variants with :meth:`with_`."""

    name: str
    mode: str  # "light" | "dark"

    # surfaces & ink
    background: str
    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    border: str

    # graph marks
    node_fill: str
    node_ring: str  # surface-colored ring separating overlapping nodes
    node_ring_width: float
    edge: str
    edge_opacity: float
    edge_highlight: str
    dim_opacity: float

    # data colors
    categorical: tuple[str, ...]
    other: str
    status: dict[str, str]
    sequential: str  # colormap name
    diverging_pair: tuple[str, str]

    # typography
    font: str = FONT_STACK
    mono: str = MONO_STACK
    font_size: float = 12.0
    label_size: float = 11.5
    title_size: float = 17.0
    subtitle_size: float = 12.5
    label_weight: int = 500
    title_weight: int = 600

    # geometry defaults
    node_size: float = 14.0
    edge_width: float = 1.25
    arrow_size: float = 8.0
    corner_radius: float = 6.0
    padding: float = 24.0

    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def with_(self, **changes: Any) -> "Theme":
        """Copy with some fields replaced: ``theme.with_(font_size=14, edge='#999')``."""
        return replace(self, **changes)

    def series(self, i: int) -> str:
        """Categorical slot *i* (0-based). Slots are never cycled; callers fold past the last."""
        return self.categorical[i]

    def role_color(self, role: str) -> str:
        """Color for a semantic role: ``neutral``, ``muted``, ``accent``, or a status name."""
        if role in self.status:
            return self.status[role]
        if role == "accent":
            return self.categorical[0]
        if role == "muted":
            return self.extras.get("muted_fill", self.other)
        if role == "neutral":
            return self.extras.get("neutral_fill", self.node_fill_neutral)
        raise ValueError(f"unknown color role {role!r}")

    @property
    def node_fill_neutral(self) -> str:
        return self.extras.get("neutral_fill", "#b7b5ac" if self.mode == "light" else "#5d5c57")

    def sequential_map(self, name: str | Colormap | None = None) -> Colormap:
        return colormap(name or self.sequential, self.mode)

    def diverging_map(self, name: str | Colormap | None = None) -> Colormap:
        if name is not None:
            return colormap(name, self.mode)
        return diverging(*self.diverging_pair, mode=self.mode)

    def css_vars(self) -> dict[str, str]:
        """Role → value map used by the HTML backend's CSS custom properties."""
        return {
            "--ag-bg": self.background,
            "--ag-surface": self.surface,
            "--ag-ink": self.ink,
            "--ag-ink-2": self.ink_secondary,
            "--ag-ink-3": self.ink_muted,
            "--ag-grid": self.grid,
            "--ag-axis": self.axis,
            "--ag-border": self.border,
            "--ag-edge": self.edge,
            "--ag-accent": self.categorical[0],
            "--ag-highlight": self.edge_highlight,
            "--ag-font": self.font,
            "--ag-mono": self.mono,
        }


LIGHT = Theme(
    name="light",
    mode="light",
    background="#f9f9f7",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    border="rgba(11,11,11,0.10)",
    node_fill=CATEGORICAL_LIGHT[0],
    node_ring="#fcfcfb",
    node_ring_width=1.75,
    edge="#9d9b93",
    edge_opacity=0.55,
    edge_highlight="#2a78d6",
    dim_opacity=0.14,
    categorical=CATEGORICAL_LIGHT,
    other=OTHER_LIGHT,
    status=dict(STATUS),
    sequential="blue",
    diverging_pair=("blue", "red"),
)

DARK = Theme(
    name="dark",
    mode="dark",
    background="#0d0d0d",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    border="rgba(255,255,255,0.10)",
    node_fill=CATEGORICAL_DARK[0],
    node_ring="#1a1a19",
    node_ring_width=1.75,
    edge="#8a8880",
    edge_opacity=0.45,
    edge_highlight="#5598e7",
    dim_opacity=0.16,
    categorical=CATEGORICAL_DARK,
    other=OTHER_DARK,
    status=dict(STATUS),
    sequential="blue",
    diverging_pair=("blue", "red"),
)

PAPER = LIGHT.with_(
    name="paper",
    background="#ffffff",
    surface="#ffffff",
    ink="#000000",
    ink_secondary="#3d3d3a",
    node_ring="#ffffff",
    edge="#7d7b74",
    edge_opacity=0.6,
    grid="#e6e5df",
)

BLUEPRINT = DARK.with_(
    name="blueprint",
    background="#0b1b33",
    surface="#0f2340",
    ink="#eef4ff",
    ink_secondary="#b5c7e6",
    ink_muted="#7f95bb",
    grid="#1c3558",
    axis="#2a4670",
    node_ring="#0f2340",
    edge="#7f9cc9",
    edge_opacity=0.5,
    edge_highlight="#9ec5f4",
    extras={"neutral_fill": "#51688f", "muted_fill": "#3c5277"},
)

THEMES: dict[str, Theme] = {t.name: t for t in (LIGHT, DARK, PAPER, BLUEPRINT)}


def get_theme(theme: str | Theme | None = None) -> Theme:
    """Resolve a theme name (``light``, ``dark``, ``paper``, ``blueprint``) or pass a Theme through."""
    if theme is None:
        return LIGHT
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme.lower()]
    except KeyError:
        raise ValueError(f"unknown theme {theme!r}; available: {', '.join(THEMES)}") from None


def register_theme(theme: Theme) -> None:
    """Make *theme* available by name to every ``theme=`` argument."""
    THEMES[theme.name.lower()] = theme


__all__ = ["Theme", "LIGHT", "DARK", "PAPER", "BLUEPRINT", "THEMES", "get_theme", "register_theme", "FONT_STACK"]
