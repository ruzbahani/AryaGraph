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

"""Palettes: categorical, sequential, diverging and status colors.

The default categorical palette is an eight-slot set whose *order* was
validated for color-vision deficiency: every adjacent pair stays ≥ 8 ΔE apart
under simulated protanopia/deuteranopia/tritanopia, in light and dark mode. The
order is the safety mechanism: slots are assigned in order and never cycled;
a ninth category folds into "Other".
"""

from __future__ import annotations

from .colors import Colormap, from_oklch, to_oklch

# --------------------------------------------------------------------------- #
# categorical (identity)
# --------------------------------------------------------------------------- #
CATEGORICAL_LIGHT: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
CATEGORICAL_DARK: tuple[str, ...] = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)
OTHER_LIGHT = "#a3a199"
OTHER_DARK = "#6b6a64"

# --------------------------------------------------------------------------- #
# status (state): reserved, never used as a series color
# --------------------------------------------------------------------------- #
STATUS: dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# --------------------------------------------------------------------------- #
# sequential (magnitude)
# --------------------------------------------------------------------------- #
BLUE_RAMP: dict[int, str] = {
    100: "#cde2fb",
    150: "#b7d3f6",
    200: "#9ec5f4",
    250: "#86b6ef",
    300: "#6da7ec",
    350: "#5598e7",
    400: "#3987e5",
    450: "#2a78d6",
    500: "#256abf",
    550: "#1c5cab",
    600: "#184f95",
    650: "#104281",
    700: "#0d366b",
}


def hue_ramp(color: str) -> dict[int, str]:
    """A 100–700 ramp in *color*'s hue with the lightness/chroma profile of the blue ramp."""
    _, _, hue = to_oklch(color)
    out = {}
    for step, ref in BLUE_RAMP.items():
        L, C, _ = to_oklch(ref)
        out[step] = from_oklch(L, C, hue)
    return out


def _ramp_map(name: str, ramp: dict[int, str], lo: int = 100, hi: int = 700) -> Colormap:
    return Colormap(name, tuple(c for s, c in sorted(ramp.items()) if lo <= s <= hi))


_SCIENTIFIC: dict[str, tuple[str, ...]] = {
    "viridis": ("#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"),
    "magma": ("#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f", "#cd4071", "#f1605d", "#fd9668", "#feca8d", "#fcfdbf"),
    "inferno": ("#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"),
    "plasma": ("#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"),
    "cividis": ("#00224e", "#123570", "#3b496c", "#575d6d", "#707173", "#8a8779", "#a69d75", "#c4b56c", "#e4cf5b", "#fee838"),
}

_HUES = {
    "blue": CATEGORICAL_LIGHT[0],
    "orange": CATEGORICAL_LIGHT[1],
    "aqua": CATEGORICAL_LIGHT[2],
    "green": "#1f8a3a",
    "violet": CATEGORICAL_LIGHT[6],
    "red": CATEGORICAL_LIGHT[7],
    "magenta": CATEGORICAL_LIGHT[4],
    "gray": "#72716b",
}

DIVERGING_MID_LIGHT = "#f0efec"
DIVERGING_MID_DARK = "#383835"


def colormap(name: str | Colormap, mode: str = "light") -> Colormap:
    """Look up a colormap by name.

    Sequential hue ramps (``blue``, ``orange``, ``aqua``, ``green``, ``violet``, ``red``,
    ``magenta``, ``gray``) run low → high as light → dark in light mode and
    dark → light in dark mode, so low values recede toward the surface in both.
    They start at step 250 (light) / stop at step 600 (dark) so every mark keeps
    ≥ 2:1 contrast against the surface. Also available: ``viridis``, ``magma``,
    ``inferno``, ``plasma``, ``cividis`` and the diverging ``blue_red`` /
    ``blue_orange``. Append ``_r`` to reverse any map.
    """
    if isinstance(name, Colormap):
        return name
    key = name.lower()
    if key.endswith("_r"):
        return colormap(key[:-2], mode).reversed()
    if key in _SCIENTIFIC:
        return Colormap(key, _SCIENTIFIC[key])
    if key in _HUES:
        ramp = BLUE_RAMP if key == "blue" else hue_ramp(_HUES[key])
        if mode == "dark":
            return _ramp_map(key, ramp, 100, 600).reversed()
        return _ramp_map(key, ramp, 250, 700)
    if key in ("blue_red", "diverging"):
        return diverging("blue", "red", mode)
    if key == "blue_orange":
        return diverging("blue", "orange", mode)
    raise ValueError(f"unknown colormap {name!r}; available: {', '.join(available_colormaps())}")


def diverging(low: str = "blue", high: str = "red", mode: str = "light") -> Colormap:
    """Two opposing hue arms through a neutral gray midpoint (equal steps per arm)."""
    lo = BLUE_RAMP if low == "blue" else hue_ramp(_HUES.get(low, low))
    hi = BLUE_RAMP if high == "blue" else hue_ramp(_HUES.get(high, high))
    mid = DIVERGING_MID_DARK if mode == "dark" else DIVERGING_MID_LIGHT
    if mode == "dark":
        arm = (200, 300, 400)
    else:
        arm = (650, 500, 350)
    stops = [lo[arm[0]], lo[arm[1]], lo[arm[2]], mid, hi[arm[2]], hi[arm[1]], hi[arm[0]]]
    return Colormap(f"{low}_{high}", tuple(stops))


def available_colormaps() -> list[str]:
    return [*_HUES, *_SCIENTIFIC, "blue_red", "blue_orange"]


__all__ = [
    "CATEGORICAL_LIGHT",
    "CATEGORICAL_DARK",
    "OTHER_LIGHT",
    "OTHER_DARK",
    "STATUS",
    "BLUE_RAMP",
    "DIVERGING_MID_LIGHT",
    "DIVERGING_MID_DARK",
    "hue_ramp",
    "colormap",
    "diverging",
    "available_colormaps",
]
