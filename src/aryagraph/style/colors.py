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

"""Color handling: parsing, perceptual (OKLab/OKLCH) math, contrast and colormaps.

Colors travel through AryaGraph as CSS hex strings (``"#2a78d6"``, or
``"#2a78d680"`` with alpha). All interpolation happens in OKLab, so gradients
change lightness evenly instead of passing through muddy midpoints.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

RGBA = tuple[float, float, float, float]

# fmt: off
_CSS_NAMED = dict(item.split(":") for item in """
aliceblue:f0f8ff antiquewhite:faebd7 aqua:00ffff aquamarine:7fffd4 azure:f0ffff beige:f5f5dc bisque:ffe4c4
black:000000 blanchedalmond:ffebcd blue:0000ff blueviolet:8a2be2 brown:a52a2a burlywood:deb887 cadetblue:5f9ea0
chartreuse:7fff00 chocolate:d2691e coral:ff7f50 cornflowerblue:6495ed cornsilk:fff8dc crimson:dc143c cyan:00ffff
darkblue:00008b darkcyan:008b8b darkgoldenrod:b8860b darkgray:a9a9a9 darkgreen:006400 darkgrey:a9a9a9
darkkhaki:bdb76b darkmagenta:8b008b darkolivegreen:556b2f darkorange:ff8c00 darkorchid:9932cc darkred:8b0000
darksalmon:e9967a darkseagreen:8fbc8f darkslateblue:483d8b darkslategray:2f4f4f darkslategrey:2f4f4f
darkturquoise:00ced1 darkviolet:9400d3 deeppink:ff1493 deepskyblue:00bfff dimgray:696969 dimgrey:696969
dodgerblue:1e90ff firebrick:b22222 floralwhite:fffaf0 forestgreen:228b22 fuchsia:ff00ff gainsboro:dcdcdc
ghostwhite:f8f8ff gold:ffd700 goldenrod:daa520 gray:808080 green:008000 greenyellow:adff2f grey:808080
honeydew:f0fff0 hotpink:ff69b4 indianred:cd5c5c indigo:4b0082 ivory:fffff0 khaki:f0e68c lavender:e6e6fa
lavenderblush:fff0f5 lawngreen:7cfc00 lemonchiffon:fffacd lightblue:add8e6 lightcoral:f08080 lightcyan:e0ffff
lightgoldenrodyellow:fafad2 lightgray:d3d3d3 lightgreen:90ee90 lightgrey:d3d3d3 lightpink:ffb6c1
lightsalmon:ffa07a lightseagreen:20b2aa lightskyblue:87cefa lightslategray:778899 lightslategrey:778899
lightsteelblue:b0c4de lightyellow:ffffe0 lime:00ff00 limegreen:32cd32 linen:faf0e6 magenta:ff00ff maroon:800000
mediumaquamarine:66cdaa mediumblue:0000cd mediumorchid:ba55d3 mediumpurple:9370db mediumseagreen:3cb371
mediumslateblue:7b68ee mediumspringgreen:00fa9a mediumturquoise:48d1cc mediumvioletred:c71585
midnightblue:191970 mintcream:f5fffa mistyrose:ffe4e1 moccasin:ffe4b5 navajowhite:ffdead navy:000080
oldlace:fdf5e6 olive:808000 olivedrab:6b8e23 orange:ffa500 orangered:ff4500 orchid:da70d6 palegoldenrod:eee8aa
palegreen:98fb98 paleturquoise:afeeee palevioletred:db7093 papayawhip:ffefd5 peachpuff:ffdab9 peru:cd853f
pink:ffc0cb plum:dda0dd powderblue:b0e0e6 purple:800080 rebeccapurple:663399 red:ff0000 rosybrown:bc8f8f
royalblue:4169e1 saddlebrown:8b4513 salmon:fa8072 sandybrown:f4a460 seagreen:2e8b57 seashell:fff5ee
sienna:a0522d silver:c0c0c0 skyblue:87ceeb slateblue:6a5acd slategray:708090 slategrey:708090 snow:fffafa
springgreen:00ff7f steelblue:4682b4 tan:d2b48c teal:008080 thistle:d8bfd8 tomato:ff6347 turquoise:40e0d0
violet:ee82ee wheat:f5deb3 white:ffffff whitesmoke:f5f5f5 yellow:ffff00 yellowgreen:9acd32
""".split())
# fmt: on

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_FUNC_RE = re.compile(r"^(rgba?|hsla?)\(\s*([^)]*)\)$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# parsing & formatting
# --------------------------------------------------------------------------- #
def parse(color: str | Sequence[float]) -> RGBA:
    """Parse a CSS color (hex, ``rgb()/rgba()``, ``hsl()/hsla()``, named, ``transparent``)
    or an RGB(A) tuple of floats in [0, 1] into an RGBA float tuple."""
    if isinstance(color, str):
        s = color.strip().lower()
        if s == "transparent" or s == "none":
            return (0.0, 0.0, 0.0, 0.0)
        if s in _CSS_NAMED:
            s = "#" + _CSS_NAMED[s]
        m = _HEX_RE.match(s)
        if m:
            h = m.group(1)
            if len(h) in (3, 4):
                h = "".join(ch * 2 for ch in h)
            vals = [int(h[i : i + 2], 16) / 255 for i in range(0, len(h), 2)]
            return (vals[0], vals[1], vals[2], vals[3] if len(vals) == 4 else 1.0)
        m = _FUNC_RE.match(s)
        if m:
            kind = m.group(1)
            parts = [p for p in re.split(r"[\s,/]+", m.group(2).strip()) if p]
            if len(parts) not in (3, 4):
                raise ValueError(f"cannot parse color {color!r}")
            alpha = _parse_alpha(parts[3]) if len(parts) == 4 else 1.0
            if kind.startswith("rgb"):
                rgb = [_parse_channel(p) for p in parts[:3]]
                return (rgb[0], rgb[1], rgb[2], alpha)
            hue = float(parts[0].replace("deg", ""))
            sat = float(parts[1].rstrip("%")) / 100
            lig = float(parts[2].rstrip("%")) / 100
            r, g, b = _hsl_to_rgb(hue, sat, lig)
            return (r, g, b, alpha)
        raise ValueError(f"cannot parse color {color!r}")
    vals = [float(v) for v in color]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4 or any(v < 0 or v > 1 for v in vals):
        raise ValueError(f"color tuples need 3-4 floats in [0, 1], got {color!r}")
    return (vals[0], vals[1], vals[2], vals[3])


def _parse_channel(p: str) -> float:
    if p.endswith("%"):
        return min(max(float(p[:-1]) / 100, 0.0), 1.0)
    return min(max(float(p) / 255, 0.0), 1.0)


def _parse_alpha(p: str) -> float:
    if p.endswith("%"):
        return min(max(float(p[:-1]) / 100, 0.0), 1.0)
    return min(max(float(p), 0.0), 1.0)


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    h = (h % 360) / 360
    if s == 0:
        return (l, l, l)
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q

    def hue(t: float) -> float:
        t %= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    return (hue(h + 1 / 3), hue(h), hue(h - 1 / 3))


def is_color(value: object) -> bool:
    """True if *value* is a string that parses as a CSS color."""
    if not isinstance(value, str):
        return False
    try:
        parse(value)
    except ValueError:
        return False
    return True


def to_hex(color: str | Sequence[float], keep_alpha: bool = True) -> str:
    """Canonical lowercase hex (``#rrggbb`` or ``#rrggbbaa`` when translucent)."""
    r, g, b, a = parse(color) if not isinstance(color, tuple) or len(color) not in (3, 4) else _tuple4(color)
    out = "#" + "".join(f"{round(min(max(c, 0.0), 1.0) * 255):02x}" for c in (r, g, b))
    if keep_alpha and a < 1.0:
        out += f"{round(a * 255):02x}"
    return out


def _tuple4(c: Sequence[float]) -> RGBA:
    return (float(c[0]), float(c[1]), float(c[2]), float(c[3]) if len(c) == 4 else 1.0)


def split_alpha(color: str) -> tuple[str, float]:
    """Split *color* into ``(opaque_hex, alpha)``; SVG renderers treat ``fill-opacity`` more uniformly than 8-digit hex."""
    r, g, b, a = parse(color)
    return to_hex((r, g, b, 1.0)), a


# --------------------------------------------------------------------------- #
# OKLab / OKLCH
# --------------------------------------------------------------------------- #
def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _from_linear(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def to_oklab(color: str | Sequence[float]) -> tuple[float, float, float]:
    """OKLab ``(L, a, b)`` of a color (alpha ignored)."""
    r, g, b, _ = parse(color) if isinstance(color, str) else _tuple4(color)
    r, g, b = _to_linear(r), _to_linear(g), _to_linear(b)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(x) ** (1 / 3), x) for x in (l, m, s))
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _oklab_to_linear_rgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    return (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )


def from_oklab(L: float, a: float, b: float, alpha: float = 1.0) -> str:
    """Hex color from OKLab, reducing chroma (not clipping channels) to stay in sRGB gamut."""
    rgb = _oklab_to_linear_rgb(L, a, b)
    if any(c < -1e-6 or c > 1 + 1e-6 for c in rgb):
        C = math.hypot(a, b)
        h = math.atan2(b, a)
        lo, hi = 0.0, C
        for _ in range(24):
            mid = (lo + hi) / 2
            test = _oklab_to_linear_rgb(L, mid * math.cos(h), mid * math.sin(h))
            if all(-1e-6 <= c <= 1 + 1e-6 for c in test):
                lo = mid
            else:
                hi = mid
        rgb = _oklab_to_linear_rgb(L, lo * math.cos(h), lo * math.sin(h))
    srgb = tuple(_from_linear(min(max(c, 0.0), 1.0)) for c in rgb)
    return to_hex((srgb[0], srgb[1], srgb[2], alpha))


def to_oklch(color: str | Sequence[float]) -> tuple[float, float, float]:
    """OKLCH ``(L, C, h°)``."""
    L, a, b = to_oklab(color)
    return (L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360)


def from_oklch(L: float, C: float, h: float, alpha: float = 1.0) -> str:
    """Hex color from OKLCH (gamut-mapped by chroma reduction)."""
    t = math.radians(h)
    return from_oklab(L, C * math.cos(t), C * math.sin(t), alpha)


def delta_e(c1: str, c2: str) -> float:
    """Perceptual distance (OKLab Euclidean × 100; ≈ 2 is just noticeable)."""
    a, b = to_oklab(c1), to_oklab(c2)
    return 100 * math.dist(a, b)


# --------------------------------------------------------------------------- #
# manipulation
# --------------------------------------------------------------------------- #
def mix(c1: str, c2: str, t: float = 0.5) -> str:
    """Interpolate from *c1* (t=0) to *c2* (t=1) in OKLab (alpha linearly)."""
    a1, a2 = parse(c1)[3], parse(c2)[3]
    L1, A1, B1 = to_oklab(c1)
    L2, A2, B2 = to_oklab(c2)
    return from_oklab(L1 + (L2 - L1) * t, A1 + (A2 - A1) * t, B1 + (B2 - B1) * t, a1 + (a2 - a1) * t)


def with_alpha(color: str, alpha: float) -> str:
    r, g, b, _ = parse(color)
    return to_hex((r, g, b, min(max(alpha, 0.0), 1.0)))


def lighten(color: str, amount: float = 0.1) -> str:
    """Raise OKLCH lightness by *amount* (0–1 scale), keeping hue and chroma."""
    L, C, h = to_oklch(color)
    return from_oklch(min(L + amount, 1.0), C, h, parse(color)[3])


def darken(color: str, amount: float = 0.1) -> str:
    return lighten(color, -amount)


def desaturate(color: str, amount: float = 0.5) -> str:
    """Scale OKLCH chroma by ``1 - amount``."""
    L, C, h = to_oklch(color)
    return from_oklch(L, C * (1 - amount), h, parse(color)[3])


def composite(color: str, background: str) -> str:
    """Flatten a translucent *color* over an opaque *background*."""
    r, g, b, a = parse(color)
    R, G, B, _ = parse(background)
    return to_hex((r * a + R * (1 - a), g * a + G * (1 - a), b * a + B * (1 - a), 1.0))


# --------------------------------------------------------------------------- #
# contrast (WCAG 2)
# --------------------------------------------------------------------------- #
def luminance(color: str) -> float:
    """WCAG relative luminance."""
    r, g, b, _ = parse(color)
    return 0.2126 * _to_linear(r) + 0.7152 * _to_linear(g) + 0.0722 * _to_linear(b)


def contrast_ratio(c1: str, c2: str) -> float:
    """WCAG contrast ratio in [1, 21]."""
    l1, l2 = sorted((luminance(c1), luminance(c2)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def readable_on(background: str, light: str = "#ffffff", dark: str = "#0b0b0b") -> str:
    """Whichever of *light* / *dark* text contrasts more with *background*."""
    return light if contrast_ratio(light, background) >= contrast_ratio(dark, background) else dark


def label_on(fill: str, light: str = "#ffffff", dark: str = "#0b0b0b") -> str:
    """Text color for a short label set *inside* a solid colored mark.

    White is kept down to 3.8:1 (short medium-weight labels on saturated fills
    read better light), and below that the higher-contrast option wins.
    """
    if contrast_ratio(light, fill) >= 3.8:
        return light
    return readable_on(fill, light, dark)


# --------------------------------------------------------------------------- #
# colormaps
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Colormap:
    """A continuous color scale over [0, 1], interpolated in OKLab between *stops*.

    >>> cmap = Colormap("heat", ["#fff5eb", "#d94801"])
    >>> cmap(0.5)
    """

    name: str
    stops: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.stops) < 2:
            raise ValueError("a colormap needs at least two stops")
        object.__setattr__(self, "stops", tuple(to_hex(s) for s in self.stops))
        object.__setattr__(self, "_lab", np.array([to_oklab(s) for s in self.stops]))

    def __call__(self, t: float) -> str:
        t = 0.0 if t is None or math.isnan(t) else min(max(float(t), 0.0), 1.0)
        lab = self._lab  # type: ignore[attr-defined]
        pos = t * (len(lab) - 1)
        i = min(int(pos), len(lab) - 2)
        f = pos - i
        L, a, b = lab[i] + (lab[i + 1] - lab[i]) * f
        return from_oklab(float(L), float(a), float(b))

    def many(self, ts: Iterable[float]) -> list[str]:
        return [self(t) for t in ts]

    def sample(self, n: int) -> list[str]:
        """*n* evenly spaced colors from the start to the end of the map."""
        if n <= 1:
            return [self(0.5)] if n == 1 else []
        return [self(i / (n - 1)) for i in range(n)]

    def reversed(self) -> "Colormap":
        return Colormap(self.name + "_r", tuple(reversed(self.stops)))

    def truncated(self, lo: float = 0.0, hi: float = 1.0, n: int = 9) -> "Colormap":
        """The sub-range [lo, hi] of this map, re-sampled with *n* stops."""
        return Colormap(f"{self.name}[{lo:g}:{hi:g}]", tuple(self(lo + (hi - lo) * i / (n - 1)) for i in range(n)))

    def css_gradient(self, direction: str = "to right", n: int = 11) -> str:
        cols = self.sample(n)
        return f"linear-gradient({direction}, " + ", ".join(f"{c} {100 * i / (n - 1):.1f}%" for i, c in enumerate(cols)) + ")"

    def _repr_svg_(self) -> str:
        n = 64
        cells = "".join(f'<rect x="{i * 4}" y="0" width="4.5" height="20" fill="{c}"/>' for i, c in enumerate(self.sample(n)))
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{n * 4}" height="20">{cells}</svg>'


__all__ = [
    "RGBA",
    "parse",
    "is_color",
    "to_hex",
    "split_alpha",
    "to_oklab",
    "from_oklab",
    "to_oklch",
    "from_oklch",
    "delta_e",
    "mix",
    "with_alpha",
    "lighten",
    "darken",
    "desaturate",
    "composite",
    "luminance",
    "contrast_ratio",
    "readable_on",
    "Colormap",
]
