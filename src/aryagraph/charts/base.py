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

"""Chart primitives shared by every chart: scales, axes, legend rows, the Chart object.

Charts follow one set of rules: hairline solid gridlines one step off the
surface; 2px lines with round joins; bars ≤ 24px thick with a 4px rounded
data end and a square baseline; a 2px surface gap between touching bars;
text always in ink tokens (a colored key beside it carries identity); a
legend whenever there are two or more series; never two y-axes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from html import escape as _esc
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape

from ..style import text as textm
from ..style.colors import split_alpha
from ..style.numbers import fmt_number, fmt_tick, nice_ticks
from ..style.themes import Theme, get_theme


def f2(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def paint(attr: str, color: str | None) -> str:
    if not color:
        return f' {attr}="none"'
    hexc, a = split_alpha(color)
    out = f' {attr}="{hexc}"'
    if a < 1:
        out += f' {attr}-opacity="{f2(a)}"'
    return out


@dataclass
class Scale:
    """Linear map from a data domain onto a pixel range."""

    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, v: float) -> float:
        if self.d1 == self.d0:
            return (self.r0 + self.r1) / 2
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)

    def invert(self, px: float) -> float:
        if self.r1 == self.r0:
            return self.d0
        return self.d0 + (px - self.r0) / (self.r1 - self.r0) * (self.d1 - self.d0)


@dataclass
class Chart:
    """A rendered chart. Save as .svg / .html / .png, or display inline in notebooks."""

    svg: str
    width: float
    height: float
    theme: Theme
    kind: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_svg(self) -> str:
        return self.svg

    def to_html(self, title: str | None = None) -> str:
        from .runtime import chart_page

        return chart_page(self, title or self.meta.get("title") or self.kind)

    def save(self, path: str | Path, scale: float = 2.0) -> Path:
        path = Path(path)
        ext = path.suffix.lower()
        if ext == ".svg":
            path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + self.svg, encoding="utf-8")
        elif ext in (".html", ".htm"):
            path.write_text(self.to_html(), encoding="utf-8")
        elif ext == ".png":
            from ..render.export import svg_to_png

            svg_to_png(self.svg, path, self.width, self.height, scale=scale)
        elif ext == ".pdf":
            from ..render.export import svg_to_pdf

            svg_to_pdf(self.svg, path, self.width, self.height)
        else:
            raise ValueError(f"unsupported extension {ext!r}")
        return path

    def _repr_html_(self) -> str:
        page = self.to_html()
        return (
            f'<iframe srcdoc="{_esc(page, quote=True)}" style="width:100%;max-width:{int(self.width) + 40}px;'
            f'height:{int(self.height) + 40}px;border:0" sandbox="allow-scripts"></iframe>'
        )

    def __repr__(self) -> str:
        return f"<Chart {self.kind} {self.width:.0f}×{self.height:.0f}>"


class Canvas:
    """Accumulates SVG fragments for one chart with the theme's typography."""

    def __init__(self, width: float, height: float, theme: Theme | str | None, *, uid: str, kind: str) -> None:
        self.w = float(width)
        self.h = float(height)
        self.th = get_theme(theme)
        self.uid = uid
        self.kind = kind
        self.parts: list[str] = []
        self.data: dict[str, Any] = {}

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(
        self,
        x: float,
        y: float,
        s: str,
        *,
        size: float | None = None,
        color: str | None = None,
        anchor: str = "start",
        weight: int = 400,
        cls: str = "",
        tabular: bool = False,
    ) -> None:
        size = size or self.th.font_size
        color = color or self.th.ink
        style = ' style="font-variant-numeric:tabular-nums"' if tabular else ""
        rtl = ' direction="rtl"' if textm.is_rtl(s) else ""
        if rtl:
            anchor = {"start": "end", "end": "start"}.get(anchor, anchor)
        w = f' font-weight="{weight}"' if weight != 400 else ""
        c = f' class="{cls}"' if cls else ""
        self.add(
            f'<text{c} x="{f2(x)}" y="{f2(y)}" font-size="{f2(size)}"{w} text-anchor="{anchor}"{rtl}{style}'
            f"{paint('fill', color)}>{escape(str(s))}</text>"
        )

    def header(self, title: str | None, subtitle: str | None, pad: float) -> float:
        """Draw title/subtitle; returns the y where content can start."""
        y = pad
        if title:
            y += self.th.title_size * 0.95
            self.text(pad, y, title, size=self.th.title_size - 2, weight=self.th.title_weight)
            y += 6
        if subtitle:
            y += self.th.subtitle_size * 1.2
            self.text(pad, y, subtitle, size=self.th.subtitle_size - 0.5, color=self.th.ink_secondary)
            y += 4
        return y + (12 if (title or subtitle) else 0)

    def legend_row(self, x: float, y: float, items: Sequence[tuple[str, str, str]], max_width: float) -> float:
        """Legend entries ``(label, color, mark)`` in a wrapping row; returns the height used."""
        fs = self.th.font_size
        cx, cy = x, y
        row_h = fs * 1.7
        for label, color, mark in items:
            w = (22 if mark == "line" else 16) + textm.text_width(label, fs) + 16
            if cx > x and cx + w > x + max_width:
                cx = x
                cy += row_h
            if mark == "line":
                self.add(f'<line x1="{f2(cx)}" y1="{f2(cy)}" x2="{f2(cx + 16)}" y2="{f2(cy)}"{paint("stroke", color)} stroke-width="2.5" stroke-linecap="round"/>')
                tx = cx + 22
            elif mark == "band":
                self.add(f'<rect x="{f2(cx)}" y="{f2(cy - 5)}" width="16" height="10" rx="2"{paint("fill", color)} fill-opacity="0.25"/>')
                tx = cx + 22
            else:
                self.add(f'<rect x="{f2(cx)}" y="{f2(cy - 5)}" width="10" height="10" rx="2"{paint("fill", color)}/>')
                tx = cx + 16
            self.text(tx, cy + fs * 0.35, label, color=self.th.ink_secondary)
            cx += w
        return cy - y + row_h

    def finish(self, title: str | None = None, desc: str | None = None, *, data: dict | None = None) -> Chart:
        th = self.th
        ttl = escape(title or self.kind)
        payload = ""
        if data is not None:
            # hover data for the HTML runtime; <metadata> is inert in every SVG consumer
            raw = json.dumps(data, separators=(",", ":"), default=float)
            payload = f'<metadata class="ag-chart-data">{escape(raw)}</metadata>'
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" class="ag-chart-svg" id="{self.uid}" width="{f2(self.w)}" height="{f2(self.h)}"'
            f' viewBox="0 0 {f2(self.w)} {f2(self.h)}" role="img" aria-labelledby="{self.uid}-t {self.uid}-d" font-family="{escape(th.font, {chr(34): "&quot;"})}">'
            f'<title id="{self.uid}-t">{ttl}</title><desc id="{self.uid}-d">{escape(desc or ttl)}</desc>{payload}'
            f'<rect width="{f2(self.w)}" height="{f2(self.h)}"{paint("fill", th.surface)}/>'
            + "".join(self.parts)
            + "</svg>"
        )
        return Chart(svg, self.w, self.h, th, self.kind, meta={"title": title, **self.data})


def y_axis(cv: Canvas, ys: Scale, ticks: list[float], x0: float, x1: float, *, fmt=None, grid: bool = True) -> None:
    """Horizontal hairline gridlines with right-aligned tick labels left of *x0*."""
    th = cv.th
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    for t in ticks:
        y = ys(t)
        if grid:
            color = th.axis if t == 0 else th.grid
            cv.add(f'<line x1="{f2(x0)}" y1="{f2(y)}" x2="{f2(x1)}" y2="{f2(y)}"{paint("stroke", color)} stroke-width="1" shape-rendering="crispEdges"/>')
        label = fmt(t) if fmt else fmt_tick(t, step)
        cv.text(x0 - 8, y + th.font_size * 0.35, label, size=th.font_size - 1, color=th.ink_muted, anchor="end", tabular=True)


def x_axis(cv: Canvas, xs: Scale, ticks: list[float], y: float, *, fmt=None, baseline: bool = True) -> None:
    th = cv.th
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    if baseline:
        cv.add(f'<line x1="{f2(xs.r0)}" y1="{f2(y)}" x2="{f2(xs.r1)}" y2="{f2(y)}"{paint("stroke", th.axis)} stroke-width="1" shape-rendering="crispEdges"/>')
    for t in ticks:
        x = xs(t)
        label = fmt(t) if fmt else fmt_tick(t, step)
        cv.text(x, y + th.font_size + 6, label, size=th.font_size - 1, color=th.ink_muted, anchor="middle", tabular=True)


def tick_label_width(ticks: list[float], th: Theme, fmt=None) -> float:
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    labels = [fmt(t) if fmt else fmt_tick(t, step) for t in ticks]
    return max((textm.text_width(s, th.font_size - 1) for s in labels), default=0.0)


def rounded_bar(x: float, y: float, w: float, h: float, r: float, end: str) -> str:
    """Bar path with rounded corners only at its data end (``top``/``right``)."""
    r = max(0.0, min(r, w / 2 if end == "top" else h / 2, h if end == "top" else w))
    if end == "top":
        return (
            f"M{f2(x)},{f2(y + h)}V{f2(y + r)}Q{f2(x)},{f2(y)} {f2(x + r)},{f2(y)}H{f2(x + w - r)}"
            f"Q{f2(x + w)},{f2(y)} {f2(x + w)},{f2(y + r)}V{f2(y + h)}Z"
        )
    return (
        f"M{f2(x)},{f2(y)}H{f2(x + w - r)}Q{f2(x + w)},{f2(y)} {f2(x + w)},{f2(y + r)}V{f2(y + h - r)}"
        f"Q{f2(x + w)},{f2(y + h)} {f2(x + w - r)},{f2(y + h)}H{f2(x)}Z"
    )


def reference_lines(cv: "Canvas", X: Scale, markers: Sequence[tuple[float, str]], top: float, bottom: float) -> None:
    """Vertical hairlines with labels; labels that would collide move down a row."""
    th = cv.th
    fs = th.font_size - 1
    rows: list[float] = []  # right edge of the last label in each row
    for mx, label in sorted(markers, key=lambda m: m[0]):
        px = X(mx)
        w = textm.text_width(label, fs)
        row = next((i for i, edge in enumerate(rows) if px + 4 > edge + 6), None)
        if row is None:
            rows.append(0.0)
            row = len(rows) - 1
        rows[row] = px + 4 + w
        cv.add(f'<line x1="{f2(px)}" y1="{f2(top)}" x2="{f2(px)}" y2="{f2(bottom)}"{paint("stroke", th.ink_secondary)} stroke-width="1"/>')
        y = top + fs * 0.9 + row * fs * 1.35
        cv.add(
            f'<text x="{f2(px + 4)}" y="{f2(y)}" font-size="{f2(fs)}" stroke="{th.surface}" stroke-width="3" paint-order="stroke"'
            f'{paint("fill", th.ink_secondary)}>{escape(label)}</text>'
        )


def uid(prefix: str, *parts: Any) -> str:
    import hashlib

    h = hashlib.sha1(("|".join(map(str, parts))).encode()).hexdigest()[:6]
    return f"{prefix}{h}"


def domain(values: Sequence[float], include_zero: bool = True) -> tuple[float, float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return (0.0, 1.0)
    lo, hi = min(finite), max(finite)
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if lo == hi:
        hi = lo + 1.0
    return (lo, hi)


__all__ = [
    "Chart",
    "Canvas",
    "Scale",
    "y_axis",
    "x_axis",
    "tick_label_width",
    "rounded_bar",
    "paint",
    "f2",
    "uid",
    "domain",
    "nice_ticks",
    "fmt_number",
]
