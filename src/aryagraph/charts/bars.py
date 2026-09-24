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

"""Bar chart (ranked values) and histogram."""

from __future__ import annotations

import math
from typing import Any, Sequence
from xml.sax.saxutils import escape

import numpy as np

from ..style import text as textm
from ..style.numbers import fmt_number, nice_ticks
from .base import Canvas, Chart, Scale, f2, paint, reference_lines, rounded_bar, tick_label_width, uid, x_axis, y_axis


def bar_chart(
    labels: Sequence[Any],
    values: Sequence[float],
    *,
    title: str | None = None,
    subtitle: str | None = None,
    color: str | None = None,
    highlight: Sequence[Any] | None = None,
    value_format: Any = None,
    width: float = 560,
    bar: float = 18.0,
    gap: float = 10.0,
    max_label_width: float = 180.0,
    theme: Any = None,
    sort: bool = True,
) -> Chart:
    """Horizontal bars, one series, every bar labeled with its value at the tip.

    *highlight* (labels) keeps those bars in the accent color and mutes the rest;
    use it when the story is about one or two entries.
    """
    cv = Canvas(width, 10, theme, uid=uid("agb", title, len(labels)), kind="bar")
    th = cv.th
    pairs = list(zip([str(l) for l in labels], [float(v) for v in values]))
    if sort:
        pairs.sort(key=lambda p: -p[1] if math.isfinite(p[1]) else math.inf)
    fmt = value_format or fmt_number
    accent = color or th.categorical[0]
    hl = set(map(str, highlight)) if highlight else None
    bar = min(bar, 24.0)
    pad = 16.0
    top = cv.header(title, subtitle, pad)
    lw = min(max((textm.text_width(l, th.font_size) for l, _ in pairs), default=0) + 4, max_label_width)
    left = pad + lw + 10
    vmax = max((v for _, v in pairs if math.isfinite(v)), default=1.0) or 1.0
    vmin = min(0.0, min((v for _, v in pairs if math.isfinite(v)), default=0.0))
    val_w = max((textm.text_width(fmt(v), th.font_size - 0.5) for _, v in pairs), default=0) + 10
    right = width - pad - val_w
    X = Scale(vmin, vmax, left, right)
    y = top + 4
    for label, v in pairs:
        shown = textm.truncate(label, lw, th.font_size)
        cv.text(left - 10, y + bar / 2 + th.font_size * 0.35, shown, anchor="end", color=th.ink_secondary)
        c = accent if hl is None or label in hl else th.other
        if math.isfinite(v) and v != 0:
            x0, x1 = X(0.0), X(v)
            if x1 >= x0:
                cv.add(f'<path d="{rounded_bar(x0, y, max(x1 - x0, 0.5), bar, 4, "right")}"{paint("fill", c)}><title>{escape(label)}: {escape(fmt(v))}</title></path>')
            else:
                cv.add(f'<rect x="{f2(x1)}" y="{f2(y)}" width="{f2(x0 - x1)}" height="{f2(bar)}" rx="2"{paint("fill", c)}/>')
            cv.text(max(x1, x0) + 6, y + bar / 2 + th.font_size * 0.35, fmt(v), size=th.font_size - 0.5, color=th.ink, tabular=True)
        y += bar + gap
    base_x = X(0.0)
    cv.add(f'<line x1="{f2(base_x)}" y1="{f2(top)}" x2="{f2(base_x)}" y2="{f2(y - gap + 4)}"{paint("stroke", th.axis)} stroke-width="1" shape-rendering="crispEdges"/>')
    cv.h = y - gap + 4 + pad
    return cv.finish(title, f"Bar chart of {len(pairs)} values.")


def histogram(
    values: Sequence[float],
    *,
    bins: int | Sequence[float] | str = "auto",
    discrete: bool | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    x_label: str | None = None,
    y_label: str | None = "count",
    width: float = 560,
    height: float = 260,
    color: str | None = None,
    log_y: bool = False,
    markers: Sequence[tuple[float, str]] = (),
    theme: Any = None,
) -> Chart:
    """Column histogram. Integer data (e.g. degrees) get one column per value when
    the range allows; columns keep a 2px surface gap and a 4px rounded top."""
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    cv = Canvas(width, height, theme, uid=uid("agh", title, arr.size), kind="histogram")
    th = cv.th
    fill = color or th.categorical[0]
    if discrete is None:
        discrete = arr.size > 0 and bool(np.all(arr == np.round(arr)))
    if arr.size == 0:
        edges = np.array([0.0, 1.0])
        counts = np.array([0])
    elif discrete and (arr.max() - arr.min()) <= 60:
        lo, hi = int(arr.min()), int(arr.max())
        edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
        counts = np.array([(arr == k).sum() for k in range(lo, hi + 1)])
    else:
        edges_or_n = bins if not isinstance(bins, str) else ("auto" if arr.size > 1 else 1)
        counts, edges = np.histogram(arr, bins=edges_or_n)
    pad = 16.0
    top = cv.header(title, subtitle, pad)
    ymax = float(counts.max()) if counts.size else 1.0
    if log_y:
        tvals = [10**k for k in range(0, max(1, math.ceil(math.log10(max(ymax, 1)))) + 1)]
        yt = [0.0] + [float(t) for t in tvals]
    else:
        yt = nice_ticks(0, max(ymax, 1), 5, include_bounds=True)
    left = pad + tick_label_width(yt, th) + 10
    right = width - pad
    bottom = height - pad - th.font_size - 8 - (th.font_size + 6 if x_label else 0)
    plot_top = top + (th.font_size + 4 if y_label else 4)
    if y_label:
        cv.text(left - 8, top + th.font_size * 0.2, y_label, size=th.font_size - 1, color=th.ink_muted)
    X = Scale(float(edges[0]), float(edges[-1]), left, right)
    if log_y:
        ly = lambda v: 0.0 if v <= 0 else math.log10(v) + 1  # noqa: E731
        Ylog = Scale(0.0, ly(yt[-1]), bottom, plot_top)
        Y = lambda v: Ylog(ly(v))  # noqa: E731
        for t in yt:
            yy = Y(t)
            cv.add(f'<line x1="{f2(left)}" y1="{f2(yy)}" x2="{f2(right)}" y2="{f2(yy)}"{paint("stroke", th.axis if t == 0 else th.grid)} stroke-width="1" shape-rendering="crispEdges"/>')
            cv.text(left - 8, yy + th.font_size * 0.35, fmt_number(int(t)), size=th.font_size - 1, color=th.ink_muted, anchor="end", tabular=True)
    else:
        Y = Scale(yt[0], yt[-1], bottom, plot_top)
        y_axis(cv, Y, yt, left, right)
    slot = (right - left) / max(len(counts), 1)
    colw = max(min(24.0, slot - 2.0), 1.0)
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        cx = X((edges[i] + edges[i + 1]) / 2)
        y0 = Y(float(c))
        h = bottom - y0
        rng = f"{fmt_number(float(edges[i] + 0.5))}" if discrete else f"{fmt_number(float(edges[i]))}–{fmt_number(float(edges[i + 1]))}"
        cv.add(f'<path d="{rounded_bar(cx - colw / 2, y0, colw, h, 4 if h > 4 else h / 2, "top")}"{paint("fill", fill)}><title>{rng}: {int(c):,}</title></path>')
    # x ticks
    if discrete and len(counts) <= 60:
        lo_v = edges[0] + 0.5
        hi_v = edges[-1] - 0.5
        xt = [t for t in nice_ticks(lo_v, hi_v, max(3, int((right - left) / 70))) if lo_v - 1e-9 <= t <= hi_v + 1e-9 and float(t).is_integer()]
    else:
        xt = [t for t in nice_ticks(float(edges[0]), float(edges[-1]), max(3, int((right - left) / 80))) if edges[0] <= t <= edges[-1]]
    x_axis(cv, X, xt, bottom, baseline=False)
    reference_lines(cv, X, markers, plot_top, bottom)
    if x_label:
        cv.text((left + right) / 2, height - pad + 2, x_label, size=th.font_size - 1, color=th.ink_muted, anchor="middle")
    return cv.finish(title, f"Histogram of {arr.size} values.")


__all__ = ["bar_chart", "histogram"]
