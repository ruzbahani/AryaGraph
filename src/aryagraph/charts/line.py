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

"""Line chart (time series, simulation curves), with optional uncertainty bands."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

import numpy as np

from ..style import text as textm
from ..style.numbers import fmt_number, nice_ticks
from .base import Canvas, Chart, Scale, f2, paint, reference_lines, tick_label_width, uid, x_axis, y_axis


def line_chart(
    x: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    colors: Mapping[str, str] | None = None,
    bands: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    outer_bands: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: float = 640,
    height: float = 300,
    theme: Any = None,
    y_min: float | None = None,
    y_max: float | None = None,
    direct_labels: bool | str = "auto",
    legend: bool | str = "auto",
    markers: Sequence[tuple[float, str]] = (),
    cursor: bool = False,
    step: bool = False,
    y_format: Any = None,
    x_format: Any = None,
    band_label: str | None = None,
) -> Chart:
    """Multi-series line chart on one y-axis.

    Parameters
    ----------
    x:
        Shared x values (e.g. simulation times).
    series:
        ``{name: y_values}``; NaNs break the line.
    colors:
        ``{name: color}``; defaults to categorical slots in series order.
    bands, outer_bands:
        ``{name: (lo, hi)}`` uncertainty envelopes drawn as 10% / 6% washes.
    markers:
        Vertical reference hairlines ``(x, label)``.
    cursor:
        Add a movable cursor line (used by the simulation player).
    step:
        Draw as a step function (for event-driven counts).
    """
    cv = Canvas(width, height, theme, uid=uid("agc", title, width, height, len(series)), kind="line")
    th = cv.th
    xs_arr = np.asarray(x, dtype=float)
    names = list(series)
    ys = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    cols = {k: (colors or {}).get(k) or th.categorical[min(i, len(th.categorical) - 1)] for i, k in enumerate(names)}
    pad = 16.0
    top = cv.header(title, subtitle, pad)
    show_legend = (legend is True) or (legend == "auto" and len(names) >= 2)
    if show_legend:
        items = [(k, cols[k], "line") for k in names]
        if bands:
            items.append((band_label or ("middle 50%" if outer_bands else "range"), th.ink_muted, "band"))
        top += cv.legend_row(pad, top + 4, items, width - 2 * pad) + 4

    # y domain
    vals = [v for arr in ys.values() for v in arr[np.isfinite(arr)]]
    for group in (bands or {}, outer_bands or {}):
        for lo, hi in group.values():
            vals += [v for v in np.asarray(lo, float) if math.isfinite(v)] + [v for v in np.asarray(hi, float) if math.isfinite(v)]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 1.0
    if y_min is not None:
        lo = y_min
    elif lo >= 0 or (hi > 0 and -lo < 0.02 * (hi - lo)):
        lo = 0.0  # counts & rates start at zero; float noise below it is not data
    if y_max is not None:
        hi = y_max
    yt = nice_ticks(lo, hi, 5, include_bounds=True)
    ylo, yhi = yt[0], yt[-1]

    # direct end labels: only when they separate cleanly at the right edge
    use_direct = False
    label_w = 0.0
    if direct_labels is True or (direct_labels == "auto" and 1 <= len(names) <= 4):
        use_direct = True
        label_w = max(textm.text_width(k, th.font_size) for k in names) + 12
    left = pad + tick_label_width(yt, th, y_format) + 10
    right = width - pad - (label_w if use_direct else 0)
    bottom = height - pad - th.font_size - 8 - (th.font_size + 6 if x_label else 0)
    plot_top = top + (th.font_size + 4 if y_label else 4)
    if y_label:
        cv.text(left - 8, top + th.font_size * 0.2, y_label, size=th.font_size - 1, color=th.ink_muted, anchor="start")

    x0v = float(np.nanmin(xs_arr)) if xs_arr.size else 0.0
    x1v = float(np.nanmax(xs_arr)) if xs_arr.size else 1.0
    if x1v == x0v:
        x1v = x0v + 1
    X = Scale(x0v, x1v, left, right)
    Y = Scale(ylo, yhi, bottom, plot_top)

    if use_direct:
        ends = []
        for k in names:
            arr = ys[k]
            idx = np.where(np.isfinite(arr))[0]
            if idx.size:
                ends.append((Y(arr[idx[-1]]), k))
        ends.sort()
        if any(b[0] - a[0] < th.font_size * 1.15 for a, b in zip(ends, ends[1:])):
            use_direct = False
            right = width - pad
            X = Scale(x0v, x1v, left, right)

    y_axis(cv, Y, yt, left, right, fmt=y_format)
    xt = [t for t in nice_ticks(x0v, x1v, max(3, int((right - left) / 90))) if x0v - 1e-9 <= t <= x1v + 1e-9]
    x_axis(cv, X, xt, bottom, fmt=x_format, baseline=False)
    if x_label:
        cv.text((left + right) / 2, height - pad + 2, x_label, size=th.font_size - 1, color=th.ink_muted, anchor="middle")

    def band_path(lo_arr, hi_arr) -> str:
        lo_a, hi_a = np.asarray(lo_arr, float), np.asarray(hi_arr, float)
        ok = np.isfinite(lo_a) & np.isfinite(hi_a)
        pts_hi = [(X(xv), Y(h)) for xv, h, g in zip(xs_arr, hi_a, ok) if g]
        pts_lo = [(X(xv), Y(l)) for xv, l, g in zip(xs_arr, lo_a, ok) if g]
        if not pts_hi:
            return ""
        return "M" + "L".join(f"{f2(a)},{f2(b)}" for a, b in pts_hi + pts_lo[::-1]) + "Z"

    for group, alpha in ((outer_bands or {}, 0.07), (bands or {}, 0.13)):
        for k, (blo, bhi) in group.items():
            d = band_path(blo, bhi)
            if d:
                cv.add(f'<path d="{d}"{paint("fill", cols.get(k, th.ink_muted))} fill-opacity="{alpha}"/>')

    reference_lines(cv, X, markers, plot_top, bottom)

    for k in names:
        arr = ys[k]
        segs: list[str] = []
        cur: list[str] = []
        prev = None
        for xv, yv in zip(xs_arr, arr):
            if not math.isfinite(yv):
                if cur:
                    segs.append("M" + "L".join(cur))
                cur, prev = [], None
                continue
            px, py = X(xv), Y(yv)
            if step and prev is not None:
                cur.append(f"{f2(px)},{f2(prev)}")
            cur.append(f"{f2(px)},{f2(py)}")
            prev = py
        if cur:
            segs.append("M" + "L".join(cur))
        cv.add(
            f'<path class="ag-series" data-name="{escape(str(k), {chr(34): "&quot;"})}" d="{"".join(segs)}" fill="none"{paint("stroke", cols[k])}'
            ' stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    if use_direct:
        for k in names:
            arr = ys[k]
            idx = np.where(np.isfinite(arr))[0]
            if not idx.size:
                continue
            j = idx[-1]
            cv.text(X(xs_arr[j]) + 8, Y(arr[j]) + th.font_size * 0.35, k, color=th.ink_secondary)

    if cursor:
        cv.add(f'<line class="ag-cursor" x1="{f2(left)}" y1="{f2(plot_top)}" x2="{f2(left)}" y2="{f2(bottom)}"{paint("stroke", th.ink_secondary)} stroke-width="1"/>')

    cv.data["x_scale"] = {"x0": left, "x1": right, "t0": x0v, "t1": x1v}
    hover = {
        "type": "line",
        "plot": [left, plot_top, right, bottom],
        "x": [float(v) for v in xs_arr],
        "xd": [x0v, x1v],
        "yd": [ylo, yhi],
        "series": [{"name": k, "color": cols[k], "y": [None if not math.isfinite(v) else float(v) for v in ys[k]]} for k in names],
    }
    desc = f"Line chart of {', '.join(names)} over {fmt_number(x0v)}–{fmt_number(x1v)}."
    return cv.finish(title, desc, data=hover)


__all__ = ["line_chart"]
