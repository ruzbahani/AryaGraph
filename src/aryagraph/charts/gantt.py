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

"""Gantt chart for DAG schedules (simulated or planned)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Sequence
from xml.sax.saxutils import escape

from ..style import text as textm
from ..style.colors import label_on
from ..style.numbers import fmt_number, tight_ticks
from .base import Canvas, Chart, Scale, f2, paint, uid, x_axis


@dataclass
class _Bar:
    task: Hashable
    lane: str
    start: float
    end: float
    status: str = "done"
    attempt: int = 1


def _lane(worker: Any) -> str:
    # simulated workers are 0-based ints; people read "Worker 1"
    if isinstance(worker, int) and not isinstance(worker, bool):
        return f"Worker {worker + 1}"
    return str(worker)


def _bars(data: Any) -> list[_Bar]:
    tasks = getattr(data, "tasks", data)
    out = []
    for t in tasks:
        get = t.get if isinstance(t, dict) else (lambda k, default=None, _t=t: getattr(_t, k, default))
        node = get("node")
        out.append(
            _Bar(
                node,
                _lane(get("worker", node)),
                float(get("start")),
                float(get("end")),
                str(get("status", "done") or "done"),
                int(get("attempt", 1) or 1),
            )
        )
    return out


def gantt(
    data: Any,
    *,
    lanes: str = "task",
    color_by: str | None = None,
    critical: Iterable[Hashable] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    width: float = 760,
    row: float = 22.0,
    theme: Any = None,
    time_label: str | None = "time",
) -> Chart:
    """Draw a schedule.

    Parameters
    ----------
    data:
        A :class:`~aryagraph.sim.ScheduleResult` or a list of task runs (objects or dicts
        with ``node``, ``worker``, ``start``, ``end``, optional ``status``/``attempt``).
    lanes:
        ``"task"`` gives one row per task (project view); ``"worker"`` gives one row
        per worker (utilisation view).
    color_by:
        Node attribute giving a categorical color (e.g. ``"team"``). Without it,
        critical-path tasks take the accent color and the rest are muted.
    critical:
        Tasks to emphasise; defaults to the result's ``critical_path``.
    """
    bars = _bars(data)
    graph = getattr(data, "graph", None)
    if critical is None:
        critical = getattr(data, "critical_path", None) or ()
    crit = set(critical)
    cv = Canvas(width, 10, theme, uid=uid("agg", title, len(bars)), kind="gantt")
    th = cv.th
    pad = 16.0
    top = cv.header(title, subtitle, pad)

    # colors
    cats: dict[Any, str] = {}
    if color_by and graph is not None:
        seen = []
        for b in bars:
            v = graph.nodes[b.task].get(color_by) if b.task in graph else None
            if v is not None and v not in seen:
                seen.append(v)
        seen.sort(key=str)
        for i, v in enumerate(seen[: len(th.categorical) - 1]):
            cats[v] = th.categorical[i]

    def color_of(b: _Bar) -> str:
        if b.status == "failed":
            return th.status["critical"]
        if color_by and graph is not None:
            return cats.get(graph.nodes[b.task].get(color_by) if b.task in graph else None, th.other)
        if crit:
            return th.categorical[0] if b.task in crit else th.other
        return th.categorical[0]

    legend_items: list[tuple[str, str, str]] = []
    if color_by and cats:
        legend_items = [(str(k), c, "rect") for k, c in cats.items()]
    elif crit:
        legend_items = [("critical path", th.categorical[0], "rect"), ("other tasks", th.other, "rect")]
    if any(b.status == "failed" for b in bars):
        legend_items.append(("failed attempt", th.status["critical"], "rect"))
    if legend_items:
        top += cv.legend_row(pad, top + 4, legend_items, width - 2 * pad) + 6

    if lanes == "worker":
        lane_names = sorted({b.lane for b in bars}, key=lambda s: (len(s), s))
        key = lambda b: b.lane  # noqa: E731
    else:
        order: dict[Hashable, float] = {}
        for b in bars:
            order[b.task] = min(order.get(b.task, math.inf), b.start)
        lane_names = [str(t) for t, _ in sorted(order.items(), key=lambda kv: (kv[1], str(kv[0])))]
        key = lambda b: str(b.task)  # noqa: E731
    lw = min(max((textm.text_width(n, th.font_size) for n in lane_names), default=0) + 6, 220)
    left = pad + lw + 10
    right = width - pad
    t0 = min((b.start for b in bars), default=0.0)
    t1 = max((b.end for b in bars), default=1.0)
    t0 = min(t0, 0.0)
    ticks = tight_ticks(t0, t1, max(3, int((right - left) / 110)))
    X = Scale(ticks[0], ticks[-1], left, right)
    lane_y = {n: top + 6 + i * (row + 6) for i, n in enumerate(lane_names)}
    bottom = top + 6 + len(lane_names) * (row + 6)
    for t in ticks:
        x = X(t)
        cv.add(f'<line x1="{f2(x)}" y1="{f2(top)}" x2="{f2(x)}" y2="{f2(bottom)}"{paint("stroke", th.grid)} stroke-width="1" shape-rendering="crispEdges"/>')
    for n, y in lane_y.items():
        cv.text(left - 10, y + row / 2 + th.font_size * 0.35, textm.truncate(n, lw, th.font_size), anchor="end", color=th.ink_secondary)
    bh = min(row, 24.0)
    for b in bars:
        y = lane_y[key(b)] + (row - bh) / 2
        x0, x1 = X(b.start), X(b.end)
        w = max(x1 - x0 - 2, 1.0)  # 2px surface gap between back-to-back runs
        c = color_of(b)
        tip = f"{b.task} · {b.lane} · {fmt_number(b.start)}–{fmt_number(b.end)}" + (f" · attempt {b.attempt}" if b.attempt > 1 else "") + (f" · {b.status}" if b.status != "done" else "")
        cv.add(f'<rect x="{f2(x0 + 1)}" y="{f2(y)}" width="{f2(w)}" height="{f2(bh)}" rx="4"{paint("fill", c)}><title>{escape(tip)}</title></rect>')
        if lanes == "worker":
            label = str(b.task)
            if textm.text_width(label, th.font_size - 1) + 10 <= w:
                cv.text(x0 + 1 + w / 2, y + bh / 2 + (th.font_size - 1) * 0.35, label, size=th.font_size - 1, anchor="middle", color=label_on(c))
    x_axis(cv, X, ticks, bottom, baseline=True)
    if time_label:
        cv.text(right, bottom + th.font_size * 2.6 + 4, time_label, size=th.font_size - 1, color=th.ink_muted, anchor="end")
    cv.h = bottom + th.font_size * 2.6 + 4 + pad
    makespan = getattr(data, "makespan", None)
    desc = f"Gantt chart of {len(bars)} task runs" + (f", makespan {fmt_number(makespan)}" if makespan is not None else "") + "."
    return cv.finish(title, desc)


__all__ = ["gantt"]
