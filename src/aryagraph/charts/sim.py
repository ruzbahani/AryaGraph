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

"""Charts for simulation results."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..style.themes import Theme, get_theme
from .base import Chart
from .bars import histogram
from .line import line_chart


def state_colors(states: list[str], roles: dict[str, str], theme: Theme) -> dict[str, str]:
    """One distinct color per state: its semantic role first, categorical slots for clashes."""
    used: set[str] = set()
    out: dict[str, str] = {}
    free = [c for c in theme.categorical]
    for s in states:
        role = roles.get(s)
        color = None
        if role:
            try:
                color = theme.role_color(role)
            except ValueError:
                color = None
        if color is None or color in used:
            color = next((c for c in free if c not in used), theme.other)
        used.add(color)
        out[s] = color
    return out


def plot_simulation(
    result: Any,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    width: float = 640,
    height: float = 280,
    theme: Any = None,
    cursor: bool = False,
    fractions: bool = False,
    legend: bool | str = "auto",
) -> Chart:
    """State counts over time (categorical models) or mean with min–max band (continuous)."""
    th = get_theme(theme)
    times = np.asarray(result.times, dtype=float)
    name = getattr(result, "model", "simulation")
    step = "gillespie" in str(getattr(result, "params", {}).get("method", "")).lower()
    if result.kind == "categorical":
        counts = result.fractions() if fractions else result.counts()
        states = [s for s in result.states if s in counts]
        colors = state_colors(states, result.roles, th)
        return line_chart(
            times,
            {s: counts[s] for s in states},
            colors=colors,
            title=title if title is not None else f"{name}: state over time",
            subtitle=subtitle,
            x_label="time",
            y_label="share of nodes" if fractions else "nodes",
            width=width,
            height=height,
            theme=th,
            cursor=cursor,
            step=step,
            legend=legend,
            direct_labels=legend is not False,
        )
    summary = result.summary()
    mean = np.asarray(summary["mean"], float)
    lo = np.asarray(summary["min"], float)
    hi = np.asarray(summary["max"], float)
    return line_chart(
        times,
        {"mean": mean},
        colors={"mean": th.categorical[0]},
        bands={"mean": (lo, hi)},
        band_label="min–max across nodes",
        legend=legend if legend != "auto" else True,
        title=title if title is not None else f"{name}: node values over time",
        subtitle=subtitle,
        x_label="time",
        y_label="value",
        width=width,
        height=height,
        theme=th,
        cursor=cursor,
    )


def plot_ensemble(ensemble: Any, *, title: str | None = None, subtitle: str | None = None, width: float = 640, height: float = 300, theme: Any = None) -> Chart:
    """Mean curve per state with 25–75% (inner) and 5–95% (outer) bands."""
    th = get_theme(theme)
    times = np.asarray(ensemble.times, dtype=float)
    states = list(ensemble.states)
    colors = state_colors(states, getattr(ensemble, "roles", {}) or {}, th)
    series = {s: ensemble.mean[s] for s in states}
    q = ensemble.quantiles
    bands = {s: (q[s][25], q[s][75]) for s in states if 25 in q.get(s, {})}
    outer = {s: (q[s][5], q[s][95]) for s in states if 5 in q.get(s, {})}
    runs = getattr(ensemble, "runs", None)
    return line_chart(
        times,
        series,
        colors=colors,
        bands=bands,
        outer_bands=outer,
        band_label="25–75% and 5–95% of runs",
        title=title if title is not None else f"{getattr(ensemble, 'model', 'Ensemble')}: {runs or ''} runs".replace(":  runs", ""),
        subtitle=subtitle,
        x_label="time",
        y_label="nodes",
        width=width,
        height=height,
        theme=th,
    )


def plot_monte_carlo(mc: Any, *, title: str | None = None, subtitle: str | None = None, width: float = 640, height: float = 280, theme: Any = None) -> Chart:
    """Distribution of simulated makespans with P50 / P80 / P95 reference lines."""
    spans = np.asarray(mc.makespans, dtype=float)
    pct = getattr(mc, "percentiles", {}) or {}
    markers = []
    for key in ("P50", "P80", "P95"):
        if key in pct:
            markers.append((float(pct[key]), f"{key} {pct[key]:.3g}"))
    return histogram(
        spans,
        discrete=False,
        title=title if title is not None else "Project duration: Monte Carlo",
        subtitle=subtitle if subtitle is not None else f"{spans.size:,} simulated schedules",
        x_label="makespan",
        y_label="runs",
        width=width,
        height=height,
        markers=markers,
        theme=theme,
    )


__all__ = ["plot_simulation", "plot_ensemble", "plot_monte_carlo", "state_colors"]
