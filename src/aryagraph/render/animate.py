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

"""Animated playback of simulation results.

The graph is drawn once; each frame only recolors nodes (and flashes the
edges that fired), so even long simulations stay light: frames are packed as
one byte per node per frame. The HTML player adds play/pause, scrubbing,
speed control, and a chart of the state counts whose cursor follows the
timeline (drag on the chart to scrub).
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from ..style.colors import label_on, mix, to_hex
from ..style.scales import by
from ..style.themes import get_theme
from .figure import Figure


def _frame_indices(T: int, max_frames: int) -> np.ndarray:
    if T <= max_frames:
        return np.arange(T)
    idx = np.unique(np.round(np.linspace(0, T - 1, max_frames)).astype(int))
    return idx


def animate(
    result: Any,
    layout: Any = None,
    *,
    theme: Any = "light",
    title: str | None = None,
    subtitle: str | None = None,
    chart: bool = True,
    max_frames: int = 600,
    static_frame: str | int = "last",
    autoplay: bool = False,
    time_format: str | None = None,
    **draw_kwargs: Any,
) -> Figure:
    """Interactive animation of a :class:`~aryagraph.sim.SimulationResult`.

    Parameters
    ----------
    layout:
        Anything :func:`aryagraph.draw` accepts (``None`` = automatic).
    chart:
        Show the synced state-count chart under the player.
    max_frames:
        Longer runs are evenly subsampled (edge activity of skipped frames is
        merged into the next kept frame, so no transmission is lost).
    static_frame:
        Which frame the static SVG/PNG export shows: ``"last"``, ``"first"`` or an index.
    **draw_kwargs:
        Passed to :func:`aryagraph.draw` (``node_size``, ``labels``, ``node_shape`` …).
    """
    from . import draw

    th = get_theme(theme)
    g = result.graph
    nodes = list(result.nodes)
    graph_nodes = list(g)
    if nodes != graph_nodes:
        # align columns with graph order (the renderer's node order)
        pos = {n: i for i, n in enumerate(nodes)}
        order = [pos[n] for n in graph_nodes]
        values = np.asarray(result.values)[:, order]
    else:
        values = np.asarray(result.values)
    T = values.shape[0]
    keep = _frame_indices(T, max_frames)
    times = np.asarray(result.times, dtype=float)[keep]
    sf = {"last": len(keep) - 1, "first": 0}.get(static_frame, static_frame) if isinstance(static_frame, str) else int(static_frame)
    sf = int(min(max(sf, 0), len(keep) - 1))
    shown = values[keep[sf]]

    if result.kind == "categorical":
        from ..charts.sim import state_colors

        states = list(result.states)
        colors = state_colors(states, result.roles, th)
        spec = by(
            {n: states[int(c)] for n, c in zip(graph_nodes, shown)},
            kind="categorical",
            palette=colors,
            domain=states,
            title=result.model,
            counts=False,
        )
        codes = values[keep].astype(np.uint8)
        swatches = [colors[s] for s in states]
    else:
        lo, hi = result.meta.get("domain") or (float(np.nanmin(values)), float(np.nanmax(values)))
        if hi <= lo:
            hi = lo + 1.0
        diverging = bool(result.meta.get("diverging"))
        cmap = th.diverging_map() if diverging else th.sequential_map()
        levels = 64
        spec = by(
            {n: float(v) for n, v in zip(graph_nodes, shown)},
            kind="diverging" if diverging else "sequential",
            domain=(lo, hi),
            title=result.model,
            midpoint=(lo + hi) / 2 if diverging else None,
        )
        norm = np.clip((values[keep] - lo) / (hi - lo), 0, 1)
        codes = np.round(norm * (levels - 1)).astype(np.uint8)
        swatches = [cmap(i / (levels - 1)) for i in range(levels)]

    draw_kwargs.setdefault("title", title if title is not None else f"{result.model} simulation")
    if subtitle is not None:
        draw_kwargs["subtitle"] = subtitle
    elif "subtitle" not in draw_kwargs:
        params = ", ".join(f"{k}={v}" for k, v in list((result.params or {}).items())[:5] if isinstance(v, (int, float, str)))
        draw_kwargs["subtitle"] = f"{len(graph_nodes):,} nodes · {T:,} frames" + (f" · {params}" if params else "")
    fig = draw(g, layout="auto" if layout is None else layout, node_color=spec, theme=th, **draw_kwargs)
    scene = fig.scene

    boxy = any(m.inside and m.shape not in ("circle", "square", "ellipse") for m in scene.nodes)
    palette = []
    for c in swatches:
        c = to_hex(c)
        if boxy:
            palette.append([to_hex(mix(th.surface, c, 0.14)), c, th.ink, c])
        else:
            palette.append([c, None, label_on(c), c])

    # edge activity, merged across skipped frames
    fire = None
    activity = getattr(result, "edge_activity", None)
    if activity:
        eindex: dict[tuple, int] = {}
        for k, e in enumerate(scene.edges):
            eindex[(e.u, e.v)] = k
            if not g.directed:
                eindex[(e.v, e.u)] = k
        fire = []
        prev = -1
        for i in keep:
            hits: set[int] = set()
            for j in range(prev + 1, int(i) + 1):
                if j < len(activity) and activity[j]:
                    for e in activity[j]:
                        k = eindex.get(tuple(e[:2]))
                        if k is not None:
                            hits.add(k)
            fire.append(sorted(hits))
            prev = int(i)

    anim: dict[str, Any] = {
        "times": [round(float(t), 6) for t in times],
        "states": list(result.states) if result.kind == "categorical" else [],
        "palette": palette,
        "frames": base64.b64encode(np.ascontiguousarray(codes).tobytes()).decode("ascii"),
        "fire": fire,
        "start": 0,
        "autoplay": autoplay,
        "time_format": time_format or "t = {t}",
    }
    if result.kind != "categorical":
        anim["values"] = base64.b64encode(np.ascontiguousarray(values[keep], dtype=np.float32).tobytes()).decode("ascii")

    chart_svg = None
    if chart:
        from ..charts.sim import plot_simulation

        cw = float(min(max(scene.width, 560), 960))
        # the player's live readout above the chart doubles as its legend
        ch = plot_simulation(result, title="", width=cw, height=210, theme=th, cursor=True, legend=result.kind != "categorical")
        chart_svg = ch.svg
        xs = ch.meta.get("x_scale")
        if xs:
            anim["chart"] = xs
    fig.extras["anim"] = anim
    fig.extras["chart_svg"] = chart_svg
    fig.extras["result"] = result
    return fig


__all__ = ["animate"]
