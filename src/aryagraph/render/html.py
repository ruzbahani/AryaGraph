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

"""Interactive HTML backend.

The page embeds the exact SVG the static backend produces (so what you see
interactively is what you export), a JSON payload describing nodes and edges,
and a small dependency-free runtime (``assets/aryagraph.js``) adding pan & zoom,
neighborhood highlighting, tooltips, a details panel, node dragging with live
edge re-routing, search, legend filtering, a sortable table view, SVG/PNG
download and (for simulations) a timeline player.

Several apps can live on one page (reports do this); assets are included once.
"""

from __future__ import annotations

import json
from functools import lru_cache
from html import escape
from importlib import resources
from typing import TYPE_CHECKING, Any

from ..style.shapes import SHAPES, Cylinder, Ellipse, Polygon, RoundRect
from ..style.themes import Theme
from .scene import Scene
from .svg import _uid, scene_to_svg

if TYPE_CHECKING:
    from .figure import Figure

_ICONS = {
    "search": '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="5"/><path d="M11 11l3.5 3.5"/></svg>',
    "fit": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"/></svg>',
    "plus": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3v10M3 8h10"/></svg>',
    "minus": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h10"/></svg>',
    "labels": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 4h10M3 8h7M3 12h9"/></svg>',
    "table": '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="3" width="12" height="10" rx="1.5"/><path d="M2 7h12M6 7v6"/></svg>',
    "download": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2v8M4.5 6.5L8 10l3.5-3.5M3 13h10"/></svg>',
    "play": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3.5v9l7-4.5z" fill="currentColor" stroke="none"/></svg>',
    "back": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3.5L5.5 8l4.5 4.5"/></svg>',
    "fwd": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5L10.5 8 6 12.5"/></svg>',
}


@lru_cache(maxsize=None)
def _asset(name: str) -> str:
    return resources.files("aryagraph.render").joinpath("assets", name).read_text(encoding="utf-8")


def _shape_specs() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, sh in SHAPES.items():
        if isinstance(sh, Ellipse):
            out[name] = {"t": "e"}
        elif isinstance(sh, RoundRect):
            out[name] = {"t": "r", "r": sh.radius}
        elif isinstance(sh, Polygon):
            out[name] = {"t": "p", "v": [list(v) for v in sh.vertices]}
        elif isinstance(sh, Cylinder):
            out[name] = {"t": "r", "r": 0}
    return out


def _r(v: float) -> float:
    return round(float(v), 2)


def scene_payload(scene: Scene, uid: str) -> dict[str, Any]:
    """JSON-ready description of the scene for the runtime."""
    nodes = []
    for m in scene.nodes:
        item: dict[str, Any] = {
            "k": m.key,
            "id": str(m.node),
            "label": m.text or str(m.node),
            "x": _r(m.x),
            "y": _r(m.y),
            "w": _r(m.w),
            "h": _r(m.h),
            "shape": m.shape,
            "tip": m.tooltip,
        }
        if m.group is not None:
            item["g"] = m.group if isinstance(m.group, (int, float, str, bool)) else str(m.group)
        nodes.append(item)
    edges = []
    for e in scene.edges:
        item = {
            "k": e.key,
            "u": e.ukey,
            "v": e.vkey,
            "style": e.style,
            "curv": _r(e.curvature),
            "al": _r(e.arrow_len),
            "tip": e.tooltip,
        }
        if e.route:
            item["route"] = [[_r(x), _r(y)] for x, y in e.route]
        if e.label_anchor:
            item["mid"] = [_r(e.label_anchor[0]), _r(e.label_anchor[1])]
        edges.append(item)
    orient = scene.meta.get("orientation")
    return {
        "uid": uid,
        "name": scene.meta.get("name") or "graph",
        "size": [scene.width, scene.height],
        "offset": [_r(scene.offset[0]), _r(scene.offset[1])],
        "directed": scene.directed,
        "axis": "x" if orient in ("LR", "RL") else "y",
        "shapes": _shape_specs(),
        "nodes": nodes,
        "edges": edges,
    }


def _json_script(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    # keep the JSON inert inside <script>
    raw = raw.replace("</", "<\\/").replace("<!--", "<\\!--")
    return f'<script type="application/json" class="ag-data">{raw}</script>'


def _css_vars(theme: Theme) -> str:
    return ";".join(f"{k}:{v}" for k, v in theme.css_vars().items())


def app_html(
    scene: Scene,
    *,
    interactive: bool = True,
    anim: dict[str, Any] | None = None,
    chart_svg: str | None = None,
    table: bool = True,
    footer: bool = True,
) -> str:
    """The ``.ag-app`` block for one figure (no ``<html>`` wrapper)."""
    th = scene.theme
    uid = _uid(scene)
    svg = scene_to_svg(scene)
    dw, dh = scene.display_size or (scene.width, scene.height)
    shell_w = max(int(dw * 1.25) + 34, 760) if interactive else int(dw) + 34
    style = f"{_css_vars(th)};--ag-max-w:{shell_w}px"
    parts = [f'<div class="ag-app" style="{escape(style)}" data-theme="{th.mode}">', '<div class="ag-shell">']
    if interactive:
        parts.append(
            '<div class="ag-toolbar" role="toolbar" aria-label="Graph controls">'
            f'<div class="ag-search">{_ICONS["search"]}<input type="search" placeholder="Search nodes…" aria-label="Search nodes" autocomplete="off">'
            '<ul class="ag-results" role="listbox" hidden></ul></div>'
            '<span class="ag-spacer"></span>'
            f'<button class="ag-btn" type="button" data-act="fit" title="Fit to view (0)">{_ICONS["fit"]}<span class="ag-txt">Fit</span></button>'
            f'<button class="ag-btn" type="button" data-act="zoom-out" title="Zoom out (−)" aria-label="Zoom out">{_ICONS["minus"]}</button>'
            f'<button class="ag-btn" type="button" data-act="zoom-in" title="Zoom in (+)" aria-label="Zoom in">{_ICONS["plus"]}</button>'
            f'<button class="ag-btn" type="button" data-act="labels" aria-pressed="false">{_ICONS["labels"]}<span class="ag-txt">Labels: auto</span></button>'
            + (f'<button class="ag-btn" type="button" data-act="table" aria-pressed="false">{_ICONS["table"]}<span class="ag-txt">Table</span></button>' if table else "")
            + f'<button class="ag-btn" type="button" data-act="svg" title="Download SVG">{_ICONS["download"]}<span class="ag-txt">SVG</span></button>'
            f'<button class="ag-btn" type="button" data-act="png" title="Download PNG">{_ICONS["download"]}<span class="ag-txt">PNG</span></button>'
            "</div>"
        )
    # interactive stages may grow a little past natural size (vector), never so much that text balloons
    grow = 1.25 if interactive else 1.0
    stage_style = (
        f"aspect-ratio:{scene.width:.1f}/{scene.height:.1f};height:auto;max-height:82vh;"
        f"max-width:{int(dw * grow)}px;margin:0 auto;width:100%"
    )
    parts.append(f'<div class="ag-stage" style="{stage_style}">{svg}')
    if interactive:
        parts.append('<div class="ag-tip" role="tooltip" hidden></div><aside class="ag-panel" aria-label="Node details" hidden></aside>')
    parts.append("</div>")
    if anim is not None:
        parts.append(
            '<div class="ag-player">'
            f'<button class="ag-btn" type="button" data-act="play" aria-pressed="false">{_ICONS["play"]}<span class="ag-txt">Play</span></button>'
            f'<span><button class="ag-btn" type="button" data-act="step-back" aria-label="Previous frame">{_ICONS["back"]}</button> '
            f'<button class="ag-btn" type="button" data-act="step-fwd" aria-label="Next frame">{_ICONS["fwd"]}</button></span>'
            f'<input type="range" min="0" max="{len(anim["times"]) - 1}" step="1" value="0" aria-label="Frame">'
            '<span class="ag-time" aria-live="polite"></span>'
            '<select aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option>'
            '<option value="2">2×</option><option value="4">4×</option></select>'
            "</div>"
        )
        if chart_svg:
            parts.append(f'<div class="ag-chart"><div class="ag-readout" aria-live="polite"></div>{chart_svg}</div>')
    if interactive and table:
        parts.append('<div class="ag-table" hidden></div>')
    if interactive and footer:
        parts.append(
            '<div class="ag-foot"><span>Scroll or pinch to zoom · drag to pan · drag nodes to move them · '
            "click a node to pin its neighborhood · click legend entries to filter</span>"
            "<span><kbd>+</kbd> <kbd>−</kbd> <kbd>0</kbd> <kbd>Esc</kbd>"
            + (" <kbd>Space</kbd>" if anim is not None else "")
            + "</span></div>"
        )
    payload = scene_payload(scene, uid)
    if anim is not None:
        payload["anim"] = anim
    parts.append(_json_script(payload))
    parts.append("</div></div>")
    return "".join(parts)


def page(body: str, theme: Theme, title: str = "AryaGraph", *, interactive: bool = True, extra_css: str = "") -> str:
    """Wrap one or more app blocks into a complete HTML document (assets included once)."""
    css = _asset("aryagraph.css")
    js = _asset("aryagraph.js") if interactive else ""
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title>"
        f"<style>html,body{{margin:0;background:{theme.background}}}{css}{extra_css}</style></head>"
        f"<body>{body}"
        + (f"<script>{js}</script>" if js else "")
        + "</body></html>"
    )


def figure_to_html(fig: "Figure", *, interactive: bool = True, title: str | None = None) -> str:
    scene = fig.scene
    name = title or scene.title or scene.meta.get("name") or "AryaGraph"
    anim = fig.extras.get("anim")
    chart = fig.extras.get("chart_svg")
    body = app_html(scene, interactive=interactive, anim=anim, chart_svg=chart)
    return page(body, scene.theme, name, interactive=interactive)


__all__ = ["app_html", "page", "figure_to_html", "scene_payload"]
