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

"""SVG backend: serialise a :class:`Scene` to a standalone, accessible SVG.

Structure (stable, so the interactive layer and user CSS can target it)::

    svg.ag-root
      title, desc, style, defs
      rect.ag-bg
      g.ag-header            title / subtitle
      g.ag-viewport          ← pan/zoom target in HTML
        g.ag-content         translate(offset)
          g.ag-edges > g.ag-e[data-k,data-u,data-v] > path.ag-edge (+ path.ag-arrow)
          g.ag-edge-labels > text
          g.ag-nodes > g.ag-n[data-k] (translate(x,y)) > path.ag-shape (+ text)
          g.ag-labels > text.ag-lbl[data-k] (translate(x,y))
      g.ag-legend
      text.ag-caption
"""

from __future__ import annotations

import hashlib
from typing import Iterable
from xml.sax.saxutils import escape

from ..style.colors import split_alpha
from ..style.numbers import fmt_number
from ..style.scales import Legend
from ..style.shapes import get_shape
from .scene import EdgeMark, LabelMark, NodeMark, Scene


def _f(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _attr(v: object) -> str:
    return escape(str(v), {'"': "&quot;"})


def _paint(attr: str, color: str | None) -> str:
    """``fill="#hex"`` plus ``fill-opacity`` for translucent colors (renderer-portable)."""
    if not color:
        return f' {attr}="none"'
    hexc, a = split_alpha(color)
    out = f' {attr}="{hexc}"'
    if a < 1.0:
        out += f' {attr}-opacity="{_f(a)}"'
    return out


def _uid(scene: Scene) -> str:
    h = hashlib.sha1()
    h.update(f"{scene.width}x{scene.height}|{scene.title}".encode())
    for m in scene.nodes[:64]:
        h.update(f"{m.key}{m.x:.1f}{m.y:.1f}".encode())
    return "ag" + h.hexdigest()[:6]


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def _text(
    lbl: LabelMark,
    *,
    cls: str,
    dx: float = 0.0,
    dy: float = 0.0,
    extra: str = "",
) -> str:
    anchor = lbl.anchor
    direction = ""
    if lbl.rtl:
        anchor = {"start": "end", "end": "start"}.get(anchor, anchor)
        direction = ' direction="rtl"'
    halo = ""
    if lbl.halo:
        hexc, _ = split_alpha(lbl.halo)
        halo = f' stroke="{hexc}" stroke-width="3" stroke-linejoin="round" paint-order="stroke"'
    weight = f' font-weight="{lbl.weight}"' if lbl.weight != 400 else ""
    x, y = lbl.x - dx, lbl.y - dy
    head = (
        f'<text class="{cls}"{extra} x="{_f(x)}" y="{_f(y)}" font-size="{_f(lbl.size)}"{weight}'
        f' text-anchor="{anchor}"{direction}{_paint("fill", lbl.color)}{halo}>'
    )
    if len(lbl.lines) == 1:
        return head + escape(lbl.lines[0]) + "</text>"
    step = lbl.size * lbl.line_height
    spans = "".join(
        f'<tspan x="{_f(x)}" dy="{_f(0 if i == 0 else step)}">{escape(line)}</tspan>' for i, line in enumerate(lbl.lines)
    )
    return head + spans + "</text>"


def _tooltip_text(info: dict) -> str:
    lines = [str(info.get("label", info.get("id", "")))]
    for key in ("degree", "in", "out"):
        if key in info:
            lines.append(f"{key}: {info[key]}")
    for k, v in (info.get("attrs") or {}).items():
        lines.append(f"{k}: {fmt_number(v) if isinstance(v, float) else v}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# marks
# --------------------------------------------------------------------------- #
def _edge(e: EdgeMark, theme_dim: float) -> str:
    op = e.opacity if not e.dimmed else min(e.opacity, theme_dim)
    hexc, a = split_alpha(e.stroke)
    op *= a
    dash = f' stroke-dasharray="{e.dash}"' if e.dash else ""
    body = (
        f'<path class="ag-edge" d="{e.d}" fill="none" stroke="{hexc}" stroke-width="{_f(e.width)}"'
        f' stroke-linecap="round" stroke-linejoin="round"{dash}/>'
    )
    if e.arrow:
        body += f'<path class="ag-arrow" d="{e.arrow}" fill="{hexc}"/>'
    hl = " ag-hl" if e.highlight else ""
    return f'<g class="ag-e{hl}" data-k="{e.key}" data-u="{e.ukey}" data-v="{e.vkey}" opacity="{_f(op)}">{body}</g>'


def _node(m: NodeMark, dim_opacity: float) -> str:
    shape = get_shape(m.shape)
    parts = []
    if m.halo_color:
        pad = 3.25
        parts.append(
            f'<path class="ag-halo" d="{shape.path(0, 0, m.w + 2 * pad, m.h + 2 * pad)}" fill="none"'
            f'{_paint("stroke", m.halo_color)} stroke-width="1.6"/>'
        )
    parts.append(
        f'<path class="ag-shape" d="{shape.path(0, 0, m.w, m.h)}"{_paint("fill", m.fill)}'
        f'{_paint("stroke", m.stroke)} stroke-width="{_f(m.stroke_width)}"/>'
    )
    if m.inside and m.label is not None:
        parts.append(_text(m.label, cls="ag-in", dx=m.x, dy=m.y))
    parts.append(f"<title>{escape(_tooltip_text(m.tooltip))}</title>")
    op = m.opacity * (0.28 if m.dimmed else 1.0)
    opacity = f' opacity="{_f(op)}"' if op < 1 else ""
    hl = " ag-hl" if m.highlight else ""
    return f'<g class="ag-n{hl}" data-k="{m.key}" transform="translate({_f(m.x)},{_f(m.y)})"{opacity}>' + "".join(parts) + "</g>"


def _side_label(m: NodeMark) -> str:
    lbl = m.label
    assert lbl is not None
    hidden = "" if lbl.visible else ' visibility="hidden"'
    op = ' opacity="0.45"' if m.dimmed else ""
    return (
        f'<g class="ag-l" data-k="{m.key}" transform="translate({_f(m.x)},{_f(m.y)})"{hidden}{op}>'
        + _text(lbl, cls="ag-lbl", dx=m.x, dy=m.y)
        + "</g>"
    )


# --------------------------------------------------------------------------- #
# legends
# --------------------------------------------------------------------------- #
def _legend(lg: Legend, x: float, y: float, w: float, h: float, scene: Scene, uid: str, n: int) -> str:
    th = scene.theme
    fs = th.font_size
    out = [f'<g class="ag-legend-block" data-channel="{_attr(lg.channel)}" transform="translate({_f(x)},{_f(y)})">']
    out.append(
        f'<text class="ag-legend-title" x="0" y="{_f(fs)}" font-size="{_f(fs - 0.5)}" font-weight="600"'
        f'{_paint("fill", th.ink_secondary)}>{escape(lg.title)}</text>'
    )
    top = fs * 1.6
    if lg.kind in ("categorical", "shape"):
        for i, e in enumerate(lg.entries):
            cy = top + 10 + 20 * i
            if lg.kind == "shape":
                sw = get_shape(str(e.value)).path(6, cy, 11, 11)
                out.append(f'<path d="{sw}"{_paint("fill", th.ink_muted)}/>')
                tx = 18
            elif lg.mark == "line":
                out.append(
                    f'<line x1="0" y1="{_f(cy)}" x2="16" y2="{_f(cy)}"{_paint("stroke", e.color)}'
                    f' stroke-width="2.5" stroke-linecap="round"/>'
                )
                tx = 24
            else:
                out.append(f'<circle cx="6" cy="{_f(cy)}" r="5"{_paint("fill", e.color)}/>')
                tx = 18
            val = "" if e.value is None else f' data-value="{_attr(e.value)}"'
            label = escape(e.label)
            out.append(
                f'<text class="ag-legend-item"{val} x="{tx}" y="{_f(cy + fs * 0.35)}" font-size="{_f(fs)}"'
                f'{_paint("fill", th.ink)}>{label}'
                + (
                    f'<tspan dx="8" font-size="{_f(fs - 1)}"{_paint("fill", th.ink_muted)}>{e.count:,}</tspan>'
                    if e.count is not None and lg.kind == "categorical"
                    else ""
                )
                + "</text>"
            )
    elif lg.kind == "colorbar" and lg.colormap is not None:
        gid = f"{uid}-grad{n}"
        stops = lg.colormap.sample(11)
        out.append(
            f'<defs><linearGradient id="{gid}" x1="0" x2="1" y1="0" y2="0">'
            + "".join(f'<stop offset="{i / 10:.2f}" stop-color="{c}"/>' for i, c in enumerate(stops))
            + "</linearGradient></defs>"
        )
        bw = w
        out.append(f'<rect x="0" y="{_f(top + 2)}" width="{_f(bw)}" height="10" rx="2" fill="url(#{gid})"/>')
        for i, (t, label) in enumerate(lg.ticks):
            tx = t * bw
            anchor = "middle"
            if tx < 12:
                anchor = "start"
            elif tx > bw - 12:
                anchor = "end"
            out.append(
                f'<line x1="{_f(tx)}" y1="{_f(top + 12)}" x2="{_f(tx)}" y2="{_f(top + 16)}"'
                f'{_paint("stroke", th.ink_muted)} stroke-width="1"/>'
            )
            out.append(
                f'<text x="{_f(tx)}" y="{_f(top + 18 + fs)}" font-size="{_f(fs - 1)}" text-anchor="{anchor}"'
                f' style="font-variant-numeric:tabular-nums"{_paint("fill", th.ink_muted)}>{escape(label)}</text>'
            )
    elif lg.kind in ("size", "width"):
        cx = 0.0
        tallest = max((e.size or 0) for e in lg.entries) if lg.kind == "size" else 8.0
        base_y = top + 4 + tallest
        for e in lg.entries:
            s = e.size or 0.0
            lw = len(e.label) * fs * 0.55
            slot = max(s if lg.kind == "size" else 26.0, lw)
            mid = cx + slot / 2
            if lg.kind == "size":
                out.append(
                    f'<circle cx="{_f(mid)}" cy="{_f(base_y - s / 2)}" r="{_f(s / 2)}"{_paint("fill", th.ink_muted)}'
                    f' fill-opacity="0.22"{_paint("stroke", th.ink_muted)} stroke-width="1"/>'
                )
            else:
                out.append(
                    f'<line x1="{_f(mid - 13)}" y1="{_f(base_y - 4)}" x2="{_f(mid + 13)}" y2="{_f(base_y - 4)}"'
                    f'{_paint("stroke", th.ink_muted)} stroke-width="{_f(s)}" stroke-linecap="round"/>'
                )
            out.append(
                f'<text x="{_f(mid)}" y="{_f(base_y + 4 + fs)}" font-size="{_f(fs - 1)}" text-anchor="middle"'
                f'{_paint("fill", th.ink_muted)}>{escape(e.label)}</text>'
            )
            cx += slot + 14
    out.append("</g>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# document
# --------------------------------------------------------------------------- #
def scene_to_svg(scene: Scene, *, xml_declaration: bool = False) -> str:
    """Serialise *scene* as an SVG document string."""
    th = scene.theme
    uid = _uid(scene)
    W, H = scene.width, scene.height
    dw, dh = scene.display_size or (W, H)
    title = scene.title or scene.meta.get("name") or "Graph"
    desc = (
        f"{'Directed' if scene.directed else 'Undirected'} graph with {scene.meta.get('nodes', 0):,} nodes and "
        f"{scene.meta.get('edges', 0):,} edges, {scene.meta.get('layout')} layout."
    )
    out: list[str] = []
    if xml_declaration:
        out.append('<?xml version="1.0" encoding="UTF-8"?>\n')
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" class="ag-root" id="{uid}" width="{_f(dw)}" height="{_f(dh)}"'
        f' viewBox="0 0 {_f(W)} {_f(H)}" role="img" aria-labelledby="{uid}-t {uid}-d"'
        f' font-family="{_attr(th.font)}" text-rendering="geometricPrecision">'
    )
    out.append(f'<title id="{uid}-t">{escape(str(title))}</title><desc id="{uid}-d">{escape(desc)}</desc>')
    out.append(
        "<style>"
        f"#{uid} text{{font-family:{th.font};}}"
        f"#{uid} .ag-legend-title,#{uid} .ag-title{{letter-spacing:-0.01em}}"
        "</style>"
    )
    if scene.background:
        out.append(f'<rect class="ag-bg" width="{_f(W)}" height="{_f(H)}"{_paint("fill", scene.background)}/>')

    pad = th.padding
    if scene.title or scene.subtitle:
        out.append('<g class="ag-header">')
        y = pad
        if scene.title:
            y += th.title_size
            out.append(
                f'<text class="ag-title" x="{_f(pad)}" y="{_f(y)}" font-size="{_f(th.title_size)}"'
                f' font-weight="{th.title_weight}"{_paint("fill", th.ink)}>{escape(scene.title)}</text>'
            )
            y += th.title_size * 0.35
        if scene.subtitle:
            y += th.subtitle_size * 1.35
            out.append(
                f'<text class="ag-subtitle" x="{_f(pad)}" y="{_f(y)}" font-size="{_f(th.subtitle_size)}"'
                f'{_paint("fill", th.ink_secondary)}>{escape(scene.subtitle)}</text>'
            )
        out.append("</g>")

    ox, oy = scene.offset
    out.append(f'<g class="ag-viewport"><g class="ag-content" transform="translate({_f(ox)},{_f(oy)})">')
    normal = [e for e in scene.edges if not e.highlight]
    lifted = [e for e in scene.edges if e.highlight]
    out.append('<g class="ag-edges">')
    out.extend(_edge(e, th.dim_opacity) for e in normal)
    out.extend(_edge(e, th.dim_opacity) for e in lifted)
    out.append("</g>")
    elabels = [e for e in scene.edges if e.label]
    if elabels:
        out.append('<g class="ag-edge-labels">')
        for e in elabels:
            assert e.label is not None
            op = ' opacity="0.35"' if e.dimmed else ""
            out.append(f'<g class="ag-el" data-k="{e.key}"{op}>' + _text(e.label, cls="ag-elbl") + "</g>")
        out.append("</g>")
    out.append('<g class="ag-nodes">')
    out.extend(_node(m, th.dim_opacity) for m in _node_order(scene.nodes))
    out.append("</g>")
    side = [m for m in scene.nodes if m.label is not None and not m.inside]
    if side:
        out.append('<g class="ag-labels">')
        out.extend(_side_label(m) for m in side)
        out.append("</g>")
    out.append("</g></g>")

    if scene.legends:
        out.append('<g class="ag-legend">')
        for n, (lg, x, y, w, h) in enumerate(scene.legends):
            out.append(_legend(lg, x, y, w, h, scene, uid, n))
        out.append("</g>")
    if scene.caption:
        out.append(
            f'<text class="ag-caption" x="{_f(pad)}" y="{_f(H - pad + th.font_size * 0.2)}" font-size="{_f(th.font_size - 1)}"'
            f'{_paint("fill", th.ink_muted)}>{escape(scene.caption)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def _node_order(nodes: Iterable[NodeMark]) -> list[NodeMark]:
    """Draw small nodes last so they are never hidden under big ones; highlighted on top."""
    return sorted(nodes, key=lambda m: (m.highlight, -(m.w * m.h), m.index))


__all__ = ["scene_to_svg"]
