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

"""Scene building: graph + layout + encodings → positioned visual marks.

A :class:`Scene` is backend-neutral. It holds every mark in final pixel
coordinates (node outlines, edge paths already clipped to node boundaries,
arrowheads, labels that were placed without collisions, legends), so the SVG
and HTML backends only serialise; they never make layout decisions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np

from ..core.exceptions import AryaGraphError
from ..layout.base import Layout
from ..style import text as textm
from ..style.colors import label_on, mix, to_hex
from ..style.scales import (
    By,
    Legend,
    LegendEntry,
    field_name,
    resolve_color,
    resolve_number,
    resolve_values,
)
from ..style.shapes import BOX_SHAPES, SHAPES, get_shape
from ..style.themes import Theme, get_theme
from . import geometry as geo

Node = Hashable
Box = tuple[float, float, float, float]  # x0, y0, x1, y1

FLOW_METHODS = {"hierarchical", "layered", "sugiyama", "tree"}
SHAPE_CYCLE = ("circle", "square", "diamond", "triangle", "hexagon", "triangle_down", "pentagon", "octagon")


# --------------------------------------------------------------------------- #
# marks
# --------------------------------------------------------------------------- #
@dataclass
class LabelMark:
    lines: list[str]
    x: float
    y: float  # baseline of the first line
    anchor: str  # start | middle | end (visual, before RTL adjustment)
    size: float
    weight: int
    color: str
    halo: str | None
    box: Box
    rtl: bool = False
    visible: bool = True
    line_height: float = 1.25


@dataclass
class NodeMark:
    node: Node
    key: str
    index: int
    x: float
    y: float
    w: float
    h: float
    shape: str
    fill: str
    stroke: str
    stroke_width: float
    opacity: float = 1.0
    label: LabelMark | None = None
    inside: bool = False
    text: str = ""
    tooltip: dict[str, Any] = field(default_factory=dict)
    group: Any = None
    highlight: bool = False
    dimmed: bool = False
    halo_color: str | None = None  # highlight ring

    @property
    def box(self) -> Box:
        return (self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2)

    def outline(self) -> str:
        return get_shape(self.shape).path(self.x, self.y, self.w, self.h)


@dataclass
class EdgeMark:
    u: Node
    v: Node
    key: str
    ukey: str
    vkey: str
    path: geo.EdgePath
    d: str
    stroke: str
    width: float
    opacity: float
    arrow: str | None = None
    style: str = "straight"
    curvature: float = 0.0
    label: LabelMark | None = None
    tooltip: dict[str, Any] = field(default_factory=dict)
    highlight: bool = False
    dimmed: bool = False
    dash: str | None = None
    route: list[tuple[float, float]] = field(default_factory=list)
    arrow_len: float = 0.0
    label_anchor: tuple[float, float] | None = None


@dataclass
class Scene:
    """Everything needed to draw a graph, in pixel coordinates.

    Graph marks live in *content* coordinates; backends translate them by
    ``offset``. Titles and legends use canvas coordinates.
    """

    width: float
    height: float
    theme: Theme
    nodes: list[NodeMark]
    edges: list[EdgeMark]
    legends: list[tuple[Legend, float, float, float, float]]  # legend, x, y, w, h
    offset: tuple[float, float]
    background: str | None
    title: str | None = None
    subtitle: str | None = None
    caption: str | None = None
    directed: bool = False
    layout: Layout | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    display_size: tuple[float, float] | None = None  # explicit width/height asked by the user

    @property
    def node_index(self) -> dict[Node, NodeMark]:
        return {m.node: m for m in self.nodes}


# --------------------------------------------------------------------------- #
# the builder
# --------------------------------------------------------------------------- #
def base_node_size(n: int) -> float:
    if n <= 30:
        return 18.0
    if n <= 100:
        return 14.0
    if n <= 300:
        return 10.0
    if n <= 1000:
        return 7.0
    if n <= 5000:
        return 5.0
    return 3.5


def _auto_method(g: Any) -> str:
    n = len(g)
    if g.directed and n <= 2000 and g.is_dag() and g.num_edges > 0:
        return "hierarchical"
    return "stress" if n <= 3000 else "forceatlas2"


def _jsonable(v: Any, depth: int = 0) -> Any:
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, np.generic):
        return _jsonable(v.item(), depth)
    if depth < 2 and isinstance(v, (list, tuple, set, frozenset)):
        items = list(v)[:12]
        return [_jsonable(x, depth + 1) for x in items]
    if depth < 2 and isinstance(v, Mapping):
        return {str(k): _jsonable(x, depth + 1) for k, x in list(v.items())[:12]}
    s = str(v)
    return s if len(s) <= 120 else s[:117] + "…"


class SceneBuilder:
    """Stateful helper behind :func:`build_scene`; each step is one method."""

    def __init__(self, g: Any, opts: dict[str, Any]) -> None:
        self.g = g
        self.o = opts
        self.theme: Theme = get_theme(opts.get("theme"))
        self.nodes: list[Node] = list(g)
        self.nattrs: list[dict] = [g._node[n] for n in self.nodes]
        self.edges: list[tuple[Node, Node, dict]] = list(g.edges.data())
        self.eattrs = [d for _, _, d in self.edges]
        self.ekeys = [(u, v) for u, v, _ in self.edges]
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.n = len(self.nodes)
        self.base = float(opts.get("base_size") or base_node_size(self.n))
        self.legends: list[Legend] = []

    # ------------------------------------------------------------------ #
    def build(self) -> Scene:
        self._resolve_method()
        self._resolve_labels()
        self._resolve_shapes()
        self._resolve_node_colors()
        self._resolve_node_sizes()
        self._compute_layout()
        self._to_pixels()
        self._make_nodes()
        self._resolve_highlight()
        self._make_edges()
        self._place_labels()
        return self._compose()

    # ------------------------------------------------------------------ #
    # 1. layout method & labels & shapes
    # ------------------------------------------------------------------ #
    def _resolve_method(self) -> None:
        lay = self.o.get("layout", "auto")
        self.given_layout: Layout | None = None
        if isinstance(lay, Layout):
            self.given_layout = lay
            self.method = lay.method
        elif isinstance(lay, Mapping):
            self.given_layout = Layout.from_positions(lay, nodes=[n for n in self.nodes if n in lay] or None)
            self.method = "custom"
        elif callable(lay):
            res = lay(self.g)
            self.given_layout = res if isinstance(res, Layout) else Layout.from_positions(res)
            self.method = self.given_layout.method
        elif isinstance(lay, str):
            self.method = _auto_method(self.g) if lay == "auto" else lay
        else:
            raise TypeError(f"layout must be a name, a Layout, a {{node: (x, y)}} mapping or a callable, got {lay!r}")
        if self.given_layout is not None:
            missing = [n for n in self.nodes if n not in self.given_layout]
            if missing:
                raise AryaGraphError(f"layout has no position for {len(missing)} node(s), e.g. {missing[:3]!r}")
        self.flow_like = self.method in FLOW_METHODS or bool(self.given_layout and self.given_layout.meta.get("orientation"))

    def _resolve_labels(self) -> None:
        spec = self.o.get("labels", "auto")
        n = self.n
        texts: list[str | None]
        if spec is False or spec is None:
            texts = [None] * n
        elif spec is True or spec == "auto" or isinstance(spec, int) and not isinstance(spec, bool):
            texts = [str(a.get("label", node)) for node, a in zip(self.nodes, self.nattrs)]
        else:
            vals = resolve_values(spec, self.nodes, self.nattrs)
            if vals is None:
                raise ValueError(f"labels={spec!r} is not an attribute present on any node")
            texts = [None if v is None else str(v) for v in vals]
        self.texts = texts
        if spec == "auto":
            self.label_quota = n if n <= 80 else 30
        elif isinstance(spec, int) and not isinstance(spec, bool):
            self.label_quota = spec
        else:
            self.label_quota = n
        self.label_size = float(self.o.get("label_size") or self.theme.label_size)

    def _resolve_shapes(self) -> None:
        spec = self.o.get("node_shape")
        n = self.n
        self.shape_legend: Legend | None = None
        if spec is None:
            all_labeled = self.label_quota >= n and any(self.texts)
            wordy = any(t is not None and len(t) > 3 for t in self.texts)
            if self.flow_like and all_labeled and n <= 150 and wordy:
                shapes = ["box"] * n  # label-sized cards; short ids read better in circles
            else:
                shapes = ["circle"] * n
        elif isinstance(spec, str) and spec in SHAPES:
            shapes = [spec] * n
        else:
            vals = resolve_values(spec, self.nodes, self.nattrs)
            if vals is None:
                raise ValueError(f"node_shape={spec!r} is neither a shape name ({', '.join(SHAPES)}) nor a node attribute")
            if all(v in SHAPES for v in vals if v is not None):
                shapes = [v or "circle" for v in vals]
            else:
                cats = list(dict.fromkeys(v for v in vals if v is not None))
                cmap = {c: SHAPE_CYCLE[i % len(SHAPE_CYCLE)] for i, c in enumerate(cats)}
                shapes = [cmap.get(v, "circle") for v in vals]
                self.shape_legend = Legend(
                    "shape",
                    field_name(spec) or "shape",
                    [LegendEntry(str(c), value=cmap[c]) for c in cats[: len(SHAPE_CYCLE)]],
                    channel="node_shape",
                )
        self.shapes = shapes

        pos = self.o.get("label_position", "auto")
        inside = []
        for s, t in zip(shapes, self.texts):
            if t is None:
                inside.append(False)
            elif pos == "center":
                inside.append(True)
            elif pos != "auto":
                inside.append(False)
            elif s in BOX_SHAPES and s not in ("ellipse",) and (spec is None or self.flow_like):
                inside.append(True)
            elif s in ("circle", "ellipse", "square") and (self.n <= 60 or (self.flow_like and self.n <= 150)) and len(t) <= 3:
                inside.append(True)
            else:
                inside.append(False)
        # inside labels are all-or-nothing per drawing so the look stays uniform
        if pos == "auto" and any(inside) and not all(inside[i] or self.texts[i] is None for i in range(n)):
            inside = [False] * n
        self.inside = inside

    # ------------------------------------------------------------------ #
    # 2. colors & sizes
    # ------------------------------------------------------------------ #
    def _resolve_node_colors(self) -> None:
        res = resolve_color(self.o.get("node_color"), self.nodes, self.nattrs, self.theme, self.theme.node_fill, channel="node_color")
        self.node_colors = res.colors
        self.node_color_values = res.values
        self.node_groups = res.values if res.kind == "categorical" else None
        if res.legend:
            self.legends.append(res.legend)

    def _resolve_node_sizes(self) -> None:
        spec = self.o.get("node_size")
        base = self.base
        res = resolve_number(spec, self.nodes, self.nattrs, base, (base * 0.55, base * 2.6), channel="node_size")
        self.size_values = res.raw
        if res.legend:
            self.legends.append(res.legend)
        sizes: list[tuple[float, float]] = []
        self.label_lines: list[list[str]] = []
        lsize = self.label_size
        maxw = float(self.o.get("label_max_width") or 160)
        for i in range(self.n):
            d = res.values[i]
            shape = self.shapes[i]
            text = self.texts[i]
            if self.inside[i] and text is not None:
                if shape in ("circle", "square", "ellipse"):
                    lines = textm.wrap(text, max(maxw * 0.6, 40), lsize - 0.5, self.theme.label_weight, max_lines=2)
                    tw, th = textm.block_size(lines, lsize - 0.5, self.theme.label_weight)
                    side = max(d, tw + 10, th + 8)
                    w = h = side
                    if shape == "ellipse":
                        w = max(d * 1.4, tw * 1.3 + 12)
                        h = max(d, th * 1.4 + 6)
                else:
                    lines = textm.wrap(text, maxw, lsize, self.theme.label_weight, max_lines=3)
                    tw, th = textm.block_size(lines, lsize, self.theme.label_weight)
                    w = max(tw + 28, 56.0)
                    h = max(th + 16, 32.0)
                    if shape == "diamond":
                        w, h = w * 1.55, h * 1.55
                    elif shape in ("hexagon", "octagon"):
                        w += h * 0.55
                    elif shape == "cylinder":
                        h += 12
                    elif shape == "pill":
                        w += h * 0.3
                self.label_lines.append(lines)
            else:
                w = h = d
                if shape == "ellipse":
                    w = d * 1.5
                self.label_lines.append([text] if text is not None else [])
            sizes.append((w, h))
        self.sizes = sizes

    # ------------------------------------------------------------------ #
    # 3. layout
    # ------------------------------------------------------------------ #
    def _layout_sizes(self) -> dict[Node, tuple[float, float]]:
        """Node boxes used for spacing by size-aware layouts (side labels included)."""
        out = {}
        orient = (self.o.get("layout_options") or {}).get("orientation", "TB")
        for i, node in enumerate(self.nodes):
            w, h = self.sizes[i]
            if not self.inside[i] and self.texts[i] is not None and self.label_quota >= self.n:
                tw = textm.text_width(self.texts[i], self.label_size, self.theme.label_weight)
                if orient in ("LR", "RL"):
                    h = h + 2 * (self.label_size * 1.3 + 4)
                    w = max(w, tw)
                else:
                    w = w + 2 * (tw + 6)
            out[node] = (w, h)
        return out

    def _compute_layout(self) -> None:
        if self.given_layout is not None:
            self.layout = self.given_layout
            return
        from ..layout import compute

        opts = dict(self.o.get("layout_options") or {})
        if self.method in ("hierarchical", "layered", "sugiyama", "tree"):
            # tighter than the engine defaults: rendered boxes already carry padding
            opts.setdefault("rank_sep", 52.0)
            opts.setdefault("node_sep", 24.0)
        elif self.method == "radial":
            biggest = max((max(w, h) for w, h in self.sizes), default=self.base)
            labeled = self.label_quota >= self.n and any(t is not None for t in self.texts)
            opts.setdefault("rank_sep", max(2.2 * biggest + 12, 80.0 if labeled else 40.0))
        seed = self.o.get("seed", 0)
        try:
            self.layout = compute(self.g, self.method, seed=seed, sizes=self._layout_sizes(), **opts)
        except TypeError as exc:
            if "sizes" not in str(exc):
                raise
            self.layout = compute(self.g, self.method, seed=seed, **opts)

    def _to_pixels(self) -> None:
        lay = self.layout
        if len(lay) == 0:
            self.pos = {}
            self.routes = {}
            return
        scale = self.o.get("scale")
        if lay.metric:
            s = float(scale or 1.0)
            lay_px = lay.scaled(s) if s != 1.0 else lay
        else:
            xy = lay.xy
            lengths = []
            idx = lay._index
            for u, v in self.ekeys:
                if u != v and u in idx and v in idx:
                    d = float(np.hypot(*(xy[idx[u]] - xy[idx[v]])))
                    if d > 1e-9:
                        lengths.append(d)
            if lengths:
                typical = float(np.median(lengths))
            elif len(lay) > 1:
                diffs = xy[:, None, :] - xy[None, :, :]
                dist = np.hypot(diffs[..., 0], diffs[..., 1])
                np.fill_diagonal(dist, np.inf)
                typical = float(np.median(dist.min(axis=1)))
            else:
                typical = 1.0
            typical = typical or 1.0
            target = max(4.2 * self.base, 48.0)
            shown = [t for t in self.texts if t is not None]
            if shown and self.label_quota >= self.n and not any(self.inside):
                widths = sorted(textm.text_width(t, self.label_size) for t in shown)
                target = max(target, 0.8 * widths[len(widths) // 2] + self.base)
            s = float(scale) if scale else target / typical
            lay_px = lay.scaled(s)
            x0, y0, x1, y1 = lay_px.bounds()
            if not scale and shown and not any(self.inside):
                # Leave room for the labels: the drawing should be ~6× the ink it must hold,
                # otherwise the collision-free placer has to hide most labels in dense cores.
                ink = sum(w * h for w, h in self.sizes)
                quota = min(self.label_quota, len(shown))
                if quota:
                    mean_w = sum(textm.text_width(t, self.label_size) for t in shown) / len(shown)
                    ink += quota * (mean_w + 6) * self.label_size * 1.3
                area = max((x1 - x0) * (y1 - y0), 1.0)
                if area < 6.0 * ink:
                    f = min(math.sqrt(6.0 * ink / area), 2.0)
                    lay_px = lay_px.scaled(f)
                    x0, y0, x1, y1 = lay_px.bounds()
            w_req, h_req = self.o.get("width"), self.o.get("height")
            if not scale and (w_req or h_req):
                margin = 2 * max(max(w, h) for w, h in self.sizes) + 40
                fx = (float(w_req) - margin - 60) / max(x1 - x0, 1e-9) if w_req else math.inf
                fy = (float(h_req) - margin - 80) / max(y1 - y0, 1e-9) if h_req else math.inf
                f = min(fx, fy)
                if math.isfinite(f) and f > 0:
                    lay_px = lay_px.scaled(f)
            elif not scale:
                cap = 2000.0
                big = max(x1 - x0, y1 - y0)
                if big > cap:
                    lay_px = lay_px.scaled(cap / big)
            if self.o.get("avoid_overlap", True) and 1 < len(lay_px) <= 3000:
                try:
                    from ..layout.overlap import remove_overlaps

                    sizes = {n: self.sizes[self.index[n]] for n in lay_px.nodes if n in self.index}
                    lay_px = remove_overlaps(lay_px, sizes, padding=3.0)
                except ImportError:
                    pass
        self.layout_px = lay_px
        self.pos = {n: (float(x), float(y)) for n, (x, y) in zip(lay_px.nodes, lay_px.xy)}
        self.routes = {e: [tuple(map(float, p)) for p in pts] for e, pts in lay_px.routes.items()}

    # ------------------------------------------------------------------ #
    # 4. node marks
    # ------------------------------------------------------------------ #
    def _make_nodes(self) -> None:
        th = self.theme
        opacity = self.o.get("node_opacity", 1.0)
        op_vals = resolve_values(opacity, self.nodes, self.nattrs) if not isinstance(opacity, (int, float)) else None
        tip_spec = self.o.get("tooltip")
        marks = []
        for i, node in enumerate(self.nodes):
            x, y = self.pos[node]
            w, h = self.sizes[i]
            color = self.node_colors[i]
            shape = self.shapes[i]
            boxy = self.inside[i] and shape not in ("circle", "square", "ellipse")
            if boxy:
                fill = mix(th.surface, color, 0.14)
                stroke = color
                sw = 1.35
            else:
                fill = color
                stroke = th.node_ring
                sw = min(th.node_ring_width, max(w, h) * 0.12) if max(w, h) < 12 else th.node_ring_width
            op = float(op_vals[i]) if op_vals is not None and op_vals[i] is not None else float(opacity if isinstance(opacity, (int, float)) else 1.0)
            mark = NodeMark(
                node=node,
                key=f"n{i}",
                index=i,
                x=x,
                y=y,
                w=w,
                h=h,
                shape=shape,
                fill=to_hex(fill),
                stroke=to_hex(stroke),
                stroke_width=sw,
                opacity=op,
                inside=self.inside[i],
                text=self.texts[i] or "",
                group=self.node_groups[i] if self.node_groups is not None else None,
            )
            if self.inside[i] and self.label_lines[i]:
                lines = self.label_lines[i]
                size = self.label_size - (0.5 if not boxy else 0)
                lh = 1.25
                total = len(lines) * size * lh
                first = y - total / 2 + size * 0.95
                if shape == "cylinder":
                    first += 4
                ink = th.ink if boxy else label_on(fill)
                mark.label = LabelMark(
                    lines,
                    x,
                    first,
                    "middle",
                    size,
                    th.label_weight,
                    ink,
                    None,
                    mark.box,
                    rtl=textm.is_rtl(" ".join(lines)),
                    line_height=lh,
                )
            mark.tooltip = self._node_tooltip(i, node, tip_spec)
            marks.append(mark)
        self.node_marks = marks
        self.mark_of = {m.node: m for m in marks}

    def _node_tooltip(self, i: int, node: Node, spec: Any) -> dict[str, Any]:
        g = self.g
        info: dict[str, Any] = {"id": str(node), "label": self.texts[i] if self.texts[i] is not None else str(node)}
        if g.directed:
            info["in"] = len(g._pred[node])
            info["out"] = len(g._succ[node])
        else:
            info["degree"] = g._degree_of(node, None)
        attrs = self.nattrs[i]
        if spec is None or spec is True:
            keys = [k for k in attrs if k not in ("label", "pos", "x", "y")][:10]
        elif spec is False:
            keys = []
        elif isinstance(spec, str):
            keys = [spec]
        else:
            keys = list(spec)
        info["attrs"] = {str(k): _jsonable(attrs.get(k)) for k in keys if k in attrs}
        for ch, vals in (("node_color", self.node_color_values), ("node_size", self.size_values)):
            name = field_name(self.o.get(ch))
            if vals is not None and name and name not in info["attrs"]:
                info["attrs"][name] = _jsonable(vals[i])
        return info

    # ------------------------------------------------------------------ #
    # 5. highlight
    # ------------------------------------------------------------------ #
    def _resolve_highlight(self) -> None:
        hl_nodes: set = set()
        hl_edges: set = set()
        spec = self.o.get("highlight")
        path = self.o.get("highlight_path")
        if spec is not None:
            items = list(spec) if not isinstance(spec, (str, int)) else [spec]
            node_items = []
            for it in items:
                if isinstance(it, tuple) and len(it) == 2 and self.g.has_edge(*it):
                    hl_edges.add(it)
                    hl_nodes.update(it)
                elif it in self.g:
                    node_items.append(it)
            hl_nodes.update(node_items)
            ns = set(node_items)
            for u, v in self.ekeys:
                if u in ns and v in ns:
                    hl_edges.add((u, v))
        if path is not None:
            p = list(path)
            hl_nodes.update(p)
            for a, b in zip(p, p[1:]):
                if self.g.has_edge(a, b):
                    hl_edges.add((a, b))
                elif not self.g.directed and self.g.has_edge(b, a):
                    hl_edges.add((b, a))
        if not self.g.directed:
            hl_edges |= {(v, u) for u, v in hl_edges}
        self.hl_nodes = hl_nodes
        self.hl_edges = hl_edges
        self.hl_color = to_hex(self.o.get("highlight_color") or self.theme.edge_highlight)
        if hl_nodes or hl_edges:
            for m in self.node_marks:
                if m.node in hl_nodes:
                    m.highlight = True
                    m.halo_color = self.hl_color
                else:
                    m.dimmed = True

    # ------------------------------------------------------------------ #
    # 6. edges
    # ------------------------------------------------------------------ #
    def _make_edges(self) -> None:
        th = self.theme
        g = self.g
        m = len(self.edges)
        style = self.o.get("edge_style", "auto")
        orient = self.layout_px.meta.get("orientation") if self.n else None
        if style == "auto":
            style = "flow" if orient and self.layout_px.metric and self.flow_like else "straight"
        self.edge_style = style
        axis = "x" if orient in ("LR", "RL") else "y"

        # colors
        ecspec = self.o.get("edge_color")
        if isinstance(ecspec, str) and ecspec in ("source", "target"):
            idx = 0 if ecspec == "source" else 1
            ecolors = [self.node_colors[self.index[e[idx]]] for e in self.ekeys]
            data_colored = True
        else:
            res = resolve_color(ecspec, self.ekeys, self.eattrs, th, th.edge, edges=True, channel="edge_color")
            ecolors = res.colors
            data_colored = res.kind != "constant" or (ecspec is not None)
            if res.legend:
                self.legends.append(res.legend)
        dense = 1.0 if m <= 400 else 0.75 if m <= 2000 else 0.55
        default_w = th.edge_width * (1.0 if m <= 1000 else 0.8)
        wres = resolve_number(self.o.get("edge_width"), self.ekeys, self.eattrs, default_w, (0.75, 5.0), edges=True, channel="edge_width", area=False)
        if wres.legend:
            self.legends.append(wres.legend)
        ospec = self.o.get("edge_opacity")
        if ospec is None:
            base_op = 0.9 if data_colored else th.edge_opacity * dense
        else:
            base_op = float(ospec)

        arrows = self.o.get("arrows")
        arrows = g.directed if arrows is None else bool(arrows)
        arrow_size = self.o.get("arrow_size")
        curv = float(self.o.get("curvature", 0.16))

        # ports for flow/orthogonal styles: spread edges along the node side
        ports = self._ports(axis) if style in ("flow", "orthogonal") and self.n else {}
        labels_spec = self.o.get("edge_label")
        elabels = resolve_values(labels_spec, self.ekeys, self.eattrs, edges=True) if labels_spec not in (None, False) else None

        out = []
        for k, (u, v, attrs) in enumerate(self.edges):
            mu, mv = self.mark_of[u], self.mark_of[v]
            width = float(wres.values[k])
            route = self.routes.get((u, v))
            reverse_route = False
            if route is None and not g.directed and (v, u) in self.routes:
                route = list(reversed(self.routes[(v, u)]))
            route = route or []
            estyle = style
            used_curv = curv
            if u == v:
                path = self._loop(mu)
                estyle = "loop"
            elif estyle in ("flow", "orthogonal"):
                pu, pv = ports.get((k, 0)), ports.get((k, 1))
                if pu is None or pv is None:
                    pu = self._boundary(mu, route[0] if route else (mv.x, mv.y))
                    pv = self._boundary(mv, route[-1] if route else (mu.x, mu.y))
                pts = [pu, *route, pv]
                delta = (pv[1] - pu[1]) if axis == "y" else (pv[0] - pu[0])
                if abs(delta) < 1.0 and not route:
                    path = geo.bent(pu, pv, 0.3)
                    estyle = "curved"
                elif estyle == "flow":
                    path = geo.flow(pts, axis)
                else:
                    path = geo.orthogonal(pts, axis, radius=8.0)
            elif estyle in ("curved",) or (g.directed and estyle == "straight" and g.has_edge(v, u)):
                c = curv if estyle == "curved" else max(curv, 0.12)
                used_curv = c
                path = geo.bent((mu.x, mu.y), (mv.x, mv.y), c)
                if route:
                    path = geo.through([(mu.x, mu.y), *route, (mv.x, mv.y)])
                path = path.clip_start(get_shape(mu.shape), mu.x, mu.y, mu.w, mu.h)
                path = path.clip_end(get_shape(mv.shape), mv.x, mv.y, mv.w, mv.h)
                estyle = "curved"
            elif estyle == "spline" and route:
                a = self._boundary(mu, route[0])
                b = self._boundary(mv, route[-1])
                path = geo.through([a, *route, b])
            else:
                a = self._boundary(mu, route[0] if route else (mv.x, mv.y))
                b = self._boundary(mv, route[-1] if route else (mu.x, mu.y))
                path = geo.polyline([a, *route, b]) if route else geo.straight(a, b)
                estyle = "straight" if not route else "polyline"

            arrow_d = None
            alen = 0.0
            if arrows:
                alen = float(arrow_size) if arrow_size else th.arrow_size + 1.1 * width
                if alen > 0:
                    path = path.trim_end(1.2)  # breathing room between tip and node outline
                    arrow_d, back = geo.arrowhead(path.end, path.end_tangent(), alen, alen * 0.78)
                    path = path.trim_end(back)
            hl = (u, v) in self.hl_edges
            dim = bool(self.hl_nodes or self.hl_edges) and not hl
            op = base_op
            color = ecolors[k]
            if hl:
                color = self.hl_color
                width = max(width + 1.0, 2.25)
                op = 1.0
            mark = EdgeMark(
                u=u,
                v=v,
                key=f"e{k}",
                ukey=mu.key,
                vkey=mv.key,
                path=path,
                d=path.to_svg(),
                stroke=to_hex(color),
                width=width,
                opacity=op,
                arrow=arrow_d,
                style=estyle,
                curvature=used_curv,
                arrow_len=alen if arrow_d else 0.0,
                highlight=hl,
                dimmed=dim,
                route=[tuple(p) for p in route],
                tooltip={"source": str(u), "target": str(v), "attrs": {str(a): _jsonable(b) for a, b in list(attrs.items())[:10]}},
            )
            if elabels is not None and elabels[k] is not None:
                (lx, ly), _ = path.midpoint()
                mark.label_anchor = (lx, ly)
                txt = _fmt_label(elabels[k])
                size = self.label_size - 1
                tw = textm.text_width(txt, size)
                mark.label = LabelMark(
                    [txt],
                    lx,
                    ly + size * 0.35,
                    "middle",
                    size,
                    500,
                    th.ink_secondary,
                    th.surface,
                    (lx - tw / 2, ly - size * 0.6, lx + tw / 2, ly + size * 0.6),
                    rtl=textm.is_rtl(txt),
                )
            out.append(mark)
        self.edge_marks = out

    def _boundary(self, m: NodeMark, toward: tuple[float, float]) -> tuple[float, float]:
        return get_shape(m.shape).boundary(m.x, m.y, m.w, m.h, toward[0], toward[1])

    def _ports(self, axis: str) -> dict[tuple[int, int], tuple[float, float]]:
        """Attachment points for flow edges, spread along each node's leading/trailing side."""
        groups: dict[tuple[Node, int], list[tuple[float, int, int]]] = defaultdict(list)
        for k, (u, v, _) in enumerate(self.edges):
            if u == v:
                continue
            mu, mv = self.mark_of[u], self.mark_of[v]
            route = self.routes.get((u, v)) or []
            first = route[0] if route else (mv.x, mv.y)
            last = route[-1] if route else (mu.x, mu.y)
            if axis == "y":
                du = 1 if first[1] > mu.y else -1
                dv = 1 if last[1] > mv.y else -1
                groups[(u, du)].append((first[0], k, 0))
                groups[(v, dv)].append((last[0], k, 1))
            else:
                du = 1 if first[0] > mu.x else -1
                dv = 1 if last[0] > mv.x else -1
                groups[(u, du)].append((first[1], k, 0))
                groups[(v, dv)].append((last[1], k, 1))
        ports: dict[tuple[int, int], tuple[float, float]] = {}
        for (node, side), items in groups.items():
            m = self.mark_of[node]
            items.sort()
            cnt = len(items)
            span = (m.w if axis == "y" else m.h) * (0.56 if m.shape not in ("circle", "ellipse", "diamond") else 0.3)
            step = min(span / max(cnt - 1, 1), 14.0) if cnt > 1 else 0.0
            for j, (_, k, end) in enumerate(items):
                off = (j - (cnt - 1) / 2) * step
                if axis == "y":
                    target = (m.x + off, m.y + side * m.h)
                else:
                    target = (m.x + side * m.w, m.y + off)
                ports[(k, end)] = get_shape(m.shape).boundary(m.x, m.y, m.w, m.h, *target)
        return ports

    def _loop(self, m: NodeMark) -> geo.EdgePath:
        g = self.g
        angles = []
        for nb in (list(g._succ[m.node]) + list(g._pred[m.node]) if g.directed else list(g._succ[m.node])):
            if nb == m.node:
                continue
            o = self.mark_of[nb]
            angles.append(math.atan2(o.y - m.y, o.x - m.x))
        if not angles:
            best = -math.pi / 4
        else:
            angles.sort()
            gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi) or 2 * math.pi for i in range(len(angles))]
            i = int(np.argmax(gaps))
            best = angles[i] + gaps[i] / 2
        size = max(10.0, min(m.w, m.h) * 0.9)
        path = geo.self_loop(m.x, m.y, m.w, m.h, best, size)
        shape = get_shape(m.shape)
        path = path.clip_start(shape, m.x, m.y, m.w, m.h)
        return path.clip_end(shape, m.x, m.y, m.w, m.h)

    # ------------------------------------------------------------------ #
    # 7. side labels without collisions
    # ------------------------------------------------------------------ #
    def _place_labels(self) -> None:
        th = self.theme
        pos = self.o.get("label_position", "auto")
        collide = self.o.get("label_collisions", "hide")
        size = self.label_size
        cands_all = ("right", "left", "above", "below", "above-right", "below-right", "above-left", "below-left")
        if pos in ("auto", "center"):
            order = cands_all
        elif pos in cands_all:
            order = (pos,)
        else:
            raise ValueError(f"label_position must be 'auto', 'center' or one of {cands_all}")

        grid = _Grid(48.0)
        for m in self.node_marks:
            x0, y0, x1, y1 = m.box
            grid.add((x0 - 1, y0 - 1, x1 + 1, y1 + 1), m.key)
        for e in self.edge_marks:
            if e.label:
                grid.add(e.label.box, e.key)
            if e.style == "loop":
                grid.add(e.path.bbox(), e.key)

        # importance: size, then degree, then graph order
        deg = [self.g._degree_of(n, None) for n in self.nodes]
        size_rank = [self.sizes[i][0] * self.sizes[i][1] for i in range(self.n)]
        todo = [i for i in range(self.n) if self.texts[i] is not None and not self.inside[i]]
        todo.sort(key=lambda i: (-size_rank[i], -deg[i], i))
        if self.hl_nodes:
            todo.sort(key=lambda i: 0 if self.nodes[i] in self.hl_nodes else 1)
        quota = self.label_quota
        gap = 4.0
        placed = 0
        for i in todo:
            m = self.node_marks[i]
            if placed >= quota:
                break
            text = self.texts[i]
            lines = [text]
            tw = textm.text_width(text, size, th.label_weight)
            tht = size * 1.2
            chosen = None
            for c in order:
                box, x, y, anchor = _label_candidate(c, m, tw, tht, size, gap)
                if not grid.hits(box, ignore=m.key):
                    chosen = (box, x, y, anchor)
                    break
            visible = True
            if chosen is None:
                if collide == "show" or (self.hl_nodes and m.node in self.hl_nodes):
                    box, x, y, anchor = _label_candidate(order[0], m, tw, tht, size, gap)
                    chosen = (box, x, y, anchor)
                else:
                    visible = False
                    box, x, y, anchor = _label_candidate(order[0], m, tw, tht, size, gap)
                    chosen = (box, x, y, anchor)
            box, x, y, anchor = chosen
            m.label = LabelMark(
                lines,
                x,
                y,
                anchor,
                size,
                th.label_weight,
                th.ink if not m.dimmed else th.ink_muted,
                th.surface,
                box,
                rtl=textm.is_rtl(text),
                visible=visible,
            )
            if visible:
                grid.add(box, "L" + m.key)
                placed += 1

    # ------------------------------------------------------------------ #
    # 8. legends, titles, canvas
    # ------------------------------------------------------------------ #
    def _compose(self) -> Scene:
        th = self.theme
        pad = float(self.o.get("padding") if self.o.get("padding") is not None else th.padding)
        if self.shape_legend:
            self.legends.append(self.shape_legend)
        legend_opt = self.o.get("legend", "auto")
        legends = self.legends if legend_opt not in (False, None, "none") else []

        # content bounds
        boxes: list[Box] = [m.box for m in self.node_marks]
        boxes += [m.label.box for m in self.node_marks if m.label and m.label.visible and not m.inside]
        for e in self.edge_marks:
            boxes.append(e.path.bbox())
            if e.label:
                boxes.append(e.label.box)
        if boxes:
            cx0 = min(b[0] for b in boxes)
            cy0 = min(b[1] for b in boxes)
            cx1 = max(b[2] for b in boxes)
            cy1 = max(b[3] for b in boxes)
        else:
            cx0 = cy0 = 0.0
            cx1 = cy1 = 120.0
        cw, ch = cx1 - cx0, cy1 - cy0

        title, subtitle, caption = self.o.get("title"), self.o.get("subtitle"), self.o.get("caption")
        header = 0.0
        title_w = 0.0
        if title:
            header += th.title_size * 1.35
            title_w = textm.text_width(title, th.title_size, th.title_weight)
        if subtitle:
            header += th.subtitle_size * 1.5
            title_w = max(title_w, textm.text_width(subtitle, th.subtitle_size))
        if header:
            header += 14.0
        cap_h = th.font_size * 1.6 + 6 if caption else 0.0

        measured = [(lg, *_legend_size(lg, th)) for lg in legends]
        where = legend_opt if legend_opt in ("right", "bottom") else ("bottom" if cw > 1.35 * ch and cw > 520 else "right")
        placed_legends: list[tuple[Legend, float, float, float, float]] = []
        min_plot_w = max(cw, 0.0)
        if measured and where == "right":
            lw = max(w for _, w, _ in measured)
            lh = sum(h for _, _, h in measured) + 18.0 * (len(measured) - 1)
            width = pad + max(cw, 0) + 36.0 + lw + pad
            plot_h = max(ch, lh)
            height = pad + header + plot_h + cap_h + pad
            width = max(width, title_w + 2 * pad)
            lx = width - pad - lw
            ly = pad + header + 4
            for lg, w, h in measured:
                placed_legends.append((lg, lx, ly, w, h))
                ly += h + 18.0
            ox = pad - cx0
            oy = pad + header - cy0 + max(0.0, (plot_h - ch) / 2)
        else:
            width = max(pad + cw + pad, title_w + 2 * pad)
            if measured:
                avail = max(width - 2 * pad, max(w for _, w, _ in measured))
                width = max(width, avail + 2 * pad)
                rows: list[list[tuple[Legend, float, float]]] = [[]]
                rw = 0.0
                for item in measured:
                    if rows[-1] and rw + 32.0 + item[1] > avail:
                        rows.append([])
                        rw = 0.0
                    rows[-1].append(item)
                    rw += item[1] + (32.0 if len(rows[-1]) > 1 else 0.0)
                lh = sum(max(h for _, _, h in row) for row in rows) + 14.0 * (len(rows) - 1)
            else:
                rows, lh = [], 0.0
            height = pad + header + ch + (24.0 + lh if measured else 0.0) + cap_h + pad
            ox = pad - cx0 + max(0.0, (width - 2 * pad - cw) / 2)
            oy = pad + header - cy0
            ly = pad + header + ch + 24.0
            for row in rows:
                lx = pad
                rh = max(h for _, _, h in row)
                for lg, w, h in row:
                    placed_legends.append((lg, lx, ly, w, h))
                    lx += w + 32.0
                ly += rh + 14.0

        display = None
        if self.o.get("width") or self.o.get("height"):
            dw = float(self.o.get("width") or width * (float(self.o["height"]) / height))
            dh = float(self.o.get("height") or height * (dw / width))
            display = (dw, dh)

        bg = self.o.get("background")
        background = None if bg in ("transparent", "none", False) else to_hex(bg) if bg else th.surface
        meta = {
            "name": self.g.name,
            "nodes": self.n,
            "edges": len(self.edges),
            "layout": self.layout.method if self.n else self.method,
            "edge_style": getattr(self, "edge_style", "straight"),
            "orientation": self.layout.meta.get("orientation") if self.n else None,
        }
        return Scene(
            width=round(width, 2),
            height=round(height, 2),
            theme=th,
            nodes=self.node_marks,
            edges=self.edge_marks,
            legends=placed_legends,
            offset=(ox, oy),
            background=background,
            title=title,
            subtitle=subtitle,
            caption=caption,
            directed=self.g.directed,
            layout=self.layout_px if self.n else None,
            meta=meta,
            display_size=display,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fmt_label(v: Any) -> str:
    if isinstance(v, float):
        from ..style.numbers import fmt_number

        return fmt_number(v)
    return str(v)


def _label_candidate(where: str, m: NodeMark, tw: float, th: float, size: float, gap: float):
    """Box, anchor point and anchor for a label at position *where* around node *m*."""
    hw, hh = m.w / 2, m.h / 2
    asc = size * 0.35  # baseline shift that vertically centers x-height on a point
    if where == "right":
        x, y = m.x + hw + gap, m.y + asc
        return (x, m.y - th / 2, x + tw, m.y + th / 2), x, y, "start"
    if where == "left":
        x, y = m.x - hw - gap, m.y + asc
        return (x - tw, m.y - th / 2, x, m.y + th / 2), x, y, "end"
    if where == "below":
        top = m.y + hh + gap * 0.5
        return (m.x - tw / 2, top, m.x + tw / 2, top + th), m.x, top + size * 0.92, "middle"
    if where == "above":
        bot = m.y - hh - gap * 0.5
        return (m.x - tw / 2, bot - th, m.x + tw / 2, bot), m.x, bot - size * 0.28, "middle"
    d = 0.72  # diagonal: tuck the label near the node's shoulder
    sx = 1 if "right" in where else -1
    sy = 1 if "below" in where else -1
    ax = m.x + sx * (hw * d + gap * 0.5)
    ay = m.y + sy * (hh * d + gap * 0.5)
    x0 = ax if sx > 0 else ax - tw
    y0 = ay if sy > 0 else ay - th
    box = (x0, y0, x0 + tw, y0 + th)
    return box, ax, y0 + size * 0.92, "start" if sx > 0 else "end"


class _Grid:
    """Uniform-grid spatial hash for rectangle overlap queries."""

    def __init__(self, cell: float) -> None:
        self.cell = cell
        self.cells: dict[tuple[int, int], list[tuple[Box, str]]] = defaultdict(list)

    def _span(self, box: Box):
        c = self.cell
        for i in range(math.floor(box[0] / c), math.floor(box[2] / c) + 1):
            for j in range(math.floor(box[1] / c), math.floor(box[3] / c) + 1):
                yield (i, j)

    def add(self, box: Box, key: str) -> None:
        for cell in self._span(box):
            self.cells[cell].append((box, key))

    def hits(self, box: Box, ignore: str | None = None) -> bool:
        for cell in self._span(box):
            for other, key in self.cells.get(cell, ()):
                if key == ignore:
                    continue
                if box[0] < other[2] and box[2] > other[0] and box[1] < other[3] and box[3] > other[1]:
                    return True
        return False


def _legend_size(lg: Legend, th: Theme) -> tuple[float, float]:
    title_h = th.font_size * 1.6
    title_w = textm.text_width(lg.title, th.font_size - 0.5, 600)
    if lg.kind in ("categorical", "shape"):
        rows = lg.entries
        wmax = 0.0
        for e in rows:
            w = textm.text_width(e.label, th.font_size)
            if e.count is not None and lg.kind == "categorical":
                w += 8 + textm.text_width(f"{e.count:,}", th.font_size - 1)
            wmax = max(wmax, w)
        swatch = 24.0 if lg.mark == "line" else 18.0
        return (max(title_w, swatch + wmax) + 2, title_h + 20.0 * len(rows))
    if lg.kind == "colorbar":
        return (max(title_w, 168.0), title_h + 12 + 6 + th.font_size * 1.3)
    if lg.kind in ("size", "width"):
        total = 0.0
        tallest = 0.0
        for e in lg.entries:
            s = e.size or 0.0
            lw = textm.text_width(e.label, th.font_size - 1)
            total += max(s if lg.kind == "size" else 26.0, lw) + 14
            tallest = max(tallest, s if lg.kind == "size" else 8.0)
        return (max(title_w, total), title_h + tallest + 6 + th.font_size * 1.3)
    return (title_w, title_h)


def build_scene(g: Any, **options: Any) -> Scene:
    """Build a :class:`Scene`; see :func:`aryagraph.draw` for the options."""
    return SceneBuilder(g, options).build()


__all__ = ["Scene", "NodeMark", "EdgeMark", "LabelMark", "build_scene", "base_node_size"]
