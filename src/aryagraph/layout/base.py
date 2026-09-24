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

"""The :class:`Layout` result type shared by every layout algorithm.

Coordinate convention (all of AryaGraph): **screen coordinates**, where x grows
to the right and y grows *downward*. A top-to-bottom hierarchy therefore has its
sources at small y.

A layout is either

* **abstract** (``metric=False``): only relative positions matter; the renderer
  scales them uniformly to fit the canvas (force-directed, spectral, …), or
* **metric** (``metric=True``): coordinates are already in pixels and respect
  the node sizes that were passed in (hierarchical, tidy tree); the renderer
  keeps the scale and sizes the canvas around them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Iterator, Mapping, Sequence

import numpy as np

Node = Hashable
Edge = tuple[Hashable, Hashable]


@dataclass
class Layout:
    """Node positions, optional edge routes, and provenance.

    Attributes
    ----------
    nodes:
        Node order; row *i* of :attr:`xy` belongs to ``nodes[i]``.
    xy:
        ``(n, 2)`` float array of positions (screen coordinates, y down).
    routes:
        Interior bend points of routed edges, ``{(u, v): (k, 2) array}``,
        excluding the endpoints themselves. Edges absent here are drawn
        directly between their endpoints.
    method:
        Name of the algorithm that produced the layout.
    metric:
        True when coordinates are pixels that respect node sizes.
    meta:
        Algorithm-specific extras, e.g. ``{"orientation": "TB", "layers": {...}}``.
    """

    nodes: list[Node]
    xy: np.ndarray
    routes: dict[Edge, np.ndarray] = field(default_factory=dict)
    method: str = "custom"
    metric: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = list(self.nodes)
        xy = np.asarray(self.xy, dtype=float)
        if xy.size == 0:
            xy = xy.reshape(0, 2)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != len(self.nodes):
            raise ValueError(f"xy must have shape ({len(self.nodes)}, 2), got {xy.shape}")
        if not np.all(np.isfinite(xy)):
            raise ValueError("layout contains non-finite coordinates")
        self.xy = xy
        self.routes = {e: np.asarray(p, dtype=float).reshape(-1, 2) for e, p in self.routes.items()}
        self._index = {n: i for i, n in enumerate(self.nodes)}

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_positions(
        cls,
        positions: Mapping[Node, Sequence[float]],
        nodes: Iterable[Node] | None = None,
        **kwargs: Any,
    ) -> "Layout":
        """Build from ``{node: (x, y)}``; *nodes* fixes the order (defaults to mapping order)."""
        order = list(positions) if nodes is None else list(nodes)
        missing = [n for n in order if n not in positions]
        if missing:
            raise KeyError(f"no position for node(s) {missing[:5]!r}")
        xy = np.array([positions[n] for n in order], dtype=float).reshape(-1, 2)
        return cls(order, xy, **kwargs)

    # ------------------------------------------------------------------ #
    # mapping protocol
    # ------------------------------------------------------------------ #
    def __getitem__(self, node: Node) -> np.ndarray:
        try:
            return self.xy[self._index[node]]
        except KeyError:
            raise KeyError(f"node {node!r} has no position in this layout") from None

    def __contains__(self, node: object) -> bool:
        try:
            return node in self._index
        except TypeError:
            return False

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def index(self, node: Node) -> int:
        return self._index[node]

    @property
    def positions(self) -> dict[Node, tuple[float, float]]:
        """``{node: (x, y)}`` as plain floats."""
        return {n: (float(x), float(y)) for n, (x, y) in zip(self.nodes, self.xy)}

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (positions, routes, method, metric, meta)."""
        return {
            "method": self.method,
            "metric": self.metric,
            "positions": {str(n): [float(x), float(y)] for n, (x, y) in zip(self.nodes, self.xy)},
            "routes": {f"{u}->{v}": p.tolist() for (u, v), p in self.routes.items()},
        }

    def __repr__(self) -> str:
        kind = "metric" if self.metric else "abstract"
        return f"<Layout {self.method!r}: {len(self.nodes)} nodes, {kind}, {len(self.routes)} routed edges>"

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def bounds(self, include_routes: bool = True) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of all positions (and route points)."""
        pts = [self.xy]
        if include_routes and self.routes:
            pts.extend(self.routes.values())
        allp = np.vstack(pts) if pts else np.zeros((0, 2))
        if allp.shape[0] == 0:
            return (0.0, 0.0, 0.0, 0.0)
        lo = allp.min(axis=0)
        hi = allp.max(axis=0)
        return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

    def _replace(self, xy: np.ndarray, routes: dict[Edge, np.ndarray], **changes: Any) -> "Layout":
        kw = dict(method=self.method, metric=self.metric, meta=dict(self.meta))
        kw.update(changes)
        return Layout(list(self.nodes), xy, routes=routes, **kw)

    def transformed(self, matrix: np.ndarray, offset: Sequence[float] = (0.0, 0.0)) -> "Layout":
        """Apply ``p ↦ M·p + offset`` to positions and routes."""
        m = np.asarray(matrix, dtype=float).reshape(2, 2)
        off = np.asarray(offset, dtype=float)
        return self._replace(self.xy @ m.T + off, {e: p @ m.T + off for e, p in self.routes.items()})

    def translated(self, dx: float, dy: float) -> "Layout":
        return self.transformed(np.eye(2), (dx, dy))

    def scaled(self, sx: float, sy: float | None = None) -> "Layout":
        """Scale about the origin (uniformly when *sy* is omitted)."""
        sy = sx if sy is None else sy
        return self.transformed(np.diag([sx, sy]))

    def rotated(self, degrees: float, center: Sequence[float] | None = None) -> "Layout":
        """Rotate clockwise on screen (y down) by *degrees* about *center* (default: centroid)."""
        c = np.asarray(center if center is not None else self.centroid(), dtype=float)
        t = math.radians(degrees)
        m = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        return self.transformed(m, c - m @ c)

    def flipped(self, axis: str = "x") -> "Layout":
        """Mirror horizontally (``axis='x'`` negates x) or vertically (``'y'``) about the centroid."""
        c = self.centroid()
        if axis == "x":
            return self.transformed(np.diag([-1.0, 1.0]), (2 * c[0], 0.0))
        if axis == "y":
            return self.transformed(np.diag([1.0, -1.0]), (0.0, 2 * c[1]))
        raise ValueError("axis must be 'x' or 'y'")

    def centroid(self) -> np.ndarray:
        return self.xy.mean(axis=0) if len(self.nodes) else np.zeros(2)

    def normalized(self) -> "Layout":
        """Centered on the origin with the larger half-extent equal to 1 (abstract layouts)."""
        if not self.nodes:
            return self._replace(self.xy.copy(), {})
        x0, y0, x1, y1 = self.bounds()
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        span = max(x1 - x0, y1 - y0) / 2 or 1.0
        return self.transformed(np.eye(2) / span, -center / span)

    def fit(self, width: float, height: float, padding: float = 0.0) -> "Layout":
        """Uniformly scale and center into the box ``[padding, width-padding] × [padding, height-padding]``."""
        x0, y0, x1, y1 = self.bounds()
        w, h = x1 - x0, y1 - y0
        aw, ah = max(width - 2 * padding, 1e-9), max(height - 2 * padding, 1e-9)
        if w <= 0 and h <= 0:
            s = 1.0
        elif w <= 0:
            s = ah / h
        elif h <= 0:
            s = aw / w
        else:
            s = min(aw / w, ah / h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return self.transformed(np.eye(2) * s, (width / 2 - s * cx, height / 2 - s * cy))

    def aligned(self) -> "Layout":
        """Rotate so the principal axis of the node cloud is horizontal (abstract layouts only)."""
        if len(self.nodes) < 3:
            return self
        centered = self.xy - self.centroid()
        cov = centered.T @ centered
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
        angle = math.degrees(math.atan2(major[1], major[0]))
        return self.rotated(-angle)

    def subset(self, nodes: Iterable[Node]) -> "Layout":
        """Layout restricted to *nodes* (routes between kept nodes are kept)."""
        keep = list(dict.fromkeys(n for n in nodes if n in self._index))
        keep_set = set(keep)
        xy = self.xy[[self._index[n] for n in keep]] if keep else np.zeros((0, 2))
        routes = {e: p for e, p in self.routes.items() if e[0] in keep_set and e[1] in keep_set}
        return Layout(keep, xy, routes=routes, method=self.method, metric=self.metric, meta=dict(self.meta))


def pack_components(
    layouts: Sequence[Layout],
    sizes: Sequence[Sequence[float]] | None = None,
    gap: float = 1.0,
    aspect: float = 1.4,
) -> Layout:
    """Arrange several component layouts side by side without overlap.

    Components are sorted by area (largest first) and placed on shelves whose
    width targets *aspect* (width / height) for the combined drawing. *sizes*
    optionally gives each component's ``(width, height)`` padding box; by
    default the bounds are used. *gap* separates boxes (same units as layouts).
    """
    if not layouts:
        return Layout([], np.zeros((0, 2)))
    if len(layouts) == 1:
        return layouts[0]
    boxes = []
    for i, lay in enumerate(layouts):
        x0, y0, x1, y1 = lay.bounds()
        w, h = (x1 - x0, y1 - y0) if sizes is None else sizes[i]
        boxes.append((i, max(w, 0.0), max(h, 0.0), x0, y0))
    boxes.sort(key=lambda b: -(b[1] + gap) * (b[2] + gap))
    total_area = sum((w + gap) * (h + gap) for _, w, h, _, _ in boxes)
    widest = max(w for _, w, _, _, _ in boxes)
    shelf_width = max(widest, math.sqrt(total_area * aspect))

    placed: dict[int, tuple[float, float]] = {}
    x = y = shelf_h = 0.0
    for i, w, h, _, _ in boxes:
        if x > 0 and x + w > shelf_width:
            y += shelf_h + gap
            x = 0.0
            shelf_h = 0.0
        placed[i] = (x, y)
        x += w + gap
        shelf_h = max(shelf_h, h)

    nodes: list[Node] = []
    xys = []
    routes: dict[Edge, np.ndarray] = {}
    for i, lay in enumerate(layouts):
        _, _, _, x0, y0 = next(b for b in boxes if b[0] == i)
        px, py = placed[i]
        moved = lay.translated(px - x0, py - y0)
        nodes.extend(moved.nodes)
        xys.append(moved.xy)
        routes.update(moved.routes)
    first = layouts[0]
    return Layout(
        nodes,
        np.vstack(xys),
        routes=routes,
        method=first.method,
        metric=first.metric,
        meta={"components": len(layouts), **{k: v for k, v in first.meta.items() if k == "orientation"}},
    )


__all__ = ["Layout", "pack_components"]
