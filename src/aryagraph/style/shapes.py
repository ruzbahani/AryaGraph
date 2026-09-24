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

"""Node shapes: outline paths, point containment and exact boundary points.

Edges end exactly on a node's outline (not at its bounding circle), so arrows
touch diamonds at the tip, rounded boxes at the curve, and ellipses on the
rim. Every shape answers three questions for a box centered at ``(cx, cy)``
with size ``w × h``: its SVG outline, whether a point is inside, and where a
ray from the center toward a point leaves it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

Point = tuple[float, float]


def _f(v: float) -> str:
    """Compact coordinate formatting for SVG path data."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


@dataclass(frozen=True)
class Shape:
    """Base class; subclasses implement :meth:`path`, :meth:`contains`, :meth:`boundary`."""

    name: str

    def path(self, cx: float, cy: float, w: float, h: float) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def contains(self, cx: float, cy: float, w: float, h: float, x: float, y: float) -> bool:  # pragma: no cover
        raise NotImplementedError

    def boundary(self, cx: float, cy: float, w: float, h: float, tx: float, ty: float) -> Point:  # pragma: no cover
        raise NotImplementedError

    # Generic helper for shapes without a closed form: bisection on the ray.
    def _bisect_ray(self, cx: float, cy: float, w: float, h: float, tx: float, ty: float) -> Point:
        dx, dy = tx - cx, ty - cy
        norm = math.hypot(dx, dy)
        if norm < 1e-12:
            return (cx, cy)
        reach = math.hypot(w, h)  # surely outside
        ux, uy = dx / norm, dy / norm
        lo, hi = 0.0, reach
        for _ in range(40):
            mid = (lo + hi) / 2
            if self.contains(cx, cy, w, h, cx + ux * mid, cy + uy * mid):
                lo = mid
            else:
                hi = mid
        return (cx + ux * lo, cy + uy * lo)


@dataclass(frozen=True)
class Ellipse(Shape):
    def path(self, cx: float, cy: float, w: float, h: float) -> str:
        rx, ry = w / 2, h / 2
        return (
            f"M{_f(cx - rx)},{_f(cy)}a{_f(rx)},{_f(ry)} 0 1,0 {_f(2 * rx)},0"
            f"a{_f(rx)},{_f(ry)} 0 1,0 {_f(-2 * rx)},0Z"
        )

    def contains(self, cx, cy, w, h, x, y):
        rx, ry = max(w / 2, 1e-9), max(h / 2, 1e-9)
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0

    def boundary(self, cx, cy, w, h, tx, ty):
        dx, dy = tx - cx, ty - cy
        rx, ry = max(w / 2, 1e-9), max(h / 2, 1e-9)
        k = math.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
        if k < 1e-12:
            return (cx, cy)
        return (cx + dx / k, cy + dy / k)


@dataclass(frozen=True)
class RoundRect(Shape):
    """Rectangle with corner radius *radius* (px, clamped); ``radius=None`` means a pill."""

    radius: float | None = 0.0

    def _r(self, w: float, h: float) -> float:
        r = min(w, h) / 2 if self.radius is None else self.radius
        return max(0.0, min(r, w / 2, h / 2))

    def path(self, cx, cy, w, h):
        r = self._r(w, h)
        x0, y0 = cx - w / 2, cy - h / 2
        if r <= 0.01:
            return f"M{_f(x0)},{_f(y0)}h{_f(w)}v{_f(h)}h{_f(-w)}Z"
        iw, ih = w - 2 * r, h - 2 * r
        return (
            f"M{_f(x0 + r)},{_f(y0)}h{_f(iw)}a{_f(r)},{_f(r)} 0 0,1 {_f(r)},{_f(r)}"
            f"v{_f(ih)}a{_f(r)},{_f(r)} 0 0,1 {_f(-r)},{_f(r)}h{_f(-iw)}"
            f"a{_f(r)},{_f(r)} 0 0,1 {_f(-r)},{_f(-r)}v{_f(-ih)}a{_f(r)},{_f(r)} 0 0,1 {_f(r)},{_f(-r)}Z"
        )

    def contains(self, cx, cy, w, h, x, y):
        hw, hh = w / 2, h / 2
        dx, dy = abs(x - cx), abs(y - cy)
        if dx > hw or dy > hh:
            return False
        r = self._r(w, h)
        if dx <= hw - r or dy <= hh - r:
            return True
        return (dx - (hw - r)) ** 2 + (dy - (hh - r)) ** 2 <= r * r

    def boundary(self, cx, cy, w, h, tx, ty):
        dx, dy = tx - cx, ty - cy
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return (cx, cy)
        hw, hh = w / 2, h / 2
        s = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
        px, py = dx * s, dy * s
        r = self._r(w, h)
        if r > 0 and abs(px) > hw - r and abs(py) > hh - r:
            # Ray hits a rounded corner: intersect with that corner's circle.
            ccx = math.copysign(hw - r, dx)
            ccy = math.copysign(hh - r, dy)
            norm = math.hypot(dx, dy)
            ux, uy = dx / norm, dy / norm
            b = ux * ccx + uy * ccy
            c = ccx * ccx + ccy * ccy - r * r
            disc = max(b * b - c, 0.0)
            t = b + math.sqrt(disc)
            return (cx + ux * t, cy + uy * t)
        return (cx + px, cy + py)


@dataclass(frozen=True)
class Polygon(Shape):
    """Convex polygon given by unit vertices in [-1, 1]² (scaled by the half-size)."""

    vertices: tuple[Point, ...] = ()

    def _pts(self, cx, cy, w, h) -> list[Point]:
        return [(cx + vx * w / 2, cy + vy * h / 2) for vx, vy in self.vertices]

    def path(self, cx, cy, w, h):
        pts = self._pts(cx, cy, w, h)
        return "M" + "L".join(f"{_f(x)},{_f(y)}" for x, y in pts) + "Z"

    def contains(self, cx, cy, w, h, x, y):
        pts = self._pts(cx, cy, w, h)
        sign = 0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if abs(cross) < 1e-12:
                continue
            s = 1 if cross > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
        return True

    def boundary(self, cx, cy, w, h, tx, ty):
        dx, dy = tx - cx, ty - cy
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return (cx, cy)
        pts = self._pts(cx, cy, w, h)
        best = math.inf
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-12:
                continue
            t = ((x1 - cx) * ey - (y1 - cy) * ex) / den
            u = ((x1 - cx) * dy - (y1 - cy) * dx) / den
            if t > 0 and -1e-9 <= u <= 1 + 1e-9:
                best = min(best, t)
        if best is math.inf:
            return self._bisect_ray(cx, cy, w, h, tx, ty)
        return (cx + dx * best, cy + dy * best)


@dataclass(frozen=True)
class Cylinder(Shape):
    """Database / storage symbol: a box with elliptical caps."""

    def _cap(self, h: float) -> float:
        return min(h * 0.14, 9.0)

    def path(self, cx, cy, w, h):
        rx, ry = w / 2, self._cap(h)
        x0, top, bot = cx - rx, cy - h / 2 + ry, cy + h / 2 - ry
        return (
            f"M{_f(x0)},{_f(top)}a{_f(rx)},{_f(ry)} 0 0,1 {_f(w)},0v{_f(bot - top)}"
            f"a{_f(rx)},{_f(ry)} 0 0,1 {_f(-w)},0Z"
            f"M{_f(x0)},{_f(top)}a{_f(rx)},{_f(ry)} 0 0,0 {_f(w)},0"
        )

    def contains(self, cx, cy, w, h, x, y):
        rx, ry = w / 2, self._cap(h)
        dx = x - cx
        if abs(dx) > rx:
            return False
        top, bot = cy - h / 2 + ry, cy + h / 2 - ry
        if top <= y <= bot:
            return True
        yc = top if y < top else bot
        return (dx / rx) ** 2 + ((y - yc) / max(ry, 1e-9)) ** 2 <= 1.0

    def boundary(self, cx, cy, w, h, tx, ty):
        return self._bisect_ray(cx, cy, w, h, tx, ty)


def _regular(n: int, rotation_deg: float) -> tuple[Point, ...]:
    pts = []
    for k in range(n):
        a = math.radians(rotation_deg + 360 * k / n)
        pts.append((math.cos(a), math.sin(a)))
    # stretch so the polygon fills the unit box in both directions
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    sx = 2 / (max(xs) - min(xs))
    sy = 2 / (max(ys) - min(ys))
    ox = (max(xs) + min(xs)) / 2
    oy = (max(ys) + min(ys)) / 2
    return tuple(((x - ox) * sx, (y - oy) * sy) for x, y in pts)


SHAPES: dict[str, Shape] = {
    "circle": Ellipse("circle"),
    "ellipse": Ellipse("ellipse"),
    "square": RoundRect("square", 0.0),
    "rect": RoundRect("rect", 0.0),
    "box": RoundRect("box", 6.0),
    "roundrect": RoundRect("roundrect", 6.0),
    "pill": RoundRect("pill", None),
    "diamond": Polygon("diamond", ((0, -1), (1, 0), (0, 1), (-1, 0))),
    "triangle": Polygon("triangle", _regular(3, -90)),
    "triangle_down": Polygon("triangle_down", _regular(3, 90)),
    "hexagon": Polygon("hexagon", _regular(6, 0)),
    "octagon": Polygon("octagon", _regular(8, 22.5)),
    "pentagon": Polygon("pentagon", _regular(5, -90)),
    "cylinder": Cylinder("cylinder"),
}

# shapes that look right when stretched around a label
BOX_SHAPES = frozenset({"rect", "box", "roundrect", "pill", "ellipse", "hexagon", "octagon", "cylinder", "diamond"})


def get_shape(name: str | Shape) -> Shape:
    if isinstance(name, Shape):
        return name
    try:
        return SHAPES[name]
    except KeyError:
        raise ValueError(f"unknown node shape {name!r}; available: {', '.join(SHAPES)}") from None


def clip_curve(
    point_at: Callable[[float], Point],
    shape: Shape,
    cx: float,
    cy: float,
    w: float,
    h: float,
    from_start: bool = True,
    iterations: int = 30,
) -> float:
    """Curve parameter where a curve leaves (or enters) a node's outline.

    *point_at(t)* evaluates the curve for t in [0, 1]. With *from_start* the
    curve starts inside the node and the exit parameter is returned; otherwise
    it ends inside the node and the entry parameter is returned.
    """
    inside = lambda t: shape.contains(cx, cy, w, h, *point_at(t))  # noqa: E731
    if from_start:
        lo, hi = 0.0, 1.0
        if inside(hi):
            return hi
        for _ in range(iterations):
            mid = (lo + hi) / 2
            if inside(mid):
                lo = mid
            else:
                hi = mid
        return hi
    lo, hi = 0.0, 1.0
    if inside(lo):
        return lo
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if inside(mid):
            hi = mid
        else:
            lo = mid
    return lo


__all__ = ["Shape", "Ellipse", "RoundRect", "Polygon", "Cylinder", "SHAPES", "BOX_SHAPES", "get_shape", "clip_curve"]
