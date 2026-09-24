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

"""Edge geometry: every edge is a chain of cubic Bézier segments.

Straight lines, quadratic bends, flow splines, orthogonal routes with rounded
corners and self-loops all reduce to cubic segments, so clipping to node
outlines, shortening for arrowheads, finding label anchors and serialising to
SVG are written once and behave identically for every edge style.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..style.shapes import Shape

Point = tuple[float, float]
Segment = tuple[Point, Point, Point, Point]  # p0, c1, c2, p3


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _fmt(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# --------------------------------------------------------------------------- #
# single-segment operations
# --------------------------------------------------------------------------- #
def point_at(seg: Segment, t: float) -> Point:
    p0, c1, c2, p3 = seg
    u = 1 - t
    a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    return (a * p0[0] + b * c1[0] + c * c2[0] + d * p3[0], a * p0[1] + b * c1[1] + c * c2[1] + d * p3[1])


def tangent_at(seg: Segment, t: float) -> Point:
    """Unit tangent; falls back to the chord when control points coincide with ends."""
    p0, c1, c2, p3 = seg
    u = 1 - t
    dx = 3 * u * u * (c1[0] - p0[0]) + 6 * u * t * (c2[0] - c1[0]) + 3 * t * t * (p3[0] - c2[0])
    dy = 3 * u * u * (c1[1] - p0[1]) + 6 * u * t * (c2[1] - c1[1]) + 3 * t * t * (p3[1] - c2[1])
    n = math.hypot(dx, dy)
    if n < 1e-9:
        dx, dy = p3[0] - p0[0], p3[1] - p0[1]
        n = math.hypot(dx, dy) or 1.0
    return (dx / n, dy / n)


def split(seg: Segment, t: float) -> tuple[Segment, Segment]:
    """De Casteljau split at *t*."""
    p0, c1, c2, p3 = seg
    a = _lerp(p0, c1, t)
    b = _lerp(c1, c2, t)
    c = _lerp(c2, p3, t)
    d = _lerp(a, b, t)
    e = _lerp(b, c, t)
    f = _lerp(d, e, t)
    return (p0, a, d, f), (f, e, c, p3)


def line(a: Point, b: Point) -> Segment:
    return (a, _lerp(a, b, 1 / 3), _lerp(a, b, 2 / 3), b)


def quad(a: Point, ctrl: Point, b: Point) -> Segment:
    """Quadratic Bézier elevated to cubic."""
    return (a, _lerp(a, ctrl, 2 / 3), _lerp(b, ctrl, 2 / 3), b)


def seg_length(seg: Segment, steps: int = 12) -> float:
    total = 0.0
    prev = seg[0]
    for i in range(1, steps + 1):
        p = point_at(seg, i / steps)
        total += math.dist(prev, p)
        prev = p
    return total


# --------------------------------------------------------------------------- #
# paths (lists of segments)
# --------------------------------------------------------------------------- #
@dataclass
class EdgePath:
    segments: list[Segment]

    @property
    def start(self) -> Point:
        return self.segments[0][0]

    @property
    def end(self) -> Point:
        return self.segments[-1][3]

    def to_svg(self) -> str:
        if not self.segments:
            return ""
        p0 = self.segments[0][0]
        parts = [f"M{_fmt(p0[0])},{_fmt(p0[1])}"]
        for _, c1, c2, p3 in self.segments:
            if _is_line(_, c1, c2, p3):
                parts.append(f"L{_fmt(p3[0])},{_fmt(p3[1])}")
            else:
                parts.append(
                    f"C{_fmt(c1[0])},{_fmt(c1[1])} {_fmt(c2[0])},{_fmt(c2[1])} {_fmt(p3[0])},{_fmt(p3[1])}"
                )
        return "".join(parts)

    def length(self) -> float:
        return sum(seg_length(s) for s in self.segments)

    def midpoint(self) -> tuple[Point, Point]:
        """Point and unit tangent halfway along the path (by arc length)."""
        lengths = [seg_length(s) for s in self.segments]
        half = sum(lengths) / 2
        acc = 0.0
        for seg, L in zip(self.segments, lengths):
            if acc + L >= half and L > 0:
                t = _t_at_length(seg, half - acc)
                return point_at(seg, t), tangent_at(seg, t)
            acc += L
        seg = self.segments[-1]
        return point_at(seg, 0.5), tangent_at(seg, 0.5)

    def end_tangent(self) -> Point:
        return tangent_at(self.segments[-1], 1.0)

    def start_tangent(self) -> Point:
        return tangent_at(self.segments[0], 0.0)

    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = [], []
        for seg in self.segments:
            for i in range(9):
                x, y = point_at(seg, i / 8)
                xs.append(x)
                ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))

    # -- trimming -------------------------------------------------------
    def clip_start(self, shape: Shape, cx: float, cy: float, w: float, h: float) -> "EdgePath":
        """Drop the part of the path inside the node at its start."""
        segs = list(self.segments)
        while len(segs) > 1 and shape.contains(cx, cy, w, h, *segs[0][3]):
            segs.pop(0)
        seg = segs[0]
        if shape.contains(cx, cy, w, h, *seg[0]):
            t = _exit_param(seg, shape, cx, cy, w, h)
            seg = split(seg, t)[1]
        segs[0] = seg
        return EdgePath(segs)

    def clip_end(self, shape: Shape, cx: float, cy: float, w: float, h: float) -> "EdgePath":
        """Drop the part of the path inside the node at its end."""
        segs = list(self.segments)
        while len(segs) > 1 and shape.contains(cx, cy, w, h, *segs[-1][0]):
            segs.pop()
        seg = segs[-1]
        if shape.contains(cx, cy, w, h, *seg[3]):
            rev = (seg[3], seg[2], seg[1], seg[0])
            t = _exit_param(rev, shape, cx, cy, w, h)
            seg = split(seg, 1 - t)[0]
        segs[-1] = seg
        return EdgePath(segs)

    def trim_end(self, distance: float) -> "EdgePath":
        """Shorten the path by *distance* (straight-line) from its end point."""
        if distance <= 0:
            return self
        segs = list(self.segments)
        end = segs[-1][3]
        # Walk backwards from the end to the *last* place the curve is `distance`
        # away from the tip; loops whose ends nearly meet are handled correctly.
        while segs:
            seg = segs[-1]
            steps = 48
            prev_t = 1.0
            for k in range(steps - 1, -1, -1):
                t = k / steps
                if math.dist(point_at(seg, t), end) >= distance:
                    lo, hi = t, prev_t
                    for _ in range(30):
                        mid = (lo + hi) / 2
                        if math.dist(point_at(seg, mid), end) >= distance:
                            lo = mid
                        else:
                            hi = mid
                    segs[-1] = split(seg, lo)[0]
                    return EdgePath(segs)
                prev_t = t
            if len(segs) == 1:
                break
            segs.pop()
        # path shorter than the trim: collapse to a stub pointing at the end
        tan = tangent_at(self.segments[-1], 1.0)
        p = (end[0] - tan[0] * distance, end[1] - tan[1] * distance)
        return EdgePath([line(p, p)])

    def trim_start(self, distance: float) -> "EdgePath":
        rev = EdgePath([(s[3], s[2], s[1], s[0]) for s in reversed(self.segments)]).trim_end(distance)
        return EdgePath([(s[3], s[2], s[1], s[0]) for s in reversed(rev.segments)])


def _is_line(p0: Point, c1: Point, c2: Point, p3: Point) -> bool:
    for c in (c1, c2):
        # control point on the chord => straight
        cross = (p3[0] - p0[0]) * (c[1] - p0[1]) - (p3[1] - p0[1]) * (c[0] - p0[0])
        if abs(cross) > 1e-6 * max(1.0, math.dist(p0, p3)) * 10:
            return False
    return True


def _exit_param(seg: Segment, shape: Shape, cx: float, cy: float, w: float, h: float) -> float:
    """First parameter where a segment starting inside the node crosses its outline.

    Scans before bisecting, so curves that leave *and re-enter* the node (self
    loops) are cut at their first exit, not at their end.
    """
    inside = lambda t: shape.contains(cx, cy, w, h, *point_at(seg, t))  # noqa: E731
    steps = 32
    lo, hi = 0.0, None
    for k in range(1, steps + 1):
        t = k / steps
        if not inside(t):
            hi = t
            break
        lo = t
    if hi is None:
        return 1.0
    for _ in range(32):
        mid = (lo + hi) / 2
        if inside(mid):
            lo = mid
        else:
            hi = mid
    return hi


def _t_at_length(seg: Segment, target: float, steps: int = 24) -> float:
    acc = 0.0
    prev = seg[0]
    for i in range(1, steps + 1):
        t = i / steps
        p = point_at(seg, t)
        d = math.dist(prev, p)
        if acc + d >= target and d > 0:
            return (i - 1) / steps + (target - acc) / d / steps
        acc += d
        prev = p
    return 1.0


# --------------------------------------------------------------------------- #
# constructors for edge styles
# --------------------------------------------------------------------------- #
def straight(a: Point, b: Point) -> EdgePath:
    return EdgePath([line(a, b)])


def bent(a: Point, b: Point, curvature: float) -> EdgePath:
    """Quadratic arc whose apex is offset by ``curvature × length`` to the left of a→b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    # normal pointing to the left of travel in screen coords (y down)
    ctrl = (mx + dy * curvature * 2, my - dx * curvature * 2)
    return EdgePath([quad(a, ctrl, b)])


def through(points: Sequence[Point], tension: float = 1.0) -> EdgePath:
    """Smooth curve through *points* (Catmull–Rom, converted to cubic segments)."""
    pts = list(points)
    if len(pts) == 2:
        return straight(pts[0], pts[1])
    segs: list[Segment] = []
    k = tension / 6
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) * k, p1[1] + (p2[1] - p0[1]) * k)
        c2 = (p2[0] - (p3[0] - p1[0]) * k, p2[1] - (p3[1] - p1[1]) * k)
        segs.append((p1, c1, c2, p2))
    return EdgePath(segs)


def flow(points: Sequence[Point], axis: str = "y", softness: float = 0.5) -> EdgePath:
    """Chain of S-curves whose tangents follow the rank *axis* at every point.

    Used for layered layouts: edges leave and enter nodes along the flow
    direction and pass smoothly through the bend points of long edges.
    """
    pts = list(points)
    segs: list[Segment] = []
    for a, b in zip(pts, pts[1:]):
        if axis == "y":
            k = (b[1] - a[1]) * softness
            segs.append((a, (a[0], a[1] + k), (b[0], b[1] - k), b))
        else:
            k = (b[0] - a[0]) * softness
            segs.append((a, (a[0] + k, a[1]), (b[0] - k, b[1]), b))
    return EdgePath(segs)


def polyline(points: Sequence[Point], radius: float = 0.0) -> EdgePath:
    """Straight segments through *points*; corners rounded with *radius* (px)."""
    pts = _dedupe(points)
    if len(pts) < 2:
        p = pts[0] if pts else (0.0, 0.0)
        return EdgePath([line(p, p)])
    if radius <= 0 or len(pts) == 2:
        return EdgePath([line(a, b) for a, b in zip(pts, pts[1:])])
    segs: list[Segment] = []
    cur = pts[0]
    kappa = 0.5523
    for i in range(1, len(pts) - 1):
        prev, corner, nxt = pts[i - 1], pts[i], pts[i + 1]
        d_in = math.dist(prev, corner)
        d_out = math.dist(corner, nxt)
        r = min(radius, d_in / 2, d_out / 2)
        if r < 0.5:
            segs.append(line(cur, corner))
            cur = corner
            continue
        a = _lerp(corner, prev, r / d_in)
        b = _lerp(corner, nxt, r / d_out)
        segs.append(line(cur, a))
        segs.append((a, _lerp(a, corner, kappa), _lerp(b, corner, kappa), b))
        cur = b
    segs.append(line(cur, pts[-1]))
    return EdgePath([s for s in segs if math.dist(s[0], s[3]) > 1e-6] or segs[:1])


def orthogonal(points: Sequence[Point], axis: str = "y", radius: float = 8.0) -> EdgePath:
    """Rank-axis / cross-axis / rank-axis route through *points* (hierarchical layouts)."""
    pts = list(points)
    corners: list[Point] = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if axis == "y":
            mid = (a[1] + b[1]) / 2
            corners += [(a[0], mid), (b[0], mid), b]
        else:
            mid = (a[0] + b[0]) / 2
            corners += [(mid, a[1]), (mid, b[1]), b]
    return polyline(_simplify(corners), radius)


def self_loop(cx: float, cy: float, w: float, h: float, angle: float, size: float) -> EdgePath:
    """Teardrop loop leaving and re-entering a node around direction *angle* (radians)."""
    spread = math.radians(32)
    reach = max(w, h) / 2 + size
    a1, a2 = angle - spread, angle + spread
    start = (cx + math.cos(a1) * max(w, h) * 0.3, cy + math.sin(a1) * max(w, h) * 0.3)
    end = (cx + math.cos(a2) * max(w, h) * 0.3, cy + math.sin(a2) * max(w, h) * 0.3)
    c1 = (cx + math.cos(a1 - 0.35) * reach * 1.9, cy + math.sin(a1 - 0.35) * reach * 1.9)
    c2 = (cx + math.cos(a2 + 0.35) * reach * 1.9, cy + math.sin(a2 + 0.35) * reach * 1.9)
    return EdgePath([(start, c1, c2, end)])


def _dedupe(points: Sequence[Point]) -> list[Point]:
    out: list[Point] = []
    for p in points:
        if not out or math.dist(out[-1], p) > 1e-6:
            out.append((float(p[0]), float(p[1])))
    return out


def _simplify(points: Sequence[Point]) -> list[Point]:
    pts = _dedupe(points)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-6:
            out.append(b)
    if len(pts) > 1:
        out.append(pts[-1])
    return out


# --------------------------------------------------------------------------- #
# arrowheads
# --------------------------------------------------------------------------- #
def arrowhead(tip: Point, direction: Point, length: float, width: float, notch: float = 0.22) -> tuple[str, float]:
    """Arrowhead path with its tip at *tip*, pointing along *direction*.

    Returns ``(svg_path, back_offset)``: the edge line should stop
    *back_offset* before the tip so its stroke hides under the head.
    """
    dx, dy = direction
    n = math.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    bx, by = tip[0] - dx * length, tip[1] - dy * length
    px, py = -dy * width / 2, dx * width / 2
    left = (bx + px, by + py)
    right = (bx - px, by - py)
    back = (tip[0] - dx * length * (1 - notch), tip[1] - dy * length * (1 - notch))
    d = (
        f"M{_fmt(tip[0])},{_fmt(tip[1])}L{_fmt(left[0])},{_fmt(left[1])}"
        f"L{_fmt(back[0])},{_fmt(back[1])}L{_fmt(right[0])},{_fmt(right[1])}Z"
    )
    return d, length * (1 - notch) + 0.3


__all__ = [
    "EdgePath",
    "Segment",
    "point_at",
    "tangent_at",
    "split",
    "line",
    "quad",
    "straight",
    "bent",
    "through",
    "flow",
    "polyline",
    "orthogonal",
    "self_loop",
    "arrowhead",
]
