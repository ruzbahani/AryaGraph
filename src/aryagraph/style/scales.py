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

"""Encodings: turning data (attributes, metrics) into colors and sizes.

A visual channel (``node_color``, ``node_size``, ``edge_width`` …) accepts:

* a constant: ``"#e34948"``, ``12``;
* an attribute name: ``"community"`` (read from each node's / edge's attrs);
* a mapping: ``{node: value}``, e.g. a :class:`NodeMap` from an algorithm;
* a callable: ``f(node)`` / ``f(node, attrs)`` (edges: ``f(u, v)`` / ``f(u, v, attrs)``);
* a sequence aligned with graph order;
* a :class:`By` spec for full control: ``by("pagerank", kind="log", palette="magma")``.

Scales are chosen by the data: strings, booleans and small integer codes are
*categorical* (fixed-order palette slots, never cycled; the tail folds into
"Other"); numbers are *sequential*; data spanning zero can be *diverging*.
Every scale also produces the legend that explains it.
"""

from __future__ import annotations

import inspect
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np

from .colors import Colormap, is_color, to_hex
from .numbers import fmt_number, nice_ticks
from .themes import Theme

CATEGORICAL_HINTS = re.compile(
    r"(^|_)(community|communities|group|cluster|class|category|type|kind|label|block|club|team|"
    r"module|partition|role|state|status|color_group|faction|party|stage|layer|level|tier|department|dept)s?($|_)",
    re.IGNORECASE,
)


@dataclass
class By:
    """Explicit encoding spec.

    Parameters
    ----------
    field:
        Attribute name, mapping, callable or sequence providing the raw values.
    kind:
        ``"categorical"``, ``"sequential"``, ``"diverging"``, ``"log"``, ``"sqrt"`` or
        ``"identity"`` (values already are colors / sizes). Inferred when omitted.
    palette:
        Colors: a list of colors (categorical), ``{value: color}``, or a colormap
        name / :class:`Colormap` (continuous).
    domain:
        Categorical: the category order. Continuous: ``(lo, hi)`` (and midpoint for
        diverging via *midpoint*).
    range:
        Size/width channels: ``(min_px, max_px)``.
    title:
        Legend title (defaults to the attribute name).
    legend:
        Set False to omit this channel from the legend.
    """

    field: Any
    kind: str | None = None
    palette: Any = None
    domain: Any = None
    range: tuple[float, float] | None = None
    title: str | None = None
    legend: bool = True
    midpoint: float | None = None
    max_categories: int = 8
    counts: bool = True


def by(field: Any, kind: str | None = None, **options: Any) -> By:
    """Shorthand for :class:`By`: ``by("pagerank", kind="log", palette="magma")``."""
    return By(field, kind, **options)


# --------------------------------------------------------------------------- #
# value extraction
# --------------------------------------------------------------------------- #
def _arity(fn: Callable) -> int:
    try:
        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
        ]
        return len(params)
    except (TypeError, ValueError):
        return 1


def extract(
    spec: Any,
    keys: Sequence[Hashable],
    attrs: Sequence[Mapping[str, Any]],
    edges: bool = False,
) -> list[Any] | None:
    """Raw per-item values for *spec*, or None when *spec* is a constant.

    *keys* are nodes (or ``(u, v)`` edges when *edges*), *attrs* their attribute dicts.
    """
    if isinstance(spec, By):
        spec = spec.field
    if spec is None:
        return None
    if isinstance(spec, str):
        if any(spec in a for a in attrs):
            return [a.get(spec) for a in attrs]
        return None  # a constant (e.g. a color); caller validates
    if isinstance(spec, Mapping):
        if edges:
            out = []
            for (u, v) in keys:
                val = spec.get((u, v))
                if val is None:
                    val = spec.get((v, u))
                out.append(val)
            return out
        return [spec.get(k) for k in keys]
    if callable(spec):
        n = _arity(spec)
        if edges:
            if n >= 3:
                return [spec(u, v, a) for (u, v), a in zip(keys, attrs)]
            if n == 2:
                return [spec(u, v) for (u, v) in keys]
            return [spec((u, v)) for (u, v) in keys]
        if n >= 2:
            return [spec(k, a) for k, a in zip(keys, attrs)]
        return [spec(k) for k in keys]
    if isinstance(spec, (np.ndarray, list, tuple)) and not isinstance(spec, str):
        vals = list(spec)
        if len(vals) != len(keys):
            raise ValueError(f"sequence encoding has {len(vals)} values for {len(keys)} items")
        return vals
    return None


def field_name(spec: Any) -> str | None:
    if isinstance(spec, By):
        if spec.title:
            return spec.title
        spec = spec.field
    if isinstance(spec, str):
        return spec
    name = getattr(spec, "name", None)
    if isinstance(name, str) and name:
        return name
    return None


def _is_number(v: Any) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(float(v))


def infer_kind(values: Sequence[Any], name: str | None = None) -> str:
    """``categorical`` / ``sequential`` / ``identity`` from the data and its field name."""
    present = [v for v in values if v is not None]
    if not present:
        return "categorical"
    if all(isinstance(v, str) and is_color(v) for v in present):
        return "identity"
    if all(_is_number(v) for v in present):
        integral = all(float(v).is_integer() for v in present)
        distinct = len(set(present))
        if integral and name and CATEGORICAL_HINTS.search(name) and distinct <= 64:
            return "categorical"
        if integral and distinct <= 2:
            return "categorical"
        return "sequential"
    return "categorical"


def _natural_key(v: Any) -> tuple:
    if isinstance(v, str):
        return (1, tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", v)))
    if isinstance(v, bool):
        return (0, int(v))
    if _is_number(v):
        return (0, float(v))
    return (2, str(v))


# --------------------------------------------------------------------------- #
# legends
# --------------------------------------------------------------------------- #
@dataclass
class LegendEntry:
    label: str
    color: str | None = None
    size: float | None = None
    count: int | None = None
    value: Any = None


@dataclass
class Legend:
    """What a renderer needs to draw one legend block."""

    kind: str  # "categorical" | "colorbar" | "size" | "width"
    title: str
    entries: list[LegendEntry] = field(default_factory=list)
    colormap: Colormap | None = None
    domain: tuple[float, float] | None = None
    ticks: list[tuple[float, str]] = field(default_factory=list)
    mark: str = "circle"  # swatch mark: circle | line | rect
    channel: str = ""


# --------------------------------------------------------------------------- #
# color scales
# --------------------------------------------------------------------------- #
@dataclass
class ColorResult:
    colors: list[str]
    legend: Legend | None
    kind: str
    values: list[Any] | None = None
    categories: dict[Any, str] | None = None  # value -> color, for interactivity


def resolve_color(
    spec: Any,
    keys: Sequence[Hashable],
    attrs: Sequence[Mapping[str, Any]],
    theme: Theme,
    default: str,
    *,
    edges: bool = False,
    channel: str = "node_color",
) -> ColorResult:
    """Colors for every item plus the legend that explains them."""
    n = len(keys)
    opts = spec if isinstance(spec, By) else By(spec)
    values = extract(spec, keys, attrs, edges=edges)
    if values is None:
        raw = opts.field
        if raw is None:
            return ColorResult([default] * n, None, "constant")
        if isinstance(raw, str):
            if is_color(raw):
                return ColorResult([to_hex(raw)] * n, None, "constant")
            raise ValueError(
                f"{channel}={raw!r} is neither a color nor an attribute present on any "
                f"{'edge' if edges else 'node'}"
            )
        raise TypeError(f"unsupported {channel} specification: {raw!r}")

    name = field_name(spec) or channel
    kind = opts.kind or infer_kind(values, field_name(spec))
    if kind == "identity":
        return ColorResult([to_hex(v) if v is not None else theme.other for v in values], None, "identity", values)
    if kind == "categorical":
        return _categorical(values, opts, theme, name, channel, edges)
    if kind in ("sequential", "log", "sqrt", "diverging"):
        return _continuous(values, opts, theme, name, kind, channel, edges)
    raise ValueError(f"unknown scale kind {kind!r}")


def _categorical(values: list[Any], opts: By, theme: Theme, name: str, channel: str, edges: bool) -> ColorResult:
    counts = Counter(v for v in values if v is not None)
    if opts.domain is not None:
        domain = [d for d in opts.domain]
    else:
        try:
            domain = sorted(counts, key=_natural_key)
        except TypeError:
            domain = list(dict.fromkeys(v for v in values if v is not None))

    palette = opts.palette
    mapping: dict[Any, str] = {}
    folded: list[Any] = []
    if isinstance(palette, Mapping):
        mapping = {k: to_hex(c) for k, c in palette.items()}
        folded = [d for d in domain if d not in mapping]
    else:
        colors = [to_hex(c) for c in palette] if palette is not None and not isinstance(palette, (str, Colormap)) else list(theme.categorical)
        slots = min(len(colors), opts.max_categories) if palette is None else len(colors)
        if len(domain) <= slots:
            kept = domain
        else:
            # Fold the tail: the most frequent categories keep a color, the rest become "Other".
            keep_n = slots - 1
            ranked = sorted(domain, key=lambda d: (-counts.get(d, 0), domain.index(d)))
            keep_set = set(ranked[:keep_n])
            kept = [d for d in domain if d in keep_set]
        mapping = {d: colors[i] for i, d in enumerate(kept)}
        folded = [d for d in domain if d not in mapping]

    missing_color = theme.other
    out = [mapping.get(v, missing_color) if v is not None else missing_color for v in values]

    legend = None
    if opts.legend:
        show = opts.counts
        entries = [
            LegendEntry(_label(d), mapping[d], count=counts.get(d, 0) if show else None, value=d) for d in domain if d in mapping
        ]
        if folded:
            k = sum(counts.get(d, 0) for d in folded)
            entries.append(LegendEntry(f"Other ({len(folded)})", missing_color, count=k if show else None))
        n_missing = sum(1 for v in values if v is None)
        if n_missing:
            entries.append(LegendEntry("Missing", missing_color, count=n_missing if show else None))
        legend = Legend("categorical", name, entries, mark="line" if edges else "circle", channel=channel)
    return ColorResult(out, legend, "categorical", values, mapping)


def _continuous(values: list[Any], opts: By, theme: Theme, name: str, kind: str, channel: str, edges: bool) -> ColorResult:
    nums = np.array([float(v) if _is_number(v) else np.nan for v in values], dtype=float)
    transform = kind if kind in ("log", "sqrt") else "linear"
    finite = nums[np.isfinite(nums)]
    if transform == "log":
        finite = finite[finite > 0]
    if opts.domain is not None:
        lo, hi = float(opts.domain[0]), float(opts.domain[-1])
    elif finite.size:
        lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = 0.0, 1.0

    if kind == "diverging":
        cmap = theme.diverging_map(opts.palette if isinstance(opts.palette, (str, Colormap)) else None)
        mid = opts.midpoint if opts.midpoint is not None else (0.0 if lo < 0 < hi else float(np.median(finite)) if finite.size else 0.0)
        half = max(abs(hi - mid), abs(mid - lo)) or 1.0

        def t_of(v: float) -> float:
            return 0.5 + 0.5 * (v - mid) / half
    else:
        cmap = theme.sequential_map(opts.palette if isinstance(opts.palette, (str, Colormap)) else None)
        f = _transform(transform)
        flo, fhi = f(lo), f(hi)

        def t_of(v: float) -> float:
            if fhi == flo:
                return 0.5
            return (f(v) - flo) / (fhi - flo)

    out = []
    for v in nums:
        if not math.isfinite(v) or (transform == "log" and v <= 0):
            out.append(theme.other)
        else:
            out.append(cmap(min(max(t_of(v), 0.0), 1.0)))

    legend = None
    if opts.legend:
        if kind == "diverging":
            dlo, dhi = mid - half, mid + half
        else:
            dlo, dhi = lo, hi
        ticks = _colorbar_ticks(dlo, dhi, transform)
        legend = Legend(
            "colorbar",
            name,
            colormap=cmap,
            domain=(dlo, dhi),
            ticks=ticks,
            channel=channel,
        )
        legend.entries = [LegendEntry("transform", value=transform)]
    return ColorResult(out, legend, kind, values)


def _transform(kind: str) -> Callable[[float], float]:
    if kind == "log":
        return lambda v: math.log10(v) if v > 0 else -math.inf
    if kind == "sqrt":
        return lambda v: math.sqrt(max(v, 0.0))
    return lambda v: v


def _colorbar_ticks(lo: float, hi: float, transform: str) -> list[tuple[float, str]]:
    if hi <= lo:
        return [(lo, fmt_number(lo))]
    if transform == "log" and lo > 0:
        e0, e1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        vals = [10.0**e for e in range(e0, e1 + 1) if lo <= 10.0**e <= hi]
        if len(vals) < 2:
            vals = [lo, hi]
        f = _transform("log")
        return [((f(v) - f(lo)) / (f(hi) - f(lo)), fmt_number(v)) for v in vals]
    vals = nice_ticks(lo, hi, 5)
    f = _transform(transform)
    return [((f(v) - f(lo)) / (f(hi) - f(lo)), fmt_number(v)) for v in vals]


def _label(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


# --------------------------------------------------------------------------- #
# numeric channels (size, width, opacity)
# --------------------------------------------------------------------------- #
@dataclass
class NumberResult:
    values: list[float]
    legend: Legend | None
    raw: list[Any] | None = None


def resolve_number(
    spec: Any,
    keys: Sequence[Hashable],
    attrs: Sequence[Mapping[str, Any]],
    default: float,
    range_: tuple[float, float],
    *,
    edges: bool = False,
    channel: str = "node_size",
    area: bool = True,
) -> NumberResult:
    """Pixel sizes/widths. Constants pass through; data are scaled into *range_*.

    With *area* (node sizes), values are scaled linearly to marker **area**
    between the smallest and the largest size, so the diameter grows with the
    square root. The smallest value gets the smallest size rather than zero,
    so a value twice as large does not get exactly twice the area.
    """
    n = len(keys)
    opts = spec if isinstance(spec, By) else By(spec)
    if isinstance(opts.field, Real) and not isinstance(opts.field, bool):
        return NumberResult([float(opts.field)] * n, None)
    values = extract(spec, keys, attrs, edges=edges)
    if values is None:
        if opts.field is None:
            return NumberResult([default] * n, None)
        raise ValueError(f"{channel}={opts.field!r} is neither a number nor an attribute present on any {'edge' if edges else 'node'}")
    if opts.kind == "identity":
        return NumberResult([float(v) if _is_number(v) else default for v in values], None, values)
    nums = np.array([float(v) if _is_number(v) else np.nan for v in values], dtype=float)
    finite = nums[np.isfinite(nums)]
    rlo, rhi = opts.range or range_
    if finite.size == 0:
        return NumberResult([default] * n, None, values)
    transform = opts.kind if opts.kind in ("log", "sqrt") else "linear"
    f = _transform(transform)
    if opts.domain is not None:
        lo, hi = float(opts.domain[0]), float(opts.domain[-1])
    else:
        pos = finite[finite > 0] if transform == "log" else finite
        lo, hi = (float(pos.min()), float(pos.max())) if pos.size else (1.0, 1.0)
    flo, fhi = f(lo), f(hi)

    def scale(v: float) -> float:
        if not math.isfinite(v) or (transform == "log" and v <= 0):
            return rlo
        t = 0.5 if fhi == flo else min(max((f(v) - flo) / (fhi - flo), 0.0), 1.0)
        if area:
            return math.sqrt(rlo**2 + (rhi**2 - rlo**2) * t)
        return rlo + (rhi - rlo) * t

    out = [scale(v) for v in nums]
    legend = None
    if opts.legend and hi > lo:
        name = field_name(spec) or channel
        # round values inside the data range; always at least two so the scale reads as a scale
        cands = [t for t in nice_ticks(lo, hi, 5) if lo - 1e-12 <= t <= hi + 1e-12]
        if len(cands) >= 3:
            ticks = [cands[0], cands[len(cands) // 2], cands[-1]]
        elif len(cands) == 2:
            ticks = cands
        else:
            ticks = [lo, hi]
        legend = Legend(
            "size" if area else "width",
            name,
            [LegendEntry(fmt_number(t), size=scale(t), value=t) for t in ticks],
            channel=channel,
        )
    return NumberResult(out, legend, values)


def resolve_values(spec: Any, keys: Sequence[Hashable], attrs: Sequence[Mapping[str, Any]], *, edges: bool = False) -> list[Any] | None:
    """Raw values for non-visual channels (labels, tooltips, shapes)."""
    return extract(spec, keys, attrs, edges=edges)


__all__ = [
    "By",
    "by",
    "Legend",
    "LegendEntry",
    "ColorResult",
    "NumberResult",
    "resolve_color",
    "resolve_number",
    "resolve_values",
    "infer_kind",
    "extract",
    "field_name",
]
