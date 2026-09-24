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

"""Number formatting and "nice" axis ticks."""

from __future__ import annotations

import math


def nice_number(x: float, round_: bool) -> float:
    """Heckbert's nice number: 1, 2, 5 or 10 times a power of ten close to *x*."""
    if x <= 0 or not math.isfinite(x):
        return 1.0
    exp = math.floor(math.log10(x))
    f = x / 10**exp
    if round_:
        nf = 1 if f < 1.5 else 2 if f < 3 else 5 if f < 7 else 10
    else:
        nf = 1 if f <= 1 else 2 if f <= 2 else 5 if f <= 5 else 10
    return nf * 10**exp


def nice_ticks(lo: float, hi: float, count: int = 5, include_bounds: bool = False) -> list[float]:
    """Round tick values spanning ``[lo, hi]`` (about *count* of them).

    With *include_bounds* the first/last ticks extend to cover the whole range
    (use it for axes); otherwise only ticks inside the range are returned
    (use it for colorbars).
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        if lo == 0:
            return [0.0, 1.0] if include_bounds else [0.0]
        span = abs(lo) * 0.5
        lo, hi = lo - span, hi + span
    rng = nice_number(hi - lo, False)
    step = nice_number(rng / max(count - 1, 1), True)
    start = math.floor(lo / step) * step if include_bounds else math.ceil(lo / step - 1e-9) * step
    stop = math.ceil(hi / step) * step if include_bounds else math.floor(hi / step + 1e-9) * step
    n = int(round((stop - start) / step)) + 1
    ticks = [start + i * step for i in range(max(n, 1))]
    digits = max(0, -math.floor(math.log10(step)) + 1) if step < 1 else 0
    return [round(t, digits + 2) + 0.0 for t in ticks]


def tight_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Axis ticks covering ``[lo, hi]`` whose end ticks overshoot the data as little as possible.

    Tries tick counts around *target* and keeps the set with the least wasted
    range (ties: closest to *target* ticks). For example, data ending at 604
    gets an axis ending at 700 with steps of 100 rather than at 800.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return nice_ticks(lo, hi, target, include_bounds=True)
    best: tuple | None = None
    for count in range(3, target + 5):
        ticks = nice_ticks(lo, hi, count, include_bounds=True)
        if len(ticks) < 3:
            continue
        waste = ((ticks[-1] - hi) + (lo - ticks[0])) / (hi - lo)
        key = (round(waste, 2), abs(len(ticks) - target))
        if best is None or key < best[0]:
            best = (key, ticks)
    return best[1] if best else nice_ticks(lo, hi, target, include_bounds=True)


def fmt_number(v: float, digits: int = 3) -> str:
    """Readable number: thousands separators for large values, *digits* significant otherwise."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if not math.isfinite(v):
        return "∞" if v > 0 else ("-∞" if v < 0 else "NaN")
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1 and float(v).is_integer():
        return f"{int(v):,}"
    if a >= 0.001:
        s = f"{v:.{digits}g}"
        return s
    return f"{v:.2e}"


def fmt_compact(v: float) -> str:
    """Compact magnitude: 1,284 → ``1,284``; 12,900 → ``12.9K``; 4.2e6 → ``4.2M``."""
    if v is None or not math.isfinite(v):
        return fmt_number(v)
    a = abs(v)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e4, "K")):
        if a >= limit:
            s = f"{v / (limit if suffix != 'K' else 1e3):.1f}".rstrip("0").rstrip(".")
            return s + suffix
    return fmt_number(v)


def fmt_tick(v: float, step: float) -> str:
    """Tick label with just enough decimals for *step*."""
    if step >= 1 or step <= 0:
        if abs(v) >= 1e4:
            return fmt_compact(v)
        return f"{v:,.0f}"
    decimals = min(6, max(0, -math.floor(math.log10(step) + 1e-9)))
    return f"{v:,.{decimals}f}"


def fmt_percent(v: float, digits: int = 0) -> str:
    return f"{100 * v:.{digits}f}%"


__all__ = ["nice_number", "nice_ticks", "tight_ticks", "fmt_number", "fmt_compact", "fmt_tick", "fmt_percent"]
