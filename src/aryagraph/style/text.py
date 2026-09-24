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

"""Text measurement without a font engine.

Layout decisions (box sizes, label collisions, legend widths) need text widths
before anything is drawn. We use the advance widths of Helvetica/Arial (the
metric template most UI sans faces are close to) with script-aware fallbacks
for Arabic/Persian, Hebrew, CJK and emoji. Estimates err slightly wide so
labels never overflow the boxes sized for them.
"""

from __future__ import annotations

import re
import unicodedata

# Helvetica advance widths (1/1000 em) for printable ASCII 32..126.
_ASCII_WIDTHS = (
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,  # ' '..'/'
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,  # '0'..'?'
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,  # '@'..'O'
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,  # 'P'..'_'
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,  # '`'..'o'
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,  # 'p'..'~'
)
_BOLD_FACTOR = 1.065  # Helvetica-Bold is ~6.5% wider on average
_SAFETY = 1.04  # system UI faces (Segoe UI, SF Pro) run a little wider than Helvetica

_RTL_RE = re.compile(r"[֐-ࣿיִ-﷿ﹰ-﻿]")


def _char_width(ch: str) -> float:
    o = ord(ch)
    if 32 <= o <= 126:
        return _ASCII_WIDTHS[o - 32] / 1000
    if ch in "‌‍‎‏﻿":  # joiners & direction marks are zero-width
        return 0.0
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
        if unicodedata.category(ch) == "Mn":  # harakat & other combining marks
            return 0.0
        return 0.52  # Arabic/Persian letters, joined forms average narrower than Latin caps
    if 0x0590 <= o <= 0x05FF:
        return 0.56
    if 0x1100 <= o <= 0x11FF or 0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFF60:
        return 1.0
    if 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF:
        return 1.2
    cat = unicodedata.category(ch)
    if cat.startswith("M"):
        return 0.0
    if cat == "Zs":
        return 0.278
    base = unicodedata.normalize("NFKD", ch)[:1]
    if base and base != ch and 32 <= ord(base) <= 126:
        return _ASCII_WIDTHS[ord(base) - 32] / 1000
    return 0.6


def text_width(text: str, size: float, weight: int | str = 400) -> float:
    """Estimated rendered width in pixels of a single line of *text* at font *size*."""
    em = sum(_char_width(ch) for ch in str(text))
    bold = weight == "bold" or (isinstance(weight, int) and weight >= 600)
    medium = isinstance(weight, int) and 500 <= weight < 600
    factor = _BOLD_FACTOR if bold else (1.03 if medium else 1.0)
    return em * size * factor * _SAFETY


def is_rtl(text: str) -> bool:
    """True when *text* contains right-to-left script (Arabic, Persian, Hebrew…)."""
    return bool(_RTL_RE.search(str(text)))


_BREAK_RE = re.compile(r"(\s+|(?<=[_\-/.])|(?<=[a-z])(?=[A-Z]))")


def wrap(text: str, max_width: float, size: float, weight: int | str = 400, max_lines: int = 3) -> list[str]:
    """Break *text* into lines no wider than *max_width*.

    Breaks prefer whitespace, then ``_ - / .`` and camelCase boundaries. The
    last allowed line is ellipsised when the text does not fit in *max_lines*.
    """
    text = str(text)
    if text_width(text, size, weight) <= max_width or max_lines <= 1:
        return [truncate(text, max_width, size, weight)] if max_lines <= 1 else [text]
    tokens = [t for t in _BREAK_RE.split(text) if t]
    lines: list[str] = []
    cur = ""
    for tok in tokens:
        candidate = cur + tok
        if cur and text_width(candidate.rstrip(), size, weight) > max_width:
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        else:
            cur = candidate
    if cur.strip():
        lines.append(cur.rstrip())
    # a single token wider than the box gets hard-split by characters
    out: list[str] = []
    for line in lines:
        while text_width(line, size, weight) > max_width and len(line) > 1:
            cut = len(line)
            while cut > 1 and text_width(line[:cut], size, weight) > max_width:
                cut -= 1
            out.append(line[:cut])
            line = line[cut:]
        out.append(line)
    if len(out) > max_lines:
        out = out[: max_lines - 1] + [truncate(" ".join(out[max_lines - 1 :]), max_width, size, weight)]
    return out


def truncate(text: str, max_width: float, size: float, weight: int | str = 400) -> str:
    """*text*, cut and suffixed with an ellipsis if it is wider than *max_width*."""
    text = str(text)
    if text_width(text, size, weight) <= max_width:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_width(text[:mid].rstrip() + ell, size, weight) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return (text[:lo].rstrip() + ell) if lo > 0 else ell


def block_size(lines: list[str], size: float, weight: int | str = 400, line_height: float = 1.25) -> tuple[float, float]:
    """``(width, height)`` of a multi-line text block."""
    if not lines:
        return (0.0, 0.0)
    return (max(text_width(line, size, weight) for line in lines), len(lines) * size * line_height)


__all__ = ["text_width", "is_rtl", "wrap", "truncate", "block_size"]
