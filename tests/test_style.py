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

import math

import pytest

from aryagraph.style import colors as C
from aryagraph.style import palettes as P
from aryagraph.style import shapes as S
from aryagraph.style import text as T
from aryagraph.style.numbers import fmt_compact, fmt_number, nice_ticks
from aryagraph.style.scales import By, by, infer_kind, resolve_color, resolve_number
from aryagraph.style.themes import DARK, LIGHT, get_theme


# ----------------------------------------------------------------- colors
@pytest.mark.parametrize(
    "spec, expected",
    [
        ("#2a78d6", "#2a78d6"),
        ("#FFF", "#ffffff"),
        ("rebeccapurple", "#663399"),
        ("rgb(255, 0, 0)", "#ff0000"),
        ("rgb(255 0 0 / 50%)", "#ff000080"),
        ("hsl(120, 100%, 25%)", "#008000"),
        ((0.0, 0.0, 1.0), "#0000ff"),
    ],
)
def test_parse_and_hex(spec, expected):
    assert C.to_hex(spec) == expected


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        C.parse("not-a-color")
    assert not C.is_color("community")
    assert C.is_color("teal")


def test_oklab_roundtrip():
    for hexc in ["#2a78d6", "#eb6834", "#000000", "#ffffff", "#1baf7a"]:
        L, a, b = C.to_oklab(hexc)
        assert C.from_oklab(L, a, b) == hexc


def test_out_of_gamut_is_chroma_reduced_not_clipped():
    out = C.from_oklch(0.7, 0.5, 145)  # far outside sRGB
    r, g, b, _ = C.parse(out)
    assert all(0 <= c <= 1 for c in (r, g, b))
    assert C.to_oklch(out)[0] == pytest.approx(0.7, abs=0.02)  # lightness preserved


def test_contrast_ratio_extremes():
    assert C.contrast_ratio("#000", "#fff") == pytest.approx(21.0)
    assert C.contrast_ratio("#777", "#777") == pytest.approx(1.0)
    assert C.readable_on("#0b0b0b") == "#ffffff"
    assert C.label_on("#2a78d6") == "#ffffff"
    assert C.label_on("#eda100") == "#0b0b0b"


def test_mix_endpoints():
    assert C.mix("#ff0000", "#0000ff", 0) == "#ff0000"
    assert C.mix("#ff0000", "#0000ff", 1) == "#0000ff"


def test_colormap_monotone_lightness():
    cmap = P.colormap("blue")
    Ls = [C.to_oklch(c)[0] for c in cmap.sample(9)]
    assert all(a > b for a, b in zip(Ls, Ls[1:]))  # light -> dark in light mode
    dark = P.colormap("blue", "dark")
    Ld = [C.to_oklch(c)[0] for c in dark.sample(9)]
    assert all(a < b for a, b in zip(Ld, Ld[1:]))  # dark -> light in dark mode


def test_diverging_has_neutral_midpoint():
    cmap = P.diverging()
    mid = cmap(0.5)
    assert C.to_oklch(mid)[1] < 0.02  # gray: no chroma


def test_colormap_reverse_and_unknown():
    assert P.colormap("viridis_r")(0) == P.colormap("viridis")(1)
    with pytest.raises(ValueError):
        P.colormap("rainbow")


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_gray_colormap_stays_neutral(mode):
    cmap = P.colormap("gray", mode)
    assert all(C.to_oklch(c)[1] < 0.04 for c in cmap.sample(9))
    Ls = [C.to_oklch(c)[0] for c in cmap.sample(9)]
    assert Ls == sorted(Ls, reverse=(mode == "light"))


def test_categorical_palette_is_the_validated_one():
    assert LIGHT.categorical[:3] == ("#2a78d6", "#eb6834", "#1baf7a")
    assert len(DARK.categorical) == 8


# ----------------------------------------------------------------- numbers
def test_nice_ticks():
    assert nice_ticks(0, 97, include_bounds=True) == [0, 20, 40, 60, 80, 100]
    assert nice_ticks(0.013, 0.87) == [0.2, 0.4, 0.6, 0.8]
    assert nice_ticks(5, 5) == [5.0] or len(nice_ticks(5, 5)) >= 1


def test_number_formatting():
    assert fmt_number(12345) == "12,345"
    assert fmt_number(0.0123456) == "0.0123"
    assert fmt_compact(12900) == "12.9K"
    assert fmt_compact(4_200_000) == "4.2M"
    assert fmt_number(float("inf")) == "∞"


# ----------------------------------------------------------------- text
def test_text_width_scales_linearly():
    w12 = T.text_width("Hello world", 12)
    assert T.text_width("Hello world", 24) == pytest.approx(2 * w12)
    assert T.text_width("", 12) == 0


def test_persian_text_is_measured_and_detected():
    s = "گره اصلی"
    assert T.is_rtl(s)
    assert 30 < T.text_width(s, 12) < 80
    assert not T.is_rtl("node")


def test_wrap_respects_width_and_lines():
    lines = T.wrap("feature_engineering_pipeline_for_customers", 90, 12)
    assert all(T.text_width(line, 12) <= 90 for line in lines)
    assert len(lines) <= 3
    assert T.truncate("a very long label indeed", 50, 12).endswith("…")


# ----------------------------------------------------------------- shapes
@pytest.mark.parametrize("name", sorted(S.SHAPES))
def test_boundary_points_lie_on_outline(name):
    shape = S.get_shape(name)
    cx, cy, w, h = 10.0, -5.0, 60.0, 34.0
    for ang in range(0, 360, 15):
        tx = cx + math.cos(math.radians(ang)) * 200
        ty = cy + math.sin(math.radians(ang)) * 200
        bx, by_ = shape.boundary(cx, cy, w, h, tx, ty)
        # just inside is inside, just outside is outside
        ux, uy = (bx - cx), (by_ - cy)
        n = math.hypot(ux, uy)
        assert n > 0
        ux, uy = ux / n, uy / n
        assert shape.contains(cx, cy, w, h, bx - ux * 0.05, by_ - uy * 0.05)
        assert not shape.contains(cx, cy, w, h, bx + ux * 0.05, by_ + uy * 0.05)


def test_unknown_shape():
    with pytest.raises(ValueError):
        S.get_shape("blob")


# ----------------------------------------------------------------- scales
def _items(n=10):
    keys = list(range(n))
    attrs = [{"community": i % 3, "score": i / n, "name": f"n{i}", "color": "#ff0000", "w": float(i)} for i in keys]
    return keys, attrs


def test_infer_kind():
    assert infer_kind([0, 1, 2, 1], "community") == "categorical"
    assert infer_kind([0.1, 0.5, 0.3], "pagerank") == "sequential"
    assert infer_kind(["a", "b"], None) == "categorical"
    assert infer_kind(["#fff", "red"], None) == "identity"
    assert infer_kind([1, 5, 9, 12, 40], "degree") == "sequential"


def test_categorical_uses_slots_in_order_and_legend():
    keys, attrs = _items()
    res = resolve_color("community", keys, attrs, LIGHT, "#000")
    assert res.kind == "categorical"
    assert res.colors[0] == LIGHT.categorical[0] and res.colors[1] == LIGHT.categorical[1]
    assert [e.label for e in res.legend.entries] == ["0", "1", "2"]
    assert sum(e.count for e in res.legend.entries) == 10


def test_categorical_folds_tail_into_other():
    keys = list(range(20))
    attrs = [{"g": i} for i in keys]  # 20 categories, each once
    res = resolve_color(by("g", kind="categorical"), keys, attrs, LIGHT, "#000")
    distinct = set(res.colors)
    assert len(distinct) == 8  # 7 kept slots + Other
    assert res.legend.entries[-1].label.startswith("Other")


def test_sequential_and_colorbar():
    keys, attrs = _items()
    res = resolve_color("score", keys, attrs, LIGHT, "#000")
    assert res.kind == "sequential"
    assert res.legend.kind == "colorbar"
    assert res.colors[0] != res.colors[-1]


def test_identity_constant_mapping_and_errors():
    keys, attrs = _items(4)
    assert resolve_color("color", keys, attrs, LIGHT, "#000").colors == ["#ff0000"] * 4
    assert resolve_color("#00ff00", keys, attrs, LIGHT, "#000").colors == ["#00ff00"] * 4
    assert resolve_color({0: "a", 1: "b"}, keys, attrs, LIGHT, "#000").colors[2] == LIGHT.other
    with pytest.raises(ValueError):
        resolve_color("nope", keys, attrs, LIGHT, "#000")


def test_size_scale_is_area_proportional():
    keys = [0, 1, 2]
    attrs = [{"v": 0.0}, {"v": 0.5}, {"v": 1.0}]
    res = resolve_number("v", keys, attrs, 10, (10, 30))
    d0, d1, d2 = res.values
    assert d0 == pytest.approx(10) and d2 == pytest.approx(30)
    assert d1**2 == pytest.approx((10**2 + 30**2) / 2)


def test_get_theme():
    assert get_theme("dark") is DARK
    with pytest.raises(ValueError):
        get_theme("neon")
    t = LIGHT.with_(font_size=14)
    assert t.font_size == 14 and LIGHT.font_size == 12
