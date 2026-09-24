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
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import aryagraph as ag
from aryagraph.render import draw
from aryagraph.render import geometry as geo
from aryagraph.render.scene import build_scene
from aryagraph.style.shapes import get_shape

SVG = "{http://www.w3.org/2000/svg}"


def _tri():
    g = ag.Graph([("a", "b"), ("b", "c"), ("c", "a")])
    pos = {"a": (0, 0), "b": (1, 0), "c": (0.5, 0.9)}
    return g, pos


def _parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


# ----------------------------------------------------------------- geometry
def test_bezier_split_is_continuous():
    seg = ((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0))
    left, right = geo.split(seg, 0.3)
    assert left[3] == pytest.approx(right[0])
    assert geo.point_at(seg, 0.3) == pytest.approx(left[3])


def test_trim_end_keeps_requested_distance():
    p = geo.straight((0, 0), (100, 0)).trim_end(10)
    assert p.end[0] == pytest.approx(90, abs=0.01)


def test_trim_end_on_loop_keeps_the_loop():
    loop = geo.self_loop(0, 0, 20, 20, -math.pi / 4, 16)
    shape = get_shape("circle")
    loop = loop.clip_start(shape, 0, 0, 20, 20).clip_end(shape, 0, 0, 20, 20)
    trimmed = loop.trim_end(8)
    assert trimmed.length() > 20  # the loop survives arrow trimming


@pytest.mark.parametrize("name", ["circle", "box", "diamond", "hexagon", "cylinder"])
def test_clip_to_shapes_lands_on_outline(name):
    shape = get_shape(name)
    path = geo.bent((0, 0), (200, 0), 0.2).clip_start(shape, 0, 0, 40, 30).clip_end(shape, 200, 0, 40, 30)
    for (cx, cy), p in (((0, 0), path.start), ((200, 0), path.end)):
        on = shape.boundary(cx, cy, 40, 30, *p)
        assert math.hypot(on[0] - p[0], on[1] - p[1]) < 0.05  # exactly on the outline
    # and the curve between the ends stays outside both nodes
    for t in (0.1, 0.5, 0.9):
        x, y = geo.point_at(path.segments[0], t)
        assert not shape.contains(0, 0, 40, 30, x, y) and not shape.contains(200, 0, 40, 30, x, y)


def test_orthogonal_route_is_axis_aligned():
    p = geo.orthogonal([(0, 0), (50, 100)], "y", radius=0)
    for s in p.segments:
        (x0, y0), (x1, y1) = s[0], s[3]
        assert x0 == pytest.approx(x1) or y0 == pytest.approx(y1)


def test_arrowhead_tip_and_back_offset():
    d, back = geo.arrowhead((10, 10), (1, 0), 8, 6)
    assert d.startswith("M10,10")
    assert 0 < back < 8


# ----------------------------------------------------------------- scene
def test_scene_positions_and_marks():
    g, pos = _tri()
    s = build_scene(g, layout=pos)
    assert len(s.nodes) == 3 and len(s.edges) == 3
    assert all(e.d.startswith("M") for e in s.edges)
    assert s.width > 0 and s.height > 0


def test_edges_end_on_node_outlines():
    g, pos = _tri()
    s = build_scene(g, layout=pos, node_shape="box", labels=False)
    marks = {m.node: m for m in s.nodes}
    for e in s.edges:
        mu, mv = marks[e.u], marks[e.v]
        start, end = e.path.start, e.path.end
        # start on u's outline (within numerical tolerance), end on v's
        sh = get_shape("box")
        for m, p in ((mu, start), (mv, end)):
            bx, by = sh.boundary(m.x, m.y, m.w, m.h, *p)
            assert math.hypot(bx - p[0], by - p[1]) < 1.0


def test_arrows_only_for_directed():
    g, pos = _tri()
    assert all(e.arrow is None for e in build_scene(g, layout=pos).edges)
    d = ag.DiGraph(g.edges)
    assert all(e.arrow for e in build_scene(d, layout=pos).edges)
    assert all(e.arrow is None for e in build_scene(d, layout=pos, arrows=False).edges)


def test_reciprocal_arcs_are_curved_apart():
    d = ag.DiGraph([("a", "b"), ("b", "a")])
    s = build_scene(d, layout={"a": (0, 0), "b": (1, 0)})
    assert {e.style for e in s.edges} == {"curved"}
    mids = [e.path.midpoint()[0][1] for e in s.edges]
    assert mids[0] * mids[1] < 0 or abs(mids[0] - mids[1]) > 5  # opposite sides


def test_self_loop_is_drawn():
    g = ag.Graph([(1, 1), (1, 2)])
    s = build_scene(g, layout={1: (0, 0), 2: (1, 0)})
    loop = [e for e in s.edges if e.style == "loop"][0]
    assert loop.path.length() > 10


def test_labels_never_overlap():
    g = ag.Graph()
    rng = np.random.default_rng(0)
    pos = {i: tuple(rng.random(2)) for i in range(60)}
    g.add_nodes(range(60))
    g.add_edges((i, (i * 7) % 60) for i in range(60))
    s = build_scene(g, layout=pos, labels=[f"node number {i}" for i in range(60)], label_position="auto")
    boxes = [m.label.box for m in s.nodes if m.label and m.label.visible and not m.inside]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            assert not (a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1])


def test_highlight_path_dims_others():
    g = ag.Graph([(1, 2), (2, 3), (3, 4), (1, 4)])
    pos = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}
    s = build_scene(g, layout=pos, highlight_path=[1, 2, 3])
    hl = {(e.u, e.v) for e in s.edges if e.highlight}
    assert hl == {(1, 2), (2, 3)}
    assert {m.node for m in s.nodes if m.dimmed} == {4}


def test_missing_positions_raise():
    g, _ = _tri()
    with pytest.raises(ag.AryaGraphError):
        build_scene(g, layout={"a": (0, 0)})


def test_legends_follow_encodings():
    g, pos = _tri()
    for n, club in zip("abc", ["x", "y", "x"]):
        g.nodes[n]["club"] = club
        g.nodes[n]["score"] = {"a": 0.1, "b": 0.5, "c": 0.9}[n]
    s = build_scene(g, layout=pos, node_color="club", node_size="score")
    kinds = {lg.kind for lg, *_ in s.legends}
    assert kinds == {"categorical", "size"}
    assert not build_scene(g, layout=pos, node_color="club", legend=False).legends


# ----------------------------------------------------------------- svg / html
def test_svg_is_well_formed_and_accessible():
    g, pos = _tri()
    fig = draw(g, layout=pos, title="Triangle & <friends>", node_color={"a": "red"})
    root = _parse(fig.to_svg())
    assert root.tag == SVG + "svg"
    assert root.get("role") == "img"
    assert root.find(SVG + "title").text == "Triangle & <friends>"
    assert len(root.findall(f".//{SVG}g[@class='ag-n']")) == 3
    assert len(root.findall(f".//{SVG}g[@class='ag-e']")) == 3


def test_unicode_and_rtl_labels_survive():
    g = ag.Graph([("گره الف", "گره ب"), ("گره ب", "node <c>")])
    fig = draw(g, layout={"گره الف": (0, 0), "گره ب": (1, 0), "node <c>": (2, 0)})
    svg = fig.to_svg()
    _parse(svg)
    assert "گره الف" in svg and 'direction="rtl"' in svg and "node &lt;c&gt;" in svg


def test_html_contains_payload_and_runtime():
    g, pos = _tri()
    html = draw(g, layout=pos).to_html()
    assert html.startswith("<!doctype html>")
    assert 'class="ag-data"' in html and "ag-app" in html and "pointerdown" in html
    static = draw(g, layout=pos).to_html(interactive=False)
    assert "pointerdown" not in static


def test_save_formats(tmp_path):
    g, pos = _tri()
    fig = draw(g, layout=pos)
    assert fig.save(tmp_path / "g.svg").read_text(encoding="utf-8").startswith("<?xml")
    assert "<html" in fig.save(tmp_path / "g.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        fig.save(tmp_path / "g.bmp")


def test_cairosvg_without_the_cairo_library_falls_back(monkeypatch):
    import builtins

    from aryagraph.render import export

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cairosvg":
            raise OSError('no library called "cairo-2" was found')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert export._cairosvg() is None
    monkeypatch.setattr(export, "find_browser", lambda: None)
    # with no browser either, the error names the real cause, not "pip install cairosvg"
    with pytest.raises(ag.DependencyError, match=r"cannot load the Cairo library: no library called \"cairo-2\""):
        export.svg_to_png("<svg xmlns='http://www.w3.org/2000/svg'/>", "unused.png", 10, 10)


def test_missing_cairosvg_and_browser_suggest_pip(monkeypatch):
    import builtins

    from aryagraph.render import export

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cairosvg":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(export, "find_browser", lambda: None)
    with pytest.raises(ag.DependencyError, match=r"\(pip install cairosvg\)"):
        export.svg_to_pdf("<svg xmlns='http://www.w3.org/2000/svg'/>", "unused.pdf", 10, 10)


def test_browser_pdf_takes_the_figure_title(monkeypatch, tmp_path):
    from urllib.parse import unquote, urlparse

    from aryagraph.render import export

    titles = []

    def fake_browser(args, timeout=120):
        page = Path(unquote(urlparse(args[-1]).path.lstrip("/")))
        if not page.exists():  # POSIX paths keep their leading slash
            page = Path(unquote(urlparse(args[-1]).path))
        text = page.read_text(encoding="utf-8")
        titles.append(text[text.index("<title>") + 7 : text.index("</title>")])
        out = next(a for a in args if a.startswith("--print-to-pdf="))
        Path(out.split("=", 1)[1]).write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(export, "_cairosvg", lambda: None)
    monkeypatch.setattr(export, "find_browser", lambda: "browser")
    monkeypatch.setattr(export, "_run_browser", fake_browser)
    g, pos = _tri()
    draw(g, layout=pos, title="Caf\u00e9 & friends").save(tmp_path / "titled.pdf")
    export.svg_to_pdf("<svg xmlns='http://www.w3.org/2000/svg'/>", tmp_path / "plain-name.pdf", 10, 10)
    assert titles == ["Caf\u00e9 &amp; friends", "plain-name"]


def test_tooltip_rejects_callables_with_a_clear_message():
    g, pos = _tri()
    assert build_scene(g, layout=pos, tooltip=["x"]).nodes
    with pytest.raises(TypeError, match="tooltip takes an attribute name"):
        build_scene(g, layout=pos, tooltip=lambda n: n)


def test_themes_change_colors():
    g, pos = _tri()
    light = draw(g, layout=pos, theme="light").scene
    dark = draw(g, layout=pos, theme="dark").scene
    assert light.background != dark.background
    assert light.nodes[0].fill != dark.nodes[0].fill


def test_explicit_size_sets_display_size():
    g, pos = _tri()
    fig = draw(g, layout=pos, width=800, height=600)
    assert (fig.width, fig.height) == (800, 600)
    root = _parse(fig.to_svg())
    assert root.get("width") == "800"


def test_empty_graph_renders():
    fig = draw(ag.Graph(), layout={})
    _parse(fig.to_svg())
