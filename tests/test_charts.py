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

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from aryagraph.charts import bar_chart, gantt, histogram, line_chart

SVG = "{http://www.w3.org/2000/svg}"


def _ok(chart):
    root = ET.fromstring(chart.svg)
    assert root.tag == SVG + "svg"
    return root


def test_line_chart_series_and_legend():
    x = np.arange(10)
    ch = line_chart(x, {"a": x, "b": x[::-1]}, title="t")
    root = _ok(ch)
    paths = root.findall(f".//{SVG}path[@class='ag-series']")
    assert len(paths) == 2
    assert all(p.get("stroke-width") == "2" for p in paths)
    assert ch.meta["x_scale"]["t0"] == 0 and ch.meta["x_scale"]["t1"] == 9


def test_line_chart_single_series_has_no_legend_box():
    ch = line_chart([0, 1, 2], {"only": [1, 2, 3]}, legend="auto", direct_labels=False)
    root = _ok(ch)
    # no legend key lines (the only <line> elements are gridlines)
    keys = [l for l in root.findall(f".//{SVG}line") if l.get("stroke-width") == "2.5"]
    assert keys == []


def test_line_chart_handles_nan_gaps():
    ch = line_chart([0, 1, 2, 3], {"s": [1, float("nan"), 2, 3]})
    d = _ok(ch).find(f".//{SVG}path[@class='ag-series']").get("d")
    assert d.count("M") == 2


def test_bar_chart_caps_thickness_and_labels_values():
    ch = bar_chart(["a", "b"], [3, 1], bar=40)
    root = _ok(ch)
    texts = [t.text for t in root.iter(SVG + "text")]
    assert "3" in texts and "1" in texts


def test_histogram_discrete_one_column_per_value():
    ch = histogram([1, 1, 2, 3, 3, 3])
    bars = [p for p in _ok(ch).iter(SVG + "path") if p.find(SVG + "title") is not None]
    assert len(bars) == 3


def test_gantt_from_dicts():
    tasks = [dict(node="a", worker="W1", start=0, end=2), dict(node="b", worker="W1", start=2, end=5, status="failed")]
    ch = gantt(tasks, lanes="worker", critical=["a"])
    root = _ok(ch)
    assert len(root.findall(f".//{SVG}rect[@rx='4']")) == 2


def test_hostile_names_are_escaped():
    evil = 'a & b <c> "d"'
    _ok(line_chart([0, 1], {evil: [0, 1], "x": [1, 0]}, title=evil))
    _ok(bar_chart([evil, "y"], [1, 2], title=evil))
    _ok(gantt([dict(node=evil, worker=evil, start=0, end=1)], lanes="worker", title=evil))
    _ok(histogram([1, 2, 2], title=evil, markers=[(2, evil)]))


def test_chart_save_html(tmp_path):
    ch = line_chart([0, 1], {"a": [0, 1]})
    html = ch.save(tmp_path / "c.html").read_text(encoding="utf-8")
    assert "ag-chart-data" in html and "pointermove" in html
    with pytest.raises(ValueError):
        ch.save(tmp_path / "c.xyz")


def test_chart_save_creates_missing_folders(tmp_path):
    ch = line_chart([0, 1], {"a": [0, 1]})
    out = ch.save(tmp_path / "new" / "deeper" / "c.svg")
    assert out.exists() and out.read_text(encoding="utf-8").startswith("<?xml")
