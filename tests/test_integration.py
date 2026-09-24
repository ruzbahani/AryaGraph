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

"""End-to-end checks across subpackages: the public API as users meet it."""

import json
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import aryagraph as ag
from aryagraph.cli import main as cli_main
from aryagraph.render.export import find_browser


def test_namespaces_exist():
    for name in ("alg", "gen", "sim", "layout", "io", "style", "charts"):
        assert hasattr(ag, name)
    assert callable(ag.draw) and callable(ag.analyze) and callable(ag.animate) and callable(ag.by)
    assert ag.__version__


def test_auto_layout_picks_hierarchical_for_dags_and_stress_otherwise():
    assert ag.draw(ag.gen.ml_pipeline()).scene.meta["layout"] == "hierarchical"
    assert ag.draw(ag.gen.ucalgary_campus()).scene.meta["layout"] == "stress"


def test_dag_drawing_uses_boxes_and_flow_edges_pointing_down():
    fig = ag.draw(ag.gen.ml_pipeline())
    assert {m.shape for m in fig.scene.nodes} == {"box"}
    marks = {m.node: m for m in fig.scene.nodes}
    for e in fig.scene.edges:
        assert e.style in ("flow", "curved")
        assert marks[e.v].y > marks[e.u].y  # TB: every arc goes down
        assert e.arrow


def test_no_node_overlaps_after_overlap_removal():
    g = ag.gen.barabasi_albert(150, 2, seed=1)
    nodes = ag.draw(g, labels=False).scene.nodes
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            assert np.hypot(a.x - b.x, a.y - b.y) >= (a.w + b.w) / 2 - 0.5


def test_large_graph_renders_quickly():
    g = ag.gen.erdos_renyi(1000, 0.004, seed=3)
    t0 = time.perf_counter()
    svg = ag.draw(g, layout="forceatlas2").to_svg()
    assert time.perf_counter() - t0 < 30
    ET.fromstring(svg)


def test_animation_categorical_payload():
    g = ag.gen.watts_strogatz(60, 4, 0.1, seed=1)
    run = ag.sim.SIR(beta=0.5, gamma=0.2).simulate(g, initial={"I": 2}, t_max=30, seed=2)
    fig = run.animate(labels=False)
    anim = fig.extras["anim"]
    assert len(anim["times"]) == run.T
    frames = np.frombuffer(__import__("base64").b64decode(anim["frames"]), dtype=np.uint8).reshape(run.T, len(g))
    assert np.array_equal(frames, run.values)
    assert anim["fire"] is not None and len(anim["fire"]) == run.T
    html = fig.to_html()
    assert "ag-player" in html and "ag-cursor" in html


def test_animation_subsamples_long_runs_and_keeps_transmissions():
    g = ag.gen.watts_strogatz(40, 4, 0.1, seed=1)
    run = ag.sim.SIR(beta=0.6, gamma=0.1).simulate(g, initial={"I": 1}, steps=400, seed=3)
    fig = run.animate(max_frames=50, chart=False)
    anim = fig.extras["anim"]
    assert len(anim["times"]) <= 50
    # every edge that fired in any frame still flashes in some kept frame
    scene_edges = [frozenset((e.u, e.v)) for e in fig.scene.edges]
    shown = {scene_edges[k] for frame in anim["fire"] for k in frame}
    fired = {frozenset(e[:2]) for frame in run.edge_activity for e in frame}
    assert fired and shown == fired


def test_animation_continuous():
    g = ag.gen.path_graph(10)
    run = ag.sim.heat_diffusion(g, {0: 1.0}, t_max=3, n_frames=20)
    fig = run.animate()
    assert "values" in fig.extras["anim"]
    ET.fromstring(fig.to_svg())


def test_schedule_animation_and_gantt():
    plan = ag.gen.project_plan()
    sched = ag.sim.simulate_schedule(plan, workers=2, seed=1)
    ET.fromstring(sched.gantt().svg)
    fig = sched.animate(static_frame="first")
    assert fig.scene.meta["layout"] == "hierarchical"


def test_report_outputs(tmp_path):
    rep = ag.analyze(ag.gen.ucalgary_campus())
    text = str(rep)
    assert "Communities" in text and "Diameter" in text
    data = rep.to_dict()
    json.dumps(data)
    assert data["summary"]["n"] == 56
    html = rep.save(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Graph report" in html and "ag-app" in html
    assert rep.save(tmp_path / "r.json").exists()


def test_report_for_dag_has_critical_path():
    rep = ag.analyze(ag.gen.project_plan())
    assert rep.dag is not None and rep.dag["critical_path"]
    assert rep.dag["critical_length"] == pytest.approx(ag.alg.critical_path(ag.gen.project_plan()).length)


def test_report_never_fails_on_odd_graphs():
    for g in (ag.Graph(), ag.Graph([(1, 1)]), ag.DiGraph([(1, 2), (2, 1)]), ag.gen.empty_graph(5)):
        rep = ag.analyze(g)
        str(rep)
        rep.to_html()


def test_cli_roundtrip(tmp_path, capsys):
    src = tmp_path / "campus.json"
    ag.write(ag.gen.ucalgary_campus(), src)
    assert cli_main(["info", str(src)]) == 0
    assert cli_main(["draw", str(src), "-o", str(tmp_path / "campus.svg"), "--color", "community", "--size", "pagerank"]) == 0
    ET.fromstring((tmp_path / "campus.svg").read_text(encoding="utf-8").split("\n", 1)[1])
    assert cli_main(["convert", str(src), str(tmp_path / "campus.graphml")]) == 0
    assert ag.read(tmp_path / "campus.graphml").num_edges == 83
    assert cli_main(["analyze", str(src), "-o", str(tmp_path / "campus.html")]) == 0
    assert cli_main(["draw", str(tmp_path / "missing.json")]) == 1
    assert "error" in capsys.readouterr().err


@pytest.mark.skipif(find_browser() is None, reason="needs a Chromium-family browser")
def test_interactive_runtime_boots_in_a_real_browser(tmp_path):
    g = ag.gen.ucalgary_campus()
    page = tmp_path / "campus.html"
    ag.draw(g, node_color="kind").save(page)
    run = ag.sim.SIR(beta=0.5, gamma=0.2).simulate(g, initial={"I": 1}, t_max=20, seed=1)
    anim_page = tmp_path / "a.html"
    run.animate().save(anim_page)
    rep_page = tmp_path / "r.html"
    ag.analyze(g).save(rep_page)
    browser = find_browser()
    for p in (page, anim_page, rep_page):
        with tempfile.TemporaryDirectory() as prof:
            proc = subprocess.run(
                [browser, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--virtual-time-budget=4000", "--dump-dom", p.resolve().as_uri()],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
            )
        out = proc.stdout
        if proc.returncode != 0 and "<html" not in out:
            # e.g. CI runners whose sandbox policy stops headless Chrome: an environment limit, not a failure
            pytest.skip(f"browser could not start ({proc.returncode}): {proc.stderr.strip()[:200]}")
        assert 'scale(1.0000)' in out, f"runtime did not initialise on {p.name}"
        assert "data-error" not in out.split("<script", 1)[0], f"runtime error on {p.name}"
