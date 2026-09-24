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

"""Figures for the gallery (source/gallery/index.md) and numbers quoted in the FAQ.

Every gallery card shows the code that makes its figure, so this script does
not repeat that code. It runs each ```` ```python ```` block of the gallery page
as printed, in a scratch directory, and turns what the block saved into files
for the page:

* each ``.svg`` becomes a PNG with the same name, rendered by
  :func:`aryagraph.render.export.svg_to_png`;
* each ``.html`` is copied unchanged for the interactive embeds, and an HTML
  page without an SVG twin (the analysis dashboard) is also captured as a PNG
  screenshot with a headless Chromium browser.

``facts.json`` records the numbers quoted in the gallery captions (read from
the snippets' own variables) and the timings and file sizes quoted in the FAQ
(source/project/faq.md), so they can be checked against a fresh run.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import aryagraph as ag
from aryagraph.render.export import HEADLESS_FLAGS, find_browser, svg_to_png

HERE = Path(__file__).resolve().parent
GALLERY = HERE.parents[1] / "source" / "gallery" / "index.md"

# the same block syntax that scripts/check_docs.py executes
FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
SVG_SIZE = re.compile(r'<svg\b[^>]*?\bwidth="([\d.]+)"[^>]*?\bheight="([\d.]+)"')

PNG_WIDTH = 1100  # target pixel width: sharp at card size on high-density screens, small files
DASHBOARD_WINDOW = (1200, 1300)  # screenshot of the top of the dashboard


def _png_from_svg(svg_path: Path, png_path: Path) -> None:
    svg = svg_path.read_text(encoding="utf-8")
    m = SVG_SIZE.search(svg)
    if m is None:
        raise ValueError(f"{svg_path.name}: no width/height on the root <svg>")
    w, h = float(m.group(1)), float(m.group(2))
    scale = min(2.0, max(1.0, PNG_WIDTH / w))
    svg_to_png(svg, png_path, w, h, scale=scale)


def _screenshot(html_path: Path, png_path: Path) -> None:
    browser = find_browser()
    if browser is None:
        raise RuntimeError("a Chromium-family browser is needed for the dashboard screenshot")
    w, h = DASHBOARD_WINDOW
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [browser, *HEADLESS_FLAGS, "--hide-scrollbars", f"--user-data-dir={profile}",
             f"--window-size={w},{h}", f"--screenshot={png_path.resolve()}", html_path.resolve().as_uri()],
            check=True, capture_output=True, timeout=120,
            creationflags=0x08000000 if sys.platform == "win32" else 0,  # no console window
        )
    if not png_path.exists():
        raise RuntimeError(f"the browser did not write {png_path.name}")


def run_gallery(out: Path) -> dict[str, dict]:
    """Run every gallery snippet; return each snippet's namespace keyed by the stem it saved."""
    blocks = FENCE.findall(GALLERY.read_text(encoding="utf-8"))
    if not blocks:
        raise RuntimeError(f"no python blocks found in {GALLERY}")
    spaces: dict[str, dict] = {}
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="ag-gallery-") as tmp:
        work = Path(tmp)
        os.chdir(work)
        try:
            for i, code in enumerate(blocks, 1):
                before = set(work.iterdir())
                ns: dict = {"__name__": "__main__"}
                exec(compile(code, f"<gallery block {i}>", "exec"), ns)
                new = sorted(set(work.iterdir()) - before)
                if not new:
                    raise RuntimeError(f"gallery block {i} saved no file")
                stems = {p.stem for p in new}
                for p in new:
                    if p.suffix == ".svg":
                        _png_from_svg(p, out / f"{p.stem}.png")
                    elif p.suffix == ".html":
                        shutil.copyfile(p, out / p.name)
                        if not (work / f"{p.stem}.svg").exists():
                            _screenshot(out / p.name, out / f"{p.stem}.png")
                    else:
                        raise RuntimeError(f"gallery block {i} saved an unexpected file: {p.name}")
                for stem in stems:
                    spaces[stem] = ns
        finally:
            os.chdir(cwd)
    return spaces


def gallery_facts(ns: dict[str, dict]) -> dict:
    """Numbers quoted in the gallery captions, read from the snippets' variables."""
    f: dict = {}
    tree = ns["layout_tree"]["tree"]
    f["tree_depth_from_MSC"] = max(ag.alg.shortest_path_length(tree, "MSC").values())
    f["radial_nodes"] = len(ns["layout_radial"]["tree"])
    les = ns["layout_stress"]
    f["lesmis"] = {"nodes": len(les["les"]), "edges": les["les"].num_edges, "communities": len(les["communities"])}
    f["planted_partition_edges"] = ns["layout_forceatlas2"]["g"].num_edges
    ws = ns["layout_circular"]["g"]
    f["circular_top_closeness"] = [[n, round(v, 3)] for n, v in ag.alg.closeness_centrality(ws).top(5)]
    # rewired shortcuts: links that join nodes more than k/2 = 2 steps apart on the ring, longest first
    ring = len(ws)
    span = {(u, v): min(abs(u - v), ring - abs(u - v)) for u, v in ws.edges}
    f["circular_chords"] = [[u, v, s] for (u, v), s in sorted(span.items(), key=lambda kv: -kv[1]) if s > 2]
    davis = ns["layout_bipartite"]["g"]
    f["davis"] = {"women": len(davis.attrs["top"]), "events": len(davis.attrs["bottom"]), "edges": davis.num_edges}
    courses = ns["edge_orthogonal"]["courses"]
    f["courses"] = {"nodes": len(courses), "edges": courses.num_edges}
    cp = ns["edge_flow"]["cp"]
    f["software_build_critical"] = {"hours": cp.length, "tasks": len(cp.path), "of": len(ns["edge_flow"]["build"])}
    f["curved_reciprocal_pairs"] = ns["edge_curved"]["pairs"]
    flor = ns["theme_light"]["g"]
    f["florentine_top_betweenness"] = [[n, round(v, 3)] for n, v in ag.alg.betweenness_centrality(flor).top(2)]
    ens = ns["chart_line"]["ens"]
    k = int(np.argmax(ens.mean["I"]))
    f["ensemble_peak"] = {"mean_infected": round(float(ens.mean["I"][k]), 1), "time": round(float(ens.times[k]), 2)}
    f["bar_top"] = [[n, round(v, 2)] for n, v in ns["chart_bar"]["top"][:2]]
    mc = ns["chart_histogram"]["mc"]
    f["monte_carlo_hours"] = {"cpm": mc.cpm_length, **{k: round(float(v)) for k, v in mc.percentiles.items()}}
    # the schedulers do not read arc lags; the caption says so and quotes them
    lags = [[u, v, lag] for u, v, lag in ns["chart_histogram"]["plan"].edges.data("lag") if lag]
    f["plan_lags"] = {"arcs": lags, "cpm_plus_lags": mc.cpm_length + sum(lag for *_, lag in lags)}
    run = ns["chart_gantt"]["run"]
    f["gantt"] = {"makespan": run.makespan, "failed_attempts": sum(1 for t in run.tasks if t.status == "failed")}
    sir = ns["sim_sir"]["run"]
    t_peak, n_peak = sir.peak("I")
    f["sir_peak"] = {"time": round(float(t_peak), 2), "infected": int(n_peak)}
    rep = ns["dashboard_lesmis"]["report"]
    f["lesmis_report"] = {
        "diameter": rep.summary.get("diameter"),
        "communities": len(rep.communities or []),
        "modularity": round(float(rep.modularity or 0.0), 3),
    }
    return f


def faq_facts(out: Path) -> dict:
    """Timings and file sizes quoted in the FAQ (they depend on the machine that runs this)."""
    import numpy

    f: dict = {
        "machine": {
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "numpy": numpy.__version__,
        },
        "draw_seconds": {},
        "analyze_seconds": {},
    }
    for n in (1000, 3000, 10000):
        g = ag.gen.barabasi_albert(n, 2, seed=1)
        t0 = time.perf_counter()
        fig = ag.draw(g, labels=False)
        html = fig.to_html()
        f["draw_seconds"][n] = {
            "layout": fig.scene.meta["layout"],
            "seconds": round(time.perf_counter() - t0, 1),
            "html_mb": round(len(html.encode("utf-8")) / 1e6, 1),
        }
    for n in (1000, 5000):
        g = ag.gen.barabasi_albert(n, 2, seed=1)
        t0 = time.perf_counter()
        rep = ag.analyze(g)
        f["analyze_seconds"][n] = {"seconds": round(time.perf_counter() - t0, 1), "notes": rep.notes}
    f["embed_sizes_kb"] = {p.name: round(p.stat().st_size / 1e3) for p in sorted(out.glob("*.html"))}
    return f


def build(out: Path) -> None:
    spaces = run_gallery(out)
    facts = {"gallery": gallery_facts(spaces), "faq": faq_facts(out)}
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
