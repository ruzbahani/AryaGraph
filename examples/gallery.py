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

"""Build the README gallery: ``python examples/gallery.py`` → docs/gallery/*.png|html.

Each figure is also a self-contained example of the API.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import aryagraph as ag  # noqa: E402

OUT = ROOT / "docs" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str, html: bool = True) -> None:
    fig.save(OUT / f"{name}.png")
    if html:
        fig.save(OUT / f"{name}.html")
    print(f"  {name}: {fig}")


def campus() -> None:
    g = ag.gen.ucalgary_campus()
    route = ag.alg.shortest_path(g, "OO", "SH", weight="length")
    metres = sum(g.edges[u, v]["length"] for u, v in zip(route, route[1:]))
    indoor = {"tunnel", "pedway", "attached"}
    fig = ag.draw(
        g,
        layout={n: d["pos"] for n, d in g.nodes.data()},  # real campus geometry (metres, y down)
        node_color="kind",
        node_size=ag.by(ag.alg.betweenness_centrality(g, weight="length"), title="betweenness"),
        edge_color=lambda u, v, d: "#52514e" if d["kind"] in indoor else "#c3c2b7",
        edge_width=ag.by(lambda u, v, d: 2.0 if d["kind"] in indoor else 1.0, kind="identity"),  # pixels, not data
        highlight_path=route,
        title="University of Calgary main campus",
        subtitle=f"Dark links: tunnels & pedways · highlighted: Olympic Oval → Scurfield Hall, {metres:,.0f} m",
    )
    save(fig, "campus")


def pipeline() -> None:
    dag = ag.gen.ml_pipeline()
    cp = ag.alg.critical_path(dag)
    color = "team" if any("team" in d for _, d in dag.nodes.data()) else None
    fig = ag.draw(
        dag,
        node_color=color,
        highlight_path=cp.path,
        title="Machine-learning pipeline",
        subtitle=f"critical path highlighted · {cp.length:g} h end to end",
    )
    save(fig, "pipeline")


def epidemic() -> None:
    g = ag.gen.watts_strogatz(160, 4, 0.08, seed=2)
    run = ag.sim.SIR(beta=0.45, gamma=0.12).simulate(g, initial={"I": 2}, t_max=45, method="gillespie", seed=5)
    fig = run.animate(labels=False, title="SIR epidemic on a small-world network", static_frame=len(run.times) // 3)
    save(fig, "epidemic")
    run.plot(title="SIR: nodes per state").save(OUT / "epidemic_curve.png")


def lesmis_dark() -> None:
    g = ag.gen.les_miserables()
    communities = ag.alg.community_labels(ag.alg.louvain_communities(g, seed=0))
    fig = ag.draw(
        g,
        node_color=ag.by(communities, kind="categorical", title="community"),
        node_size=g.degree(weight="weight"),
        edge_width="weight",
        theme="dark",
        title="Les Misérables",
        subtitle="character co-occurrence · width = scenes shared",
    )
    save(fig, "lesmis_dark")


def schedule() -> None:
    plan = ag.gen.project_plan()
    sched = ag.sim.simulate_schedule(plan, workers=3, policy="critical_path", failure_rate=0.06, max_retries=2, seed=3)
    sched.gantt(
        lanes="worker",
        title="House construction, simulated with 3 crews",
        subtitle=f"makespan {sched.makespan:g} days · 6% failure risk per attempt, up to 2 retries",
    ).save(OUT / "gantt.png")
    mc = ag.sim.monte_carlo_schedule(plan, runs=2000, seed=1)  # PERT durations from min/mode/max
    mc.plot(subtitle=f"2,000 PERT simulations · deterministic plan says {mc.cpm_length:g} days").save(OUT / "montecarlo.png")
    fig = sched.animate(title="House construction: execution", static_frame="first")
    save(fig, "schedule")


def report() -> None:
    rep = ag.analyze(ag.gen.les_miserables())
    path = rep.save(OUT / "report.html")
    from aryagraph.render.export import find_browser, _run_browser

    browser = find_browser()
    if browser:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _run_browser(
                [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={tmp}",
                 "--window-size=1200,1500", f"--screenshot={OUT / 'report.png'}", path.resolve().as_uri()]
            )
    print(f"  report: {path}")


def layouts() -> None:
    g = ag.gen.barabasi_albert(60, 2, seed=4)
    for method in ("stress", "forceatlas2", "circular", "spectral"):
        fig = ag.draw(g, layout=method, labels=False, title=method, node_size=g.degree(), legend=False)
        save(fig, f"layout_{method}", html=False)
    tree = ag.gen.balanced_tree(2, 4)
    save(ag.draw(tree, layout="tree", title="tree"), "layout_tree", html=False)
    wide_tree = ag.gen.balanced_tree(3, 4)  # 121 nodes, 5 levels: what radial layouts are for
    save(ag.draw(wide_tree, layout="radial", labels=False, title="radial", legend=False), "layout_radial", html=False)


def persian() -> None:
    dag = ag.DAG(name="برنامه‌ی درسی")
    dag.add_edges(
        [
            ("ریاضی ۱", "ریاضی ۲"),
            ("ریاضی ۲", "آمار و احتمال"),
            ("مبانی برنامه‌نویسی", "ساختمان داده"),
            ("ساختمان داده", "طراحی الگوریتم"),
            ("ریاضی گسسته", "طراحی الگوریتم"),
            ("آمار و احتمال", "یادگیری ماشین"),
            ("طراحی الگوریتم", "یادگیری ماشین"),
            ("ساختمان داده", "پایگاه داده"),
        ]
    )
    fig = ag.draw(dag, title="پیش‌نیازهای درسی", subtitle="چیدمان لایه‌ای با برچسب‌های فارسی")
    save(fig, "persian")


if __name__ == "__main__":
    t0 = time.perf_counter()
    for fn in (campus, pipeline, epidemic, lesmis_dark, schedule, report, layouts, persian):
        print(fn.__name__)
        fn()
    print(f"done in {time.perf_counter() - t0:.1f} s → {OUT}")
