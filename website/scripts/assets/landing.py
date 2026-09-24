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

"""Figures and interactive embeds for the landing page (source/index.md)."""

from __future__ import annotations

import json
from pathlib import Path

import aryagraph as ag

INDOOR = {"tunnel", "pedway", "attached"}


def campus_figure(**kwargs):
    g = ag.gen.ucalgary_campus()
    route = ag.alg.shortest_path(g, "OO", "SH", weight="length")
    metres = sum(g.edges[u, v]["length"] for u, v in zip(route, route[1:]))
    fig = ag.draw(
        g,
        layout={n: d["pos"] for n, d in g.nodes.data()},
        node_color="kind",
        node_size=ag.by(ag.alg.betweenness_centrality(g, weight="length"), title="betweenness"),
        edge_color=lambda u, v, d: "#52514e" if d["kind"] in INDOOR else "#c3c2b7",
        edge_width=ag.by(lambda u, v, d: 2.0 if d["kind"] in INDOOR else 1.0, kind="identity"),
        highlight_path=route,
        title="University of Calgary main campus",
        subtitle=f"Tunnels and pedways in dark gray · shortest walk Olympic Oval → Scurfield Hall, {metres:,.0f} m",
        **kwargs,
    )
    return fig, route, metres


def build(out: Path) -> None:
    facts: dict[str, object] = {}

    fig, route, metres = campus_figure()
    fig.save(out / "campus.png")
    fig.save(out / "campus.html")
    facts["route"] = route
    facts["route_m"] = round(metres)

    dag = ag.gen.ml_pipeline()
    cp = ag.alg.critical_path(dag)
    ag.draw(
        dag,
        node_color="team",
        highlight_path=cp.path,
        title="Machine-learning pipeline",
        subtitle=f"Critical path highlighted · {cp.length:g} h end to end",
    ).save(out / "pipeline.png")
    facts["pipeline_hours"] = cp.length

    g = ag.gen.watts_strogatz(160, 4, 0.08, seed=2)
    run = ag.sim.SIR(beta=0.45, gamma=0.12).simulate(g, initial={"I": 2}, t_max=45, method="gillespie", seed=5)
    anim = run.animate(labels=False, title="SIR epidemic on a small-world network", static_frame=len(run.times) // 3)
    anim.save(out / "epidemic.png")
    anim.save(out / "epidemic.html")

    les = ag.gen.les_miserables()
    communities = ag.alg.community_labels(ag.alg.louvain_communities(les, seed=0))
    ag.draw(
        les,
        node_color=ag.by(communities, kind="categorical", title="community"),
        node_size=les.degree(weight="weight"),
        edge_width="weight",
        theme="dark",
        title="Les Misérables",
        subtitle="Character co-occurrence · width = scenes shared",
    ).save(out / "lesmis_dark.png")

    plan = ag.gen.project_plan()
    mc = ag.sim.monte_carlo_schedule(plan, runs=2000, seed=1)
    # project_plan() durations are working hours (40 h = one week); the plan's
    # curing and drying lags are not part of this simulation.
    mc.plot(subtitle=f"2,000 PERT simulations · deterministic plan: {mc.cpm_length:g} working hours").save(
        out / "montecarlo.png"
    )
    facts["plan_hours"] = mc.cpm_length
    facts["plan_p50"] = round(mc.percentiles["P50"])
    facts["plan_p95"] = round(mc.percentiles["P95"])

    course = ag.DAG(name="Course prerequisites")
    course.add_edges(
        [
            ("ریاضی ۱", "ریاضی ۲"),
            ("ریاضی ۲", "آمار و احتمال"),
            ("مبانی برنامه‌نویسی", "ساختمان داده"),
            ("ساختمان داده", "طراحی الگوریتم"),
            ("ریاضی گسسته", "طراحی الگوریتم"),
            ("آمار و احتمال", "یادگیری ماشین"),
            ("طراحی الگوریتم", "یادگیری ماشین"),
        ]
    )
    ag.draw(
        course,
        layout_options={"orientation": "RL"},
        title="پیش‌نیازهای درسی",
        subtitle="Persian labels, laid out right to left",
    ).save(out / "persian.png")

    rep = ag.analyze(ag.gen.ucalgary_campus())
    rep.save(out / "campus_report.html")

    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
