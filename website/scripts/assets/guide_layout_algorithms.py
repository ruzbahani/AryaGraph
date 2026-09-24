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

"""Figures for the user-guide pages on layouts, algorithms, analysis and charts.

Every figure is built with the same calls and seeds as the code shown on
``user-guide/layouts.md``, ``algorithms.md``, ``analysis.md`` and ``charts.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import aryagraph as ag

OSM = "Positions © OpenStreetMap contributors (ODbL)"


# --------------------------------------------------------------------------- #
# layouts.md
# --------------------------------------------------------------------------- #
def _layouts(out: Path) -> None:
    # one graph, six engines: four planted groups of 16 nodes
    g = ag.gen.planted_partition(4, 16, 0.35, 0.02, seed=3)
    for method in ("stress", "forceatlas2", "fruchterman_reingold", "spectral", "circular", "shell"):
        options = {"order": "auto"} if method == "circular" else None
        fig = ag.draw(
            g,
            layout=method,
            layout_options=options,
            node_color=ag.by("block", kind="categorical"),
            labels=False,
            legend=False,
            title=method,
            width=400,
        )
        fig.save(out / f"compare_{method}.png")

    # hierarchical: layering, orientation and fixed ranks
    etl = ag.gen.data_warehouse_etl()
    for method in ("network_simplex", "coffman_graham"):
        ag.draw(
            etl,
            layout="hierarchical",
            layout_options={"layering": method},
            node_color="kind",
            labels=False,
            title="Data-warehouse ETL",
            subtitle=f'layering="{method}"',
        ).save(out / f"etl_{method}.png")

    courses = ag.gen.course_prerequisites()
    ag.draw(
        courses,
        layout="hierarchical",
        layout_options={"orientation": "LR"},
        node_color="department",
        title="Course prerequisites",
        subtitle="orientation LR",
    ).save(out / "courses_lr.png")
    ranks = {c: d["level"] // 100 - 1 for c, d in courses.nodes.data()}
    ag.draw(
        courses,
        layout="hierarchical",
        layout_options={"ranks": ranks},
        node_color="department",
        title="Course prerequisites",
        subtitle="one layer per course level (100 to 400)",
    ).save(out / "courses_ranks.png")

    # trees: breadth-first tree of the campus from the student centre; a wide tree
    campus = ag.gen.ucalgary_campus()
    bfs = ag.alg.bfs_tree(campus, "MSC")
    ag.draw(
        bfs,
        layout="tree",
        layout_options={"orientation": "LR", "node_sep": 12, "rank_sep": 28},
        title="Breadth-first tree of the campus from MSC",
    ).save(out / "campus_bfs_tree.png")
    wide = ag.gen.balanced_tree(3, 4)
    ag.draw(wide, layout="radial", labels=False, legend=False, title="radial").save(out / "radial.png")

    # stress with edge lengths against the real geometry
    pos = {n: d["pos"] for n, d in campus.nodes.data()}
    ag.draw(campus, layout=pos, node_color="kind", title="Real positions", subtitle=OSM, width=760).save(
        out / "campus_positions.png"
    )
    ag.draw(
        campus,
        layout="stress",
        layout_options={"weight": "length"},
        node_color="kind",
        title="Stress layout",
        subtitle='weight="length"',
        width=760,
    ).save(out / "campus_stress.png")

    # ForceAtlas2 options
    lesmis = ag.gen.les_miserables()
    for name, options in (("default", {}), ("lin_log", {"lin_log": True})):
        ag.draw(
            lesmis,
            layout="forceatlas2",
            layout_options=options,
            node_size=lesmis.degree(),
            labels=False,
            legend=False,
            title="ForceAtlas2",
            subtitle="default options" if not options else "lin_log=True",
            width=520,
            height=520,
        ).save(out / f"lesmis_fa2_{name}.png")

    # circular order
    tree = ag.gen.random_tree(30, seed=2)
    for order in (None, "auto"):
        lay = ag.layout.circular(tree, order=order)
        ag.draw(tree, layout=lay, labels=False, width=380, height=380).save(out / f"tree_circular_{order or 'graph'}.png")

    # bipartite
    davis = ag.gen.davis_southern_women()
    women = [n for n, d in davis.nodes.data() if d["bipartite"] == 0]
    ag.draw(
        davis,
        layout="bipartite",
        layout_options={"top": women},
        node_color="bipartite",
        legend=False,
        title="Davis Southern Women",
        subtitle="bipartite layout: 18 women (left), 14 events (right)",
    ).save(out / "davis_bipartite.png")

    # disconnected graph: components laid out one by one and packed
    indoor = ag.gen.ucalgary_campus(indoor_only=True)
    ag.draw(
        indoor,
        layout="stress",
        node_color="kind",
        title="Indoor-only campus network",
        subtitle="19 components, packed",
        width=760,
    ).save(out / "indoor_packed.png")


# --------------------------------------------------------------------------- #
# algorithms.md
# --------------------------------------------------------------------------- #
def _algorithms(out: Path) -> None:
    campus = ag.gen.ucalgary_campus()
    pos = {n: d["pos"] for n, d in campus.nodes.data()}

    def indoors(u, v, d):
        return None if d["kind"] == "outdoor" else d["length"]

    inside = ag.alg.shortest_path(campus, "MSC", "TFDL", weight=indoors)
    meters = sum(campus.edges[u, v]["length"] for u, v in zip(inside, inside[1:]))
    ag.draw(
        campus,
        layout=pos,
        node_color="kind",
        highlight_path=inside,
        title="Indoor route from MSC to TFDL",
        subtitle=f"{meters:,.1f} m through tunnels, pedways and attached buildings · {OSM}",
        width=760,
    ).save(out / "campus_indoor_route.png")

    bc = ag.alg.betweenness_centrality(campus, weight="length")
    ag.draw(
        campus,
        layout=pos,
        node_size=ag.by(bc, title="betweenness"),
        node_color=ag.by(bc, title="betweenness"),
        title="Betweenness by link length",
        subtitle=OSM,
        width=760,
    ).save(out / "campus_betweenness.png")

    cut = ag.alg.minimum_cut(campus, "OO", "SH", capacity=lambda u, v, d: 1)
    side = {n: ("Oval side" if n in cut.source_side else "rest of campus") for n in campus.nodes}
    cut_edges = {frozenset(e) for e in cut.cut_edges}
    ag.draw(
        campus,
        layout=pos,
        node_color=ag.by(side, kind="categorical", title="side of the cut"),
        edge_color=lambda u, v, d: "#d03b3b" if frozenset((u, v)) in cut_edges else "#c3c2b7",
        edge_width=ag.by(lambda u, v, d: 3.0 if frozenset((u, v)) in cut_edges else 1.0, kind="identity"),
        title="Minimum cut between OO and SH",
        subtitle=f"{cut.value} links: {', '.join(f'{u}–{v}' for u, v in cut.cut_edges)} · {OSM}",
        width=760,
    ).save(out / "campus_mincut.png")

    mst = ag.alg.minimum_spanning_tree(campus, weight="length")
    total = sum(d["length"] for _, _, d in mst.edges.data())
    ag.draw(
        mst,
        layout=pos,
        node_color="kind",
        title="Campus minimum spanning tree",
        subtitle=f"{mst.num_edges} links, {total:,.1f} m · {OSM}",
        width=760,
    ).save(out / "campus_mst.png")

    lesmis = ag.gen.les_miserables()
    label = ag.alg.community_labels(ag.alg.louvain_communities(lesmis, seed=0))
    fig = ag.draw(
        lesmis,
        node_color=ag.by(label, kind="categorical", title="community"),
        node_size=lesmis.degree(weight="weight"),
        edge_width="weight",
        title="Les Misérables: Louvain communities",
        subtitle="node size: weighted degree · edge width: chapters shared",
    )
    fig.save(out / "lesmis_communities.html")


# --------------------------------------------------------------------------- #
# analysis.md
# --------------------------------------------------------------------------- #
def _analysis(out: Path) -> None:
    ag.analyze(ag.gen.les_miserables()).save(out / "lesmis_report.html")
    ag.analyze(ag.gen.project_plan()).save(out / "plan_report.html")


# --------------------------------------------------------------------------- #
# charts.md
# --------------------------------------------------------------------------- #
def _charts(out: Path) -> None:
    # line chart: the small-world effect
    ps = np.logspace(-3, 0, 13)
    lattice = ag.gen.watts_strogatz(200, 6, 0.0, seed=0)
    c0 = ag.alg.average_clustering(lattice)
    l0 = ag.alg.average_shortest_path_length(lattice)
    clust, path = [], []
    for p in ps:
        graphs = [ag.gen.watts_strogatz(200, 6, p, seed=s) for s in range(5)]
        clust.append([ag.alg.average_clustering(g) / c0 for g in graphs])
        path.append([ag.alg.average_shortest_path_length(g) / l0 for g in graphs])
    clust, path = np.array(clust), np.array(path)
    ag.charts.line_chart(
        np.log10(ps),
        {"clustering C(p) / C(0)": clust.mean(axis=1), "path length L(p) / L(0)": path.mean(axis=1)},
        bands={
            "clustering C(p) / C(0)": (clust.min(axis=1), clust.max(axis=1)),
            "path length L(p) / L(0)": (path.min(axis=1), path.max(axis=1)),
        },
        x_format=lambda v: f"{10 ** v:g}",
        x_label="rewiring probability p (log scale)",
        title="The small-world effect",
        subtitle="Watts–Strogatz graphs, n = 200, k = 6 · mean of 5 seeds, band = range",
        y_min=0,
    ).save(out / "small_world.png")

    # bar chart: betweenness on campus
    campus = ag.gen.ucalgary_campus()
    bc = ag.alg.betweenness_centrality(campus, weight="length")
    top = bc.top(10)
    ag.charts.bar_chart(
        [campus.nodes[n]["name"] for n, _ in top],
        [v for _, v in top],
        highlight=[campus.nodes["PF"]["name"], campus.nodes["KNB"]["name"]],
        value_format=lambda v: f"{v:.2f}",
        title="Buildings on the most shortest routes",
        subtitle="Betweenness centrality by link length, top 10 of 56",
        max_label_width=260,
        width=640,
    ).save(out / "betweenness_bars.png")
    ag.charts.bar_chart(
        [n for n, _ in top],
        [v for _, v in top],
        theme="dark",
        width=420,
        value_format=lambda v: f"{v:.2f}",
        title="Top 10 by betweenness",
    ).save(out / "betweenness_dark.png")

    # histogram: shortest-path lengths between all pairs of buildings
    dist = ag.alg.all_pairs_shortest_path_length(campus, weight="length")
    lengths = [d for u, row in dist.items() for v, d in row.items() if u < v]
    median = float(np.median(lengths))
    ag.charts.histogram(
        lengths,
        bins=np.arange(0, 2251, 125),
        markers=[(median, f"median {median:,.0f} m")],
        title="How far apart are campus buildings?",
        subtitle="Shortest-path length through the network for every pair of buildings",
        x_label="meters",
        y_label="pairs",
    ).save(out / "walk_histogram.png")

    # Gantt charts: CPM plan and a simulated schedule
    plan = ag.gen.project_plan()
    cp = ag.alg.critical_path(plan)
    rows = [
        {"node": task, "start": cp.earliest_start[task], "end": cp.earliest_finish[task]}
        for task in ag.alg.topological_sort(plan)
    ]
    ag.charts.gantt(
        rows,
        critical=cp.path,
        title="House construction: earliest-start schedule",
        subtitle=f"critical path {cp.length:g} h, unlimited crews",
        time_label="working hours",
    ).save(out / "plan_gantt.png")
    sched = ag.sim.simulate_schedule(plan, workers=3, policy="critical_path", seed=3)
    sched.gantt(lanes="worker", title="House construction with 3 crews", time_label="working hours").save(
        out / "crews_gantt.png"
    )

    # simulation plots
    g = ag.gen.watts_strogatz(300, 6, 0.08, seed=1)
    sir = ag.sim.SIR(beta=0.35, gamma=0.1)
    run = sir.simulate(g, initial={"I": 3}, t_max=60, method="gillespie", seed=7)
    run.plot(title="SIR epidemic, one run").save(out / "sir_run.png")
    ensemble = ag.sim.run_ensemble(sir.simulate, runs=100, seed=1, g=g, initial={"I": 3}, t_max=60, method="gillespie")
    chart = ensemble.plot(title="SIR epidemic, 100 runs")
    chart.save(out / "sir_ensemble.png")
    chart.save(out / "sir_ensemble.html")
    mc = ag.sim.monte_carlo_schedule(plan, runs=2000, seed=1)
    note = f"2,000 PERT simulations, makespan in working hours · deterministic plan: {mc.cpm_length:g} h"
    mc.plot(subtitle=note).save(out / "montecarlo.png")


def build(out: Path) -> None:
    _layouts(out)
    _algorithms(out)
    _analysis(out)
    _charts(out)
