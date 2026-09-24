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

"""Figures and interactive embeds for the first four tutorials.

Pages: tutorials/first-network.md, tutorials/campus-routing.md,
tutorials/pipeline-dag.md and tutorials/publication-figures.md. Each figure
repeats the drawing code shown on its page, so what readers run is what they
see. Randomized layouts use fixed seeds (``ag.draw`` defaults to ``seed=0``).
"""

from __future__ import annotations

from pathlib import Path

import aryagraph as ag


# --------------------------------------------------------------------------- #
# tutorials/first-network.md
# --------------------------------------------------------------------------- #
def science_core() -> ag.Graph:
    """The nine-building network the first tutorial builds by hand."""
    g = ag.Graph(name="Science core")
    g.add_node("MSC", name="MacEwan Student Centre", kind="student-life")
    g.add_node("MH", name="MacEwan Hall", kind="student-life")
    g.add_node("TI", name="Taylor Institute for Teaching and Learning", kind="academic")
    g.add_node("SB", name="Science B", kind="academic")
    g.add_node("SA", name="Science A", kind="academic")
    g.add_node("ES", name="Earth Sciences", kind="academic")
    g.add_node("MS", name="Mathematical Sciences", kind="academic")
    g.add_node("ST", name="Science Theatres", kind="academic")
    g.add_node("SS", name="Social Sciences", kind="academic")
    g.add_edges(
        [
            ("MSC", "MH", {"kind": "attached", "length": 97.4}),
            ("MSC", "TI", {"kind": "outdoor", "length": 108.1}),
            ("MH", "TI", {"kind": "outdoor", "length": 83.4}),
            ("MH", "SB", {"kind": "tunnel", "length": 121.2}),
            ("SB", "ES", {"kind": "attached", "length": 86.5}),
            ("SB", "SA", {"kind": "attached", "length": 92.7}),
            ("ES", "MS", {"kind": "pedway", "length": 90.5}),
            ("MS", "ST", {"kind": "attached", "length": 58.0}),
            ("SA", "ST", {"kind": "attached", "length": 88.7}),
            ("SA", "SS", {"kind": "attached", "length": 82.4}),
            ("ST", "SS", {"kind": "attached", "length": 58.8}),
        ]
    )
    return g


def first_network(out: Path) -> None:
    g = science_core()
    campus = ag.gen.ucalgary_campus()
    for u, v, d in g.edges.data():  # the hand-typed edges must match the dataset
        assert campus.edges[u, v] == {**d, "weight": d["length"]}, (u, v)

    bc = ag.alg.betweenness_centrality(g, weight="length")
    ag.draw(
        g,
        node_color="kind",
        node_size=ag.by(bc, title="betweenness"),
        edge_label="length",
        label_position="below",
        title="Nine buildings in the science core",
        subtitle="Node size: betweenness by length · edge labels: meters",
    ).save(out / "first_core.png")

    campus_bc = ag.alg.betweenness_centrality(campus, weight="length")
    pos = {n: d["pos"] for n, d in campus.nodes.data()}
    fig = ag.draw(
        campus,
        layout=pos,
        node_color="kind",
        node_size=ag.by(campus_bc, title="betweenness (by length)"),
        title="University of Calgary main campus",
        subtitle="56 buildings and 83 links · positions © OpenStreetMap contributors",
    )
    fig.save(out / "first_campus.png")
    fig.save(out / "first_campus.html")


# --------------------------------------------------------------------------- #
# tutorials/campus-routing.md
# --------------------------------------------------------------------------- #
def campus_routing(out: Path) -> None:
    campus = ag.gen.ucalgary_campus()
    indoor = ag.gen.ucalgary_campus(indoor_only=True)
    route = ag.alg.shortest_path(campus, "OO", "SH", weight="length")
    inside = ag.alg.shortest_path(indoor, "OO", "SH", weight="length")

    # Step 4: both routes on the map
    shortest_links = {frozenset(e) for e in zip(route, route[1:])}
    indoor_links = {frozenset(e) for e in zip(inside, inside[1:])}

    def which_route(u, v):
        e = frozenset((u, v))
        if e in shortest_links and e in indoor_links:
            return "both routes"
        if e in shortest_links:
            return "shortest route"
        if e in indoor_links:
            return "indoor route"
        return "other links"

    on_a_route = set(route) | set(inside)

    def building_group(n):
        return "on a route" if n in on_a_route else "elsewhere"

    def link_width(u, v):
        return 1.0 if which_route(u, v) == "other links" else 3.5

    building_colors = {"on a route": "#52514e", "elsewhere": "#c3c2b7"}
    link_colors = {
        "shortest route": "#2a78d6",
        "indoor route": "#eb6834",
        "both routes": "#4a3aa7",
        "other links": "#d6d5ce",
    }
    fig = ag.draw(
        campus,
        layout={n: d["pos"] for n, d in campus.nodes.data()},
        node_color=ag.by(building_group, title="building", domain=list(building_colors), palette=building_colors),
        labels={n: n for n in on_a_route},
        edge_color=ag.by(which_route, title="link", domain=list(link_colors), palette=link_colors),
        edge_width=ag.by(link_width, kind="identity", legend=False),
        title="Olympic Oval to Scurfield Hall",
        subtitle="Shortest route 1,002 m (266 m outdoors) · indoor route 1,132 m",
    )
    fig.save(out / "routing_routes.png")
    fig.save(out / "routing_routes.html")

    # Step 7: indoor distance from MSC
    from_msc = ag.alg.shortest_path_length(indoor, "MSC", weight="length")
    ag.draw(
        indoor,
        layout={n: d["pos"] for n, d in indoor.nodes.data()},
        node_color=ag.by(from_msc, title="meters from MSC, indoors"),
        title="How far can you walk indoors from MacEwan Student Centre?",
        subtitle="Tunnels, pedways and attached buildings only · gray: no indoor connection",
    ).save(out / "routing_indoor_reach.png")

    # Step 9: chokepoints
    bc = ag.alg.betweenness_centrality(campus, weight="length")
    cut_nodes = ag.alg.articulation_points(campus)
    role_colors = {"articulation point": "#e34948", "other building": "#2a78d6"}

    def role(n):
        return "articulation point" if n in cut_nodes else "other building"

    ag.draw(
        campus,
        layout={n: d["pos"] for n, d in campus.nodes.data()},
        node_size=ag.by(bc, title="betweenness (by length)"),
        node_color=ag.by(role, title="building", palette=role_colors),
        labels={n: n for n in cut_nodes + [b for b, _ in bc.top(5)]},
        label_position="below",
        title="Where campus routes concentrate",
        subtitle="Size: betweenness by length · red: articulation points",
    ).save(out / "routing_chokepoints.png")


# --------------------------------------------------------------------------- #
# tutorials/pipeline-dag.md
# --------------------------------------------------------------------------- #
def nightly_report() -> ag.DAG:
    """The hand-built job of Step 1, after removing its redundant edge (Step 3)."""
    job = ag.DAG(name="Nightly sales report")
    job.add_node("download sales", duration=0.5)
    job.add_node("download inventory", duration=0.25)
    job.add_node("clean sales", duration=1.0)
    job.add_node("join tables", duration=0.5)
    job.add_node("compute KPIs", duration=0.75)
    job.add_node("forecast demand", duration=2.0)
    job.add_node("render dashboard", duration=0.25)
    job.add_node("email summary", duration=0.1)
    job.add_edges(
        [
            ("download sales", "clean sales"),
            ("clean sales", "join tables"),
            ("download inventory", "join tables"),
            ("join tables", "compute KPIs"),
            ("join tables", "forecast demand"),
            ("compute KPIs", "render dashboard"),
            ("forecast demand", "render dashboard"),
            ("join tables", "render dashboard"),
            ("render dashboard", "email summary"),
        ]
    )
    assert sorted(set(job.edges) - set(job.transitive_reduction().edges)) == [("join tables", "render dashboard")]
    job.remove_edge("join tables", "render dashboard")
    return job


def pipeline_dag(out: Path) -> None:
    job = nightly_report()
    cp_job = job.critical_path()
    ag.draw(
        job,
        highlight_path=cp_job.path,
        title="Nightly sales report",
        subtitle=f"Critical path: {cp_job.length:g} h",
    ).save(out / "pipeline_job.png")

    dag = ag.gen.ml_pipeline()
    cp = ag.alg.critical_path(dag)
    fig = ag.draw(
        dag,
        node_color="team",
        highlight_path=cp.path,
        title="ML pipeline",
        subtitle=f"Critical path highlighted · {cp.length:g} h end to end",
    )
    fig.save(out / "pipeline_critical.png")
    fig.save(out / "pipeline_critical.html")

    two = ag.sim.simulate_schedule(dag, workers=2)
    two.gantt(
        lanes="worker",
        title="ML pipeline on 2 workers",
        subtitle=f"Makespan {two.makespan:g} h · utilization {two.utilization['overall']:.0%}",
        time_label="hours",
    ).save(out / "pipeline_gantt_2.png")
    four = ag.sim.simulate_schedule(dag, workers=4)
    four.gantt(
        lanes="worker",
        title="ML pipeline on 4 workers",
        subtitle=f"Makespan {four.makespan:g} h · utilization {four.utilization['overall']:.0%}",
        time_label="hours",
    ).save(out / "pipeline_gantt_4.png")
    two.gantt(lanes="task", color_by="team", title="ML pipeline by task", time_label="hours").save(
        out / "pipeline_gantt_tasks.png"
    )

    hours = list(range(17))
    lengths = []
    for h in hours:
        variant = dag.copy()
        variant.nodes["tune gradient boosting"]["duration"] = h
        lengths.append(ag.alg.critical_path(variant).length)
    ag.charts.line_chart(
        hours,
        {"pipeline length": lengths},
        title="What if tuning took longer or shorter?",
        subtitle="Critical-path length of the ML pipeline",
        x_label="hours spent tuning gradient boosting",
        y_label="hours",
        y_min=45,
        markers=[(12, "current plan")],
    ).save(out / "pipeline_whatif.png")


# --------------------------------------------------------------------------- #
# tutorials/publication-figures.md
# --------------------------------------------------------------------------- #
def publication_figures(out: Path) -> None:
    les = ag.gen.les_miserables()
    communities = ag.alg.louvain_communities(les, seed=0)
    strength = les.degree(weight="weight")
    group, leaders = {}, []
    for members in communities:
        leader = max(sorted(members), key=strength.__getitem__)
        leaders.append(leader)
        for m in members:
            group[m] = leader

    for method in ("stress", "forceatlas2", "circular"):
        ag.draw(
            les,
            layout=method,
            seed=0,
            node_color=ag.by(group, domain=leaders),
            node_size=strength,
            labels=False,
            legend=False,
            theme="paper",
            title=method,
        ).save(out / f"pub_layout_{method}.png")

    top12 = [name for name, _ in strength.top(12)]
    fig = ag.draw(
        les,
        node_color=ag.by(group, title="community", domain=leaders),
        node_size=ag.by(strength, title="chapters shared"),
        edge_width=ag.by("weight", title="chapters together"),
        labels={name: name for name in top12},
        theme="paper",
        width=672,
    )
    hidden = [m.node for m in fig.scene.nodes if m.label is not None and not m.label.visible]
    assert hidden == ["Combeferre", "Bossuet"], hidden  # the page text quotes this
    fig.save(out / "pub_lesmis.png")
    fig.save(out / "pub_lesmis_300dpi.png", scale=300 / 96)
    fig.save(out / "pub_lesmis.pdf")
    fig.save(out / "pub_lesmis.svg")


# --------------------------------------------------------------------------- #
def build(out: Path) -> None:
    first_network(out)
    campus_routing(out)
    pipeline_dag(out)
    publication_figures(out)
