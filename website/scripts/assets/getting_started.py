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

"""Figures and interactive embeds for the Getting started section.

The code mirrors the examples on ``getting-started/quickstart.md`` and
``getting-started/concepts.md`` so that every figure shows exactly what the
page's code produces.
"""

from __future__ import annotations

from pathlib import Path

import aryagraph as ag


# --------------------------------------------------------------------------- #
# quickstart
# --------------------------------------------------------------------------- #
def campus_figure() -> tuple[ag.Graph, dict, ag.Figure]:
    """Step 5 of the quickstart: the campus on its real geometry."""
    campus = ag.gen.ucalgary_campus()
    route = ag.alg.shortest_path(campus, "OO", "SH", weight="length")
    bc = ag.alg.betweenness_centrality(campus, weight="length")
    geometry = {n: d["pos"] for n, d in campus.nodes.data()}
    fig = ag.draw(
        campus,
        layout=geometry,
        node_color="kind",
        node_size=ag.by(bc, title="betweenness"),
        highlight_path=route,
        title="University of Calgary main campus",
        subtitle="Shortest walk from the Olympic Oval to Scurfield Hall",
    )
    return campus, geometry, fig


def paper_plan() -> tuple[ag.DAG, object]:
    """Step 8 of the quickstart: a seven-task project plan."""
    paper = ag.DAG(name="Conference paper")
    paper.add_edges(
        [
            ("data", "analysis"),
            ("analysis", "figures"),
            ("analysis", "draft"),
            ("literature", "draft"),
            ("draft", "review"),
            ("figures", "review"),
            ("review", "submit"),
        ]
    )
    days = {"literature": 3, "data": 4, "analysis": 5, "figures": 2, "draft": 6, "review": 2, "submit": 1}
    for task, d in days.items():
        paper.nodes[task]["duration"] = d
    return paper, ag.alg.critical_path(paper)


def quickstart(out: Path) -> None:
    campus, geometry, fig = campus_figure()
    fig.save(out / "campus.png", scale=1.5)
    fig.save(out / "campus.html")

    ag.analyze(campus).save(out / "campus_report.html")

    paper, cp = paper_plan()
    ag.draw(
        paper,
        labels=lambda n, d: f"{n} ({d['duration']} d)",
        highlight_path=cp.path,
        layout_options={"orientation": "LR"},
        title="Conference paper plan",
        subtitle=f"Critical path highlighted: {cp.length:g} days",
    ).save(out / "paper.png")

    model = ag.sim.SIR(beta=0.4, gamma=0.1)
    run = model.simulate(campus, initial={"I": ["MSC"]}, t_max=60, seed=2)
    run.plot(
        title="SIR on the campus network",
        subtitle="Seeded at MacEwan Student Centre · beta 0.4, gamma 0.1, seed 2",
    ).save(out / "sir.png")
    run.animate(layout=geometry, title="SIR on the campus network").save(out / "sir.html")


# --------------------------------------------------------------------------- #
# concepts
# --------------------------------------------------------------------------- #
def concepts(out: Path) -> None:
    """A diagram of how the main objects and functions connect, drawn by AryaGraph."""
    flow = ag.DAG(name="How the pieces fit")
    roles = {
        "object": ["Graph / DiGraph / DAG", "NodeMap / EdgeMap", "Layout", "Scene", "Figure", "SimulationResult"],
        "function": ["ag.alg.*", "ag.layout.compute", "ag.draw", "ag.sim models", "animate()"],
        "output": [".svg", ".html", ".png / .pdf"],
    }
    for role, names in roles.items():
        flow.add_nodes(names, role=role)
    flow.add_edges(
        [
            ("Graph / DiGraph / DAG", "ag.alg.*"),
            ("ag.alg.*", "NodeMap / EdgeMap"),
            ("Graph / DiGraph / DAG", "ag.layout.compute"),
            ("ag.layout.compute", "Layout"),
            ("Graph / DiGraph / DAG", "ag.draw"),
            ("NodeMap / EdgeMap", "ag.draw"),
            ("Layout", "ag.draw"),
            ("ag.draw", "Scene"),
            ("Scene", "Figure"),
            ("Figure", ".svg"),
            ("Figure", ".html"),
            ("Figure", ".png / .pdf"),
            ("Graph / DiGraph / DAG", "ag.sim models"),
            ("ag.sim models", "SimulationResult"),
            ("SimulationResult", "animate()"),
            ("animate()", "Figure"),
        ]
    )
    ag.draw(
        flow,
        node_color=ag.by(
            "role",
            palette={"object": "#2a78d6", "function": "#eb6834", "output": "#1baf7a"},
            domain=["object", "function", "output"],
            counts=False,
            title="role",
        ),
        layout_options={"orientation": "LR"},
        title="How AryaGraph's pieces fit together",
        subtitle="Objects hold data, functions transform them, a Figure writes files",
    ).save(out / "concepts_flow.png")


def build(out: Path) -> None:
    quickstart(out)
    concepts(out)
