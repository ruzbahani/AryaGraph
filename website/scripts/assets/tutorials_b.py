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

"""Figures and interactive embeds for the second group of tutorials.

Pages: source/tutorials/epidemics.md, communities.md, influence.md and
interop.md. Every model, seed and parameter below matches the code shown on
those pages, so the figures show the same runs the reader reproduces.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import aryagraph as ag


# ---------------------------------------------------------------------- #
# epidemics.md
# ---------------------------------------------------------------------- #
def epidemics(out: Path, facts: dict) -> None:
    g = ag.gen.watts_strogatz(400, 6, 0.1, seed=3)
    sir = ag.sim.SIR(beta=0.08, gamma=0.1)
    exact = sir.simulate(g, initial={"I": 3}, method="gillespie", seed=7)
    exact.plot(title="SIR on a small-world network", subtitle="One Gillespie run, seed 7").save(out / "sir_counts.png")
    facts["sir_peak"] = exact.peak("I")

    ens = ag.sim.run_ensemble(sir.simulate, runs=200, g=g, initial={"I": 3}, method="gillespie", seed=11)
    ens.plot(title="SIR: 200 Gillespie runs", subtitle="Mean with 25–75% and 5–95% bands").save(out / "sir_ensemble.png")
    final = ens.final_sizes["R"]
    ag.charts.histogram(
        final, bins=20, x_label="nodes ever infected", y_label="runs", title="Final epidemic size",
        subtitle="200 Gillespie runs of the same SIR model",
    ).save(out / "sir_final_sizes.png")
    facts["sir_minor"] = int((final < 40).sum())

    player = exact.animate(
        layout="circular",
        avoid_overlap=False,
        labels=False,
        node_size=6,
        width=640,
        height=640,
        title="SIR on a small-world network",
        subtitle="Nodes sit on the ring; the chords are the rewired shortcuts",
    )
    player.save(out / "sir_player.html")

    seir = ag.sim.SEIR(beta=0.08, sigma=0.2, gamma=0.1)
    ens_seir = ag.sim.run_ensemble(seir.simulate, runs=200, g=g, initial={"I": 3}, method="gillespie", seed=11)
    grid = np.linspace(0, 150, 151)
    curves = {
        "SIR, infected": np.interp(grid, ens.times, ens.mean["I"]),
        "SEIR, infected": np.interp(grid, ens_seir.times, ens_seir.mean["I"]),
        "SEIR, exposed": np.interp(grid, ens_seir.times, ens_seir.mean["E"]),
    }
    ag.charts.line_chart(
        grid, curves, title="Adding a latent period", subtitle="Same beta and gamma; SEIR adds a mean latent period of 5",
        x_label="time", y_label="nodes (mean of 200 runs)",
    ).save(out / "sir_vs_seir.png")

    siqr = ag.sim.CompartmentalModel(
        ["S", "I", "Q", "R"],
        spontaneous=[("I", "R", 0.1), ("I", "Q", 0.05), ("Q", "R", 0.1)],
        induced=[("S", "I", "I", 0.08)],
        name="SIQR",
        roles={"Q": "serious"},
    )
    ens_q = ag.sim.run_ensemble(siqr.simulate, runs=200, g=g, initial={"I": 3}, method="gillespie", seed=11)
    ens_q.plot(title="SIQR: isolating infected nodes", subtitle="200 Gillespie runs").save(out / "siqr_ensemble.png")
    facts["siqr_final_mean"] = float(ens_q.final_sizes["R"].mean())


# ---------------------------------------------------------------------- #
# communities.md
# ---------------------------------------------------------------------- #
def communities(out: Path, facts: dict) -> None:
    les = ag.gen.les_miserables()
    strength = les.degree(weight="weight")
    partitions = {
        "louvain": ("Louvain", ag.alg.louvain_communities(les, seed=0)),
        "greedy": ("greedy modularity", ag.alg.greedy_modularity_communities(les, weight="weight")),
        "lpa": ("label propagation", ag.alg.label_propagation_communities(les, weight="weight", seed=0)),
    }
    for key, (name, parts) in partitions.items():
        labels = ag.alg.community_labels(parts)
        q = ag.alg.modularity(les, parts)
        fig = ag.draw(
            les,
            node_color=ag.by(labels, kind="categorical", title="community"),
            node_size=ag.by(strength, title="co-appearances"),
            edge_width=ag.by("weight", title="chapters shared"),
            title=f"Les Misérables: {name}",
            subtitle=f"{len(parts)} communities, modularity {q:.3f} · node size = co-appearances",
        )
        fig.save(out / f"lesmis_{key}.png")
        facts[f"lesmis_{key}"] = {"communities": len(parts), "modularity": round(q, 4)}

    louvain = partitions["louvain"][1]
    labels = ag.alg.community_labels(louvain)
    bc = ag.alg.betweenness_centrality(les)
    dark = ag.draw(
        les,
        node_color=ag.by(labels, kind="categorical", title="community"),
        node_size=ag.by(bc, title="betweenness"),
        edge_width=ag.by("weight", title="chapters shared"),
        theme="dark",
        title="Les Misérables",
        subtitle="Louvain communities · node size = betweenness centrality",
    )
    dark.save(out / "lesmis_dark.png")
    dark.save(out / "lesmis_dark.html")


# ---------------------------------------------------------------------- #
# influence.md
# ---------------------------------------------------------------------- #
def _activation(les, seeds, runs: int = 2000) -> dict:
    e = ag.sim.run_ensemble(ag.sim.independent_cascade, runs=runs, g=les, seeds=seeds, p=0.15, seed=3, keep=True)
    share = np.mean([r.values[-1] for r in e.results], axis=0)
    return dict(zip(e.results[0].nodes, share.tolist()))


def _reach_figure(les, seeds, reach: dict, title: str, subtitle: str):
    role = {v: ("seed" if v in seeds else "other") for v in les}
    return ag.draw(
        les,
        node_color=ag.by(reach, kind="sequential", domain=(0, 1), title="P(active)"),
        node_shape=ag.by(role, title="role"),
        labels={v: v for v in seeds},
        label_collisions="show",
        title=title,
        subtitle=subtitle,
    )


def influence(out: Path, facts: dict) -> None:
    les = ag.gen.les_miserables()
    best = ag.sim.greedy_influence_maximization(les, k=5, model="ic", p=0.15, runs=1000, seed=1)
    by_degree = [v for v, _ in les.degree().top(5)]
    for key, name, seeds in [("greedy", "Greedy (CELF) seeds", best.seeds), ("degree", "Top-degree seeds", by_degree)]:
        est = ag.sim.influence_spread(les, seeds, model="ic", p=0.15, runs=10_000, seed=2)
        reach = _activation(les, seeds)
        fig = _reach_figure(
            les, seeds, reach, f"{name}: {', '.join(seeds)}",
            f"Independent cascade, p = 0.15 · expected spread {est.mean:.1f} ± {est.stderr:.2f} characters",
        )
        fig.save(out / f"reach_{key}.png")
        fig.save(out / f"reach_{key}.html")
        facts[f"reach_{key}"] = {"seeds": seeds, "spread": round(est.mean, 2), "stderr": round(est.stderr, 3)}

    greedy8 = ag.sim.greedy_influence_maximization(les, k=8, model="ic", p=0.15, runs=1000, seed=1)
    degree8 = [v for v, _ in les.degree().top(8)]
    ks = list(range(1, 9))
    curve: dict[str, list[float]] = {"greedy (CELF)": [], "top degree": []}
    for k in ks:
        curve["greedy (CELF)"].append(ag.sim.influence_spread(les, greedy8.seeds[:k], p=0.15, runs=5000, seed=2).mean)
        curve["top degree"].append(ag.sim.influence_spread(les, degree8[:k], p=0.15, runs=5000, seed=2).mean)
    ag.charts.line_chart(
        ks, curve, title="Expected spread by seed-set size",
        subtitle="Independent cascade on Les Misérables, p = 0.15 · 5,000 runs per point",
        x_label="seeds", y_label="expected active characters",
    ).save(out / "spread_by_k.png")
    facts["spread_by_k"] = {name: [round(v, 2) for v in vals] for name, vals in curve.items()}

    run = ag.sim.independent_cascade(les, ["Valjean"], p=0.15, seed=1)
    run.animate(title="Independent cascade from Valjean", time_format="step {t}").save(out / "cascade_player.html")


# ---------------------------------------------------------------------- #
# interop.md
# ---------------------------------------------------------------------- #
MERMAID_FLOW = """
flowchart LR
    raw[Raw events] --> clean[Clean] --> features[Features]
    labels[Labels] --> join[Join] --> features
    clean --> join
    features --> train[Train model] --> evaluate[Evaluate]
    features --> baseline[Baseline] --> evaluate
    evaluate -->|approved| deploy[Deploy]
"""


def interop(out: Path, facts: dict) -> None:
    plan = ag.DAG(ag.io.from_mermaid(MERMAID_FLOW))
    ag.draw(
        plan,
        labels="label",
        layout_options={"orientation": plan.attrs["direction"]},
        title="Pipeline read from Mermaid",
    ).save(out / "mermaid_flow.png")

    # The pipeline reads the same two tables the page writes to CSV.
    nodes_df, edges_df = ag.io.to_pandas(ag.gen.ucalgary_campus())
    buildings = nodes_df.drop(columns="pos").rename(columns={"id": "code"})
    links = edges_df.drop(columns="weight")
    indoor_links = links[links["kind"].isin(["tunnel", "pedway", "attached"])]
    indoor = ag.io.from_pandas(buildings, indoor_links, node_id="code")
    parts = ag.alg.connected_components(indoor)
    main = max(parts, key=len)
    core = indoor.subgraph(main)
    bc_indoor = ag.alg.betweenness_centrality(core, weight="length")
    lat0, lon0 = buildings["lat"].mean(), buildings["lon"].mean()
    pos = {
        row.code: ((row.lon - lon0) * 111_320 * math.cos(math.radians(lat0)), (lat0 - row.lat) * 110_574)
        for row in buildings.itertuples()
    }
    status = {v: ("indoor network" if v in main else "no indoor link") for v in indoor}
    fig = ag.draw(
        indoor,
        layout=pos,
        node_color=ag.by(status, title="building", palette={"indoor network": "#2a78d6", "no indoor link": "#a3a199"}),
        node_size=ag.by({v: bc_indoor.get(v, 0.0) for v in indoor}, title="betweenness"),
        edge_color=ag.by("kind", title="link", palette={"attached": "#1baf7a", "pedway": "#eb6834", "tunnel": "#4a3aa7"}),
        title="Indoor routes on the University of Calgary campus",
        subtitle=f"{len(main)} of {len(indoor)} buildings connected without going outside",
        caption="Building positions © OpenStreetMap contributors (ODbL)",
    )
    fig.save(out / "indoor_routes.png")
    fig.save(out / "indoor_routes.html")
    facts["indoor_main"] = len(main)
    facts["indoor_top"] = [(v, round(b, 3)) for v, b in bc_indoor.top(5)]


def build(out: Path) -> None:
    facts: dict[str, object] = {}
    epidemics(out, facts)
    communities(out, facts)
    influence(out, facts)
    interop(out, facts)
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
