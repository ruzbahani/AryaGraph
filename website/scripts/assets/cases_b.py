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

"""Figures and interactive embeds for two case studies.

* ``source/case-studies/epidemic-interventions.md``: SIR outbreaks on a
  Watts–Strogatz and a Barabási–Albert network, and 10% immunization chosen at
  random, by acquaintance, by degree and by betweenness;
* ``source/case-studies/les-miserables.md``: rankings, communities, bridges and
  k-cores of the Les Misérables co-appearance network.

The computations mirror the code shown on the two pages (same parameters and
seeds), and ``facts.json`` records the numbers quoted there so they can be
compared after a rebuild.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import aryagraph as ag

# ---------------------------------------------------------------------- #
# epidemic interventions (mirrors the code on epidemic-interventions.md)
# ---------------------------------------------------------------------- #
N = 1000
RUNS = 200
K = N // 10  # 10% of the nodes are immunized
LABEL = {"WS": "Watts–Strogatz", "BA": "Barabási–Albert"}
NAMES = {
    "none": "no immunization",
    "random": "random",
    "acquaintance": "acquaintance",
    "degree": "highest degree",
    "betweenness": "highest betweenness",
}


def outbreak(ens) -> dict[str, np.ndarray]:
    """Final size, peak prevalence (both as shares of N) and peak day of every run."""
    infected = ens.samples[:, :, ens.states.index("I")]
    return {
        "final": ens.final_sizes["R"] / N,
        "peak": infected.max(axis=1) / N,
        "peak_day": ens.times[infected.argmax(axis=1)],
    }


def infected_on(ens, days: np.ndarray) -> np.ndarray:
    """Share infected in every run on a common time grid (counts hold between frames)."""
    idx = np.clip(np.searchsorted(ens.times, days, side="right") - 1, 0, None)
    return ens.samples[:, idx, ens.states.index("I")] / N


def random_targets(g, rng):
    nodes = list(g)
    return [nodes[i] for i in rng.choice(len(nodes), size=K, replace=False)]


def acquaintance_targets(g, rng):
    """Pick a node at random and immunize one of its neighbors, until K are immunized."""
    nodes, chosen = list(g), set()
    while len(chosen) < K:
        v = nodes[rng.integers(len(nodes))]
        neighbors = list(g.neighbors(v))
        chosen.add(neighbors[rng.integers(len(neighbors))])
    return list(chosen)


def _pct(v: float) -> str:
    return f"{v:.0%}" if v >= 0.1 else f"{v:.1%}"


def _stats(f: np.ndarray, base: np.ndarray | None) -> dict:
    row = {
        "mean": round(float(f.mean()), 4),
        "p5": round(float(np.percentile(f, 5)), 4),
        "p95": round(float(np.percentile(f, 95)), 4),
    }
    if base is not None:
        diff = base.mean() - f.mean()
        half = 1.96 * np.sqrt(base.var(ddof=1) / base.size + f.var(ddof=1) / f.size)
        row.update(reduction=round(float(diff), 4), ci=round(float(half), 4), relative=round(float(diff / base.mean()), 4))
    return row


def epidemic(out: Path, facts: dict) -> None:
    networks = {
        "WS": ag.gen.watts_strogatz(N, 6, 0.1, seed=1),
        "BA": ag.gen.barabasi_albert(N, 3, seed=1),
    }
    # nodes by degree as labeled bars: the Barabási–Albert rows widen with the
    # degree, so its few hubs stay visible on a linear scale
    first_degree = {"WS": [4, 5, 6, 7, 8, 9], "BA": [3, 4, 5, 6, 10, 20, 50]}
    for name, g in networks.items():
        k = np.array(list(g.degree().values()))
        edges = [*first_degree[name], int(k.max()) + 1]
        rows = [str(a) if b == a + 1 else f"{a}–{b - 1}" for a, b in zip(edges, edges[1:])]
        counts = [int(np.sum((k >= a) & (k < b))) for a, b in zip(edges, edges[1:])]
        assert sum(counts) == N, (name, counts)
        facts[f"{name}_degree_rows"] = dict(zip(rows, counts))
        ag.charts.bar_chart(
            rows,
            counts,
            title=f"{LABEL[name]}: nodes by degree",
            subtitle=f"{N:,} nodes · mean degree {k.mean():.2f} · rows: degree · bars: nodes",
            sort=False,
            width=460,
        ).save(out / f"degrees_{name.lower()}.png")

    beta, gamma = 0.15, 0.25
    model = ag.sim.CompartmentalModel(
        ["S", "I", "R", "V"],
        spontaneous=[("I", "R", gamma)],
        induced=[("S", "I", "I", beta)],
        roles={"V": "accent"},
    )

    def immunize_and_simulate(g, pick, seed):
        rng = np.random.default_rng(seed)  # a new immunized set in every run
        return model.simulate(g, initial={"V": pick(g, rng), "I": 5}, method="gillespie", seed=rng)

    baseline = {
        name: ag.sim.run_ensemble(model.simulate, runs=RUNS, g=g, initial={"I": 5}, method="gillespie", seed=7, n_points=301)
        for name, g in networks.items()
    }
    days = np.linspace(0, 100, 401)
    series, bands = {}, {}
    for name, ens in baseline.items():
        share = infected_on(ens, days) * 100
        series[LABEL[name]] = share.mean(axis=0)
        bands[LABEL[name]] = (np.percentile(share, 5, axis=0), np.percentile(share, 95, axis=0))
    ag.charts.line_chart(
        days,
        series,
        bands=bands,
        band_label="5–95% of runs",
        title="Share of nodes infected over time",
        subtitle=f"SIR, β = {beta}, γ = {gamma} · {RUNS} runs per network · mean with 5–95% band",
        x_label="day",
        y_label="% infected",
        width=680,
        height=320,
    ).save(out / "curves_structure.png")

    ensembles = {}
    for name, g in networks.items():
        betweenness = ag.alg.betweenness_centrality(g)
        targets = {
            "degree": [v for v, _ in g.degree().top(K)],
            "betweenness": [v for v, _ in betweenness.top(K)],
        }
        facts[f"{name}_shared_targets"] = len(set(targets["degree"]) & set(targets["betweenness"]))
        ensembles[name] = {"none": baseline[name]}
        for strategy, pick in [("random", random_targets), ("acquaintance", acquaintance_targets)]:
            ensembles[name][strategy] = ag.sim.run_ensemble(immunize_and_simulate, runs=RUNS, g=g, pick=pick, seed=7, n_points=301)
        for strategy, nodes in targets.items():
            ensembles[name][strategy] = ag.sim.run_ensemble(
                model.simulate, runs=RUNS, g=g, initial={"V": nodes, "I": 5}, method="gillespie", seed=7, n_points=301
            )

    horizon = {"WS": 100, "BA": 60}
    for name, by_strategy in ensembles.items():
        base = outbreak(by_strategy["none"])["final"]
        facts[name] = {}
        for strategy, ens in by_strategy.items():
            o = outbreak(ens)
            facts[name][strategy] = {
                "final": _stats(o["final"], None if strategy == "none" else base),
                "peak": _stats(o["peak"], None),
                "peak_day_median": round(float(np.median(o["peak_day"])), 2),
                "below_5pct": round(float(np.mean(o["final"] < 0.05)), 3),
            }
        ag.charts.bar_chart(
            [NAMES[s] for s in by_strategy],
            [outbreak(e)["final"].mean() for e in by_strategy.values()],
            title=f"{LABEL[name]}: final outbreak size",
            subtitle=f"mean share of nodes ever infected, {RUNS} runs each",
            value_format=_pct,
            sort=False,
            width=460,
        ).save(out / f"final_{name.lower()}.png")
        t = np.linspace(0, horizon[name], 301)
        ag.charts.line_chart(
            t,
            {NAMES[s]: infected_on(e, t).mean(axis=0) * 100 for s, e in by_strategy.items()},
            title=f"{LABEL[name]}: mean share infected",
            subtitle=f"10% of nodes immunized ({K} of {N:,}) · {RUNS} runs per strategy",
            x_label="day",
            y_label="% infected",
            width=680,
            height=320,
        ).save(out / f"curves_{name.lower()}.png")

    ba = networks["BA"]
    rng = np.random.default_rng(1)
    runs = {
        "random": model.simulate(ba, initial={"V": random_targets(ba, rng), "I": 5}, method="gillespie", seed=rng),
        "degree": model.simulate(ba, initial={"V": [v for v, _ in ba.degree().top(K)], "I": 5}, method="gillespie", seed=1),
    }
    for strategy, run in runs.items():
        infected = run.summary()["R"]["final"]
        facts[f"BA_single_{strategy}"] = infected
        player = run.animate(
            labels=False,
            node_size=ag.by(ba.degree(), title="degree"),
            title=f"Barabási–Albert network, 10% immunized: {NAMES[strategy]}",
            subtitle=f"one run · blue: immunized (V) · green: infected, then recovered (R) · {infected} of {N:,} infected",
            width=620,
        )
        player.save(out / f"network_ba_{strategy}.png")
        player.save(out / f"network_ba_{strategy}.html")


# ---------------------------------------------------------------------- #
# Les Misérables
# ---------------------------------------------------------------------- #
def lesmis(out: Path, facts: dict) -> None:
    g = ag.gen.les_miserables()
    strength = g.degree(weight="weight")
    betweenness = ag.alg.betweenness_centrality(g)
    pagerank = ag.alg.pagerank(g)
    communities = ag.alg.louvain_communities(g, seed=0)
    labels = ag.alg.community_labels(communities)
    # each community is named after its member with the largest weighted degree
    # (the first one in graph order on ties, so the name does not depend on set order)
    names = {i: max((v for v in g if labels[v] == i), key=strength.__getitem__) for i in range(len(communities))}
    q = ag.alg.modularity(g, communities)
    facts["lesmis_modularity"] = round(q, 4)
    facts["lesmis_sizes"] = [len(c) for c in communities]
    facts["lesmis_names"] = names

    # 1. communities (interactive)
    community = ag.by({v: f"{labels[v]} · {names[labels[v]]}" for v in g}, kind="categorical", title="community")
    chapters = ag.by("weight", title="shared chapters")
    fig = ag.draw(
        g,
        node_color=community,
        node_size=ag.by(strength, title="weighted degree"),
        edge_width=chapters,
        title="Les Misérables: six communities",
        subtitle=f"Louvain (seed 0), modularity {q:.3f} · size: weighted degree · width: shared chapters",
    )
    fig.save(out / "lesmis_communities.html")

    # 2. betweenness (size) and PageRank (color)
    ag.draw(
        g,
        node_size=ag.by(betweenness, title="betweenness"),
        node_color=ag.by(pagerank, kind="log", title="PageRank"),
        edge_width=chapters,
        title="Betweenness and PageRank",
        subtitle="size: betweenness centrality · color: weighted PageRank (log scale)",
    ).save(out / "lesmis_centrality.png")

    # 3. bridges between communities: crossing ties dark, size = number of crossing ties
    inter = [(u, v, d["weight"]) for u, v, d in g.edges.data() if labels[u] != labels[v]]
    crossing = Counter(x for u, v, _ in inter for x in (u, v))
    facts["lesmis_crossing_top"] = crossing.most_common(5)
    facts["lesmis_no_crossing"] = sum(1 for v in g if crossing[v] == 0)
    ag.draw(
        g,
        node_color=community,
        node_size=ag.by({v: crossing[v] for v in g}, title="ties to other communities"),
        edge_color=ag.by(
            lambda u, v, d: "between communities" if labels[u] != labels[v] else "within a community",
            kind="categorical",
            palette={"between communities": "#1f1e1c", "within a community": "#d3d1c8"},
            title="tie",
        ),
        edge_width=chapters,
        title="Bridges between communities",
        subtitle=f"dark: the {len(inter)} ties that join two communities · size: ties to other communities",
    ).save(out / "lesmis_bridges.png")

    # 4. community graph
    between = Counter()
    for u, v, w in inter:
        between[tuple(sorted((labels[u], labels[v])))] += w
    cg = ag.Graph(name="Les Misérables communities")
    for i, c in enumerate(communities):
        cg.add_node(f"{i} · {names[i]}", members=len(c))
    for (a, b), w in between.items():
        cg.add_edge(f"{a} · {names[a]}", f"{b} · {names[b]}", weight=w)
    ag.draw(
        cg,
        node_color=ag.by({n: n for n in cg}, kind="categorical", legend=False),
        node_size=ag.by("members", title="characters"),
        edge_width=ag.by("weight", title="shared chapters"),
        edge_label="weight",
        layout="stress",
        title="The six communities as one graph",
        subtitle="size: characters in the community · width and label: co-appearances between communities",
    ).save(out / "lesmis_community_graph.png")
    facts["lesmis_between"] = {f"{a}-{b}": w for (a, b), w in sorted(between.items())}

    # 5. k-cores
    core = ag.alg.core_number(g)
    kmax = max(core.values())
    innermost = [v for v, c in core.items() if c == kmax]
    ag.draw(
        g,
        node_color=ag.by(core, kind="sequential", title="core number"),
        highlight=innermost,
        title=f"k-cores: the {kmax}-core",
        subtitle=f"color: core number · highlighted: the {len(innermost)} characters of the innermost core",
    ).save(out / "lesmis_cores.png")

    # 6. Valjean's ego network
    ego = g.subgraph(["Valjean", *g.neighbors("Valjean")])
    ag.draw(
        ego,
        node_color=community,
        node_size=ag.by({v: strength[v] for v in ego}, title="weighted degree"),
        edge_width=chapters,
        title="Valjean's ego network",
        subtitle=f"his {len(ego) - 1} neighbors and the ties among them · color: community in the full network",
    ).save(out / "lesmis_valjean.png")

    # the one-call report as an interactive dashboard
    ag.analyze(g).save(out / "lesmis_report.html")


def build(out: Path) -> None:
    facts: dict = {}
    epidemic(out, facts)
    lesmis(out, facts)
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
