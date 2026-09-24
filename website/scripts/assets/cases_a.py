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

"""Figures for the first three case studies (source/case-studies/).

* ``ucalgary-campus.md``: connectivity of the University of Calgary indoor network;
* ``construction-schedule.md``: CPM, crews, PERT Monte Carlo and rework risk for
  the house-construction plan;
* ``ml-pipeline.md``: critical path, parallelism, workers, policies and retries
  for the machine-learning pipeline.

The analyses repeat the code shown on the pages, with the same seeds, so every
number in a figure matches the text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import aryagraph as ag

INDOOR = {"tunnel", "pedway", "attached"}
LINK_COLORS = {"tunnel": "#4a3aa7", "pedway": "#2a78d6", "attached": "#1baf7a", "outdoor": "#c9c8c0"}
STATUS_COLORS = {"indoor network": "#2a78d6", "no indoor link": "#eb6834"}
MAP_WIDTH = 880
OSM = "Positions © OpenStreetMap contributors (ODbL)"
POLICIES = ("critical_path", "fifo", "longest_first", "shortest_first")
PHASES = {
    "planning": "design & permit",
    "permit": "design & permit",
    "sitework": "structure",
    "structure": "structure",
    "envelope": "envelope & systems",
    "systems": "envelope & systems",
    "interior": "interior",
    "exterior": "exterior & handover",
    "inspection": "exterior & handover",
    "wait": "wait",
}


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# =============================================================================
# 1. University of Calgary indoor network
# =============================================================================
def campus_analysis() -> dict:
    full = ag.gen.ucalgary_campus()
    indoor = ag.gen.ucalgary_campus(indoor_only=True)
    comps = ag.alg.connected_components(indoor)
    main = indoor.subgraph(n for n in indoor if n in comps[0])  # keep graph order: ties break reproducibly
    nodes = list(indoor)
    pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    base = ag.alg.all_pairs_shortest_path_length(indoor, weight="length")
    closures = []
    for u, v, kind in indoor.edges.data("kind"):
        g = indoor.copy()
        g.remove_edge(u, v)
        after = ag.alg.all_pairs_shortest_path_length(g, weight="length")
        lost, detour, pair = 0, 0.0, None
        for a, b in pairs:
            if b not in base[a]:
                continue  # no indoor route even before the closure
            if b not in after[a]:
                lost += 1
            elif after[a][b] - base[a][b] > detour + 1e-9:
                detour, pair = after[a][b] - base[a][b], (a, b)
        closures.append({"link": (u, v), "kind": kind, "lost": lost, "detour": detour, "pair": pair})
    dist = ag.alg.all_pairs_shortest_path_length(main, weight="length")
    a, b, meters = max(((s, t, d) for s, row in dist.items() for t, d in row.items()), key=lambda x: x[2])
    return {
        "full": full,
        "indoor": indoor,
        "comps": comps,
        "main": main,
        "bridges": ag.alg.bridges(indoor),
        "cut": ag.alg.articulation_points(indoor),
        "betweenness": ag.alg.betweenness_centrality(main, weight="length"),
        "closures": closures,
        "longest": (a, b, meters, ag.alg.shortest_path(main, a, b, weight="length")),
    }


def _geo(g) -> dict:
    return {n: d["pos"] for n, d in g.nodes.data()}


def campus_figures(out: Path) -> None:
    c = campus_analysis()
    full, indoor, main = c["full"], c["indoor"], c["main"]
    grounds = [n for n in full if n != "OVC"]  # OVC stands 626 m south, at McMahon Stadium
    fmap = full.subgraph(grounds)
    imap = indoor.subgraph(grounds)
    connected = set(c["comps"][0])
    status = {n: ("indoor network" if n in connected else "no indoor link") for n in imap}
    status_spec = ag.by(status, kind="categorical", palette=STATUS_COLORS, domain=list(STATUS_COLORS), title="building")

    # 1. every modeled link, by kind
    ag.draw(
        fmap,
        layout=_geo(fmap),
        node_color="#6f6e69",
        edge_color=ag.by("kind", kind="categorical", palette=LINK_COLORS, domain=list(LINK_COLORS), title="link"),
        edge_width=ag.by(lambda u, v, d: 2.6 if d["kind"] in INDOOR else 1.2, kind="identity"),
        width=MAP_WIDTH,
        title="University of Calgary main campus: every modeled link",
        subtitle=f"Olympic Volunteer Centre (626 m south) not shown · {OSM}",
    ).save(out / "campus_links.png")

    # 2. indoor islands
    ag.draw(
        imap,
        layout=_geo(imap),
        node_color=status_spec,
        edge_color="#52514e",
        edge_width=2.2,
        width=MAP_WIDTH,
        title="Indoor links only: one network and 18 islands",
        subtitle=f"{len(connected)} buildings share one indoor network · Olympic Volunteer Centre not shown · {OSM}",
    ).save(out / "indoor_islands.png")

    # 3. single points of failure
    bridge_set = {frozenset(e) for e in c["bridges"]}
    cut = set(c["cut"])
    role = {n: ("cut building" if n in cut else "other building") for n in main}
    ag.draw(
        main,
        layout=_geo(main),
        node_color=ag.by(role, kind="categorical", palette={"cut building": "#eb6834", "other building": "#a3a199"},
                         domain=["cut building", "other building"], title="building"),
        edge_color=ag.by(lambda u, v, d: "bridge" if frozenset((u, v)) in bridge_set else "on a loop",
                         kind="categorical", palette={"bridge": "#d03b3b", "on a loop": "#2a78d6"},
                         domain=["bridge", "on a loop"], title="indoor link"),
        edge_width=2.6,
        width=MAP_WIDTH,
        title="Single points of failure in the indoor network",
        subtitle=(
            f"{len(c['bridges'])} of {len(main.edges)} links are bridges; "
            f"{len(cut)} of {len(main)} buildings are cut buildings · {OSM}"
        ),
    ).save(out / "single_points.png")

    # 4. betweenness
    bc = c["betweenness"]
    ag.draw(
        main,
        layout=_geo(main),
        node_color=ag.by(bc, kind="sequential", title="betweenness"),
        node_size=ag.by(bc, legend=False),
        edge_color="#8d8c85",
        edge_width=2.0,
        width=MAP_WIDTH,
        title="Where indoor routes converge",
        subtitle=f"Length-weighted betweenness on the {len(main)}-building indoor network · {OSM}",
    ).save(out / "betweenness.png")

    # 5. closures that cut the most building pairs off
    worst = sorted(c["closures"], key=lambda r: -r["lost"])[:12]
    connected_pairs = len(main) * (len(main) - 1) // 2
    ag.charts.bar_chart(
        [f"{r['link'][0]}–{r['link'][1]} ({r['kind']})" for r in worst],
        [r["lost"] for r in worst],
        title="Building pairs that lose their indoor route",
        subtitle=f"One indoor link closed at a time; {connected_pairs} pairs are connected indoors before any closure",
        width=640,
        max_label_width=200,
    ).save(out / "closures.png")

    # 6. the longest indoor walk
    a, b, meters, route = c["longest"]
    ag.draw(
        imap,
        layout=_geo(imap),
        node_color="#6f6e69",
        edge_color="#8d8c85",
        edge_width=2.0,
        highlight_path=route,
        width=MAP_WIDTH,
        title=f"Longest indoor walk: {full.nodes[a]['name']} to {full.nodes[b]['name']}",
        subtitle=f"{meters:,.0f} m of center-to-center links through {len(route)} buildings · {OSM}",
    ).save(out / "longest_walk.png")

    # 7. interactive view of the indoor network
    ag.draw(
        imap,
        layout=_geo(imap),
        node_color=status_spec,
        node_size=ag.by({n: bc.get(n, 0.0) for n in imap}, title="betweenness"),
        edge_color=ag.by("kind", kind="categorical", palette=LINK_COLORS, domain=["tunnel", "pedway", "attached"], title="link"),
        edge_width=2.4,
        tooltip=["name", "kind"],
        title="University of Calgary indoor network",
        subtitle=f"Size shows betweenness; hover a building for its name · {OSM}",
    ).save(out / "indoor_network.html")


# =============================================================================
# 2. House construction
# =============================================================================
def with_lags(dag):
    """Turn every arc's ``lag`` into an explicit wait task (fixed duration, no crew)."""
    out = dag.copy()
    for u, v, lag in dag.edges.data("lag"):
        if lag:
            wait = f"wait after {u}"
            out.remove_edge(u, v)
            out.add_node(wait, duration=lag, min=lag, mode=lag, max=lag, kind="wait", team="none")
            out.add_edges([(u, wait), (wait, v)])
    for n in out:
        out.nodes[n]["crew"] = 0 if out.nodes[n]["kind"] == "wait" else 1
    return out


def crew_rows(res, crews: int) -> list[dict]:
    """Assign every task run to a crew lane (greedy interval coloring) for a Gantt chart."""
    free_at = [0.0] * crews
    rows = []
    for t in sorted(res.tasks, key=lambda r: (r.start, r.end)):
        if str(t.node).startswith("wait"):
            lane = "waiting"
        else:
            i = next(j for j in range(crews) if free_at[j] <= t.start + 1e-9)
            free_at[i] = t.end
            lane = f"crew {i + 1}"
        rows.append({"node": t.node, "worker": lane, "start": t.start, "end": t.end, "status": t.status})
    return rows


def construction_figures(out: Path) -> None:
    house = with_lags(ag.gen.project_plan())
    real = [n for n in house if house.nodes[n]["kind"] != "wait"]
    cp = ag.alg.critical_path(house)

    # 1. the plan and its critical path
    for ext in ("png", "html"):
        ag.draw(
            house,
            node_color=ag.by(lambda n, d: PHASES[d["kind"]], kind="categorical", title="phase"),
            layout_options={"rank_sep": 48},
            highlight_path=cp.path,
            title="House construction plan",
            subtitle=f"Critical path highlighted · {cp.length:g} working hours including the two waits",
        ).save(out / f"house_plan.{ext}")

    # 2. makespan against crews
    crews = [1, 2, 3, 4, 5]
    det, p50, p80 = [], [], []
    for k in crews:
        det.append(ag.sim.simulate_schedule(house, workers=None, resources={"crew": k}).makespan)
        mc = ag.sim.monte_carlo_schedule(house, runs=1000, resources={"crew": k}, seed=7)
        p50.append(mc.percentiles["P50"])
        p80.append(mc.percentiles["P80"])
    ag.charts.line_chart(
        crews,
        {"planned (no variation)": det, "P50 with PERT": p50, "P80 with PERT": p80},
        title="Makespan against the number of crews",
        subtitle="Critical-path-first dispatching · PERT: 1,000 runs per point, seed 7",
        x_label="crews",
        y_label="working hours",
        x_format=lambda v: f"{v:g}",
        y_min=600,
        width=640,
        height=300,
    ).save(out / "house_crews.png")

    # 3. two-crew schedule
    two = ag.sim.simulate_schedule(house, workers=None, resources={"crew": 2})
    ag.charts.gantt(
        crew_rows(two, 2),
        lanes="worker",
        critical=cp.critical,
        title="Planned schedule with two crews",
        subtitle=f"Makespan {two.makespan:g} h · waits need no crew · critical activities in the accent color",
        time_label="working hours",
        width=760,
    ).save(out / "house_gantt.png")

    # 4. PERT Monte Carlo
    mc = ag.sim.monte_carlo_schedule(house, runs=5000, seed=7)
    pct = mc.percentiles
    ag.charts.histogram(
        mc.makespans,
        title="House construction: 5,000 PERT simulations",
        subtitle=f"Unlimited crews · the plan's {mc.cpm_length:g} h is met in {_pct(mc.probability(mc.cpm_length))} of runs",
        x_label="makespan (working hours)",
        y_label="runs",
        markers=[(mc.cpm_length, f"plan {mc.cpm_length:.4g}")] + [(v, f"{k} {v:.4g}") for k, v in pct.items()],
        width=640,
        height=280,
    ).save(out / "house_montecarlo.png")

    # 5. criticality index
    crit = sorted(((n, mc.criticality[n]) for n in real), key=lambda kv: -kv[1])
    ag.charts.bar_chart(
        [n for n, _ in crit],
        [100 * v for _, v in crit],
        title="Criticality index",
        subtitle="Share of the 5,000 runs in which the task had zero total float",
        value_format=lambda v: f"{v:.1f}%",
        width=640,
    ).save(out / "house_criticality.png")

    # 6. which uncertainty matters: pin one task at its most likely duration
    ref = ag.sim.monte_carlo_schedule(house, runs=20000, seed=7).percentiles["P80"]
    gains = []
    for n in real:
        pinned = house.copy()
        pinned.nodes[n]["min"] = pinned.nodes[n]["max"] = pinned.nodes[n]["mode"]
        m = ag.sim.monte_carlo_schedule(pinned, runs=20000, seed=7)
        gains.append((n, ref - m.percentiles["P80"]))
    gains.sort(key=lambda kv: -kv[1])
    top = gains[:8]
    ag.charts.bar_chart(
        [n for n, _ in top],
        [v for _, v in top],
        title="Hours saved at P80 by removing one task's uncertainty",
        subtitle="Each task pinned to its most likely duration in turn · 20,000 runs each, seed 7",
        value_format=lambda v: f"{v:.1f} h",
        width=640,
    ).save(out / "house_sensitivity.png")

    # 7. rework risk
    rates = [0.0, 0.05, 0.10, 0.15, 0.20]
    bands = {"P50": [], "P80": [], "P95": []}
    for p in rates:
        m = ag.sim.monte_carlo_schedule(
            house, runs=2000, resources={"crew": 2}, failure_rate={n: p for n in real}, max_retries=10, seed=7
        )
        for k in bands:
            bands[k].append(m.percentiles[k])
    ag.charts.line_chart(
        [100 * p for p in rates],
        bands,
        title="Rework risk: makespan against the chance a task must be redone",
        subtitle="Two crews, PERT durations, up to 10 retries per task · 2,000 runs per point, seed 7",
        x_label="per-attempt rework probability (%)",
        y_label="working hours",
        x_format=lambda v: f"{v:g}%",
        y_min=600,
        width=640,
        height=300,
    ).save(out / "house_rework.png")


# =============================================================================
# 3. Machine-learning pipeline
# =============================================================================
def pert_draws(dag, rng) -> dict:
    """One PERT duration per task (the same Beta-PERT the simulator uses)."""
    out = {}
    for n, d in dag.nodes.data():
        lo, mode, hi = d["min"], d["mode"], d["max"]
        a = 1 + 4 * (mode - lo) / (hi - lo)
        b = 1 + 4 * (hi - mode) / (hi - lo)
        out[n] = lo + (hi - lo) * rng.beta(a, b)
    return out


def ml_figures(out: Path) -> None:
    dag = ag.gen.ml_pipeline()
    cp = ag.alg.critical_path(dag)

    # 1. the pipeline
    for ext in ("png", "html"):
        ag.draw(
            dag,
            node_color="team",
            highlight_path=cp.path,
            title="Machine-learning pipeline",
            subtitle=f"Colored by team · critical path highlighted · {cp.length:g} h end to end",
        ).save(out / f"ml_pipeline.{ext}")

    # 2. parallelism profile with unlimited workers
    free = ag.sim.simulate_schedule(dag, workers=None)
    times = sorted({t.start for t in free.tasks} | {t.end for t in free.tasks})
    running = [sum(1 for t in free.tasks if t.start <= x < t.end) for x in times]
    ag.charts.line_chart(
        times,
        {"running tasks": running},
        step=True,
        title="How many tasks can run at once",
        subtitle="Every task starts as soon as its inputs are ready (unlimited workers)",
        x_label="hours",
        y_label="tasks",
        legend=False,
        direct_labels=False,
        width=640,
        height=240,
    ).save(out / "ml_parallelism.png")

    # 3. two-worker Gantt
    two = ag.sim.simulate_schedule(dag, workers=2)
    ag.charts.gantt(
        two,
        lanes="worker",
        color_by="team",
        title="Two workers, critical-path-first",
        subtitle=f"Makespan {two.makespan:g} h · worker utilization {_pct(two.utilization['overall'])}",
        time_label="hours",
        width=760,
    ).save(out / "ml_gantt.png")

    # 4. workers against makespan
    rng = np.random.default_rng(3)
    samples = [pert_draws(dag, rng) for _ in range(2000)]  # the same samples at every pool size
    workers = [1, 2, 3, 4, 5, 6]
    det, p50, p80 = [], [], []
    for w in workers:
        det.append(ag.sim.simulate_schedule(dag, workers=w).makespan)
        spans = [ag.sim.simulate_schedule(dag, workers=w, duration=d).makespan for d in samples]
        p50.append(float(np.percentile(spans, 50)))
        p80.append(float(np.percentile(spans, 80)))
    ag.charts.line_chart(
        workers,
        {"planned (no variation)": det, "P50 with PERT": p50, "P80 with PERT": p80},
        title="Makespan against the number of workers",
        subtitle="Critical-path-first dispatching · the same 2,000 PERT samples at every point (seed 3)",
        x_label="workers",
        y_label="hours",
        x_format=lambda v: f"{v:g}",
        y_min=50,
        width=640,
        height=300,
    ).save(out / "ml_workers.png")

    # 5. Monte Carlo
    mc = ag.sim.monte_carlo_schedule(dag, runs=5000, seed=11)
    pct = mc.percentiles
    ag.charts.histogram(
        mc.makespans,
        title="ML pipeline: 5,000 PERT simulations",
        subtitle=f"Unlimited workers · the plan's {mc.cpm_length:g} h is met in {_pct(mc.probability(mc.cpm_length))} of runs",
        x_label="makespan (hours)",
        y_label="runs",
        markers=[(mc.cpm_length, f"plan {mc.cpm_length:.4g}")] + [(v, f"{k} {v:.4g}") for k, v in pct.items()],
        width=640,
        height=280,
    ).save(out / "ml_montecarlo.png")

    # 6. criticality
    crit = sorted(mc.criticality.items(), key=lambda kv: -kv[1])
    ag.charts.bar_chart(
        [n for n, _ in crit],
        [100 * v for _, v in crit],
        title="Criticality index",
        subtitle="Share of the 5,000 runs in which the task had zero total float",
        value_format=lambda v: f"{v:.1f}%",
        width=640,
    ).save(out / "ml_criticality.png")

    # 7. policies on a shared pool: three regional pipelines
    shared = ag.DAG(name="three regional pipelines")
    for region in ("us", "eu", "apac"):
        shared = shared.compose(dag.relabel(lambda n, r=region: f"{r}: {n}"))
    work = sum(d["duration"] for _, d in shared.nodes.data())
    pool = list(range(2, 9))
    bound = {w: max(cp.length, work / w) for w in pool}
    series = {
        p: [ag.sim.simulate_schedule(shared, workers=w, policy=p).makespan - bound[w] for w in pool] for p in POLICIES
    }
    ag.charts.line_chart(
        pool,
        series,
        title="Three pipelines on one worker pool: policy matters",
        subtitle=f"Planned durations · 48 tasks, {work:g} h of work · lower bound = max(critical path, work / workers)",
        x_label="workers",
        y_label="hours above the lower bound",
        x_format=lambda v: f"{v:g}",
        width=640,
        height=320,
    ).save(out / "ml_policies.png")

    # 8. retries under failures
    rates = (0.05, 0.10, 0.20)
    budgets = [0, 1, 2, 3]
    done = {}
    for p in rates:
        row = []
        for r in budgets:
            ok = 0
            for s in range(1000):
                res = ag.sim.simulate_schedule(dag, workers=2, failure_rate=p, max_retries=r, seed=s)
                ok += sum(1 for t in res.tasks if t.status == "done") == len(dag)
            row.append(100 * ok / 1000)
        done[f"p = {p:.0%}"] = row
    ag.charts.line_chart(
        budgets,
        done,
        title="Completed runs against the retry budget",
        subtitle="Two workers, planned durations, each attempt fails with probability p · 1,000 runs per point",
        x_label="retries allowed per task",
        y_label="runs that finish every task (%)",
        x_format=lambda v: f"{v:g}",
        y_min=0,
        y_max=100,
        width=640,
        height=300,
    ).save(out / "ml_retries.png")


def build(out: Path) -> None:
    campus_figures(out)
    construction_figures(out)
    ml_figures(out)
