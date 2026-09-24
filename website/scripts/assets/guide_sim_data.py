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

"""Figures, embeds and measured numbers for the user-guide pages on simulation,
DAGs, datasets, file formats, the command line and reproducibility.

Pages: ``source/user-guide/{simulation,dags,datasets,io,cli,reproducibility}.md``.
Besides the figures, ``facts.json`` records every number quoted in the prose
that does not come from a code block on the page, and ``timings.json`` holds
the performance measurements quoted in ``reproducibility.md``.
"""

from __future__ import annotations

import contextlib
import io
import json
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

import aryagraph as ag
from aryagraph import cli

INDOOR = {"tunnel", "pedway", "attached"}
OSM = "Positions © OpenStreetMap contributors (ODbL)"


def campus_layout(g):
    return {n: d["pos"] for n, d in g.nodes.data()}


# ---------------------------------------------------------------------- #
# simulation.md
# ---------------------------------------------------------------------- #
def simulation(out: Path, facts: dict) -> None:
    g = ag.gen.watts_strogatz(300, 6, 0.08, seed=1)
    run = ag.sim.SIR(beta=0.1, gamma=0.1).simulate(g, initial={"I": 3}, method="gillespie", seed=7)
    t_peak, n_peak = run.peak("I")
    facts["sir"] = {"repr": repr(run), "peak_time": t_peak, "peak": n_peak, "final_R": int(run.counts()["R"][-1])}

    run.plot(
        title="SIR epidemic: nodes per state",
        subtitle="Watts–Strogatz graph, 300 nodes · β = 0.1, γ = 0.1 · Gillespie, seed 7",
    ).save(out / "sir_curve.png")

    anim = run.animate(
        labels=False,
        title="SIR epidemic on a small-world network",
        subtitle="300 nodes · β = 0.1, γ = 0.1 · press play or drag the timeline",
        static_frame=int(run.index_at(t_peak)),
    )
    anim.save(out / "sir_player.html")

    ens = ag.sim.run_ensemble(
        ag.sim.sir, runs=200, g=g, beta=0.1, gamma=0.1, initial={"I": 3}, method="gillespie", seed=11
    )
    ens.plot(
        title="SIR ensemble: 200 Gillespie runs",
        subtitle="Mean count per state with 25–75% and 5–95% bands · master seed 11",
    ).save(out / "sir_ensemble.png")

    vacc = ag.sim.CompartmentalModel(
        ["S", "V", "I", "R"],
        spontaneous=[("S", "V", 0.02), ("I", "R", 0.1)],
        induced=[("S", "I", "I", 0.1)],
        name="SIR + vaccination",
    )
    vr = vacc.simulate(g, initial={"I": 3}, method="gillespie", seed=7)
    vr.plot(
        fractions=True,
        title="Custom model: SIR with vaccination",
        subtitle="S → V at rate 0.02 per day competes with infection",
    ).save(out / "vaccination_curve.png")

    campus = ag.gen.ucalgary_campus()
    heat = ag.sim.heat_diffusion(campus, "MSC", rate=1.0, t_max=5.0)
    snapshot = heat.at(1.0)
    ag.draw(
        campus,
        layout=campus_layout(campus),
        node_color=ag.by({n: float(np.log10(v)) for n, v in snapshot.items()}, title="log10 heat, t = 1"),
        labels={n: n for n in ("MSC", "SS", "TFDL", "ICT", "OO", "OVC")},
        title="Heat diffusion on the UCalgary campus graph",
        subtitle=f"One unit of heat released at MacEwan Student Centre (MSC), after 1 time unit · {OSM}",
    ).save(out / "heat_campus.png")
    facts["heat"] = {"MSC_t1": snapshot["MSC"], "min_t1": min(snapshot.values()), "total_t1": sum(snapshot.values())}

    series = {}
    for K in (0.5, 1.0, 2.0, 4.0):
        res = ag.sim.kuramoto(g, coupling=K, t_max=20, seed=2)
        series[f"K = {K:g}"] = res.meta["order_parameter"]
        times = res.times
    ag.charts.line_chart(
        times,
        series,
        title="Kuramoto oscillators: synchrony over time",
        subtitle="Order parameter r(t) on the 300-node small world · seed 2",
        x_label="time",
        y_label="r(t)",
        y_min=0,
        y_max=1,
    ).save(out / "kuramoto.png")


# ---------------------------------------------------------------------- #
# dags.md
# ---------------------------------------------------------------------- #
def dags(out: Path, facts: dict) -> None:
    ml = ag.gen.ml_pipeline()
    cp = ag.alg.critical_path(ml)
    ag.draw(
        ml,
        highlight_path=cp.path,
        title="Machine-learning pipeline: critical path",
        subtitle=f"The critical chain is highlighted · {cp.length:g} h end to end",
    ).save(out / "pipeline_cpm.png")

    plan = [
        {"node": n, "worker": n, "start": cp.earliest_start[n], "end": cp.earliest_finish[n]}
        for n in ml
    ]
    ag.charts.gantt(
        plan,
        critical=cp.critical,
        title="CPM schedule of the pipeline (earliest start)",
        subtitle="Blue tasks have zero slack; gray tasks can slip without delaying the end",
        time_label="hours",
    ).save(out / "cpm_gantt.png")

    courses_lr = ag.draw(
        ag.gen.course_prerequisites(),
        layout_options={"orientation": "LR"},
        edge_style="flow",
        node_color="department",
        title="Course prerequisites, left to right",
    )
    courses_lr.save(out / "courses_lr.png")
    # The page explains why the ML pipeline is not drawn left to right: its size that way.
    ml_lr = ag.draw(
        ml,
        layout_options={"orientation": "LR"},
        edge_style="flow",
        node_color="team",
        title="The same pipeline, left to right",
    )
    facts["drawing_lr"] = {"courses": repr(courses_lr), "ml_pipeline": repr(ml_lr)}

    sched = ag.sim.simulate_schedule(ml, workers=2, failure_rate=0.15, max_retries=1, seed=4)
    sched.gantt(
        lanes="worker",
        title="Pipeline run on 2 workers",
        subtitle=f"15% failure risk per attempt, 1 retry · makespan {sched.makespan:g} h · seed 4",
        time_label="hours",
    ).save(out / "gantt.png")
    facts["gantt"] = {"repr": repr(sched), "makespan": sched.makespan}

    mc = ag.sim.monte_carlo_schedule(ml, runs=2000, seed=1)
    mc.plot(
        title="Pipeline duration: 2,000 PERT simulations",
        subtitle=f"Deterministic critical path: {mc.cpm_length:g} h · seed 1",
    ).save(out / "montecarlo.png")
    facts["montecarlo"] = {"repr": repr(mc), "cpm": mc.cpm_length}


# ---------------------------------------------------------------------- #
# datasets.md
# ---------------------------------------------------------------------- #
def datasets(out: Path, facts: dict) -> None:
    campus = ag.gen.ucalgary_campus()
    ag.draw(
        campus,
        layout=campus_layout(campus),
        node_color="kind",
        edge_color=lambda u, v, d: "#52514e" if d["kind"] in INDOOR else "#c3c2b7",
        edge_width=ag.by(lambda u, v, d: 2.0 if d["kind"] in INDOOR else 1.0, kind="identity"),
        title="ucalgary_campus(): 56 buildings, 83 links",
        subtitle=f"Dark links: tunnels, pedways, attached buildings · light: modeled outdoor walks · {OSM}",
    ).save(out / "campus.png")

    indoor = ag.gen.ucalgary_campus(indoor_only=True)
    ag.draw(
        indoor,
        layout=campus_layout(indoor),
        node_color="kind",
        title="ucalgary_campus(indoor_only=True)",
        subtitle=f"40 indoor links; 18 buildings have no indoor connection · {OSM}",
    ).save(out / "campus_indoor.png")

    etl = ag.gen.data_warehouse_etl()
    ag.draw(
        etl,
        node_color="kind",
        title="data_warehouse_etl(): 25 tasks, 41 dependencies",
        subtitle="Extract and stage per source, dimensions, facts, a quality gate, aggregates",
    ).save(out / "etl.png")

    thumbs = {
        "erdos_renyi": ("erdos_renyi(60, 0.06)", ag.gen.erdos_renyi(60, 0.06, seed=1)),
        "barabasi_albert": ("barabasi_albert(60, 2)", ag.gen.barabasi_albert(60, 2, seed=1)),
        "watts_strogatz": ("watts_strogatz(60, 4, 0.1)", ag.gen.watts_strogatz(60, 4, 0.1, seed=1)),
        "planted_partition": ("planted_partition(3, 20, 0.3, 0.01)", ag.gen.planted_partition(3, 20, 0.3, 0.01, seed=1)),
        "random_geometric": ("random_geometric(60, 0.2)", ag.gen.random_geometric(60, 0.2, seed=1)),
        "random_tree": ("random_tree(60)", ag.gen.random_tree(60, seed=1)),
    }
    for name, (call, g) in thumbs.items():
        opts = {}
        if name == "planted_partition":
            opts["node_color"] = ag.by("block", kind="categorical")
        layout = campus_layout(g) if name == "random_geometric" else ("tree" if name == "random_tree" else "auto")
        ag.draw(
            g,
            layout=layout,
            labels=False,
            legend=False,
            title=call,
            subtitle=f"{len(g)} nodes · {g.num_edges} edges · seed=1",
            **opts,
        ).save(out / f"gen_{name}.png")
        facts.setdefault("generators", {})[call] = [len(g), g.num_edges]


# ---------------------------------------------------------------------- #
# cli.md
# ---------------------------------------------------------------------- #
def command_line(out: Path, facts: dict) -> None:
    """Run the command-line entry point itself, as `aryagraph draw ...` would."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ag.write(ag.gen.ucalgary_campus(), tmp / "campus.json")
        ag.write(ag.gen.les_miserables(), tmp / "lesmis.graphml")
        with contextlib.redirect_stdout(io.StringIO()):  # the CLI reports the files it wrote
            assert cli.main(["draw", str(tmp / "campus.json"), "-o", str(out / "cli_campus.png"),
                             "--color", "kind", "--size", "betweenness", "--title", "UCalgary campus"]) == 0
            assert cli.main(["analyze", str(tmp / "lesmis.graphml"), "-o", str(out / "cli_report.html")]) == 0


# ---------------------------------------------------------------------- #
# reproducibility.md: measured timings
# ---------------------------------------------------------------------- #
def _median_time(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def timings(out: Path) -> None:
    rows: list[dict] = []

    def record(task: str, size: str, fn, repeat: int = 3) -> None:
        rows.append({"task": task, "size": size, "seconds": round(_median_time(fn, repeat), 4), "repeat": repeat})

    for n in (1_000, 10_000, 100_000):
        g = ag.gen.gnm_random_graph(n, 5 * n, seed=1)
        size = f"{n:,} nodes, {5 * n:,} edges"
        record("generate gnm_random_graph", size, lambda: ag.gen.gnm_random_graph(n, 5 * n, seed=1), 1)
        record("bfs_order", size, lambda: ag.alg.bfs_order(g, 0))
        record("dijkstra (one source)", size, lambda: ag.alg.dijkstra(g, 0, weight=None))
        record("pagerank", size, lambda: ag.alg.pagerank(g))
        record("connected_components", size, lambda: ag.alg.connected_components(g))
        if n <= 10_000:
            record("louvain_communities", size, lambda: ag.alg.louvain_communities(g, seed=0), 1)

    tracemalloc.start()
    big = ag.gen.gnm_random_graph(100_000, 500_000, seed=1)
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory = {"graph": "100,000 nodes, 500,000 edges", "megabytes": round(current / 1e6, 1),
              "bytes_per_edge": round(current / big.num_edges)}
    del big

    for n in (1_000, 3_000):
        g = ag.gen.gnm_random_graph(n, 5 * n, seed=1)
        record("betweenness_centrality (exact)", f"{n:,} nodes, {5 * n:,} edges",
               lambda: ag.alg.betweenness_centrality(g), 1)
    g = ag.gen.gnm_random_graph(10_000, 50_000, seed=1)
    record("betweenness_centrality (k=200 sources)", "10,000 nodes, 50,000 edges",
           lambda: ag.alg.betweenness_centrality(g, k=200, seed=0), 1)
    for n in (500, 2_000):
        g = ag.gen.gnm_random_graph(n, 3 * n, seed=1)
        record("stress layout", f"{n:,} nodes, {3 * n:,} edges", lambda: ag.layout.compute(g, "stress"), 1)
    for n in (2_000, 10_000):
        g = ag.gen.gnm_random_graph(n, 3 * n, seed=1)
        record("forceatlas2 layout", f"{n:,} nodes, {3 * n:,} edges", lambda: ag.layout.compute(g, "forceatlas2"), 1)
    for n in (300, 1_000):
        d = ag.gen.random_task_dag(n, seed=1)
        record("hierarchical layout", f"{n:,} tasks, {d.num_edges:,} arcs", lambda: ag.layout.compute(d, "hierarchical"), 1)
    g = ag.gen.gnm_random_graph(1_000, 3_000, seed=1)
    record("draw + SVG", "1,000 nodes, 3,000 edges", lambda: ag.draw(g, labels=False).to_svg(), 1)
    g = ag.gen.barabasi_albert(1_000, 3, seed=1)
    record("analyze", f"1,000 nodes, {g.num_edges:,} edges", lambda: ag.analyze(g), 1)
    g = ag.gen.watts_strogatz(10_000, 10, 0.05, seed=1)
    record("SIR, Gillespie", "10,000 nodes, 50,000 edges",
           lambda: ag.sim.sir(g, 0.1, 0.1, initial={"I": 10}, method="gillespie", seed=1))
    record("SIR, discrete (dt = 1)", "10,000 nodes, 50,000 edges",
           lambda: ag.sim.sir(g, 0.1, 0.1, initial={"I": 10}, seed=1))
    d = ag.gen.random_task_dag(200, seed=1)
    record("monte_carlo_schedule, 10,000 runs", f"200 tasks, {d.num_edges} arcs, unlimited workers",
           lambda: ag.sim.monte_carlo_schedule(d, runs=10_000, seed=1))
    record("monte_carlo_schedule, 1,000 runs", f"200 tasks, {d.num_edges} arcs, 4 workers",
           lambda: ag.sim.monte_carlo_schedule(d, runs=1_000, workers=4, seed=1), 1)

    try:
        import subprocess

        cpu = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or platform.processor()
    except Exception:  # pragma: no cover - informational only
        cpu = platform.processor()
    info = {
        "cpu": cpu,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "aryagraph": ag.__version__,
    }
    (out / "timings.json").write_text(json.dumps({"machine": info, "memory": memory, "rows": rows}, indent=2),
                                      encoding="utf-8")


# ---------------------------------------------------------------------- #
def build(out: Path) -> None:
    facts: dict = {}
    simulation(out, facts)
    dags(out, facts)
    datasets(out, facts)
    command_line(out, facts)
    timings(out)
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
