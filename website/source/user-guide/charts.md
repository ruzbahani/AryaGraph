# Charts

`ag.charts` draws line charts, ranked bar charts, histograms and Gantt charts in the same visual language as AryaGraph's graph drawings. The analysis dashboard and the simulation plots use these charts, and you can call them directly for your own data.

Each function returns a {py:class}`aryagraph.charts.base.Chart`. Like a figure, a chart saves to `.svg`, `.png`, `.pdf` or `.html` (the format follows the extension) and displays inline in Jupyter.

## Line charts

{py:func}`aryagraph.charts.line_chart` draws several series against one shared x axis. `series` is a `{name: values}` mapping; `bands` adds a shaded range around a series, for example the spread over repeated runs.

This example measures the small-world effect: rewiring a few edges of a ring lattice shortens paths long before it destroys clustering. Each point averages five Watts–Strogatz graphs, and the band shows the range of the five.

```python
import numpy as np
import aryagraph as ag

ps = np.logspace(-3, 0, 13)                       # rewiring probabilities 0.001 … 1
lattice = ag.gen.watts_strogatz(200, 6, 0.0, seed=0)
C0 = ag.alg.average_clustering(lattice)
L0 = ag.alg.average_shortest_path_length(lattice)

clust, path = [], []
for p in ps:
    graphs = [ag.gen.watts_strogatz(200, 6, p, seed=s) for s in range(5)]
    clust.append([ag.alg.average_clustering(g) / C0 for g in graphs])
    path.append([ag.alg.average_shortest_path_length(g) / L0 for g in graphs])
clust, path = np.array(clust), np.array(path)
print(f"p = {ps[4]:g}: clustering {clust[4].mean():.2f}, "
      f"path length {path[4].mean():.2f}")

chart = ag.charts.line_chart(
    np.log10(ps),
    {"clustering C(p) / C(0)": clust.mean(axis=1),
     "path length L(p) / L(0)": path.mean(axis=1)},
    bands={"clustering C(p) / C(0)": (clust.min(axis=1), clust.max(axis=1)),
           "path length L(p) / L(0)": (path.min(axis=1), path.max(axis=1))},
    x_format=lambda v: f"{10 ** v:g}",
    x_label="rewiring probability p (log scale)",
    title="The small-world effect",
    subtitle="Watts–Strogatz graphs, n = 200, k = 6 · mean of 5 seeds, band = range",
    y_min=0,
)
chart.save("small_world.svg")
```

```text
p = 0.01: clustering 0.97, path length 0.54
```

```{figure} ../_static/generated/guide_layout_algorithms/small_world.png
:alt: Line chart of relative clustering and relative path length against the rewiring probability on a log scale; path length drops well before clustering
:width: 90%

At p = 0.01, paths are about half as long as in the lattice, while clustering is nearly unchanged.
```

The chart has no logarithmic axis. Plot `log10(x)` and label the ticks with `x_format`, as above, when the data spans several orders of magnitude. The crosshair of the HTML version (see [Simulation plots](#simulation-plots)) shows the plotted value, here log10 p.

Options you will use often:

| option | effect |
|---|---|
| `colors` | `{name: color}`; by default series take the categorical palette in order |
| `bands`, `outer_bands` | `{name: (low, high)}` shaded ranges (an inner and an outer band, such as 25–75% and 5–95%) |
| `band_label` | legend text for the bands |
| `markers` | vertical reference lines `[(x, label), ...]` |
| `step` | draw series as step functions (event-driven counts) |
| `y_min`, `y_max` | fix the y range; by default the axis starts at zero when no value is negative |
| `x_format`, `y_format` | tick formatters (callables that return a string) |
| `x_label`, `y_label`, `title`, `subtitle` | text |
| `direct_labels`, `legend` | label lines at their right end (with up to four series, when the labels do not collide) and a legend row (with two or more series) |
| `width`, `height`, `theme` | size in pixels and theme |

A `NaN` value breaks a line, which is how you show missing data.

## Bar charts

{py:func}`aryagraph.charts.bar_chart` draws one horizontal bar per label, sorted from largest to smallest, with the value at the tip of each bar. `highlight` keeps the named bars in the accent color and mutes the others, which puts the focus on one or two entries:

```python
campus = ag.gen.ucalgary_campus()
bc = ag.alg.betweenness_centrality(campus, weight="length")
top = bc.top(10)

chart = ag.charts.bar_chart(
    [campus.nodes[n]["name"] for n, _ in top],
    [v for _, v in top],
    highlight=[campus.nodes["PF"]["name"], campus.nodes["KNB"]["name"]],
    value_format=lambda v: f"{v:.2f}",
    title="Buildings on the most shortest routes",
    subtitle="Betweenness centrality by link length, top 10 of 56",
    max_label_width=260,
    width=640,
)
chart.save("betweenness_bars.svg")
```

```{figure} ../_static/generated/guide_layout_algorithms/betweenness_bars.png
:alt: Horizontal bar chart of the ten campus buildings with the highest betweenness, with the top two highlighted
:width: 90%

The ten campus buildings with the highest betweenness, with link lengths in meters as distances.
```

Pass `sort=False` to keep your own order. Labels longer than `max_label_width` pixels are shortened with an ellipsis.

## Histograms

{py:func}`aryagraph.charts.histogram` counts values into columns. Integer data such as degrees get one column per value when the range is 60 or less; other data are binned (`bins="auto"`, a number of bins, or explicit edges). `markers` adds reference lines and `log_y=True` a logarithmic count axis for heavy-tailed data.

This histogram shows the shortest-path length between every pair of campus buildings, in 125 m bins. Each campus link has the straight-line distance between its two buildings as its length, so these values are network distances, not measured walking routes.

```python
dist = ag.alg.all_pairs_shortest_path_length(campus, weight="length")
lengths = [d for u, row in dist.items() for v, d in row.items() if u < v]
median = float(np.median(lengths))
print(len(lengths), "pairs; median", round(median), "m")

chart = ag.charts.histogram(
    lengths,
    bins=np.arange(0, 2251, 125),                 # explicit bin edges, 125 m apart
    markers=[(median, f"median {median:,.0f} m")],
    title="How far apart are campus buildings?",
    subtitle="Shortest-path length through the network for every pair of buildings",
    x_label="meters",
    y_label="pairs",
)
chart.save("walk_histogram.svg")
```

```text
1540 pairs; median 607 m
```

```{figure} ../_static/generated/guide_layout_algorithms/walk_histogram.png
:alt: Histogram of the shortest-path lengths between all pairs of campus buildings in 125 meter bins, with a marker at the median of 607 meters
:width: 90%

Shortest-path lengths through the campus network for all 1,540 pairs of buildings.
```

## Gantt charts

{py:func}`aryagraph.charts.gantt` draws schedules. It accepts a simulated schedule, or any list of task runs given as dicts (or objects) with `node`, `start` and `end`, plus optional `worker`, `status` and `attempt`. `lanes="task"` gives one row per task; `lanes="worker"` gives one row per worker.

The critical-path method gives each task its earliest start. Built into rows, it becomes a project plan. The durations of the bundled house plan are working hours (40 h is one week):

```python
plan = ag.gen.project_plan()
cp = ag.alg.critical_path(plan)
rows = [
    {"node": task, "start": cp.earliest_start[task], "end": cp.earliest_finish[task]}
    for task in ag.alg.topological_sort(plan)
]
chart = ag.charts.gantt(
    rows,
    critical=cp.path,
    title="House construction: earliest-start schedule",
    subtitle=f"critical path {cp.length:g} h, unlimited crews",
    time_label="working hours",
)
chart.save("plan_gantt.svg")
```

```{figure} ../_static/generated/guide_layout_algorithms/plan_gantt.png
:alt: Gantt chart of the 18 house-construction tasks at their earliest start times, with the critical path in the accent color
:width: 100%

The earliest-start schedule of the house plan. Tasks on the critical path are in the accent color.
```

A {py:func}`aryagraph.sim.simulate_schedule` result has a `gantt()` method with the same options. With `lanes="worker"`, it shows what each crew did and when it waited:

```python
sched = ag.sim.simulate_schedule(plan, workers=3, policy="critical_path", seed=3)
print(sched)
chart = sched.gantt(lanes="worker", title="House construction with 3 crews",
                    time_label="working hours")
chart.save("crews_gantt.svg")
```

```text
<ScheduleResult: 18 tasks, workers=3, makespan=604, utilization=42.6%, 18 done, 0 failed>
```

```{figure} ../_static/generated/guide_layout_algorithms/crews_gantt.png
:alt: Gantt chart with one row per crew showing the tasks each of the three crews ran
:width: 100%

The same plan run by three crews, one row per crew.
```

With three crews and fixed durations, the plan takes 604 working hours instead of the 572 hours of the critical path, because some tasks wait for a free crew.

Without `color_by`, critical tasks take the accent color and the others are muted; `color_by="team"` colors bars by a node attribute instead. Failed attempts are drawn in the theme's alert color. The [DAG workflows](dags.md) page covers scheduling policies, failures and resources.

## Simulation plots

Simulation results plot themselves with these charts:

| result | method | chart |
|---|---|---|
| `SimulationResult` | `plot()` | state counts over time (`fractions=True` for shares) |
| `EnsembleResult` | `plot()` | mean per state with 25–75% and 5–95% bands |
| `MonteCarloResult` | `plot()` | histogram of completion times with P50, P80 and P95 markers |
| `ScheduleResult` | `gantt()` | the executed schedule |

```python
g = ag.gen.watts_strogatz(300, 6, 0.08, seed=1)
sir = ag.sim.SIR(beta=0.35, gamma=0.1)

run = sir.simulate(g, initial={"I": 3}, t_max=60, method="gillespie", seed=7)
print("peak (time, infected):", run.peak("I"))
run.plot(title="SIR epidemic, one run").save("sir_run.svg")

ensemble = ag.sim.run_ensemble(sir.simulate, runs=100, seed=1,
                               g=g, initial={"I": 3}, t_max=60, method="gillespie")
print(ensemble)
ensemble.plot(title="SIR epidemic, 100 runs").save("sir_ensemble.html")
```

```text
peak (time, infected): (6.0, 204)
<EnsembleResult SIR: 100 runs, 300 nodes; final S=0.05±0.261, I=1.26±1.04, R=298.7±1.05>
```

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/generated/guide_layout_algorithms/sir_run.png
:alt: Line chart of susceptible, infected and recovered counts over time in one SIR run
One run.
```
:::

:::{grid-item}
```{figure} ../_static/generated/guide_layout_algorithms/sir_ensemble.png
:alt: Mean SIR curves over 100 runs with shaded 25-75 and 5-95 percent bands
100 runs: mean, 25–75% and 5–95% bands.
```
:::
::::

Line charts saved as HTML get a crosshair: hover over the plot to read every series at the nearest x value. Try it on the ensemble:

<iframe class="ag-embed" src="../_static/generated/guide_layout_algorithms/sir_ensemble.html" height="380" loading="lazy" title="Interactive line chart of an SIR ensemble with a crosshair tooltip"></iframe>
<p class="ag-embed-note">Hover to read the mean counts at any time. <a href="../_static/generated/guide_layout_algorithms/sir_ensemble.html">Open full screen</a></p>

The Monte Carlo plot of a project's duration puts the percentiles on the histogram:

```python
mc = ag.sim.monte_carlo_schedule(plan, runs=2000, seed=1)
print(mc)
note = f"2,000 PERT simulations, makespan in working hours · deterministic plan: {mc.cpm_length:g} h"
mc.plot(subtitle=note).save("montecarlo.svg")
```

```text
<MonteCarloResult: 2000 runs, makespan 609.4 ± 36.2; P50=607.1, P80=639.6, P95=673>
```

```{figure} ../_static/generated/guide_layout_algorithms/montecarlo.png
:alt: Histogram of 2,000 simulated durations of the house project with P50, P80 and P95 lines
:width: 90%

Simulated completion times of the house plan in working hours, with PERT durations for every task.
```

The [Simulation](simulation.md) page covers the models behind these results.

## Saving and styling charts

| call | result |
|---|---|
| `chart.save("x.svg")` | standalone SVG |
| `chart.save("x.png", scale=2.0)` | PNG at twice the pixel size (cairosvg, or a Chrome, Edge or Chromium browser) |
| `chart.save("x.pdf")` | vector PDF |
| `chart.save("x.html")` | self-contained page; line charts get the crosshair tooltip |
| `chart.to_svg()`, `chart.to_html()` | the markup as a string |

Every chart function takes `theme=`, with the same themes as drawings (`"light"`, `"dark"`, `"paper"`, `"blueprint"` or a custom {py:class}`aryagraph.style.themes.Theme`), and `width=` in pixels. The height of bar and Gantt charts follows from the number of rows (`bar`, `gap` and `row` set the row sizes):

```python
dark = ag.charts.bar_chart(
    [n for n, _ in top], [v for _, v in top],
    theme="dark", width=420,
    value_format=lambda v: f"{v:.2f}", title="Top 10 by betweenness",
)
print(dark, dark.kind)
dark.save("betweenness_dark.svg")
```

```text
<Chart bar 420×344> bar
```

The charts follow a few fixed rules so that they read the same everywhere: hairline gridlines, one y axis, a legend whenever there are two or more series, bars at most 24 pixels thick, and text in ink colors while the color keys carry the series identity. See [Styling](styling.md) for themes and palettes.
