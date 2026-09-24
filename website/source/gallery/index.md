# Gallery

A visual tour of what AryaGraph draws: layouts, encodings, edge styles, themes, right-to-left labels, charts, a
simulation and an analysis dashboard. Every image was produced by the code on its card; open **Code** to see it.

Each snippet is self-contained and saves an SVG in your working directory. Change the extension to `.png`, `.pdf` or
`.html` to get another format (see [Exporting](../user-guide/exporting.md)). When this website is built, a script
runs these snippets as printed and converts each SVG to the PNG you see here, so the pictures and the code stay in
step. All randomness is seeded, so with AryaGraph {{ version }} your output matches the images. Click an image to open
it at full size.

## Layouts

Pass a method name as `layout=` to {py:func}`aryagraph.render.draw`, and engine options as `layout_options=`. The
[Layouts](../user-guide/layouts.md) guide explains when to use each one.

::::::{grid} 1 1 2 2
:gutter: 3

:::::{grid-item-card} Hierarchical
::::{figure} ../_static/generated/gallery_faq/layout_hierarchical.png
:target: ../_static/generated/gallery_faq/layout_hierarchical.png
:alt: A data-warehouse ETL drawn in layers from left to right, extract tasks on the left and publishing tasks on the right, boxes colored by task kind

The Sugiyama pipeline layers a DAG, reduces crossings and routes long edges between the layers. Here the nightly ETL
runs left to right with `orientation="LR"`.
::::
::::{dropdown} Code
```python
import aryagraph as ag

etl = ag.gen.data_warehouse_etl()
fig = ag.draw(
    etl,
    layout="hierarchical",
    layout_options={"orientation": "LR"},
    node_color="kind",
    title="Nightly data-warehouse ETL",
    subtitle=f"{len(etl)} tasks, {etl.num_edges} dependencies, layered left to right",
)
fig.save("layout_hierarchical.svg")
```
::::
:::::

:::::{grid-item-card} Tree
::::{figure} ../_static/generated/gallery_faq/layout_tree.png
:target: ../_static/generated/gallery_faq/layout_tree.png
:alt: A tidy tree rooted at the MacEwan Student Centre with every University of Calgary building as a labeled box, colored by building kind

The breadth-first tree of the campus network, rooted at the MacEwan Student Centre (MSC), drawn by the tidy-tree
algorithm. The deepest building is 9 links from MSC.
::::
::::{dropdown} Code
```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
tree = ag.alg.bfs_tree(campus, "MSC")
fig = ag.draw(
    tree,
    layout="tree",
    node_color="kind",
    title="Breadth-first tree of the University of Calgary campus",
    subtitle="Every building, reached from the MacEwan Student Centre (MSC)",
)
fig.save("layout_tree.svg")
```
::::
:::::

:::::{grid-item-card} Radial
::::{figure} ../_static/generated/gallery_faq/layout_radial.png
:target: ../_static/generated/gallery_faq/layout_radial.png
:alt: A balanced ternary tree drawn as concentric rings around its root, nodes shaded from light to dark blue by depth

A radial tidy tree puts the root in the center and each depth on its own ring. Every subtree gets a wedge in
proportion to its leaves, so subtrees do not interleave. This balanced ternary tree has 121 nodes.
::::
::::{dropdown} Code
```python
import aryagraph as ag

tree = ag.gen.balanced_tree(3, 4)
depth = ag.alg.shortest_path_length(tree, 0)
fig = ag.draw(
    tree,
    layout="radial",
    node_color=ag.by(depth, title="depth"),
    labels=False,
    title="Radial tidy tree",
    subtitle=f"Balanced ternary tree: {len(tree)} nodes on four rings around the root",
)
fig.save("layout_radial.svg")
```
::::
:::::

:::::{grid-item-card} Stress
::::{figure} ../_static/generated/gallery_faq/layout_stress.png
:target: ../_static/generated/gallery_faq/layout_stress.png
:alt: The Les Misérables co-appearance network in a stress layout, characters colored by their Louvain community and sized by weighted degree

Stress majorization places nodes so that drawn distances follow graph distances; it is the default for general
graphs of up to 3,000 nodes. The 77 characters of *Les Misérables* fall into 6 Louvain communities (seed 0).
::::
::::{dropdown} Code
```python
import aryagraph as ag

les = ag.gen.les_miserables()
communities = ag.alg.louvain_communities(les, seed=0)
fig = ag.draw(
    les,
    layout="stress",
    node_color=ag.by(ag.alg.community_labels(communities), kind="categorical", title="community"),
    node_size=les.degree(weight="weight"),
    title="Les Misérables: stress majorization",
    subtitle=f"{len(les)} characters, {les.num_edges} co-appearances, {len(communities)} communities",
)
fig.save("layout_stress.svg")
```
::::
:::::

:::::{grid-item-card} ForceAtlas2
::::{figure} ../_static/generated/gallery_faq/layout_forceatlas2.png
:target: ../_static/generated/gallery_faq/layout_forceatlas2.png
:alt: Four dense clusters of 40 nodes each, colored by planted group, pulled apart by the ForceAtlas2 layout

ForceAtlas2 separates dense groups clearly and switches to Barnes–Hut approximation above 1,500 nodes. The planted
partition has 4 groups of 40 nodes and 858 edges.
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.planted_partition(4, 40, 0.25, 0.01, seed=3)
fig = ag.draw(
    g,
    layout="forceatlas2",
    node_color="block",
    labels=False,
    title="ForceAtlas2",
    subtitle=f"Planted partition: 4 groups of 40 nodes, {g.num_edges} edges",
)
fig.save("layout_forceatlas2.svg")
```
::::
:::::

:::::{grid-item-card} Circular
::::{figure} ../_static/generated/gallery_faq/layout_circular.png
:target: ../_static/generated/gallery_faq/layout_circular.png
:alt: Thirty numbered nodes on a circle; most links join near neighbors on the ring and a few long chords cross it

In graph order, a circular layout shows the structure of a Watts–Strogatz small world: a ring lattice plus a few
rewired shortcuts. The four ends of the two longest chords, 2–19 and 18–29, have the highest closeness: 0.39 for
node 19 and 0.38 for nodes 2, 18 and 29.
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.watts_strogatz(30, 4, 0.1, seed=3)
fig = ag.draw(
    g,
    layout="circular",
    node_color=ag.by(ag.alg.closeness_centrality(g), title="closeness"),
    title="Circular layout of a small world",
    subtitle="Watts–Strogatz ring lattice (k = 4), rewiring probability 0.1",
)
fig.save("layout_circular.svg")
```
::::
:::::

:::::{grid-item-card} Bipartite
::::{figure} ../_static/generated/gallery_faq/layout_bipartite.png
:target: ../_static/generated/gallery_faq/layout_bipartite.png
:alt: Two columns, eighteen women on the left and fourteen social events on the right, with attendance links between them

Two columns with barycenter ordering, which removes most crossings. Davis's Southern Women links 18 women to the
14 events they attended (89 attendances). `label_position="left"` keeps the names clear of the edges, and `width=`
spreads the columns apart.
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.davis_southern_women()
side = {n: "event" if d["bipartite"] else "woman" for n, d in g.nodes.data()}
fig = ag.draw(
    g,
    layout="bipartite",
    layout_options={"top": g.attrs["top"]},
    node_color=ag.by(side, title="side"),
    label_position="left",
    width=560,
    title="Davis's Southern Women",
    subtitle=f"{len(g.attrs['top'])} women, {len(g.attrs['bottom'])} events, {g.num_edges} attendances",
)
fig.save("layout_bipartite.svg")
```
::::
:::::

:::::{grid-item-card} Arc
::::{figure} ../_static/generated/gallery_faq/layout_arc.png
:target: ../_static/generated/gallery_faq/layout_arc.png
:alt: The Les Misérables characters on one horizontal line, joined by arcs whose thickness shows how many chapters two characters share

An arc diagram puts the nodes on one line. `edge_style="curved"` with a large curvature draws each link as an arc
above the line, and `avoid_overlap=False` keeps every node on the baseline.
::::
::::{dropdown} Code
```python
import aryagraph as ag

les = ag.gen.les_miserables()
communities = ag.alg.community_labels(ag.alg.louvain_communities(les, seed=0))
fig = ag.draw(
    les,
    layout="arc",
    edge_style="curved",
    curvature=0.5,
    avoid_overlap=False,
    labels=False,
    node_color=ag.by(communities, kind="categorical", title="community"),
    edge_width="weight",
    title="Les Misérables as an arc diagram",
    subtitle="Characters in dataset order; arc width counts the chapters two characters share",
)
fig.save("layout_arc.svg")
```
::::
:::::

::::::

## Data-driven encodings

Every visual channel accepts a constant, an attribute name, a `{node: value}` mapping such as an algorithm result, a
callable, or an {py:func}`aryagraph.style.scales.by` spec. Legends follow from the data. See
[Styling](../user-guide/styling.md) for the scales.

::::::{grid} 1 1 2 2
:gutter: 3

:::::{grid-item-card} Color, size and edge encodings
::::{figure} ../_static/generated/gallery_faq/encoding_campus.png
:target: ../_static/generated/gallery_faq/encoding_campus.png
:alt: Map-like drawing of 56 University of Calgary buildings at their real positions, colored by kind and sized by betweenness, with indoor links drawn darker

The University of Calgary campus at its real geometry: color shows the building kind, size shows betweenness with
link length as distance, and dark links are indoor connections. Building positions © OpenStreetMap contributors
(ODbL).
::::
::::{dropdown} Code
```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
indoor = {"tunnel", "pedway", "attached"}
fig = ag.draw(
    campus,
    layout={n: d["pos"] for n, d in campus.nodes.data()},
    node_color="kind",
    node_size=ag.by(ag.alg.betweenness_centrality(campus, weight="length"), title="betweenness"),
    edge_color=lambda u, v, d: "#52514e" if d["kind"] in indoor else "#c3c2b7",
    edge_width=ag.by(lambda u, v, d: 2.0 if d["kind"] in indoor else 1.0, kind="identity"),
    title="University of Calgary main campus",
    subtitle="Color: building kind · size: betweenness · dark links: tunnels, pedways and attached buildings",
    caption="Building positions © OpenStreetMap contributors (ODbL)",
)
fig.save("encoding_campus.svg")
```
::::
:::::

:::::{grid-item-card} Color and shape
::::{figure} ../_static/generated/gallery_faq/encoding_shapes.png
:target: ../_static/generated/gallery_faq/encoding_shapes.png
:alt: A machine-learning pipeline in layers, each step drawn with a shape for its kind and a color for its team

Two categorical channels on one DAG: color for the team that owns a step and shape for the kind of step. Each channel
gets its own legend.
::::
::::{dropdown} Code
```python
import aryagraph as ag

pipeline = ag.gen.ml_pipeline()
fig = ag.draw(
    pipeline,
    node_color="team",
    node_shape="kind",
    title="Machine-learning pipeline",
    subtitle="Color encodes the owning team, shape the kind of step",
)
fig.save("encoding_shapes.svg")
```
::::
:::::

::::::

## Edge styles

`edge_style=` accepts `"straight"`, `"curved"`, `"flow"`, `"orthogonal"` and `"spline"`; `"auto"` picks flow for
hierarchies and straight lines otherwise.

::::::{grid} 1 1 2 3
:gutter: 3

:::::{grid-item-card} Flow
::::{figure} ../_static/generated/gallery_faq/edge_flow.png
:target: ../_static/generated/gallery_faq/edge_flow.png
:alt: A software build graph in layers with smooth S-shaped edges; the critical path is highlighted in blue and the rest is dimmed

Smooth S-curves along the layers, with ports spread across each node side. The critical path of the software build
(1.57 h through 9 of the 20 targets) is highlighted.
::::
::::{dropdown} Code
```python
import aryagraph as ag

build = ag.gen.software_build()
cp = ag.alg.critical_path(build)
fig = ag.draw(
    build,
    edge_style="flow",
    node_color="team",
    highlight_path=cp.path,
    title="Software build: flow edges",
    subtitle=f"Critical path highlighted: {cp.length:g} h from fetch to release",
)
fig.save("edge_flow.svg")
```
::::
:::::

:::::{grid-item-card} Orthogonal
::::{figure} ../_static/generated/gallery_faq/edge_orthogonal.png
:target: ../_static/generated/gallery_faq/edge_orthogonal.png
:alt: Course prerequisites in layers connected by right-angled edges with rounded corners

Right-angle routing with rounded corners, a common choice for technical diagrams. The course catalog has 22 courses
and 31 prerequisite links.
::::
::::{dropdown} Code
```python
import aryagraph as ag

courses = ag.gen.course_prerequisites()
fig = ag.draw(
    courses,
    edge_style="orthogonal",
    node_color="department",
    title="Course prerequisites: orthogonal edges",
    subtitle=f"{len(courses)} courses, {courses.num_edges} prerequisite links",
)
fig.save("edge_orthogonal.svg")
```
::::
:::::

:::::{grid-item-card} Curved
::::{figure} ../_static/generated/gallery_faq/edge_curved.png
:target: ../_static/generated/gallery_faq/edge_curved.png
:alt: A small random directed graph drawn with curved arrows; nodes are shaded by PageRank

Each arc bends to its left, so the two arcs of a reciprocal pair separate instead of overlapping. Node color is
PageRank.
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.gnm_random_graph(14, 32, directed=True, seed=11)
pairs = sum(1 for u, v in g.edges if g.has_edge(v, u)) // 2
fig = ag.draw(
    g,
    edge_style="curved",
    node_color=ag.by(ag.alg.pagerank(g), title="PageRank"),
    width=640,
    title="Curved edges",
    subtitle=f"Random directed graph: {g.num_edges} arcs, {pairs} reciprocal pairs",
)
fig.save("edge_curved.svg")
```
::::
:::::

::::::

## Themes

The same drawing in the four built-in themes. A theme sets every color, font and size by role; the dark themes use
their own tuned color steps. You can derive your own with {py:meth}`aryagraph.style.themes.Theme.with_`.

::::::{grid} 1 1 2 2
:gutter: 3

:::::{grid-item-card} Light
::::{figure} ../_static/generated/gallery_faq/theme_light.png
:target: ../_static/generated/gallery_faq/theme_light.png
:alt: Florentine families network on a light background, nodes shaded blue by betweenness and sized by degree

`theme="light"`, the default. The Medici family has the highest betweenness among the 15 families (0.52).
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.florentine_families()
fig = ag.draw(
    g,
    node_color=ag.by(ag.alg.betweenness_centrality(g), title="betweenness"),
    node_size=g.degree(),
    theme="light",
    title="Florentine families",
    subtitle='theme="light"',
)
fig.save("theme_light.svg")
```
::::
:::::

:::::{grid-item-card} Dark
::::{figure} ../_static/generated/gallery_faq/theme_dark.png
:target: ../_static/generated/gallery_faq/theme_dark.png
:alt: Florentine families network on a near-black background with light labels

`theme="dark"` for dark slides and dashboards.
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.florentine_families()
fig = ag.draw(
    g,
    node_color=ag.by(ag.alg.betweenness_centrality(g), title="betweenness"),
    node_size=g.degree(),
    theme="dark",
    title="Florentine families",
    subtitle='theme="dark"',
)
fig.save("theme_dark.svg")
```
::::
:::::

:::::{grid-item-card} Paper
::::{figure} ../_static/generated/gallery_faq/theme_paper.png
:target: ../_static/generated/gallery_faq/theme_paper.png
:alt: Florentine families network on a pure white background with black text

`theme="paper"`: a white surface, black ink and slightly darker edges for print and journals.
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.florentine_families()
fig = ag.draw(
    g,
    node_color=ag.by(ag.alg.betweenness_centrality(g), title="betweenness"),
    node_size=g.degree(),
    theme="paper",
    title="Florentine families",
    subtitle='theme="paper"',
)
fig.save("theme_paper.svg")
```
::::
:::::

:::::{grid-item-card} Blueprint
::::{figure} ../_static/generated/gallery_faq/theme_blueprint.png
:target: ../_static/generated/gallery_faq/theme_blueprint.png
:alt: Florentine families network on a deep blue background with pale blue edges and text

`theme="blueprint"`: a navy surface with pale blue ink.
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.florentine_families()
fig = ag.draw(
    g,
    node_color=ag.by(ag.alg.betweenness_centrality(g), title="betweenness"),
    node_size=g.degree(),
    theme="blueprint",
    title="Florentine families",
    subtitle='theme="blueprint"',
)
fig.save("theme_blueprint.svg")
```
::::
:::::

::::::

## Right-to-left labels

::::::{grid} 1 1 2 2
:gutter: 3

:::::{grid-item-card} Persian labels
::::{figure} ../_static/generated/gallery_faq/persian.png
:target: ../_static/generated/gallery_faq/persian.png
:alt: A layered prerequisite graph whose title, subtitle and node labels are written in Persian

Node labels, title and subtitle in Persian. Text width is estimated per script so boxes fit their labels, and
right-to-left text is written with the correct direction. The title reads "Course prerequisites" and the subtitle
"Layered layout with Persian labels". More in [Languages](../user-guide/languages.md).
::::
::::{dropdown} Code
```python
import aryagraph as ag

courses = ag.DAG(name="برنامه‌ی درسی")
courses.add_edges(
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
fig = ag.draw(courses, title="پیش‌نیازهای درسی", subtitle="چیدمان لایه‌ای با برچسب‌های فارسی")
fig.save("persian.svg")
```
::::
:::::

::::::

## Charts

The `ag.charts` functions {py:func}`~aryagraph.charts.line_chart`, {py:func}`~aryagraph.charts.bar_chart`,
{py:func}`~aryagraph.charts.histogram` and {py:func}`~aryagraph.charts.gantt` draw in the same visual language as the
graphs, and their HTML versions add a crosshair tooltip. See [Charts](../user-guide/charts.md).

::::::{grid} 1 1 2 2
:gutter: 3

:::::{grid-item-card} Line chart with uncertainty bands
::::{figure} ../_static/generated/gallery_faq/chart_line.png
:target: ../_static/generated/gallery_faq/chart_line.png
:alt: Line chart of infected and recovered counts over time, each line surrounded by shaded bands showing the spread across simulation runs

Mean infected and recovered counts across 100 seeded SIR runs on a 300-node small world, with 25–75% and 5–95%
bands. The mean number infected peaks at 179.4 nodes at time 12.
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.watts_strogatz(300, 6, 0.08, seed=1)
ens = ag.sim.run_ensemble(
    ag.sim.SIR(beta=0.35, gamma=0.1).simulate, runs=100, g=g, initial={"I": 3}, t_max=60, seed=7
)
fig = ag.charts.line_chart(
    ens.times,
    {"infected": ens.mean["I"], "recovered": ens.mean["R"]},
    bands={"infected": ens.band("I", 25, 75), "recovered": ens.band("R", 25, 75)},
    outer_bands={"infected": ens.band("I", 5, 95), "recovered": ens.band("R", 5, 95)},
    band_label="25–75% and 5–95% of runs",
    title=f"SIR epidemic: mean of {ens.runs} seeded runs",
    x_label="time",
    y_label="nodes",
)
fig.save("chart_line.svg")
```
::::
:::::

:::::{grid-item-card} Ranked bar chart
::::{figure} ../_static/generated/gallery_faq/chart_bar.png
:target: ../_static/generated/gallery_faq/chart_bar.png
:alt: Horizontal bar chart of the ten campus buildings with the highest betweenness, Professional Faculties first

The ten campus buildings with the highest betweenness when link length is the distance. Professional Faculties
(0.31) and the Dr. Roger Jackson Kinesiology Complex, Block B (0.30) lead. Link lengths are straight-line distances
between building positions © OpenStreetMap contributors (ODbL).
::::
::::{dropdown} Code
```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
top = ag.alg.betweenness_centrality(campus, weight="length").top(10)
fig = ag.charts.bar_chart(
    [campus.nodes[n]["name"] for n, _ in top],
    [value for _, value in top],
    value_format=lambda v: f"{v:.2f}",
    max_label_width=300,
    width=680,
    title="Campus buildings on the most shortest paths",
    subtitle="Betweenness centrality with link length (meters) as distance",
)
fig.save("chart_bar.svg")
```
::::
:::::

:::::{grid-item-card} Histogram with reference lines
::::{figure} ../_static/generated/gallery_faq/chart_histogram.png
:target: ../_static/generated/gallery_faq/chart_histogram.png
:alt: Histogram of simulated project durations with vertical reference lines for the critical-path length and the 50th, 80th and 95th percentiles

Durations of a house-construction plan in 2,000 PERT simulations. The critical path says 572 working hours; the
simulated median is 607 h and the 95th percentile 673 h. These numbers leave out the plan's two lags (72 h of
concrete curing and 24 h of drying); the [construction case study](../case-studies/construction-schedule.md) adds
them as wait tasks, which puts the plan at 668 h.
::::
::::{dropdown} Code
```python
import aryagraph as ag

plan = ag.gen.project_plan()
mc = ag.sim.monte_carlo_schedule(plan, runs=2000, seed=1)
markers = [(mc.cpm_length, "CPM")] + [(mc.percentiles[k], k) for k in ("P50", "P80", "P95")]
fig = ag.charts.histogram(
    mc.makespans,
    discrete=False,
    markers=markers,
    title=f"House construction: {mc.runs:,} PERT simulations",
    x_label="project duration (working hours)",
    y_label="runs",
)
fig.save("chart_histogram.svg")
```
::::
:::::

:::::{grid-item-card} Gantt chart
::::{figure} ../_static/generated/gallery_faq/chart_gantt.png
:target: ../_static/generated/gallery_faq/chart_gantt.png
:alt: Gantt chart of eighteen construction tasks with the critical path in blue, other tasks in gray and three failed attempts in red

The same plan, again without the lags, executed by 3 crews with an 8% chance that an attempt fails and up to 2
retries. In this seeded run, 3 attempts fail and the project takes 676 h.
::::
::::{dropdown} Code
```python
import aryagraph as ag

plan = ag.gen.project_plan()
run = ag.sim.simulate_schedule(plan, workers=3, failure_rate=0.08, max_retries=2, seed=6)
fig = ag.charts.gantt(
    run,
    lanes="task",
    title="House construction with 3 crews",
    subtitle=f"8% failure risk per attempt, up to 2 retries · makespan {run.makespan:g} h",
    time_label="working hours",
)
fig.save("chart_gantt.svg")
```
::::
:::::

::::::

## Simulation

Every simulator returns a result that plots and animates. {py:func}`aryagraph.render.animate.animate` draws the graph
once and recolors its nodes frame by frame; see [Simulation](../user-guide/simulation.md).

::::::{grid} 1 1 2 2
:gutter: 3

:::::{grid-item-card} Epidemic on a spatial network
::::{figure} ../_static/generated/gallery_faq/sim_sir.png
:target: ../_static/generated/gallery_faq/sim_sir.png
:alt: Four hundred points joined to their near neighbors; a band of red infected nodes spreads outward from a region of green recovered nodes, with gray susceptible nodes beyond

One frame of an SIR epidemic on a random geometric graph of 400 nodes, a quarter of the way through the run. Infection
moves outward as a front; the number infected peaks at 217 nodes at time 21.6.
::::
::::{dropdown} Code
```python
import aryagraph as ag

g = ag.gen.random_geometric(400, 0.08, seed=4)
run = ag.sim.SIR(beta=0.3, gamma=0.1).simulate(
    g, initial={"I": 1}, t_max=80, method="gillespie", seed=2
)
fig = run.animate(
    layout={n: d["pos"] for n, d in g.nodes.data()},
    labels=False,
    static_frame=len(run.times) // 4,
    title="SIR epidemic on a random geometric graph",
)
fig.save("sim_sir.svg")   # the frame chosen by static_frame
fig.save("sim_sir.html")  # the interactive player
```
::::
:::::

::::::

The player below is the `sim_sir.html` file saved by the snippet. Press **Play**, drag the slider, or drag across the
chart to scrub through time.

<div style="container-type: inline-size">
<iframe class="ag-embed" src="../_static/generated/gallery_faq/sim_sir.html" height="1330" style="height: calc(83cqw + 560px)" loading="lazy" title="Interactive player for the SIR epidemic on a random geometric graph"></iframe>
</div>
<p class="ag-embed-note">A self-contained HTML file: it needs no server or network connection. <a href="../_static/generated/gallery_faq/sim_sir.html">Open full screen</a></p>

## Analysis dashboard

{py:func}`aryagraph.analysis.report.analyze` computes the measures that apply to a graph and saves them as an
interactive dashboard of stat tiles, the explorable graph, distributions and rankings. See
[Analysis reports](../user-guide/analysis.md).

::::::{grid} 1 1 2 2
:gutter: 3

:::::{grid-item-card} Les Misérables report
::::{figure} ../_static/generated/gallery_faq/dashboard_lesmis.png
:target: ../_static/generated/gallery_faq/dashboard_lesmis.png
:alt: Screenshot of the analysis dashboard: a row of stat tiles above an interactive network drawing with a community legend

The top of the dashboard for *Les Misérables*: 77 nodes, 254 edges, diameter 5 and 6 communities with modularity
0.567. The screenshot was taken from the saved HTML file with headless Chrome.
::::
::::{dropdown} Code
```python
import aryagraph as ag

report = ag.analyze(ag.gen.les_miserables())
report.save("dashboard_lesmis.html")
```
::::
:::::

::::::

<iframe class="ag-embed" src="../_static/generated/gallery_faq/dashboard_lesmis.html" height="800" loading="lazy" title="Interactive analysis dashboard for the Les Misérables network"></iframe>
<p class="ag-embed-note">The complete dashboard, as saved by the snippet above. <a href="../_static/generated/gallery_faq/dashboard_lesmis.html">Open full screen</a></p>
