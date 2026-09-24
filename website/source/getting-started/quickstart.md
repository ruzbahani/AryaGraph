# Quickstart

This ten-minute tour covers the main parts of AryaGraph on one real dataset, the University of Calgary main campus. You will load the network, query it, draw it on its real geometry, export the drawing, summarize the graph in one call, plan a small project as a DAG and run a simulation.

Before you start, [install AryaGraph](installation.md). Run the code blocks in order, in a Python session or a Jupyter notebook: each block uses the variables created by the blocks before it. Every output on this page was produced by the code shown above it.

## Load the campus network

```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
print(campus)
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
```

{py:func}`~aryagraph.generators.datasets.ucalgary_campus` returns a new undirected {py:class}`~aryagraph.core.graph.Graph` built from data bundled with the package, so nothing is downloaded. Each node is a building, named by its official code (`MSC`, `TFDL`, `ICT`, …). Each edge is a way between two buildings: an underground tunnel, an enclosed pedway, an internal door between attached buildings, or a modeled outdoor walking link.

Building positions come from OpenStreetMap (© OpenStreetMap contributors, ODbL). The dataset is a schematic model compiled from public sources: every edge length is the straight-line distance between two buildings, not a surveyed walking distance.

## Inspect nodes, edges and attributes

A graph behaves like a collection of nodes. `campus.nodes[n]` and `campus.edges[u, v]` return the attribute dictionaries stored on a node or an edge:

```python
print(list(campus)[:8])
print(campus.nodes["TFDL"])
print(campus.edges["OO", "KNB"])
print(campus.degree("SS"))
```

```text
['AB', 'AD', 'AU', 'BI', 'CC', 'CCIT', 'CD', 'CDC']
{'name': 'Taylor Family Digital Library', 'kind': 'library', 'lat': 51.0774272, 'lon': -114.1299715, 'pos': (98.8, 51.8)}
{'kind': 'attached', 'length': 169.8, 'weight': 169.8}
4
```

Nodes come back in the order they were added, which for this dataset is alphabetical by code. Each building carries its `name`, its `kind`, its latitude and longitude, and `pos`, a position in meters on a local plane centered on the campus. Each edge carries its `kind` and its `length` in meters (`weight` repeats the length for algorithms that look for a `weight` attribute).

`nodes.data(key)` and `edges.data(key)` iterate over one attribute at a time, which makes quick tallies easy:

```python
from collections import Counter

print(Counter(kind for _, _, kind in campus.edges.data("kind")))
print(Counter(kind for _, kind in campus.nodes.data("kind")).most_common(3))
```

```text
Counter({'outdoor': 43, 'attached': 26, 'pedway': 11, 'tunnel': 3})
[('academic', 25), ('residence', 9), ('services', 5)]
```

Of the 83 edges, 40 are indoor links (tunnels, pedways and attached buildings) and 43 are outdoor walks.

## Find the shortest walk

How far is it from the Olympic Oval (`OO`) to Scurfield Hall (`SH`)? {py:func}`~aryagraph.algorithms.paths.shortest_path` answers with a list of nodes:

```python
route = ag.alg.shortest_path(campus, "OO", "SH", weight="length")
meters = ag.alg.shortest_path_length(campus, "OO", "SH", weight="length")
print(" → ".join(route))
print(f"{meters:,.0f} m in {len(route) - 1} links")
print(Counter(campus.edges[u, v]["kind"] for u, v in zip(route, route[1:])))
```

```text
OO → KNB → KNA → IH → RC → RT → CH → MFH → PF → EDT → SH
1,002 m in 10 links
Counter({'attached': 4, 'outdoor': 3, 'pedway': 3})
```

`weight="length"` tells the algorithm to add up the `length` attribute, so the route minimizes meters. Without `weight`, every edge counts as 1 and the route minimizes the number of links. For this pair both criteria pick the same route:

```python
print(ag.alg.shortest_path(campus, "OO", "SH") == route)
```

```text
True
```

Seven of the ten links on the route are indoors. If you would rather rank routes by comfort than by distance, `weight` also accepts a function of the edge; the [Concepts](concepts.md#algorithms-are-functions) page shows one that penalizes outdoor links.

## Find the buildings that hold the campus together

Betweenness centrality measures how often a building lies on the shortest walks between other buildings:

```python
bc = ag.alg.betweenness_centrality(campus, weight="length")
print(bc)
for code, score in bc.top(5):
    print(f"{code:4} {score:.3f}  {campus.nodes[code]['name']}")
```

```text
NodeMap('betweenness_centrality', 56 nodes; top: {'PF': 0.3071, 'KNB': 0.301, 'IH': 0.2175, …})
PF   0.307  Professional Faculties
KNB  0.301  Dr. Roger Jackson Kinesiology Complex (Block B)
IH   0.218  International House
RC   0.209  Rozsa Centre
MFH  0.207  Murray Fraser Hall
```

The result is a {py:class}`~aryagraph.core.results.NodeMap`: a regular `dict` from node to score, with helpers such as `top()`, `rank()` and `describe()`. Professional Faculties lies on about 31% of the length-weighted shortest walks between pairs of other buildings. All five also lie on the Olympic Oval route above, which crosses the middle of campus.

## Draw the campus on its real geometry

{py:func}`~aryagraph.render.draw` turns a graph into a {py:class}`~aryagraph.render.figure.Figure`. Here the layout is the buildings' own positions, colors come from the `kind` attribute, sizes from the betweenness scores, and the route is highlighted:

```python
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
print(fig)
```

```text
<Figure 1208×895: 56 nodes, 83 edges, custom layout, theme 'light'>
```

```{figure} ../_static/generated/getting_started/campus.png
:alt: Map-like drawing of 56 University of Calgary buildings colored by kind and sized by betweenness, with the shortest walk from the Olympic Oval to Scurfield Hall highlighted in blue and the other buildings dimmed.
:width: 100%

The campus at its real geometry, north up. Colors show each building's kind, sizes its length-weighted betweenness, and the highlighted chain is the 1,002 m route. The isolated building at the bottom right is the Olympic Volunteer Centre at McMahon Stadium, joined to Mathison Hall by a modeled 626 m outdoor link. Building positions © OpenStreetMap contributors (ODbL).
```

A few things happened without extra code:

- **Positions as a layout.** Any `{node: (x, y)}` mapping works as a layout. AryaGraph uses screen coordinates (x to the right, y downward), and the dataset's `pos` has y pointing south, so the drawing comes out north up.
- **Scales chosen from the data.** `kind` holds strings, so it gets a categorical palette and a legend with counts. The campus has nine kinds; the seven most frequent keep their own colors and the two smallest (library and administration, three buildings together) share the "Other" entry.
- **Sizes by area.** {py:func}`ag.by <aryagraph.style.scales.by>` wraps the `NodeMap` to give the legend a readable title. Scores map to circle *area* rather than diameter, from the smallest circle for the lowest score to the largest for the highest, so high scores do not look disproportionately large.
- **Emphasis.** `highlight_path` draws the route's edges and nodes in the highlight color and dims everything else.

The interactive version below supports pan and zoom, search and a table view. Hover a building to highlight its neighbors, and click it to pin a details panel.

<iframe class="ag-embed" src="../_static/generated/getting_started/campus.html" height="720" loading="lazy" title="Interactive map of the University of Calgary campus graph with the shortest walk highlighted"></iframe>
<p class="ag-embed-note">Interactive figure produced by <code>fig.save("campus.html")</code>. Building positions © OpenStreetMap contributors (ODbL). <a href="../_static/generated/getting_started/campus.html">Open full screen</a></p>

## Save the figure

The file extension picks the format:

```python
fig.save("campus.svg")
fig.save("campus.html")
fig.save("campus.png")
```

| Extension | What you get | Extra requirements |
|---|---|---|
| `.svg` | Vector image, with a title and description for screen readers. | None |
| `.html` | Self-contained interactive page, the same view as the embed above. It loads no scripts, fonts or styles from the network, so it works offline and on any static web host. | None |
| `.png` | Raster image at 2× the figure's pixel size by default (`fig.save("campus.png", scale=3)` for more). | cairosvg, or Chrome, Edge, Chromium or Brave |
| `.pdf` | Vector PDF at the figure's natural size. | Same as PNG |

If PNG export raises an error about `cairosvg`, see [PNG and PDF export](installation.md#png-and-pdf-export). In Jupyter you do not need to save at all: a `Figure` as the last expression of a cell renders inline as the interactive view.

## Summarize the graph with analyze()

{py:func}`~aryagraph.analysis.report.analyze` computes the measures that apply to a graph and returns a {py:class}`~aryagraph.analysis.report.GraphReport`:

```python
report = ag.analyze(campus)
print(report)
report.save("campus_report.html")
```

```text
Graph report: University of Calgary main campus
───────────────────────────────────────────────
Graph                   Undirected · 56 nodes · 83 edges
Density                 0.0539
Average degree          2.96
Max degree              5
Components              1 (largest: 56 nodes, 100%)
Diameter                15
Radius                  8
Avg path length         5.9
Avg clustering          0.322
Transitivity            0.293
Degree assortativity    -0.171
Algebraic connectivity  0.0477
Communities             8 (modularity 0.732)
Top degree              ES (5), CD (4), CH (4)
Top pagerank            ES (0.0279), MTH (0.0258), EDT (0.0239)
Top betweenness         KNB (0.4), PF (0.282), SB (0.269)
Top closeness           KNB (0.226), MH (0.224), SB (0.224)
Top eigenvector         ES (0.227), ENE (0.223), GL (0.223)
Bridges / cut nodes     4 / 6
Max k-core              2
```

The campus is one connected piece and sparse: each building has about three neighbors on average, and 15 links separate the two buildings that are farthest apart. Louvain community detection splits the buildings into 8 groups (modularity 0.732).

```{note}
`analyze()` computes centralities without edge weights unless you pass `weight=`, so its betweenness counts links, not meters. That is why the Kinesiology Complex (`KNB`) leads here while Professional Faculties (`PF`) led the length-weighted ranking earlier. Community detection is the exception: it reads the `weight` attribute when edges have one.
```

`report.save("campus_report.html")` writes the same results as an interactive dashboard with stat tiles, the explorable graph, distributions and rankings. `.json`, `.txt` and `.md` work too.

<p class="ag-embed-note"><a href="../_static/generated/getting_started/campus_report.html">Open the campus dashboard</a> produced by the code above.</p>

## Plan a project as a DAG

A {py:class}`~aryagraph.core.dag.DAG` is a directed graph that refuses to contain a cycle. Here is the plan for a conference paper, with each task's duration in days stored as a node attribute:

```python
paper = ag.DAG(name="Conference paper")
paper.add_edges([
    ("data", "analysis"),
    ("analysis", "figures"),
    ("analysis", "draft"),
    ("literature", "draft"),
    ("draft", "review"),
    ("figures", "review"),
    ("review", "submit"),
])
days = {
    "literature": 3, "data": 4, "analysis": 5, "figures": 2,
    "draft": 6, "review": 2, "submit": 1,
}
for task, d in days.items():
    paper.nodes[task]["duration"] = d

print(paper)
print(paper.topological_order())
```

```text
<DAG 'Conference paper': 7 nodes, 7 edges>
['data', 'literature', 'analysis', 'figures', 'draft', 'review', 'submit']
```

An edge `u → v` means that `v` cannot start before `u` finishes. The critical-path method finds the chain of tasks that sets the earliest finish date:

```python
cp = ag.alg.critical_path(paper)
print(cp)
print(dict(cp.slack))
```

```text
CriticalPath(length=18, path=['data' → 'analysis' → 'draft' → 'review' → 'submit'], critical=5 of 7 activities)
{'data': 0.0, 'analysis': 0.0, 'figures': 4.0, 'draft': 0.0, 'literature': 6.0, 'review': 0.0, 'submit': 0.0}
```

The paper takes 18 days. The literature review can slip by up to 6 days and the figures by up to 4 without moving the submission date; any delay on the other five tasks delays the whole project.

A dependency that would close a loop is rejected, and the error names the loop:

```python
try:
    paper.add_edge("submit", "data")
except ag.CycleError as err:
    print(err)
print(paper)
```

```text
edge 'submit' → 'data' would close the cycle 'submit' → 'data' → 'analysis' → 'draft' → 'review' → 'submit'
<DAG 'Conference paper': 7 nodes, 7 edges>
```

The graph is left unchanged. To draw the plan, `draw()` picks a layered (hierarchical) layout for DAGs; the layout option `orientation="LR"` runs it left to right, and a callable builds each label from the node's attributes:

```python
plan = ag.draw(
    paper,
    labels=lambda n, d: f"{n} ({d['duration']} d)",
    highlight_path=cp.path,
    layout_options={"orientation": "LR"},
    title="Conference paper plan",
    subtitle=f"Critical path highlighted: {cp.length:g} days",
)
plan.save("paper.png")
print(plan)
```

```text
<Figure 720×192: 7 nodes, 7 edges, hierarchical layout, theme 'light'>
```

```{figure} ../_static/generated/getting_started/paper.png
:alt: Left-to-right diagram of seven tasks in rounded boxes with durations; data, analysis, draft, review and submit are highlighted as the critical path, while literature and figures are dimmed.
:width: 100%

The conference paper plan in a layered layout. The five tasks on the critical path are highlighted; the two tasks with slack (literature and figures) are dimmed.
```

## Run a small simulation

Simulations live in `ag.sim`. This one runs an SIR (susceptible, infected, recovered) contagion model on the campus network, starting from a single infected building, MacEwan Student Centre. It is a toy that shows the API, not a model of disease on campus:

```python
model = ag.sim.SIR(beta=0.4, gamma=0.1)
run = model.simulate(campus, initial={"I": ["MSC"]}, t_max=60, seed=2)
print(run)
print(run.peak("I"))
print(run.summary()["R"])
```

```text
<SimulationResult SIR: 56 nodes, 53 frames, t=0…52; final S=7, I=0, R=49>
(15.0, 31)
{'initial': 0, 'final': 49, 'peak': 49, 'peak_time': 52.0}
```

The default method advances in discrete time steps. At each step, a susceptible building with *k* infected neighbors becomes infected with probability 1 − e<sup>−0.4k</sup>, and an infected building recovers with probability 1 − e<sup>−0.1</sup>, about 9.5%. In this run the number of infected buildings peaks at 31 at step 15, and the outbreak ends at step 52, after 49 of the 56 buildings have been infected; the other 7 stayed susceptible.

The run returns a {py:class}`~aryagraph.sim.base.SimulationResult`: the state of every building at every step, with methods for counts, peaks and exports. `plot()` charts the counts, and `animate()` replays the run on any layout:

```python
chart = run.plot(
    title="SIR on the campus network",
    subtitle="Seeded at MacEwan Student Centre · beta 0.4, gamma 0.1, seed 2",
)
chart.save("sir.png")
run.animate(layout=geometry, title="SIR on the campus network").save("sir.html")
print(chart)
```

```text
<Chart line 640×280>
```

```{figure} ../_static/generated/getting_started/sir.png
:alt: Line chart of susceptible, infected and recovered building counts over 52 time steps; infected rises to a peak of 31 at step 15 and falls to zero.
:width: 100%

Buildings in each state over time for the seeded run above.
```

<iframe class="ag-embed" src="../_static/generated/getting_started/sir.html" height="760" loading="lazy" title="Animated playback of the SIR simulation on the campus map"></iframe>
<p class="ag-embed-note">Press play, or drag the timeline, to watch the outbreak move between buildings; transmissions flash along the links. Produced by <code>run.animate(...).save("sir.html")</code>. Building positions © OpenStreetMap contributors (ODbL). <a href="../_static/generated/getting_started/sir.html">Open full screen</a></p>

The `seed` makes a run repeatable: with the same versions of AryaGraph and numpy, `seed=2` reproduces this run. Other seeds give other outbreaks:

```python
for seed in range(1, 6):
    trial = model.simulate(campus, initial={"I": ["MSC"]}, t_max=60, seed=seed)
    print(seed, trial.summary()["R"]["final"], trial.peak("I"))
```

```text
1 33 (7.0, 18)
2 49 (15.0, 31)
3 54 (12.0, 23)
4 52 (18.0, 23)
5 47 (10.0, 26)
```

Across these five seeds, between 33 and 54 buildings are infected by the end. To summarize many runs with mean curves and percentile bands, use {py:func}`~aryagraph.sim.run_ensemble` (see [Simulation](../user-guide/simulation.md)).

## Where to go next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Concepts
:link: concepts
:link-type: doc

The ideas behind the API: graph classes, views, result maps, layouts, scenes, encodings, themes and seeds.
:::

:::{grid-item-card} Drawing
:link: ../user-guide/drawing
:link-type: doc

Every option of `draw()`: labels, shapes, edge styles, highlighting and legends.
:::

:::{grid-item-card} Campus routing tutorial
:link: ../tutorials/campus-routing
:link-type: doc

Indoor-only routes, weighted paths and route comparisons on the same dataset.
:::

:::{grid-item-card} Pipeline DAG tutorial
:link: ../tutorials/pipeline-dag
:link-type: doc

Layered layouts, critical paths and schedule simulation for an example ML pipeline.
:::
::::
