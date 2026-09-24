# Analysis reports

{py:func}`aryagraph.analysis.report.analyze` runs the measures that apply to a graph in one call and returns a {py:class}`aryagraph.analysis.report.GraphReport`. You can print the report as text, export it as data, or save it as an interactive dashboard.

## A first report

```python
import aryagraph as ag

lesmis = ag.gen.les_miserables()     # 77 characters, 254 co-appearance edges
report = ag.analyze(lesmis)
print(report)
```

```text
Graph report: Les Misérables
────────────────────────────────────────
Graph                   Undirected · 77 nodes · 254 edges
Density                 0.0868
Average degree          6.6
Max degree              36
Components              1 (largest: 77 nodes, 100%)
Diameter                5
Radius                  3
Avg path length         2.64
Avg clustering          0.573
Transitivity            0.499
Degree assortativity    -0.165
Algebraic connectivity  0.205
Communities             6 (modularity 0.567)
Top degree              Valjean (36), Gavroche (22), Marius (19)
Top pagerank            Valjean (0.0754), Myriel (0.0428), Gavroche (0.0358)
Top betweenness         Valjean (0.57), Myriel (0.177), Gavroche (0.165)
Top closeness           Valjean (0.644), Marius (0.531), Thenardier (0.517)
Top eigenvector         Gavroche (0.318), Valjean (0.268), Enjolras (0.267)
Bridges / cut nodes     18 / 8
Max k-core              9
```

Every line comes from an algorithm in `ag.alg`: the summary statistics from {py:func}`aryagraph.algorithms.structure.summary`, the distances from `diameter`, `radius` and `average_shortest_path_length`, the rankings from the centrality functions, the communities from Louvain, and so on. The [Algorithms](algorithms.md) page describes each of them.

## Three ways to use the result

### Text

`print(report)` or `str(report)` gives the summary above. `report.save("report.txt")` writes it to a file, and `report.save("report.md")` wraps it in a Markdown code block, ready to paste into an issue or a notebook.

### Data

The report's attributes hold the underlying values, and `to_dict()` / `to_json()` give a JSON-ready copy in which node keys become strings:

```python
print(report.summary["diameter"], round(report.modularity, 3))
print([(n, round(v, 3)) for n, v in report.centrality["betweenness"].top(3)])
print(len(report.communities), "communities; Valjean is in community",
      report.community_of["Valjean"])

data = report.to_dict()
print(list(data))
report.save("lesmis.json")
```

```text
5 0.567
[('Valjean', 0.57), ('Myriel', 0.177), ('Gavroche', 0.165)]
6 communities; Valjean is in community 0
['name', 'summary', 'centrality', 'communities', 'modularity', 'structure', 'dag', 'degree_histogram', 'notes', 'timings']
```

| attribute | contents |
|---|---|
| `summary` | size, density, degrees, components, distances, clustering, assortativity, algebraic connectivity |
| `centrality` | `{name: NodeMap}` for degree, PageRank, betweenness, closeness and eigenvector |
| `communities`, `community_of`, `modularity` | Louvain partition, node → community index, and its modularity |
| `structure` | core numbers and the largest k-core; bridges and articulation points for undirected graphs |
| `dag` | DAG measures (see [DAGs](#dags)); `None` for other graphs |
| `degree_histogram` | number of nodes of each degree |
| `notes` | why anything was skipped or sampled |
| `timings` | seconds spent in each step |

Because centralities are `NodeMap` objects, you can pass them straight to a drawing, for example `ag.draw(lesmis, node_size=report.centrality["pagerank"])`. `report.figure()` returns the drawing that the dashboard uses: communities as colors, PageRank as size, and every centrality in the tooltips. It accepts any {py:func}`aryagraph.render.draw` option.

### Interactive dashboard

`report.save("report.html")` writes a self-contained page. It needs no server and no network connection, so you can open it locally, attach it to an email, or upload it to any static web host. It contains:

- stat tiles for the headline numbers;
- the interactive graph (for graphs of up to 5,000 nodes), with search, hover, a details panel and a table view;
- the degree distribution and the top 10 nodes by PageRank and by betweenness;
- a table of the communities (up to 12) with their most central members;
- for DAGs, the sources, sinks, level sizes, critical path and an earliest-start Gantt chart;
- the notes and the time the analysis took.

```python
report.save("lesmis_report.html")
```

<iframe class="ag-embed" src="../_static/generated/guide_layout_algorithms/lesmis_report.html" height="1400" loading="lazy" title="Interactive analysis dashboard of the Les Miserables network"></iframe>
<p class="ag-embed-note">The dashboard for Les Misérables, embedded as a static HTML file. The dashboard is taller than this frame: scroll inside the frame to reach the charts and the community table, or <a href="../_static/generated/guide_layout_algorithms/lesmis_report.html">open it full screen</a>.</p>

`analyze(g, theme="dark")` renders the dashboard, its charts and `report.figure()` in another theme; see [Styling](styling.md).

## What is computed, and when

`analyze` computes what is defined for the graph and affordable at its size. The table lists the conditions; every measure skipped or sampled because of the graph's size is recorded in `notes`.

| section | computed | condition |
|---|---|---|
| summary | nodes, edges, density, average and maximum degree, self-loops, components, largest component | every graph |
| directed graphs | strongly connected components, reciprocity | directed graphs |
| distances | diameter, radius, average path length (in hops), on the largest component | largest component up to 3,000 nodes |
| clustering | average clustering, transitivity | every graph (on its undirected version) |
| mixing | degree assortativity | at least one edge |
| spectrum | algebraic connectivity | 3 to 1,500 nodes |
| centrality | degree, PageRank | `centrality=True` (default) |
| | betweenness | exact up to 3,000 nodes; above that, estimated from 500 sampled sources |
| | closeness | up to 3,000 nodes |
| | eigenvector | graphs that are not DAGs |
| communities | Louvain partition and its modularity | `communities=True` (default), at least one edge, not a DAG |
| structure | core numbers, largest k-core | every graph |
| | bridges, articulation points | undirected graphs |
| DAG | see [DAGs](#dags) | directed acyclic graphs |

Each step runs on its own. If one fails, for example because an iterative method does not converge, the report records `"<step> skipped: <reason>"` in `notes` and continues with the rest.

For a large graph, the notes show what changed. Here a 4,000-node preferential-attachment graph is analyzed with communities switched off:

```python
big = ag.gen.barabasi_albert(4000, 2, seed=1)
big_report = ag.analyze(big, communities=False, seed=0)
for note in big_report.notes:
    print(note)
print("closeness" in big_report.centrality)
```

```text
distance measures skipped: largest component has 4,000 nodes (> 3,000)
algebraic connectivity skipped: 4,000 nodes (> 1,500)
betweenness estimated from 500 sampled sources
closeness skipped: 4,000 nodes (> 3,000)
False
```

Sampled betweenness uses the `seed` argument of `analyze` (default 0), so the estimate is reproducible. Pass `centrality=False` as well to skip the centralities on very large graphs.

## Weights

`analyze(g, weight=None)` (the default) computes distances and centralities without weights. Community detection is the exception: Louvain and modularity read the edge attribute named `"weight"` whenever it exists, because they treat weights as connection strengths.

With `weight="<attribute>"`, the attribute is used as a **distance** for betweenness and closeness, and as a **strength** for PageRank, eigenvector centrality and communities. Choose an attribute whose meaning suits both uses, or keep the default. In Les Misérables, the `weight` attribute counts shared chapters, a strength, so weighted betweenness treats close companions as far apart, and the ranking changes:

```python
weighted = ag.analyze(lesmis, weight="weight")
for r in (report, weighted):
    print([(n, round(v, 3)) for n, v in r.centrality["betweenness"].top(3)])

print(weighted.summary["diameter"], ag.alg.diameter(lesmis, weight="weight"))
```

```text
[('Valjean', 0.57), ('Myriel', 0.177), ('Gavroche', 0.165)]
[('Valjean', 0.454), ('Gavroche', 0.285), ('Javert', 0.193)]
5 14.0
```

The diameter, radius and average path length in the report count hops, whatever you pass as `weight`: the weighted report above still gives a diameter of 5. For a weighted distance measure, call the algorithm directly, as in the last line, which gives 14.0.

```{note}
On the campus graph, each edge has a `weight` attribute equal to its length in meters. The default report therefore finds communities with lengths as strengths. To group buildings by proximity instead, run {py:func}`aryagraph.algorithms.community.louvain_communities` yourself with a weight that decreases with length, such as `lambda u, v, d: 1 / d["length"]`.
```

## DAGs

For a directed acyclic graph, `analyze` adds a `dag` section and skips community detection and eigenvector centrality:

```python
plan = ag.gen.project_plan()          # 18 tasks of a house build, in working hours
plan_report = ag.analyze(plan)
d = plan_report.dag
print(sorted(d))
print(d["depth"], "levels, level sizes", d["level_sizes"])
print("width:", d["width"], " redundant edges:", d["redundant_edges"])
print("critical path:", d["critical_length"], "h,",
      len(d["critical_path"]), "tasks")
```

```text
['critical_length', 'critical_path', 'depth', 'level_sizes', 'longest_path', 'redundant_edges', 'schedule', 'sinks', 'slack', 'sources', 'width']
12 levels, level sizes [1, 1, 1, 1, 1, 1, 5, 2, 1, 1, 2, 1]
width: 5  redundant edges: 0
critical path: 572.0 h, 12 tasks
```

| key | meaning | condition |
|---|---|---|
| `depth`, `level_sizes` | number of generations and the number of nodes in each | every DAG |
| `sources`, `sinks` | nodes without predecessors / successors | every DAG |
| `longest_path` | a longest path by number of edges | every DAG |
| `width` | size of a largest antichain: the most tasks that can run at once | up to 1,500 nodes |
| `redundant_edges` | edges implied by other paths (removed by the transitive reduction) | up to 2,000 nodes |
| `critical_path`, `critical_length`, `slack`, `schedule` | critical-path method on the node durations | nodes have the duration attribute |

The duration attribute is chosen with `analyze(g, duration="hours")`; the default is `"duration"`, and `duration=None` skips the critical-path analysis. In the dashboard, the critical path is highlighted on the graph and the earliest-start schedule (with unlimited workers) is drawn as a Gantt chart:

```python
plan_report.save("plan_report.html")
```

<iframe class="ag-embed" src="../_static/generated/guide_layout_algorithms/plan_report.html" height="1400" loading="lazy" title="Interactive analysis dashboard of the house construction plan"></iframe>
<p class="ag-embed-note">The dashboard of the house-construction DAG. The dashboard is taller than this frame: scroll inside the frame to reach the critical path and the earliest-start schedule, or <a href="../_static/generated/guide_layout_algorithms/plan_report.html">open it full screen</a>.</p>

For scheduling with limited workers, random durations and Monte Carlo risk, see [DAG workflows](dags.md).

## From the command line

The `aryagraph analyze` command writes the same outputs for a graph file; the extension of `-o` chooses the format, and `--theme` sets the theme:

```bash
aryagraph analyze network.graphml -o report.html
aryagraph analyze network.graphml -o report.json --theme dark
```

See [Command line](cli.md) for the other commands.
