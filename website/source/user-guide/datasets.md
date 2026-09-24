# Datasets and generators

`ag.gen` provides graphs to learn with, test against and benchmark on: a real-world campus network, three classic social networks, five example task DAGs, and generators for classic families and random models. The datasets ship inside the package, so they load without a download and without networkx, and every function returns a fresh graph that you can modify freely.

## At a glance

| Function | Type | Nodes | Edges | What it is |
|---|---|---|---|---|
| {py:func}`~aryagraph.generators.datasets.ucalgary_campus` | Graph | 56 | 83 | University of Calgary buildings, tunnels, pedways and walking links |
| {py:func}`~aryagraph.generators.datasets.les_miserables` | Graph | 77 | 254 | character co-appearances in Victor Hugo's novel |
| {py:func}`~aryagraph.generators.datasets.florentine_families` | Graph | 15 | 20 | marriage alliances in Renaissance Florence |
| {py:func}`~aryagraph.generators.datasets.davis_southern_women` | Graph | 32 | 89 | 18 women and the 14 social events they attended (bipartite) |
| {py:func}`~aryagraph.generators.dags.ml_pipeline` | DAG | 16 | 19 | training and shipping a churn model |
| {py:func}`~aryagraph.generators.dags.software_build` | DAG | 20 | 33 | build targets of a client/server product |
| {py:func}`~aryagraph.generators.dags.project_plan` | DAG | 18 | 23 | building a single-family house |
| {py:func}`~aryagraph.generators.dags.data_warehouse_etl` | DAG | 25 | 41 | a nightly data-warehouse load |
| {py:func}`~aryagraph.generators.dags.course_prerequisites` | DAG | 22 | 31 | prerequisites of a computer-science degree |

```python
import aryagraph as ag

for fn in (ag.gen.ucalgary_campus, ag.gen.les_miserables, ag.gen.florentine_families,
           ag.gen.davis_southern_women, ag.gen.ml_pipeline, ag.gen.software_build,
           ag.gen.project_plan, ag.gen.data_warehouse_etl, ag.gen.course_prerequisites):
    g = fn()
    print(f"{fn.__name__:22} {type(g).__name__:5} {len(g):3} nodes {g.num_edges:4} edges")
```

```text
ucalgary_campus        Graph  56 nodes   83 edges
les_miserables         Graph  77 nodes  254 edges
florentine_families    Graph  15 nodes   20 edges
davis_southern_women   Graph  32 nodes   89 edges
ml_pipeline            DAG    16 nodes   19 edges
software_build         DAG    20 nodes   33 edges
project_plan           DAG    18 nodes   23 edges
data_warehouse_etl     DAG    25 nodes   41 edges
course_prerequisites   DAG    22 nodes   31 edges
```

## University of Calgary campus

{py:func}`~aryagraph.generators.datasets.ucalgary_campus` is a real-world spatial network: 56 main-campus buildings of the University of Calgary and the ways between them. It is the showcase dataset of AryaGraph and appears throughout the tests, tutorials and case studies, because it is small enough to read and rich enough to route on, rank and cluster.

```python
campus = ag.gen.ucalgary_campus()
print(campus)
print(campus.nodes["MSC"])
print(campus.edges["AB", "CH"])
print(sorted(campus.attrs))
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
{'name': 'MacEwan Student Centre', 'kind': 'student-life', 'lat': 51.0781866, 'lon': -114.131706, 'pos': (-22.3, -32.7)}
{'kind': 'pedway', 'length': 141.2, 'weight': 141.2}
['attribution', 'description', 'name', 'sources']
```

**Nodes** are official building codes (`"MSC"`, `"TFDL"`, `"ICT"`, `"SS"`, `"OO"`, `"SH"`, …) with these attributes:

| Attribute | Content |
|---|---|
| `name` | the building's name, e.g. `"Taylor Family Digital Library"` |
| `kind` | `"academic"`, `"residence"`, `"student-life"`, `"athletics"`, `"arts"`, `"library"`, `"research"`, `"administration"` or `"services"` |
| `lat`, `lon` | WGS84 coordinates in degrees |
| `pos` | `(x, y)` in meters on a local plane centered on the campus; x points east and y points **south**, the screen convention, so drawings come out north-up |

**Edges** are undirected. `kind` says what connects the two buildings, `length` is the straight-line distance in meters, and `weight` equals `length` so that weighted algorithms work without extra arguments.

| Edge kind | Meaning |
|---|---|
| `"tunnel"` | underground pedestrian tunnel |
| `"pedway"` | enclosed bridge or link corridor |
| `"attached"` | adjoining buildings with an internal door |
| `"outdoor"` | a modeled walking link: each building joins its two nearest neighbors within 250 m, plus the shortest links needed to connect the campus |

```python
from collections import Counter

print(Counter(kind for _, kind in campus.nodes.data("kind")).most_common())
print(Counter(kind for _, _, kind in campus.edges.data("kind")).most_common())
lengths = [d["length"] for _, _, d in campus.edges.data()]
print(min(lengths), max(lengths), round(sum(lengths), 1))
```

```text
[('academic', 25), ('residence', 9), ('services', 5), ('arts', 4), ('student-life', 4), ('research', 3), ('athletics', 3), ('administration', 2), ('library', 1)]
[('outdoor', 43), ('attached', 26), ('pedway', 11), ('tunnel', 3)]
32.4 626.2 8855.2
```

Because `pos` holds real geometry, you can draw the campus as a map by passing the positions as the layout:

```python
indoor = {"tunnel", "pedway", "attached"}
fig = ag.draw(
    campus,
    layout={n: d["pos"] for n, d in campus.nodes.data()},
    node_color="kind",
    edge_color=lambda u, v, d: "#52514e" if d["kind"] in indoor else "#c3c2b7",
    title="ucalgary_campus(): 56 buildings, 83 links",
)
fig.save("campus.svg")
```

```{figure} ../_static/generated/guide_sim_data/campus.png
:alt: Map-like drawing of 56 UCalgary buildings colored by kind, with dark indoor links and light outdoor walking links; the Olympic Volunteer Centre sits far to the south.
:width: 100%

The campus drawn at its real positions, north up. Dark links are indoor connections; light links are modeled outdoor walks. With nine building kinds, the two smallest (administration and library) fold into "Other". Positions © OpenStreetMap contributors (ODbL).
```

### The indoor network

Calgary winters make the indoor network a question of its own. `indoor_only=True` keeps only tunnels, pedways and attached buildings, and keeps all 56 buildings, so those without an indoor connection become isolated nodes:

```python
inside = ag.gen.ucalgary_campus(indoor_only=True)
parts = ag.alg.connected_components(inside)
print(inside)
print(len(parts), max(len(p) for p in parts), sum(1 for n in inside if inside.degree(n) == 0))

def meters(g, route):
    return round(sum(g.edges[u, v]["length"] for u, v in zip(route, route[1:])), 1)

warm = ag.alg.shortest_path(inside, "ICT", "TFDL", weight="length")
any_way = ag.alg.shortest_path(campus, "ICT", "TFDL", weight="length")
print(meters(inside, warm), warm)
print(meters(campus, any_way), any_way)
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 40 edges>
19 38 18
795.9 ['ICT', 'ES', 'MS', 'ST', 'SS', 'AD', 'PF', 'MFH', 'MT', 'HNSC', 'TFDL']
728.9 ['ICT', 'ES', 'MS', 'ST', 'SS', 'AD', 'PF', 'MFH', 'MT', 'TFDL']
```

38 buildings form one connected indoor network. Staying inside from the ICT building to the Taylor Family Digital Library takes 795.9 m against 728.9 m by the shortest route overall. Both routes agree as far as MacKimmie Tower (MT); from there the shortest route takes a 76.7 m outdoor link, while the indoor one continues through Hunter Student Commons (HNSC), a 67 m detour.

```{figure} ../_static/generated/guide_sim_data/campus_indoor.png
:alt: The campus map with only indoor links; a large connected cluster in the center and eighteen isolated buildings around it.
:width: 100%

`ucalgary_campus(indoor_only=True)`: the 38-building indoor cluster and 18 buildings without an indoor link. Positions © OpenStreetMap contributors (ODbL).
```

### Provenance and limits

The dataset is a schematic model compiled from public sources, not an official or surveyed product of the university. Its graph attributes record the sources (`campus.attrs["sources"]` lists 12 URLs) and the attribution.

* Building codes and names follow the university's Main Campus Map (July 2023) and Facilities directory, updated for later renamings: Kinesiology A and B (`"KNA"`, `"KNB"`) have been the Dr. Roger Jackson Kinesiology Complex since February 2026.
* Indoor links follow the university's Indoor Building Routes map (November 2025), cross-checked against OpenStreetMap.
* Positions are the centers of the buildings' OpenStreetMap bounding boxes (© OpenStreetMap contributors, available under the Open Database License); they can sit a few meters off an irregular building's centroid.
* Every `length` is a straight-line distance between two positions, not a measured walking distance, and `"outdoor"` edges are modeled, not surveyed paths.
* Craigie Hall's blocks are merged into one node, `"CH"`. The Olympic Volunteer Centre (`"OVC"`) stands at McMahon Stadium south of 24 Avenue NW; its only edge, a 626.2 m outdoor link to Mathison Hall, is the longest in the graph and stretches north-up drawings.

The repository script `tools/build_ucalgary_campus.py` rebuilds the dataset from its sources.

## Classic social networks

Three small networks from the social-network literature are bundled as copies of networkx's versions: same nodes in the same order, same edges and attributes (the test suite compares them), so results can be cross-checked. Each graph's `citation` attribute names the original study.

```python
les = ag.gen.les_miserables()
print(les, les.edges["Valjean", "Javert"])
florence = ag.gen.florentine_families()
print(florence, ag.alg.betweenness_centrality(florence).top(1))
davis = ag.gen.davis_southern_women()
print(davis, davis.nodes["Evelyn Jefferson"], davis.nodes["E1"])
print(len(davis.attrs["top"]), len(davis.attrs["bottom"]), ag.alg.is_bipartite(davis))
print(florence.attrs["citation"])
```

```text
<Graph 'Les Misérables': 77 nodes, 254 edges> {'weight': 17}
<Graph 'Florentine families': 15 nodes, 20 edges> [('Medici', 0.521978021978022)]
<Graph 'Davis Southern Women': 32 nodes, 89 edges> {'bipartite': 0} {'bipartite': 1}
18 14 True
Marriage ties among Renaissance Florentine families; R. Breiger & P. Pattison, Cumulated social roles: the duality of persons and their algebras, Social Networks 8, 215-256 (1986).
```

* **Les Misérables** ({py:func}`~aryagraph.generators.datasets.les_miserables`): 77 characters; an edge's `weight` counts the chapters in which the two appear together. A standard test case for community detection and weighted centrality.
* **Florentine families** ({py:func}`~aryagraph.generators.datasets.florentine_families`): marriage alliances among 15 families; the Medici's central position is the textbook example for betweenness. The isolated Pucci family is omitted, as in networkx.
* **Davis Southern Women** ({py:func}`~aryagraph.generators.datasets.davis_southern_women`): a two-mode network. Women carry `bipartite=0`, events (`"E1"` to `"E14"`) carry `bipartite=1`, and the graph attributes `top` and `bottom` list the two sides.

## Example DAGs

Five hand-crafted DAGs with realistic names serve the [DAG workflows](dags.md) features. Every task node carries:

* `duration`: the planned time in hours (equal to `mode`);
* `min`, `mode`, `max`: optimistic, most likely and pessimistic hours, a three-point estimate for PERT and Monte Carlo scheduling;
* `kind` and `team`: categories for coloring and grouping.

```python
examples = (ag.gen.ml_pipeline, ag.gen.software_build, ag.gen.project_plan,
            ag.gen.data_warehouse_etl, ag.gen.course_prerequisites)
print(f"{'dataset':22} {'depth':>5} {'width':>5} {'critical path':>13} {'total work':>10}")
for fn in examples:
    d = fn()
    cp = ag.alg.critical_path(d)
    work = sum(x["duration"] for _, x in d.nodes.data())
    print(f"{fn.__name__:22} {d.depth():5} {ag.alg.dag_width(d):5} {cp.length:13g} {work:10g}")
```

```text
dataset                depth width critical path total work
ml_pipeline               12     3         58.75      72.25
software_build             9     8          1.57       3.03
project_plan              12     5           572        772
data_warehouse_etl         7     7             5       12.1
course_prerequisites       5     9           855       3420
```

| Dataset | Extra attributes |
|---|---|
| `ml_pipeline()` | edges: `artifact`, the data product handed downstream |
| `software_build()` | edges: `artifact`; durations are fractions of an hour (a compile step takes minutes) |
| `project_plan()` | edges: `lag`, a mandatory wait in hours (concrete curing, drying), 0 elsewhere; 40 h is one working week |
| `data_warehouse_etl()` | nodes: `table`, the table each task writes |
| `course_prerequisites()` | nodes: `title`, `credits`, `level`, `department`; edges: `requirement` (`"required"` or `"recommended"`); `duration` is the semester workload at 45 h per credit, with `min`/`max` at 80% and 150% of it |

```{figure} ../_static/generated/guide_sim_data/etl.png
:alt: Layered drawing of the data-warehouse ETL: six extract tasks, six staging tasks, four dimensions, three fact tables, a data-quality gate, three aggregates and two publishing tasks, colored by kind.
:width: 90%

`ag.draw(ag.gen.data_warehouse_etl(), node_color="kind")`: fan-out from the staging tables and dimensions, fan-in at the fact tables and aggregates.
```

## Classic graph families

Deterministic generators for textbook graphs. Node labels are integers unless noted, and most take `directed=True` for a directed version.

| Function | Graph |
|---|---|
| `empty_graph(n)` | *n* nodes, no edges |
| `path_graph(n)`, `cycle_graph(n)` | a path or a cycle through 0 … *n*−1 |
| `complete_graph(n)` | every pair linked |
| `complete_bipartite_graph(n1, n2)` | every cross pair linked |
| `complete_multipartite_graph(*sizes)`, `turan_graph(n, r)` | complete multipartite; parts as equal as possible |
| `star_graph(n)`, `wheel_graph(n)` | hub 0 with *n* leaves; hub joined to a cycle |
| `grid_graph(rows, cols)` | lattice with `(r, c)` nodes and a `pos` attribute; `periodic=True` for a torus |
| `hypercube_graph(d)` | the *d*-dimensional hypercube |
| `balanced_tree(r, h)`, `binomial_tree(k)` | complete *r*-ary tree of height *h*; binomial tree of order *k* |
| `ladder_graph(n)`, `circular_ladder_graph(n)` | two rails with *n* rungs; closed rails (prism) |
| `lollipop_graph(m, n)`, `barbell_graph(m1, m2)` | a clique with a tail; two cliques joined by a path |
| `petersen_graph()` | the Petersen graph |

Each graph's `name` attribute records the call that built it:

```python
examples = [ag.gen.empty_graph(4), ag.gen.cycle_graph(6), ag.gen.complete_graph(5),
            ag.gen.complete_bipartite_graph(3, 4), ag.gen.turan_graph(10, 3), ag.gen.star_graph(5),
            ag.gen.grid_graph(3, 4), ag.gen.hypercube_graph(3), ag.gen.balanced_tree(2, 3),
            ag.gen.ladder_graph(4), ag.gen.barbell_graph(4, 2), ag.gen.petersen_graph()]
for g in examples:
    print(f"{g.name:32} {len(g):3} nodes {g.num_edges:3} edges")
print(ag.gen.grid_graph(3, 4).nodes[(1, 2)], ag.gen.balanced_tree(2, 3, directed=True))
```

```text
empty_graph(4)                     4 nodes   0 edges
cycle_graph(6)                     6 nodes   6 edges
complete_graph(5)                  5 nodes  10 edges
complete_bipartite_graph(3, 4)     7 nodes  12 edges
turan_graph(10, 3)                10 nodes  33 edges
star_graph(5)                      6 nodes   5 edges
grid_graph(3, 4)                  12 nodes  17 edges
hypercube_graph(3)                 8 nodes  12 edges
balanced_tree(2, 3)               15 nodes  14 edges
ladder_graph(4)                    8 nodes  10 edges
barbell_graph(4, 2)               10 nodes  15 edges
petersen_graph()                  10 nodes  15 edges
{'pos': (2.0, 1.0)} <DiGraph 'balanced_tree(2, 3)': 15 nodes, 14 edges>
```

## Random graph models

Every random generator takes a `seed`: the same arguments and seed produce the same graph, node for node and edge for edge.

| Function | Model |
|---|---|
| `erdos_renyi(n, p)` | G(n, p): every pair linked independently with probability *p* |
| `gnm_random_graph(n, m)` | G(n, m): uniform among graphs with *m* edges |
| `barabasi_albert(n, m)` | preferential attachment, *m* edges per new node |
| `powerlaw_cluster(n, m, p)` | Holme–Kim: preferential attachment plus triangle closure with probability *p* |
| `watts_strogatz(n, k, p)` | ring lattice with *k* neighbors, each edge rewired with probability *p* |
| `stochastic_block_model(sizes, p)` | blocks linked with probability `p[i][j]`; nodes carry `block` |
| `planted_partition(l, k, p_in, p_out)` | *l* equal blocks of *k* nodes |
| `random_geometric(n, radius)` | points in the unit square, linked within *radius*; nodes carry `pos` |
| `random_regular(d, n)` | uniform *d*-regular graph |
| `configuration_model(degrees)` | simple graph with the given degree sequence |
| `random_tree(n)` | uniform labeled tree (random Prüfer sequence) |

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item}
```{image} ../_static/generated/guide_sim_data/gen_erdos_renyi.png
:alt: Erdős–Rényi random graph with 60 nodes.
```
:::
:::{grid-item}
```{image} ../_static/generated/guide_sim_data/gen_barabasi_albert.png
:alt: Barabási–Albert graph with 60 nodes and a few hubs.
```
:::
:::{grid-item}
```{image} ../_static/generated/guide_sim_data/gen_watts_strogatz.png
:alt: Watts–Strogatz small world with 60 nodes.
```
:::
:::{grid-item}
```{image} ../_static/generated/guide_sim_data/gen_planted_partition.png
:alt: Planted partition with three colored blocks of 20 nodes.
```
:::
:::{grid-item}
```{image} ../_static/generated/guide_sim_data/gen_random_geometric.png
:alt: Random geometric graph drawn at its points in the unit square.
```
:::
:::{grid-item}
```{image} ../_static/generated/guide_sim_data/gen_random_tree.png
:alt: Uniform random tree with 60 nodes in a tidy tree layout.
```
:::
::::

```python
a = ag.gen.barabasi_albert(1000, 3, seed=42)
b = ag.gen.barabasi_albert(1000, 3, seed=42)
print(a, list(a.edges) == list(b.edges))
print(ag.gen.barabasi_albert(1000, 3, seed=43).degree().top(1), a.degree().top(1))
```

```text
<Graph 'barabasi_albert(1000, 3)': 1000 nodes, 2991 edges> True
[(0, 101)] [(4, 103)]
```

The same seed rebuilds the graph edge for edge; a different seed gives another graph from the same model, here with its largest hub at degree 101 instead of 103.

### Random DAGs

| Function | Structure |
|---|---|
| `random_dag(n, p=None, m=None)` | a hidden random order and only forward arcs; about three arcs per node by default |
| `layered_dag(layers, p=0.4)` | arcs between consecutive layers only; nodes carry `layer` and `index`; repaired to be connected by default |
| `random_task_dag(n)` | a project network numbered in topological order, with task 0 as the only source and task *n*−1 as the only sink; tasks depend on a few recent tasks and carry `duration`, `min`, `mode`, `max` in hours, ready for CPM and Monte Carlo scheduling |

```python
tasks = ag.gen.random_task_dag(40, seed=1)
print(tasks, tasks.sources(), tasks.sinks())
print(tasks.nodes[5])
print(ag.gen.layered_dag([3, 5, 5, 2], p=0.3, seed=4).generations())
```

```text
<DAG 'random_task_dag(40)': 40 nodes, 70 edges> [0] [39]
{'duration': 9.0, 'min': 5.0, 'mode': 9.0, 'max': 15.5}
[[0, 1, 2], [4, 6, 3, 5, 7], [8, 11, 12, 9, 10], [13, 14]]
```

Generated graphs are ordinary `Graph`, `DiGraph` or `DAG` objects: combine them with `compose`, relabel them, or save them with {py:func}`ag.write <aryagraph.io.write>` (see [File formats and interoperability](io.md)).
