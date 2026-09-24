# Moving data in and out

AryaGraph reads and writes the common graph file formats and converts to and from pandas, networkx, numpy and SciPy. In this tutorial you move one network through each of them, check what survives every round trip, and finish with a small pipeline that goes from two CSV files to an analysis and a figure.

**Goal:** move a graph between AryaGraph, pandas, networkx, numpy, SciPy and graph files, and know what each conversion keeps.

**Prerequisites:**

- [Your first network](first-network.md), or familiarity with `ag.Graph` and node and edge attributes.
- Basic pandas: selecting rows and columns, `groupby`.
- pandas, networkx and SciPy. numpy is AryaGraph's only required dependency; the converters import the other packages when you call them. Install all three with `pip install ".[interop]"` from the source directory. Without them, the converters raise {py:class}`~aryagraph.core.exceptions.DependencyError` with the command to run.

The code blocks build on each other: run them in order in one Python session or notebook. They write a few small files into the current working directory.

## The example network

The examples use the University of Calgary campus network that ships with AryaGraph: 56 buildings, with the tunnels, pedways, attached buildings and modeled outdoor walking links between them. Buildings carry a name, a kind and their coordinates; links carry a kind and a `length`, the straight-line distance between the two buildings in meters.

```python
import math

import numpy as np
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
print(campus)
print(campus.nodes["MSC"])
print(campus.edges["MSC", "KNB"])
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
{'name': 'MacEwan Student Centre', 'kind': 'student-life', 'lat': 51.0781866, 'lon': -114.131706, 'pos': (-22.3, -32.7)}
{'kind': 'pedway', 'length': 145.8, 'weight': 145.8}
```

Building positions come from OpenStreetMap (© OpenStreetMap contributors, available under the Open Database License).

## pandas: tables of nodes and edges

{py:func}`~aryagraph.io.interop.to_pandas` returns two DataFrames: one row per node and one row per edge, with a column for every attribute.

```python
nodes_df, edges_df = ag.io.to_pandas(campus)
print(nodes_df.head(3).to_string())
print(edges_df.head(3).to_string())
print(edges_df.groupby("kind")["length"].agg(["count", "sum"]).round(1))
```

```text
   id                        name            kind        lat         lon              pos
0  AB  Art Building & Art Parkade            arts  51.075439 -114.130149    (86.5, 272.9)
1  AD     Administration Building  administration  51.078155 -114.127135   (297.0, -29.2)
2  AU                 Aurora Hall       residence  51.074896 -114.134045  (-185.7, 333.3)
  source target      kind  length  weight
0     AB     CH    pedway   141.2   141.2
1     AB     RT   outdoor   104.3   104.3
2     AD     PF  attached    85.7    85.7
          count     sum
kind                   
attached     26  1839.5
outdoor      43  5477.3
pedway       11  1160.7
tunnel        3   377.7
```

Once the edges are a DataFrame, pandas does the bookkeeping: the campus has 43 outdoor links and 40 indoor ones, and the three tunnels add up to 377.7 m. To go the other way, {py:func}`~aryagraph.io.interop.from_pandas_edgelist` builds a graph from any table with one row per edge. Filter first, then convert:

```python
tunnel_rows = edges_df[edges_df["kind"] == "tunnel"]
tunnels = ag.io.from_pandas_edgelist(tunnel_rows, edge_attr=["kind", "length"])
print(tunnels, list(tunnels.edges.data()))
```

```text
<Graph: 5 nodes, 3 edges> [('AU', 'DC', {'kind': 'tunnel', 'length': 114.1}), ('DC', 'KNA', {'kind': 'tunnel', 'length': 142.4}), ('MH', 'SB', {'kind': 'tunnel', 'length': 121.2})]
```

`source` and `target` are the default column names; pass `source=` and `target=` for others, and `directed=True` or `dag=True` for directed graphs. {py:func}`~aryagraph.io.interop.from_pandas` takes both tables and reverses `to_pandas` for nodes and edges (graph-level attributes, such as the graph's name, are not part of the tables):

```python
def same(a, b):
    """Same nodes and edges, with attributes and in the same order?"""
    nodes = list(a.nodes.data()) == list(b.nodes.data())
    return nodes, list(a.edges.data()) == list(b.edges.data())

again = ag.io.from_pandas(nodes_df, edges_df)
print(same(again, campus))
```

```text
(True, True)
```

Algorithm results convert too. A {py:class}`~aryagraph.core.results.NodeMap` has a `to_pandas()` method that returns a Series named after the metric, ready to join onto the node table:

```python
bc = ag.alg.betweenness_centrality(campus, weight="length")
table = nodes_df.set_index("id").join(bc.to_pandas())
top = table.sort_values("betweenness_centrality", ascending=False)
print(top[["name", "betweenness_centrality"]].head(3))
```

```text
                                                name  betweenness_centrality
id                                                                          
PF                            Professional Faculties                0.307071
KNB  Dr. Roger Jackson Kinesiology Complex (Block B)                0.301010
IH                               International House                0.217508
```

## networkx: round trips and cross-checks

{py:func}`~aryagraph.io.interop.to_networkx` and {py:func}`~aryagraph.io.interop.from_networkx` copy the graph with all node, edge and graph attributes, so you can move a graph into networkx for a function you need and bring the result back.

```python
import networkx as nx

G = ag.io.to_networkx(campus)
print(G)
theirs = nx.betweenness_centrality(G, weight="length")
print(max(abs(theirs[v] - bc[v]) for v in campus))

back = ag.io.from_networkx(G)
print(same(back, campus), back.attrs == campus.attrs)
```

```text
Graph named 'University of Calgary main campus' with 56 nodes and 83 edges
0.0
(True, True) True
```

The two libraries return identical betweenness values here: the largest difference is 0.0. That is by design: AryaGraph follows networkx's definitions where networkx has an established one, and its test suite checks those functions against networkx on seeded random graphs.

Any graph class accepts a networkx graph directly, which is the shortest way to bring in a networkx dataset:

```python
karate = ag.Graph(nx.karate_club_graph())
print(karate)
parts = ag.alg.louvain_communities(karate, seed=0)
print(len(parts), round(ag.alg.modularity(karate, parts), 3))
```

```text
<Graph "Zachary's Karate Club": 34 nodes, 78 edges>
3 0.435
```

A few rules keep the conversion predictable:

- `nx.Graph` becomes a {py:class}`~aryagraph.core.graph.Graph`, `nx.DiGraph` a {py:class}`~aryagraph.core.graph.DiGraph`, and `from_networkx(G, as_dag=True)` a {py:class}`~aryagraph.core.dag.DAG` (raising `CycleError` if the edges contain a cycle).
- AryaGraph graphs have at most one edge between two nodes. A networkx multigraph raises `GraphTypeError`, unless you pass `collapse=True` to merge parallel edges (later attributes win).
- Attribute dictionaries are copied shallowly: a list stored as an attribute is shared by both graphs.

## Files: choose a format for the job

{py:func}`~aryagraph.io.read` and {py:func}`~aryagraph.io.write` choose the format from the file extension. To see what each format keeps, write the campus, read it back and compare:

```python
def round_trip(path):
    ag.write(campus, path)
    h = ag.read(path)
    edges = {frozenset(e) for e in h.edges} == {frozenset(e) for e in campus.edges}
    lengths = all(h.edges[u, v].get("length") == d["length"]
                  for u, v, d in campus.edges.data())
    kept = [k for k in ("name", "kind", "lat", "lon", "pos")
            if all(h.nodes[n].get(k) == d[k] for n, d in campus.nodes.data())]
    return edges, lengths, kept, h.nodes["MSC"].get("pos")

for path in ["campus.json", "campus.graphml", "campus.gexf", "campus.dot", "campus.csv"]:
    print(f"{path:15}", round_trip(path))
```

```text
campus.json     (True, True, ['name', 'kind', 'lat', 'lon'], [-22.3, -32.7])
campus.graphml  (True, True, ['name', 'kind', 'lat', 'lon'], '[-22.3, -32.7]')
campus.gexf     (True, True, ['name', 'kind', 'lat', 'lon', 'pos'], (-22.3, -32.7))
campus.dot      (True, True, ['name', 'kind', 'lat', 'lon'], '-22.3,-32.7')
campus.csv      (True, True, [], None)
```

Every format keeps the 83 edges and their lengths, with numbers read back as numbers. They differ in what else they can hold:

| Format | Extension | Keeps | Use it for |
|---|---|---|---|
| Node-link JSON | `.json` | every attribute; tuples come back as lists | exchange with networkx and d3, archiving |
| GraphML | `.graphml` | scalar attributes with their types; other values as JSON text | yEd, Cytoscape, Gephi, networkx |
| GEXF | `.gexf` | scalar node and edge attributes; `pos` as a Gephi position; of the graph attributes only the name | Gephi |
| DOT | `.dot`, `.gv` | attributes as Graphviz attributes; `pos` as `"x,y"` text | Graphviz and diagram tools |
| Edge list | `.csv`, `.tsv`, `.txt` | edges and edge attributes only | spreadsheets, databases |

The `pos` attribute shows the differences: JSON returns the tuple as a list, GraphML as JSON text and DOT in Graphviz's own `"x,y"` notation, while GEXF stores it as the node's drawing position and returns the tuple. An edge list stores no node attributes at all, so building names are lost; keep a separate node table, as in the pipeline below.

### Diagrams: Mermaid and DOT

Mermaid and DOT are text languages for diagrams, and AryaGraph reads and writes both. That turns a diagram from a README into a graph you can analyze. Parse a small Mermaid flowchart:

```python
text = """
flowchart LR
    raw[Raw events] --> clean[Clean] --> features[Features]
    labels[Labels] --> join[Join] --> features
    clean --> join
    features --> train[Train model] --> evaluate[Evaluate]
    features --> baseline[Baseline] --> evaluate
    evaluate -->|approved| deploy[Deploy]
"""
flow = ag.io.from_mermaid(text)
print(flow, flow.attrs)
print(flow.nodes["raw"], flow.edges["evaluate", "deploy"])
plan = ag.DAG(flow)
print(plan.topological_order())
ag.draw(
    plan,
    labels="label",
    layout_options={"orientation": flow.attrs["direction"]},
    title="Pipeline read from Mermaid",
).save("mermaid_flow.svg")
```

```text
<DiGraph: 9 nodes, 10 edges> {'direction': 'LR'}
{'label': 'Raw events', 'shape': 'rect'} {'label': 'approved'}
['raw', 'labels', 'clean', 'join', 'features', 'train', 'baseline', 'evaluate', 'deploy']
```

```{figure} ../_static/generated/tutorials_b/mermaid_flow.png
:alt: Layered drawing of a nine-step data pipeline read from a Mermaid flowchart, from raw events and labels on the left to deploy on the right.
:width: 100%

The flowchart, laid out by AryaGraph's hierarchical layout in the direction the Mermaid header asked for (`LR`). Mermaid node texts become `label` attributes.
```

Wrapping the graph in {py:class}`~aryagraph.core.dag.DAG` checks that it has no cycle and gives you the DAG methods, such as `topological_order()` and `critical_path()`.

Writing works the same way. {py:func}`~aryagraph.io.mermaid.to_mermaid` returns text that GitHub, GitLab and the Mermaid live editor render:

```python
pipeline = ag.gen.ml_pipeline()
lines = ag.io.to_mermaid(pipeline, direction="LR").splitlines()
print("\n".join(lines[:3] + ["    ..."] + lines[-2:]))

ag.write(pipeline, "pipeline.mmd", direction="LR")
reread = ag.read("pipeline.mmd", node_names="label")
print(reread, set(reread) == set(pipeline))
```

```text
flowchart LR
    n0["ingest events"]
    n1["ingest labels"]
    ...
    n13 --> n14
    n14 --> n15
<DiGraph: 16 nodes, 19 edges> True
```

Node names that are not valid Mermaid ids are written as `n0`, `n1`, … with the name as the displayed text. Here that applies to every name, because they contain spaces; Mermaid keywords are replaced too (the campus building code `END` is one). Reading with `node_names="label"` restores the original names. Mermaid has no place for other attributes, so durations and teams are lost; DOT keeps them:

```python
ag.write(pipeline, "pipeline.dot")
print("\n".join(open("pipeline.dot", encoding="utf-8").read().splitlines()[:3]))
restored = ag.read("pipeline.dot", dag=True)
print(restored, restored.nodes["ingest events"])
print(restored.critical_path().length == pipeline.critical_path().length)
```

```text
digraph "ML pipeline" {
  graph [description="Train, compare and ship a churn-prediction model."];
  "ingest events" [duration=2.0, min=1.5, mode=2.0, max=4.0, kind=ingest, team="data-eng"];
<DAG 'ML pipeline': 16 nodes, 19 edges> {'duration': 2.0, 'min': 1.5, 'mode': 2.0, 'max': 4.0, 'kind': 'ingest', 'team': 'data-eng'}
True
```

`dag=True` returns a {py:class}`~aryagraph.core.dag.DAG`, so the critical path of the restored plan can be checked against the original.

## numpy and SciPy: matrices

For linear algebra, convert to an adjacency matrix. {py:func}`~aryagraph.io.interop.to_numpy` returns a dense array whose rows and columns follow the graph's node order (or the `nodes` you pass); {py:func}`~aryagraph.io.interop.to_scipy_sparse` returns a sparse array.

```python
order = list(campus)
A = ag.io.to_numpy(campus, weight="length")
print(A.shape, order[:4], A[order.index("MSC"), order.index("KNB")])

from scipy.sparse import csgraph

S = ag.io.to_scipy_sparse(campus, weight="length")
dist = csgraph.shortest_path(S, directed=False)
i, j = order.index("OO"), order.index("SH")
ours = ag.alg.shortest_path_length(campus, "OO", "SH", weight="length")
print(round(dist[i, j], 1), round(ours, 1))
```

```text
(56, 56) ['AB', 'AD', 'AU', 'BI'] 145.8
1001.8 1001.8
```

SciPy and AryaGraph find the same shortest-path length from the Olympic Oval (OO) to Scurfield Hall (SH): 1001.8 m, summed over the straight-line lengths of the links on the path. Keep `order` next to the matrix: it is the only link between row numbers and building codes.

The way back labels the rows with `nodes`, and every non-zero entry becomes an edge:

```python
rebuilt = ag.io.from_numpy(A, nodes=order, weight="length")
print(rebuilt, rebuilt.edges["MSC", "KNB"])

L = ag.alg.laplacian_matrix(campus)
eigenvalues = np.linalg.eigvalsh(L)
print(round(eigenvalues[1], 4), round(ag.alg.algebraic_connectivity(campus), 4))
```

```text
<Graph: 56 nodes, 83 edges> {'length': 145.8}
0.0477 0.0477
```

A symmetric matrix gives an undirected {py:class}`~aryagraph.core.graph.Graph`, an asymmetric one a {py:class}`~aryagraph.core.graph.DiGraph`. Because zero means "no edge", an edge whose weight is 0 does not survive the trip through a matrix. The last line shows the same quantity computed both ways: the second-smallest eigenvalue of the Laplacian, from numpy, equals {py:func}`~aryagraph.algorithms.matrix.algebraic_connectivity`.

## A mini pipeline: from CSV files to a figure

Real data often arrives as tables. To have files to work with, first write the campus as two CSV files, the way a facilities database might export them:

```python
export = nodes_df.drop(columns="pos").rename(columns={"id": "code"})
export.to_csv("buildings.csv", index=False)
edges_df.drop(columns="weight").to_csv("links.csv", index=False)
print("\n".join(open("buildings.csv", encoding="utf-8").read().splitlines()[:3]))
print("\n".join(open("links.csv", encoding="utf-8").read().splitlines()[:3]))
```

```text
code,name,kind,lat,lon
AB,Art Building & Art Parkade,arts,51.075439,-114.1301486
AD,Administration Building,administration,51.0781552,-114.1271352
source,target,kind,length
AB,CH,pedway,141.2
AB,RT,outdoor,104.3
```

The pipeline answers one question: **which buildings can you reach without going outside, and which ones carry the most indoor traffic?** Read the tables, keep the indoor links and build the graph:

```python
import pandas as pd

buildings = pd.read_csv("buildings.csv")
links = pd.read_csv("links.csv")
indoor_links = links[links["kind"].isin(["tunnel", "pedway", "attached"])]
indoor = ag.io.from_pandas(buildings, indoor_links, node_id="code")
print(indoor)

parts = ag.alg.connected_components(indoor)
main = max(parts, key=len)
others = [len(p) for p in parts if p is not main]
print(len(main), "buildings in the largest indoor network")
print(len(others), "other components, of sizes", set(others))
```

```text
<Graph: 56 nodes, 40 edges>
38 buildings in the largest indoor network
18 other components, of sizes {1}
```

Passing the building table as well keeps the buildings without an indoor link in the graph, as isolated nodes. Next, rank the buildings of the main indoor network by betweenness, with each link's straight-line `length` in meters as its distance, and save the ranking as a table:

```python
core = indoor.subgraph(main)
bc_indoor = ag.alg.betweenness_centrality(core, weight="length")
ranking = buildings.set_index("code").join(bc_indoor.to_pandas().rename("betweenness"))
ranking = ranking.sort_values("betweenness", ascending=False)
ranking.to_csv("indoor_betweenness.csv")
print(ranking[["name", "betweenness"]].head(5).round(3))
```

```text
                         name  betweenness
code                                      
PF     Professional Faculties        0.498
SS            Social Sciences        0.483
AD    Administration Building        0.468
ES             Earth Sciences        0.392
SB                  Science B        0.363
```

Professional Faculties, Social Sciences and the Administration Building lie on about half of the shortest indoor routes between the other buildings of the network. The reason is structural: the indoor network is close to a tree, so most routes have no alternative. {py:func}`~aryagraph.algorithms.connectivity.bridges` finds the links whose loss would split it:

```python
print(len(ag.alg.bridges(core)), "of", core.num_edges, "indoor links are bridges")
cut = core.copy()
cut.remove_edge("AD", "SS")
print(sorted(len(p) for p in ag.alg.connected_components(cut)))
```

```text
30 of 40 indoor links are bridges
[14, 24]
```

Closing the pedway between the Administration Building and Social Sciences would split the indoor network into two parts of 14 and 24 buildings.

Finally, draw the result in real geometry. Convert latitude and longitude to meters around the campus center (x to the east, y to the south, because AryaGraph uses screen coordinates with y pointing down) and pass the positions as the layout:

```python
lat0, lon0 = buildings["lat"].mean(), buildings["lon"].mean()
kx = 111_320 * math.cos(math.radians(lat0))  # meters per degree of longitude
ky = 110_574                                 # meters per degree of latitude
pos = {
    row.code: ((row.lon - lon0) * kx, (lat0 - row.lat) * ky)
    for row in buildings.itertuples()
}
building_colors = {"indoor network": "#2a78d6", "no indoor link": "#a3a199"}
link_colors = {"attached": "#1baf7a", "pedway": "#eb6834", "tunnel": "#4a3aa7"}
status = {v: ("indoor network" if v in main else "no indoor link") for v in indoor}
fig = ag.draw(
    indoor,
    layout=pos,
    node_color=ag.by(status, title="building", palette=building_colors),
    node_size=ag.by({v: bc_indoor.get(v, 0.0) for v in indoor}, title="betweenness"),
    edge_color=ag.by("kind", title="link", palette=link_colors),
    title="Indoor routes on the University of Calgary campus",
    subtitle=f"{len(main)} of {len(indoor)} buildings connected without going outside",
    caption="Building positions © OpenStreetMap contributors (ODbL)",
)
fig.save("indoor_routes.svg")
fig.save("indoor_routes.html")
```

Each `palette` maps categories to colors: gray keeps the buildings without an indoor link in the background, and each kind of link gets its own color.

```{figure} ../_static/generated/tutorials_b/indoor_routes.png
:alt: Map-like drawing of University of Calgary buildings; 38 buildings joined by tunnels, pedways and attached links form one indoor network, 18 have no indoor link, and node size shows betweenness.
:width: 100%

The indoor network in real campus geometry. Building positions © OpenStreetMap contributors (ODbL). <a href="../_static/generated/tutorials_b/indoor_routes.html">Interactive version</a>.
```

Every intermediate result in this pipeline is a plain DataFrame, graph or `NodeMap`, so you can inspect or test each step on its own, and the saved `indoor_betweenness.csv` goes back to whoever sent you the data.

## Next steps

- [File formats and interoperability](../user-guide/io.md) in the user guide covers each format, including GEXF and adjacency lists.
- [Exporting](../user-guide/exporting.md) covers SVG, PNG, PDF and interactive HTML output.
- [Campus routing](campus-routing.md) works with the same network in depth.
- [Datasets and generators](../user-guide/datasets.md) lists the other bundled networks and example DAGs.
