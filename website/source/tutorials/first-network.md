# Your first network

In this tutorial you build a small network of University of Calgary buildings by hand, then load the full campus dataset that ships with AryaGraph. Along the way you inspect a graph, measure it, draw it and save the drawing in several formats.

**Goal:** go from an empty graph to a saved, interactive drawing of a real network, and understand each object you touch on the way.

**Prerequisites:**

- AryaGraph installed (see [Installation](../getting-started/installation.md)); numpy is its only runtime dependency.
- Python 3.10 or newer and basic Python (lists, dicts, loops).
- For PNG and PDF export: `cairosvg` or an installed Chrome, Edge or Chromium. SVG and HTML export need nothing extra.

No graph theory is assumed. Every code block on this page runs as shown, top to bottom, in one Python session.

## Step 1: Create a graph and add nodes

A graph is a set of *nodes* joined by *edges*. Here, each node is a campus building, identified by its official building code. Create an empty undirected graph and add nine buildings in the science core, each with a full `name` and a `kind`:

```python
import aryagraph as ag

g = ag.Graph(name="Science core")
g.add_node("MSC", name="MacEwan Student Centre", kind="student-life")
g.add_node("MH", name="MacEwan Hall", kind="student-life")
g.add_node("TI", name="Taylor Institute for Teaching and Learning", kind="academic")
g.add_node("SB", name="Science B", kind="academic")
g.add_node("SA", name="Science A", kind="academic")
g.add_node("ES", name="Earth Sciences", kind="academic")
g.add_node("MS", name="Mathematical Sciences", kind="academic")
g.add_node("ST", name="Science Theatres", kind="academic")
g.add_node("SS", name="Social Sciences", kind="academic")
print(g)
```

```text
<Graph 'Science core': 9 nodes, 0 edges>
```

Any hashable value can be a node (strings, numbers, tuples). Keyword arguments become the node's *attributes*, stored in a plain dict. Nodes keep their insertion order, and every algorithm, layout and drawing in AryaGraph follows that order, so the same input produces the same output.

## Step 2: Connect the buildings

Each edge below is a real connection from the campus dataset: `kind` says how you get from one building to the other (`attached` buildings share an internal door, a `pedway` is an enclosed bridge, a `tunnel` runs underground and `outdoor` is a walking link), and `length` is the straight-line distance between the two buildings in meters.

```python
g.add_edges([
    ("MSC", "MH", {"kind": "attached", "length": 97.4}),
    ("MSC", "TI", {"kind": "outdoor", "length": 108.1}),
    ("MH", "TI", {"kind": "outdoor", "length": 83.4}),
    ("MH", "SB", {"kind": "tunnel", "length": 121.2}),
    ("SB", "ES", {"kind": "attached", "length": 86.5}),
    ("SB", "SA", {"kind": "attached", "length": 92.7}),
    ("ES", "MS", {"kind": "pedway", "length": 90.5}),
    ("MS", "ST", {"kind": "attached", "length": 58.0}),
    ("SA", "ST", {"kind": "attached", "length": 88.7}),
    ("SA", "SS", {"kind": "attached", "length": 82.4}),
    ("ST", "SS", {"kind": "attached", "length": 58.8}),
])
print(g)
```

```text
<Graph 'Science core': 9 nodes, 11 edges>
```

An edge is a tuple `(u, v)`, `(u, v, attrs)` or `(u, v, weight)`. Adding an edge creates any endpoint that does not exist yet, so for quick experiments you can skip `add_node` altogether. Because the graph is undirected, `("MSC", "MH")` and `("MH", "MSC")` are the same edge.

## Step 3: Inspect what you built

The `nodes` and `edges` views give you attribute dicts; `neighbors` and `degree` describe the structure around a node:

```python
print(g.nodes["SB"])
print(g.edges["MH", "SB"])
print(g.edges["SB", "MH"] is g.edges["MH", "SB"])
print(list(g.neighbors("SA")))
print(g.degree("SA"))
```

```text
{'name': 'Science B', 'kind': 'academic'}
{'kind': 'tunnel', 'length': 121.2}
True
['SB', 'ST', 'SS']
3
```

The third line prints `True`: an undirected edge has one attribute dict, reachable from both ends, so updating it from either side updates the same data.

To look at every node or edge at once, iterate over `.data()`. With an attribute name, it yields just that attribute:

```python
for u, v, kind in g.edges.data("kind"):
    if kind != "attached":
        print(f"{u}-{v}: {kind}, {g.edges[u, v]['length']} m")
```

```text
MSC-TI: outdoor, 108.1 m
MH-TI: outdoor, 83.4 m
MH-SB: tunnel, 121.2 m
ES-MS: pedway, 90.5 m
```

`g.degree()` without an argument returns the degree of every node as a {py:class}`~aryagraph.core.results.NodeMap`: a dict with helpers such as `top()`, `rank()` and `describe()`.

```python
deg = g.degree()
print(deg)
print(deg.top(3))
```

```text
NodeMap('degree', 9 nodes; top: {'MH': 3, 'SB': 3, 'SA': 3, …})
[('MH', 3), ('SB', 3), ('SA', 3)]
```

Ties keep insertion order, so `top()` returns the same answer on every run.

## Step 4: Measure the network

Algorithms live under `ag.alg`. Each takes the graph first and returns plain data or a `NodeMap`. Start with two global numbers and one route:

```python
print(round(ag.alg.density(g), 3))
print(ag.alg.is_connected(g))


def route_length(graph, path):
    """Total length in meters of a route given as a list of nodes."""
    return sum(graph.edges[u, v]["length"] for u, v in zip(path, path[1:]))


by_hops = ag.alg.shortest_path(g, "ES", "SS")
by_meters = ag.alg.shortest_path(g, "ES", "SS", weight="length")
print(by_hops, round(route_length(g, by_hops), 1))
print(by_meters, round(route_length(g, by_meters), 1))
```

```text
0.306
True
['ES', 'SB', 'SA', 'SS'] 261.6
['ES', 'MS', 'ST', 'SS'] 207.3
```

Density is the share of possible edges that exist: 11 of the 36 possible pairs of nine buildings. `is_connected` confirms that every building can reach every other one.

The two routes from Earth Sciences (ES) to Social Sciences (SS) both take three steps. Without a weight, {py:func}`~aryagraph.algorithms.paths.shortest_path` counts steps and returns the first three-step route it finds, through Science B and Science A. With `weight="length"` it adds up meters and returns the route through Mathematical Sciences and Science Theatres, which is 54.3 m shorter. Whenever edges have a real cost, pass it as `weight`.

Next, find out which buildings the short routes pass through. *Betweenness centrality* is the share of shortest paths between other pairs of nodes that run through a node; with `weight="length"` those paths are measured in meters:

```python
bc = ag.alg.betweenness_centrality(g, weight="length")
for building, score in bc.top(3):
    print(building, round(score, 3))
```

```text
SB 0.571
MH 0.429
SA 0.286
```

The scores are shares of the 28 pairs of other buildings. Science B lies on 16 of those 28 shortest routes (0.571), including all 15 that join MacEwan Student Centre, MacEwan Hall or the Taylor Institute to the five buildings beyond Science B. The MacEwan Hall to Science B tunnel is the only link between those two groups, so every route from one group to the other uses it. Pass `normalized=False` to get the raw counts instead of shares.

## Step 5: Draw it

{py:func}`ag.draw <aryagraph.render.draw>` turns a graph into a {py:class}`~aryagraph.render.figure.Figure`. Each visual channel accepts a constant, an attribute name, a `{node: value}` mapping (such as the `NodeMap` you just computed) or an {py:func}`ag.by <aryagraph.style.scales.by>` spec when you want to set the legend title or the scale yourself:

```python
fig = ag.draw(
    g,
    node_color="kind",
    node_size=ag.by(bc, title="betweenness"),
    edge_label="length",
    label_position="below",
    title="Nine buildings in the science core",
    subtitle="Node size: betweenness by length · edge labels: meters",
)
print(fig)
```

```text
<Figure 540×322: 9 nodes, 11 edges, stress layout, theme 'light'>
```

```{figure} ../_static/generated/tutorials_a/first_core.png
:alt: Nine campus buildings drawn as circles colored by kind, with edge labels giving lengths in meters; Science B and MacEwan Hall are the largest nodes.
:width: 640px

The hand-built network. Color comes from the `kind` attribute and size from betweenness; both get a legend automatically. Positions come from the default layout for undirected graphs (stress majorization), computed with the default `seed=0`.
```

Two choices in this call are worth a note. `label_position="below"` keeps the building codes outside the circles: a label drawn inside a node enlarges the node to fit its text, which would distort a size encoding. Edge labels are numbers formatted for display, so 121.2 appears as 121. And because no `layout` was given, AryaGraph picked one: stress majorization for general graphs, a layered layout for DAGs. See [Layouts](../user-guide/layouts.md) for the available engines.

## Step 6: Save the drawing

`fig.save()` picks the format from the file extension:

```python
fig.save("science-core.svg")    # vector, for the web and for editing
fig.save("science-core.html")   # interactive page: pan, zoom, hover, search, table view
fig.save("science-core.png")    # raster, 2x the display size by default
fig.save("science-core.pdf")    # vector, for print
```

SVG and HTML are written directly. PNG and PDF go through `cairosvg` when it is installed, otherwise through a headless Chrome, Edge or Chromium; if neither is available, those two calls raise `DependencyError` with installation advice. The HTML file is self-contained (no CDN, no server), so you can open it offline, email it or upload it to any static web host. See [Exporting](../user-guide/exporting.md) for the export options.

## Step 7: Load the full campus

The network you typed is a small piece of a bundled dataset: 56 main-campus buildings of the University of Calgary and the tunnels, pedways, attached-building doors and walking links between them. Load it with one call:

```python
campus = ag.gen.ucalgary_campus()
print(campus)
print(campus.nodes["TFDL"])
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
{'name': 'Taylor Family Digital Library', 'kind': 'library', 'lat': 51.0774272, 'lon': -114.1299715, 'pos': (98.8, 51.8)}
```

Besides `name` and `kind`, each building carries its latitude and longitude and a `pos`: its position in meters on a local plane centered on the campus (x points east, y points south), derived from OpenStreetMap. Check that your hand-typed edges match the dataset:

```python
for u, v, d in g.edges.data():
    assert campus.edges[u, v] == {**d, "weight": d["length"]}
core = campus.subgraph(g.nodes)
print(core.num_edges == g.num_edges)
```

```text
True
```

Every edge agrees (the dataset also stores `length` as `weight`, the attribute that functions such as `dijkstra` and `k_shortest_paths` read by default), and the campus has no other links among those nine buildings.

## Step 8: Inspect and measure the campus

The same tools scale up. Count buildings and links by kind:

```python
from collections import Counter

print(Counter(kind for _, kind in campus.nodes.data("kind")).most_common())
print(Counter(kind for _, _, kind in campus.edges.data("kind")).most_common())
```

```text
[('academic', 25), ('residence', 9), ('services', 5), ('arts', 4), ('student-life', 4), ('research', 3), ('athletics', 3), ('administration', 2), ('library', 1)]
[('outdoor', 43), ('attached', 26), ('pedway', 11), ('tunnel', 3)]
```

Academic buildings make up 25 of the 56. Of the 83 links, 43 are modeled outdoor walks and 40 are indoor connections (26 attached-building links, 11 pedways and 3 tunnels).

Then measure it:

```python
print(ag.alg.is_connected(campus), round(ag.alg.density(campus), 4))
print(campus.degree().top(3))

campus_bc = ag.alg.betweenness_centrality(campus, weight="length")
print([(b, round(s, 3)) for b, s in campus_bc.top(3)])
```

```text
True 0.0539
[('ES', 5), ('CD', 4), ('CH', 4)]
[('PF', 0.307), ('KNB', 0.301), ('IH', 0.218)]
```

Every building can reach every other one, but the network is sparse: about 5.4% of all possible building pairs are directly linked. Earth Sciences (ES) has the most links, five. Measured in meters, Professional Faculties (PF) and Block B of the Dr. Roger Jackson Kinesiology Complex (KNB) each lie on about 30% of the shortest routes between other buildings, which makes them the campus's main crossroads.

For a broader first look, {py:func}`ag.analyze <aryagraph.analysis.report.analyze>` runs the measures that apply to the graph and returns a report; printing it gives a text summary:

```python
report = ag.analyze(campus)
print(report)
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

Called without `weight`, the report measures distances and centralities in hops, not meters. That is why its top betweenness building differs from the length-weighted ranking above. One section differs: community detection falls back to the edge attribute named `weight`, which holds meters on the campus graph, so the 8 communities above treat long links as strong ties.

Passing `weight="length"` to `analyze` does not turn the whole report into a distance analysis. Betweenness and closeness then use meters, but PageRank, eigenvector centrality and community detection read the weight as a connection strength, so a long walk counts as a strong tie, and diameter, radius and average path length stay in hops. For rankings by distance, call the measure directly with `weight="length"`, as you did with `betweenness_centrality` above. To keep the report, call `report.save("campus-report.html")`; it writes an interactive dashboard.

## Step 9: Draw the campus on its real geometry

Any `{node: (x, y)}` mapping can serve as a layout. Passing the `pos` attribute draws the campus north-up, to scale:

```python
pos = {n: d["pos"] for n, d in campus.nodes.data()}
fig = ag.draw(
    campus,
    layout=pos,
    node_color="kind",
    node_size=ag.by(campus_bc, title="betweenness (by length)"),
    title="University of Calgary main campus",
    subtitle="56 buildings and 83 links · positions © OpenStreetMap contributors",
)
fig.save("campus.html")
fig.save("campus.svg")
```

```{figure} ../_static/generated/tutorials_a/first_campus.png
:alt: Map-like drawing of 56 University of Calgary buildings colored by kind, with node size showing length-weighted betweenness.
:width: 100%

The campus drawn with its real building positions. Node size is betweenness computed on edge lengths. Building positions © OpenStreetMap contributors (ODbL).
```

The campus has nine kinds of building. The categorical palette gives the seven most frequent their own colors and folds the other two (administration and library) into "Other (2)", so the legend stays readable.

The interactive version below is the file `campus.html` from the code above. Hover a building to highlight its neighbors, click to pin its details, drag buildings to rearrange them, or open the table view and sort the buildings by any column.

<iframe class="ag-embed" src="../_static/generated/tutorials_a/first_campus.html" height="700" loading="lazy" title="Interactive drawing of the University of Calgary campus network"></iframe>
<p class="ag-embed-note">Interactive: pan, zoom, hover, search and table view. Building positions © OpenStreetMap contributors (ODbL). <a href="../_static/generated/tutorials_a/first_campus.html">Open full screen</a></p>

## Recap

You have:

- created a {py:class}`~aryagraph.core.graph.Graph`, added nodes and edges with attributes, and read them back through `g.nodes`, `g.edges`, `neighbors()` and `degree()`;
- measured density, connectivity, shortest paths (by hops and by meters) and betweenness with `ag.alg`;
- mapped data to color and size with `ag.draw`, and saved SVG, HTML, PNG and PDF files;
- loaded the University of Calgary campus with {py:func}`ag.gen.ucalgary_campus <aryagraph.generators.datasets.ucalgary_campus>`, verified your edges against it, summarized it with `ag.analyze`, and drawn it on its real geometry.

## Next steps

- [Campus routing](campus-routing.md): indoor-only routes, alternative routes and chokepoints on the same dataset.
- [Pipelines as DAGs](pipeline-dag.md): directed acyclic graphs, critical paths and schedules.
- [Publication figures](publication-figures.md): turn a drawing into a figure ready for a paper.
- User guide: [Graphs](../user-guide/graphs.md), [Drawing](../user-guide/drawing.md), [Algorithms](../user-guide/algorithms.md) and [Datasets and generators](../user-guide/datasets.md).
