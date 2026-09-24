# Concepts

AryaGraph is built from a small set of ideas: graphs that keep their insertion order, algorithms that are plain functions returning dictionaries, layouts that are data, and a renderer that turns everything into one backend-neutral scene. This page explains each idea, why it works the way it does, and what you can rely on. The code continues from block to block, like the [Quickstart](quickstart.md).

```{figure} ../_static/generated/getting_started/concepts_flow.png
:alt: Left-to-right diagram. A Graph, DiGraph or DAG feeds ag.alg functions, which return NodeMap or EdgeMap results; ag.layout.compute, which returns a Layout; ag.sim models, which return a SimulationResult; and ag.draw. ag.draw also takes the results and the layout and builds a Scene, which becomes a Figure. A SimulationResult reaches a Figure through animate(). A Figure writes .svg, .html and .png or .pdf files.
:width: 100%

How the main objects (blue) and functions (orange) connect, from a graph to the files a figure writes (green). The diagram is itself a DAG drawn by AryaGraph.
```

## Graph classes

AryaGraph has three graph classes. All of them are *simple*: at most one edge joins a pair of nodes (one per direction for directed graphs).

| Class | Edges | Typical use |
|---|---|---|
| {py:class}`~aryagraph.core.graph.Graph` | Undirected; self-loops allowed | Symmetric relations: roads, walkways, collaborations |
| {py:class}`~aryagraph.core.graph.DiGraph` | Directed; self-loops allowed | Flows, citations, links that can form loops |
| {py:class}`~aryagraph.core.dag.DAG` | Directed; no cycles, no self-loops | Pipelines, schedules, prerequisites, build systems |

Nodes can be any hashable value except `None`: strings, numbers, tuples. Adding an edge creates missing endpoints. An edge can be written as `(u, v)`, `(u, v, {attributes})` or `(u, v, number)`, which stores the number as `weight`:

```python
import aryagraph as ag

road = ag.Graph([("a", "b"), ("b", "c", 2.5)])
links = ag.DiGraph([("x", "y"), ("y", "x")])
plan = ag.DAG([("design", "build"), ("build", "ship")])
print(road, links, plan, sep="\n")
print(road.edges["c", "b"])
```

```text
<Graph: 3 nodes, 2 edges>
<DiGraph: 2 nodes, 2 edges>
<DAG: 3 nodes, 2 edges>
{'weight': 2.5}
```

`DAG` is a `DiGraph` that checks every change. An edge that would close a cycle raises {py:class}`~aryagraph.core.exceptions.CycleError`, whose `cycle` attribute lists the loop. A batch added with `add_edges()` is validated once, in linear time, and applied completely or not at all:

```python
try:
    plan.add_edges([("ship", "review"), ("review", "design")])
except ag.CycleError as err:
    print(type(err).__name__, "| the loop visits", len(err.cycle) - 1, "nodes:", sorted(set(err.cycle)))
print(plan, "review" in plan)
```

```text
CycleError | the loop visits 4 nodes: ['build', 'design', 'review', 'ship']
<DAG: 3 nodes, 2 edges> False
```

The plan still has its original three nodes: the new node `review` was rolled back along with the edges. The error message and `err.cycle` spell out the loop as a closed list of nodes, with the first node repeated at the end. On top of the `DiGraph` interface, a `DAG` adds methods such as `topological_order()`, `sources()`, `sinks()`, `levels()`, `ancestors()` and `critical_path()`. See [Graphs](../user-guide/graphs.md) and [DAG workflows](../user-guide/dags.md) for the full interface.

## Insertion order is graph order

Nodes iterate in the order they were first added, and each node's neighbors iterate in the order their edges were added:

```python
g = ag.Graph()
g.add_edge("b", "a")
g.add_edge("c", "a")
g.add_node("z")
print(list(g))
print(list(g.neighbors("a")))
print(g.node_index())

fixed = ag.Graph([("c", "a"), ("b", "a")], nodes=["a", "b", "c"])
print(list(fixed))
```

```text
['b', 'a', 'c', 'z']
['b', 'c']
{'b': 0, 'a': 1, 'c': 2, 'z': 3}
['a', 'b', 'c']
```

This order carries through the library. Algorithms visit nodes in graph order, matrices use it as their row order (`node_index()` gives the mapping), topological sorts use it to break ties, layout engines return nodes in it, and simulation results use it for their columns. The reason is reproducibility: the same graph, built in the same order, gives the same results, including tie-breaks. Pass `nodes=[...]` to the constructor when you want to fix the order yourself.

## Attributes and views

Every graph, node and edge has a plain `dict` of attributes. `g.attrs` holds the graph's; `g.nodes[n]` and `g.edges[u, v]` return the node's and edge's own dictionaries, so assigning into them edits the graph in place. An undirected edge has a single dictionary, reachable from both orientations:

```python
g = ag.Graph(name="demo")
g.add_node("a", color="red")
g.add_edge("a", "b", weight=2.0)
print(g.attrs)
print(g.nodes["a"], g.edges["b", "a"])

g.nodes["a"]["color"] = "blue"
nodes = g.nodes
g.add_node("c", color="green")
print(len(nodes), list(nodes.data("color")))
print(g.nodes(data=True))
```

```text
{'name': 'demo'}
{'color': 'red'} {'weight': 2.0}
3 [('a', 'blue'), ('b', None), ('c', 'green')]
[('a', {'color': 'blue'}), ('b', {}), ('c', {'color': 'green'})]
```

`g.nodes`, `g.edges`, `g.adj`, `g.succ` and `g.pred` are *views*: live, read-only windows onto the graph. The view stored in `nodes` above saw node `c` appear without being fetched again, because a view holds a reference to the graph rather than a copy. The structure itself is meant to change only through graph methods such as `add_edge()` and `remove_node()`. `data(key)` iterates over one attribute, and the call form `g.nodes(data=True)` returns a list of `(node, attrs)` pairs; networkx returns a view with the same pairs.

Lookups of missing nodes raise {py:class}`~aryagraph.core.exceptions.NodeNotFound`, which is also a `KeyError`, so existing `except KeyError` code keeps working:

```python
try:
    g.nodes["missing"]
except KeyError as err:
    print(type(err).__name__, "|", err)
```

```text
NodeNotFound | node 'missing' is not in the graph
```

## Results: NodeMap and EdgeMap

Per-node results are {py:class}`~aryagraph.core.results.NodeMap` objects and per-edge results are {py:class}`~aryagraph.core.results.EdgeMap` objects. Both are `dict` subclasses, so anything that accepts a dictionary accepts them, and both add the operations people usually need next:

```python
campus = ag.gen.ucalgary_campus()
pr = ag.alg.pagerank(campus)
print(pr)
print(pr.top(3))
print(pr.rank()["ES"], round(pr.normalized()["ES"], 3))
print({k: round(v, 4) for k, v in pr.describe().items()})
print(pr.to_array(["ES", "MSC"]))
```

```text
NodeMap('pagerank', 56 nodes; top: {'MTH': 0.03484, 'CR': 0.02946, 'ES': 0.028, …})
[('MTH', 0.03484234822228259), ('CR', 0.029458538108272647), ('ES', 0.027999638254937433)]
3 0.804
{'count': 56, 'mean': 0.0179, 'std': 0.0057, 'min': 0.0057, '25%': 0.0137, '50%': 0.0172, '75%': 0.0211, 'max': 0.0348}
[0.02799964 0.01850369]
```

| Method | Returns |
|---|---|
| `top(k)`, `bottom(k)` | The *k* highest or lowest `(key, value)` pairs |
| `argmax()`, `argmin()` | The key with the largest or smallest value |
| `rank()` | Dense 1-based ranks (1 = highest) |
| `describe()` | Count, mean, standard deviation, min, quartiles, max |
| `normalized(method)` | Values divided by the largest absolute value or by the sum, mapped to [0, 1], or standardized |
| `to_array(keys)` | A numpy array in the order you give (graph order by default) |
| `map(fn)`, `filter(pred)` | Transformed or filtered copies |
| `to_pandas()` | A `pandas.Series` (needs the `interop` extra) |

A map's `name` (here `'pagerank'`) becomes the legend title when you pass it to a drawing. An `EdgeMap` is keyed by `(u, v)` tuples:

```python
eb = ag.alg.edge_betweenness_centrality(campus, weight="length")
print(eb)
print(isinstance(pr, dict), isinstance(eb, dict))
```

```text
EdgeMap('edge_betweenness_centrality', 83 edges; top: {('IH', 'RC'): 0.2227, ('MFH', 'PF'): 0.2188, ('RC', 'RT'): 0.2169, …})
True True
```

## Algorithms are functions

Every algorithm is a function in `ag.alg` that takes the graph as its first argument and options as keywords. Algorithms read the graph without modifying it. They return plain values (numbers, lists, sets), `NodeMap` or `EdgeMap` objects, new graphs (a spanning tree, a transitive reduction), or small result classes with named fields, such as {py:class}`~aryagraph.algorithms.dag.CriticalPath`. Functions are grouped into modules (paths, centrality, communities, flow and so on), which the [API reference](../reference/algorithms.rst) follows.

The `weight` argument follows one convention across the library:

- `None`: every edge counts as 1;
- a string: the name of an edge attribute (edges without it count as 1);
- a function `f(u, v, attrs)`: any rule you like, computed per edge.

A function lets you encode preferences. This one counts outdoor meters 1.5 times, which favors tunnels and pedways on a cold day:

```python
indoor = {"tunnel", "pedway", "attached"}

def effort(u, v, d):
    """Meters walked, with outdoor links counted 1.5 times."""
    return d["length"] * (1.0 if d["kind"] in indoor else 1.5)

def meters(path, kinds=None):
    steps = zip(path, path[1:])
    return sum(campus.edges[u, v]["length"] for u, v in steps if kinds is None or campus.edges[u, v]["kind"] in kinds)

shortest = ag.alg.shortest_path(campus, "OO", "SH", weight="length")
sheltered = ag.alg.shortest_path(campus, "OO", "SH", weight=effort)
print(" → ".join(sheltered))
for path in (shortest, sheltered):
    print(f"{meters(path):,.0f} m in total, {meters(path, {'outdoor'}):,.0f} m outdoors")
```

```text
OO → KNB → MSC → MH → SB → SA → SS → AD → PF → EDT → SH
1,002 m in total, 266 m outdoors
1,073 m in total, 68 m outdoors
```

The sheltered route is 71 m longer but cuts the outdoor distance from 266 m to 68 m.

Failures raise typed exceptions that carry their evidence instead of returning a sentinel value. All of them derive from {py:class}`~aryagraph.core.exceptions.AryaGraphError`, so you can catch the library's errors without hiding unrelated ones:

```python
try:
    ag.alg.shortest_path(campus, "OO", "XYZ")
except ag.AryaGraphError as err:
    print(type(err).__name__, "|", err)
```

```text
NodeNotFound | node 'XYZ' is not in the graph
```

Where networkx has an established definition (betweenness normalization, PageRank's handling of dangling nodes, and so on), AryaGraph follows it and its test suite compares the two on seeded random graphs. See [Algorithms](../user-guide/algorithms.md) for the catalog.

## Layouts are data

A layout engine returns a {py:class}`~aryagraph.layout.base.Layout`: the node order, an `(n, 2)` array of positions, optional bend points for routed edges, the method name, and a `metric` flag. {py:func}`~aryagraph.layout.compute` picks an engine by name:

```python
lay = ag.layout.compute(campus, "stress", seed=0)
print(lay)
print(lay.nodes[:3], lay.xy.shape)
print(lay["MSC"])

tree = ag.layout.compute(ag.gen.balanced_tree(2, 3), "tree")
print(tree)
print(ag.layout.auto_method(campus), ag.layout.auto_method(plan))
```

```text
<Layout 'stress': 56 nodes, abstract, 0 routed edges>
['AB', 'AD', 'AU'] (56, 2)
[-1.34263778  0.07048128]
<Layout 'tree': 15 nodes, metric, 0 routed edges>
stress hierarchical
```

Three rules explain how layouts behave in a drawing:

- **Screen coordinates.** Every layout uses x to the right and y *downward*, as SVG and HTML do. A top-to-bottom hierarchy has its sources at small y. Data in map coordinates (y pointing north) needs its y flipped; the campus dataset stores `pos` with y pointing south for this reason.
- **Abstract or metric.** An *abstract* layout (stress, force-directed, spectral, circular, or your own positions) only fixes relative positions. `draw()` scales it uniformly, so shapes and proportions are kept, until a typical edge is a comfortable length on screen; for up to 3,000 nodes it then nudges apart nodes that overlap. Pass `avoid_overlap=False` to keep your coordinates untouched. A *metric* layout (hierarchical, tree, radial) is already in pixels, spaced around each node's box and label, and `draw()` keeps its scale.
- **Automatic choice.** `layout="auto"`, the default, uses the hierarchical engine for DAGs of up to 2,000 nodes, stress for other graphs of up to 3,000 nodes, and ForceAtlas2 beyond that.

Because a layout is data, you can build one from coordinates you already have, inspect it and transform it:

```python
geo = ag.Layout.from_positions({n: d["pos"] for n, d in campus.nodes.data()})
print(geo)
x0, y0, x1, y1 = geo.bounds()
print(f"{x1 - x0:,.0f} m east-west, {y1 - y0:,.0f} m north-south")
```

```text
<Layout 'custom': 56 nodes, abstract, 0 routed edges>
1,522 m east-west, 1,220 m north-south
```

`rotated()`, `flipped()`, `scaled()`, `translated()` and `subset()` return new layouts. `draw(layout=...)` accepts a method name, a `Layout`, a `{node: (x, y)}` mapping or a function `f(g)`, and `layout_options={...}` passes engine options such as `orientation`. See [Layouts](../user-guide/layouts.md).

## From draw() to pixels: scene, figure and backends

{py:func}`~aryagraph.render.draw` works in two stages. First it resolves every encoding, computes or accepts the layout, converts it to pixels, clips edges to node outlines, places labels without collisions and sizes the legends. The result is a {py:class}`~aryagraph.render.scene.Scene`: a list of positioned marks in final pixel coordinates. Then it wraps the scene in a {py:class}`~aryagraph.render.figure.Figure`, and each output format serializes that same scene:

```python
fig = ag.draw(campus, node_color="kind")
scene = fig.scene
print(fig)
print(len(scene.nodes), len(scene.edges), [lg[0].title for lg in scene.legends])
mark = scene.node_index["MSC"]
print(mark.node, mark.shape, mark.fill, mark.w)
svg, page = fig.to_svg(), fig.to_html()
print(svg[:4], svg in page)
```

```text
<Figure 861×747: 56 nodes, 83 edges, stress layout, theme 'light'>
56 83 ['kind']
MSC circle #4a3aa7 14.0
<svg True
```

- **SVG** writes the marks as SVG elements.
- **HTML** embeds that same SVG, unchanged, in a self-contained page and adds a script for pan, zoom, hover, search, dragging and the table view.
- **PNG and PDF** convert the SVG, with cairosvg when it is installed or with a headless Chromium-family browser otherwise.

Because the backends only serialize, a node sits at the same place in every format. The scene is also the place to look when you need the computed geometry, and `fig.layout` returns the pixel-space layout that was drawn. See [Drawing](../user-guide/drawing.md) and [Exporting](../user-guide/exporting.md).

## Encodings map data to visual channels

Each visual channel of `draw()` (`node_color`, `node_size`, `node_shape`, `node_opacity`, `labels`, `edge_color`, `edge_width`, `edge_label`, `tooltip`) accepts the same kinds of value:

| You pass | Example | Meaning |
|---|---|---|
| A constant | `edge_width=1.5`, `node_color="#e34948"` | Same value everywhere |
| An attribute name | `node_color="kind"` | Read from each node's or edge's attributes |
| A mapping | `node_size=campus.degree()` | `{node: value}`, such as any `NodeMap` |
| A function | `lambda n, d: ...` | Nodes: `f(node)` or `f(node, attrs)`; edges: `f(u, v)` or `f(u, v, attrs)` |
| A sequence | `[3, 1, 2, ...]` | One value per node, in graph order |
| An `ag.by(...)` spec | `ag.by("kind", palette={...})` | Any of the above, plus explicit scale, palette, domain, range and legend title |

```python
deg = campus.degree()
fig = ag.draw(
    campus,
    node_color="kind",
    node_size=deg,
    node_shape=lambda n, d: "square" if d["kind"] == "residence" else "circle",
    edge_color=ag.by(
        "kind",
        palette={"tunnel": "#2a78d6", "pedway": "#1baf7a", "attached": "#52514e", "outdoor": "#c3c2b7"},
        title="link kind",
    ),
    edge_width=1.5,
)
print([(lg[0].title, lg[0].kind) for lg in fig.scene.legends])
```

```text
[('kind', 'categorical'), ('degree', 'size'), ('link kind', 'categorical')]
```

The scale is chosen from the data unless you set it with `ag.by(..., kind=...)`:

- strings and booleans are **categorical**, as are integer codes whose attribute name suggests a category (`community`, `group`, `kind`, `cluster`, …). Categories get fixed palette slots in a stable order; beyond eight, the seven most frequent keep their colors and the rest share an "Other" entry;
- other numbers are **sequential** on a perceptual color scale, or mapped to *area* for sizes; `kind="diverging"`, `"log"` and `"sqrt"` are available;
- values that are already colors, or shape names for `node_shape`, are used as they are (**identity**).

Data-driven color, size, width and shape encodings produce their legends automatically. Identity values explain themselves, which is why the shape function in the example, which returns shape names, adds no legend entry. See [Styling](../user-guide/styling.md).

## Themes

A {py:class}`~aryagraph.style.themes.Theme` names every color, font and size a rendering needs by its role (background, ink, edge, categorical palette, …). Renderers ask the theme for a role instead of hard-coding a color, so changing the theme restyles drawings, charts and animations consistently. Four themes are built in:

```python
for name in ("light", "dark", "paper", "blueprint"):
    th = ag.get_theme(name)
    print(f"{name:9} {th.mode:5} background {th.background}  ink {th.ink}")

night = ag.get_theme("dark").with_(name="night", edge="#6c7a89")
ag.register_theme(night)
print(ag.draw(campus, theme="night"))
```

```text
light     light background #f9f9f7  ink #0b0b0b
dark      dark  background #0d0d0d  ink #ffffff
paper     light background #ffffff  ink #000000
blueprint dark  background #0b1b33  ink #eef4ff
<Figure 861×544: 56 nodes, 83 edges, stress layout, theme 'night'>
```

The dark theme has its own tuned palette steps rather than inverted light colors. `with_()` copies a theme with some fields changed, and {py:func}`~aryagraph.style.themes.register_theme` makes it available by name to every `theme=` argument.

## Seeds and reproducibility

Every source of randomness in AryaGraph takes a `seed` argument:

- **Layouts** default to `seed=0`, in `draw()` and in `ag.layout.compute()`. The same graph (with the same insertion order), the same parameters and the same seed give the same coordinates, so a drawing does not change between runs.
- **Simulations, random graph generators and randomized algorithms** (Louvain and the other randomized community methods, sampled betweenness) default to `seed=None`, which draws fresh entropy, because repeated runs are usually meant to differ. Pass an integer to make a result repeatable. A simulation records it in `result.seed`, which is `None` for an unseeded run.

```python
import numpy as np

a = ag.layout.compute(campus, "forceatlas2", seed=1)
b = ag.layout.compute(campus, "forceatlas2", seed=1)
c = ag.layout.compute(campus, "forceatlas2", seed=2)
print(np.array_equal(a.xy, b.xy), np.array_equal(a.xy, c.xy))

model = ag.sim.SIR(beta=0.4, gamma=0.1)
r1 = model.simulate(campus, initial={"I": ["MSC"]}, t_max=60, seed=7)
r2 = model.simulate(campus, initial={"I": ["MSC"]}, t_max=60, seed=7)
r3 = model.simulate(campus, initial={"I": ["MSC"]}, t_max=60)
print(np.array_equal(r1.values, r2.values), r1.seed, r3.seed)
```

```text
True False
True 7 None
```

Seeds make results repeatable for a given version of AryaGraph and numpy. For published work, record both versions along with your seeds; [Reproducibility and performance](../user-guide/reproducibility.md) has a checklist.

## Simulations return a SimulationResult

Every simulator in `ag.sim` returns a {py:class}`~aryagraph.sim.base.SimulationResult`: a `(T, N)` array holding the state of each of *N* nodes in each of *T* frames, the frame times, and the names of the states. Categorical models (epidemics, cascades, walks, voter models) store small integer state codes; continuous models (opinion averaging, heat diffusion, oscillators) store floats.

```python
run = model.simulate(campus, initial={"I": ["MSC"]}, t_max=60, seed=2)
print(run.T, run.N, run.states, run.kind)
print(run.times[:5], run.values.shape, run.values.dtype)
print(run.history("MSC")[:12])
print({s: int(c[15]) for s, c in run.counts().items()})
first = run.first_time("I")
print(first["MSC"], first["SH"], first["OVC"])
```

```text
53 56 ['S', 'I', 'R'] categorical
[0. 1. 2. 3. 4.] (53, 56) int8
['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I']
{'S': 14, 'I': 31, 'R': 11}
0.0 20.0 nan
```

This is the seeded run from the [Quickstart](quickstart.md#run-a-small-simulation). MacEwan Student Centre was infected in each of the first 12 frames, Scurfield Hall was first infected at time 20, and the Olympic Volunteer Centre was not infected at all (`nan`).

| Method | Returns |
|---|---|
| `counts()`, `fractions()` | Number or share of nodes in each state, per frame |
| `peak(state)` | `(time, count)` of the largest count |
| `first_time(state)` | A `NodeMap` of the first time each node entered the state |
| `frame(i)`, `at(t)`, `final()` | `{node: state}` for a frame, a time, or the end |
| `history(node)` | One node's states over time |
| `summary()` | Initial, final and peak counts per state |
| `to_records()`, `to_pandas()` | Long-format export |
| `plot()`, `animate()` | A state chart, or an interactive player on any layout |

Schedule simulations of DAGs and Monte Carlo schedule estimates return their own result classes with makespans, Gantt charts and criticality indices; see [DAG workflows](../user-guide/dags.md#simulating-execution).

## Next steps

- [Quickstart](quickstart.md): the ten-minute tour, if you skipped it.
- [User guide](../user-guide/index.md): one page per topic, from graphs to the command line.
- [Tutorials](../tutorials/index.md): longer worked examples.
- [API reference](../reference/index.rst): every public function and class.
