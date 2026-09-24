# Layouts

A layout gives every node a position and, for layered drawings, gives long edges their bend points. This page covers the fourteen layout engines in `ag.layout`, how AryaGraph picks one for you, and how to pass, transform and post-process positions yourself.

## How a layout is chosen

{py:func}`aryagraph.render.draw` computes a layout for you. Its `layout=` argument accepts:

- a method name such as `"stress"` or `"hierarchical"` (the default is `"auto"`);
- a {py:class}`aryagraph.layout.base.Layout` you computed earlier;
- a `{node: (x, y)}` mapping, for example real coordinates;
- a callable `f(g)` that returns either of the above.

Engine options go in `layout_options=`, and the `seed=` argument of `draw()` is forwarded to the engine. To get positions without drawing, call {py:func}`aryagraph.layout.compute` with the same arguments.

```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()     # 56 buildings, 83 connections
pipeline = ag.gen.ml_pipeline()       # a 16-task DAG

print(ag.layout.auto_method(campus), ag.layout.auto_method(pipeline))

lay = ag.layout.compute(campus)       # method="auto"
print(lay)
print(lay["MSC"])                     # the MacEwan Student Centre
```

```text
stress hierarchical
<Layout 'stress': 56 nodes, abstract, 0 routed edges>
[-1.34263778  0.07048128]
```

With `method="auto"`, {py:func}`aryagraph.layout.auto_method` decides:

| graph | method |
|---|---|
| directed and acyclic, up to 2,000 nodes | `hierarchical` |
| any other graph up to 3,000 nodes | `stress` |
| larger graphs | `forceatlas2` |

When `"auto"` resolves to an engine that does not accept one of your options, that option is dropped instead of raising an error. You can therefore pass `layout_options={"orientation": "LR"}` in a function that draws both DAGs and general graphs.

`draw()` applies the same rule with one extra condition: a DAG without any edge is drawn with stress.

### The engines

| method | also accepted as | kind | typical use |
|---|---|---|---|
| `hierarchical` | `layered`, `sugiyama` | metric | DAGs, pipelines, flows, dependency graphs |
| `tree` | | metric | trees and hierarchies, top-down or left-to-right |
| `radial` | | metric | wide trees with many leaves |
| `stress` | `kamada_kawai` | abstract | general graphs up to a few thousand nodes |
| `forceatlas2` | `force_atlas2` | abstract | clustered networks, large graphs |
| `fruchterman_reingold` | `force`, `spring` | abstract | small graphs, pinning some nodes |
| `spectral` | | abstract | revealing global structure and bottlenecks |
| `circular`, `shell`, `grid`, `spiral`, `bipartite`, `arc`, `random` | | abstract | fixed geometric arrangements |

```python
engines = set(ag.layout.METHODS.values())
print(len(engines), "engines,", len(ag.layout.METHODS), "accepted names")
```

```text
14 engines, 20 accepted names
```

*Metric* layouts are in pixels and respect the node sizes that were passed in, so the renderer keeps their scale. *Abstract* layouts only fix relative positions; the renderer scales them so that a typical edge is a readable length. Every engine uses screen coordinates: x grows to the right and **y grows downward**.

## Six engines on one graph

The graph below has four planted groups of 16 nodes, with many edges inside each group and few between them. Each engine answers a different question about it.

```python
g = ag.gen.planted_partition(4, 16, 0.35, 0.02, seed=3)  # 64 nodes

methods = ("stress", "forceatlas2", "fruchterman_reingold", "spectral", "circular", "shell")
for method in methods:
    options = {"order": "auto"} if method == "circular" else None
    fig = ag.draw(
        g,
        layout=method,
        layout_options=options,
        node_color=ag.by("block", kind="categorical"),
        labels=False,
        legend=False,
        title=method,
        width=400,              # same display width for every panel
    )
    fig.save(f"compare_{method}.svg")
```

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item}
```{image} ../_static/generated/guide_layout_algorithms/compare_stress.png
:alt: Stress layout of a graph with four planted groups; the groups sit next to each other with even edge lengths
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_layout_algorithms/compare_forceatlas2.png
:alt: ForceAtlas2 layout of the same graph; the four groups form compact clusters
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_layout_algorithms/compare_fruchterman_reingold.png
:alt: Fruchterman-Reingold layout of the same graph
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_layout_algorithms/compare_spectral.png
:alt: Spectral layout of the same graph; each group collapses into a tight cluster in its own corner
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_layout_algorithms/compare_circular.png
:alt: Circular layout with an automatic order; each group occupies its own arc of the circle
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_layout_algorithms/compare_shell.png
:alt: Shell layout; nodes sit on rings by distance from a central node, mixing the groups
```
:::
::::

<p class="ag-caption">One graph, six engines. Colors show the four planted groups.</p>

- **Stress** follows graph distances: it aims to draw nodes two hops apart about twice as far apart as neighbors. It is the default for general graphs because it balances local detail with the overall shape.
- **ForceAtlas2** and **Fruchterman–Reingold** are force simulations. ForceAtlas2 pulls groups into compact clusters, which makes community structure visible at a glance.
- **Spectral** places nodes with eigenvectors of the Laplacian. Weakly joined groups separate strongly, which is useful for spotting bottlenecks.
- **Circular** with `order="auto"` keeps each group on its own arc; **shell** arranges nodes on rings by their distance from a central node.

## Hierarchical layouts

{py:func}`aryagraph.layout.hierarchical` draws directed graphs in layers, with edges pointing the same way. It follows the Sugiyama framework in four phases:

1. **Cycle removal.** A small set of arcs that breaks every directed cycle is reversed for the layout (Eades–Lin–Smyth heuristic); those arcs are drawn pointing back. Undirected graphs are oriented by breadth-first depth from a peripheral node.
2. **Layering.** Network simplex assigns the layers that minimize total edge length.
3. **Crossing reduction.** Weighted-median sweeps, sifting and the *transpose* step reorder each layer; the order with the fewest crossings found is kept.
4. **Coordinates.** The Brandes–Köpf method, refined by network-simplex positioning, keeps long edges straight and centers parents over their children.

Edges that span several layers get bend points in `Layout.routes`, and `Layout.meta` records what the algorithm found:

```python
etl = ag.gen.data_warehouse_etl()          # 25 tasks, 41 dependencies
lay = ag.layout.hierarchical(etl)

print(lay)
print("crossings:", lay.meta["crossings"])
print("layer of 'extract_crm':", lay.meta["layers"]["extract_crm"])
print(sorted(lay.meta))
```

```text
<Layout 'hierarchical': 25 nodes, metric, 10 routed edges>
crossings: 12
layer of 'extract_crm': 0
['components', 'crossings', 'flat_edges', 'layers', 'node_sep', 'orientation', 'rank_positions', 'rank_sep', 'reversed_edges']
```

### Layering methods

`layering=` chooses how nodes are assigned to layers:

- `"network_simplex"` (default) minimizes the total number of layers that edges span, so related tasks stay close;
- `"longest_path"` puts every node in the earliest possible layer, which is fast but tends to crowd the top layers;
- `"coffman_graham"` limits each layer to `max_width` nodes (by default ⌈√n⌉), trading depth for width.

```python
from collections import Counter

for method in ("network_simplex", "longest_path", "coffman_graham"):
    layer = ag.layout.hierarchical(etl, layering=method).meta["layers"]
    widest = max(Counter(layer.values()).values())
    span = sum(layer[v] - layer[u] for u, v in etl.edges)
    layers = max(layer.values()) + 1
    print(f"{method:16} layers={layers}  widest={widest}  total span={span}")

    fig = ag.draw(etl, layout="hierarchical", layout_options={"layering": method},
                  node_color="kind", labels=False,
                  title="Data-warehouse ETL", subtitle=f'layering="{method}"')
    fig.save(f"etl_{method}.svg")
```

```text
network_simplex  layers=7  widest=6  total span=52
longest_path     layers=7  widest=7  total span=59
coffman_graham   layers=8  widest=5  total span=54
```

Network simplex gives the smallest total span of the three (52 layer-to-layer steps summed over all edges). Coffman–Graham caps each layer at 5 nodes, the ceiling of √25, at the cost of one extra layer.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/generated/guide_layout_algorithms/etl_network_simplex.png
:alt: Data-warehouse ETL drawn in seven layers with network-simplex layering
Network simplex: 7 layers, up to 6 nodes wide.
```
:::

:::{grid-item}
```{figure} ../_static/generated/guide_layout_algorithms/etl_coffman_graham.png
:alt: Data-warehouse ETL drawn in eight layers with Coffman-Graham layering
Coffman–Graham: 8 layers, at most 5 nodes wide.
```
:::
::::

### Orientation and spacing

`orientation=` is `"TB"` (sources on top, the default), `"BT"`, `"LR"` or `"RL"`. Left-to-right suits graphs with many layers and short labels, such as timelines and prerequisite chains.

```python
courses = ag.gen.course_prerequisites()     # 22 courses
fig = ag.draw(
    courses,
    layout="hierarchical",
    layout_options={"orientation": "LR"},
    node_color="department",
    title="Course prerequisites",
    subtitle="orientation LR",
)
fig.save("courses_lr.svg")
```

```{figure} ../_static/generated/guide_layout_algorithms/courses_lr.png
:alt: Course prerequisite DAG drawn from left to right, with CS, MATH and STAT courses in different colors
:width: 80%

The course prerequisite DAG with `orientation="LR"`.
```

Spacing is set in pixels: `node_sep` is the gap between neighbors in a layer and `rank_sep` the gap between layers. The engine defaults are 28 and 72; `draw()` uses 24 and 52 unless you pass your own, because rendered node boxes already include padding. For example, `layout_options={"node_sep": 40, "rank_sep": 90}` gives a more open drawing. Pass `sizes={node: (w, h)}` to {py:func}`aryagraph.layout.compute` to reserve room for large nodes; `draw()` does this for you from the rendered boxes and labels.

### Fixing the layers yourself

`ranks={node: layer}` overrides the layering. Here each course sits in the layer of its level (100 to 400). Prerequisites within one level become *flat edges*, drawn inside the layer:

```python
ranks = {c: d["level"] // 100 - 1 for c, d in courses.nodes.data()}
lay = ag.layout.hierarchical(courses, ranks=ranks)

print("crossings:", lay.meta["crossings"])
print("flat edges:", lay.meta["flat_edges"])

fig = ag.draw(courses, layout="hierarchical", layout_options={"ranks": ranks},
              node_color="department", title="Course prerequisites",
              subtitle="one layer per course level (100 to 400)")
fig.save("courses_ranks.svg")
```

```text
crossings: 10
flat edges: [('MATH101', 'MATH102'), ('CS101', 'CS102'), ('CS210', 'CS220'), ('CS330', 'CS340')]
```

```{figure} ../_static/generated/guide_layout_algorithms/courses_ranks.png
:alt: Course prerequisites in four layers, one per course level, with same-level prerequisites drawn as arcs
:width: 80%

Layers fixed by course level with `ranks=`. The four prerequisites between courses of the same level are drawn as arcs within their layer.
```

### Graphs with cycles

`hierarchical` also accepts directed graphs that contain cycles. The arcs it reverses are listed in `meta["reversed_edges"]` and are drawn pointing back up:

```python
review = ag.DiGraph([
    ("request", "review"), ("review", "revise"), ("revise", "review"),
    ("review", "approve"), ("approve", "publish"),
])
lay = ag.layout.hierarchical(review)
print(lay.meta["reversed_edges"])
print(dict(lay.meta["layers"]))
```

```text
[('revise', 'review')]
{'request': 0, 'review': 1, 'revise': 2, 'approve': 2, 'publish': 3}
```

Other options: `crossing_passes` (maximum sweeps, default 24), `compact=False` to keep the plain Brandes–Köpf coordinates, and `seed` for the extra random initial orders tried on small graphs.

## Trees and radial trees

{py:func}`aryagraph.layout.tree` draws a tidy tree in linear time (Walker's algorithm with Buchheim's improvements): parents are centered over their children, and neighbors on a level keep `node_sep` between their boxes, whatever their sizes. It accepts `root`, `orientation`, `node_sep`, `rank_sep` and `sizes`. Without `root`, it uses the unique source of a directed graph, or else a node of minimum eccentricity.

A graph that is not a tree is laid out along a breadth-first spanning tree from the root; `draw()` still draws all edges. Here the breadth-first tree of the campus from the student center shows how many links separate each building from MSC. With 56 labeled nodes, a left-to-right tree with tighter spacing fits the page better than a top-down one:

```python
bfs = ag.alg.bfs_tree(campus, "MSC")
lay = ag.layout.tree(bfs)

print(lay.meta["roots"], "depth:", max(lay.meta["depth"].values()))
fig = ag.draw(bfs, layout="tree",
              layout_options={"orientation": "LR", "node_sep": 12, "rank_sep": 28},
              title="Breadth-first tree of the campus from MSC")
fig.save("campus_bfs_tree.svg")
```

```text
['MSC'] depth: 9
```

```{figure} ../_static/generated/guide_layout_algorithms/campus_bfs_tree.png
:alt: Tidy tree of the campus buildings drawn from left to right, rooted at MSC, with nine levels after the root
Tidy tree layout of the breadth-first tree from MSC, with `orientation="LR"`. Each column is one more link away from the student center.
```

{py:func}`aryagraph.layout.radial` puts the root in the center and each depth on a ring. Every node owns an angular wedge proportional to its number of leaves, so subtrees do not interleave. Radial trees suit wide, shallow hierarchies, where a top-down tree would become a long strip:

```python
wide = ag.gen.balanced_tree(3, 4)           # 121 nodes, 5 levels
lay = ag.layout.radial(wide)
print(len(lay.meta["radii"][0]), "rings")

fig = ag.draw(wide, layout="radial", labels=False, legend=False, title="radial")
fig.save("radial.svg")
```

```text
5 rings
```

```{figure} ../_static/generated/guide_layout_algorithms/radial.png
:alt: Balanced ternary tree of 121 nodes drawn as concentric rings around the root
:width: 60%

A balanced ternary tree of 121 nodes in the radial layout.
```

By default the rings are evenly spaced (`uniform=True`); `uniform=False` lets each ring take only the radius its nodes need.

## Stress majorization

{py:func}`aryagraph.layout.stress` places nodes so that drawn distances approximate graph distances, by minimizing the *stress* between the two. It starts from classical multidimensional scaling (pivot MDS above 1,000 nodes) and iterates the SMACOF update, then rotates the result so its principal axis is horizontal.

Options:

- `weight`: an edge attribute or callable giving edge **lengths**; `None` (default) treats every edge as length 1;
- `iterations` (default 300) and `tol` (default 1e-5): stop criteria;
- `init`: `"mds"` (default), `"random"`, or starting positions.

`meta["stress"]` reports the final normalized stress, so you can compare runs. With `weight="length"`, stress uses the link lengths in meters (straight-line distances between buildings) and recovers much of the campus geometry. The correlation between drawn and real pairwise distances shows how much:

```python
import numpy as np

pos = {n: d["pos"] for n, d in campus.nodes.data()}
real = np.array([pos[n] for n in campus.nodes])

def distance_correlation(lay):
    xy = np.array([lay[n] for n in campus.nodes])
    i, j = np.triu_indices(len(xy), 1)
    a = np.hypot(*(xy[i] - xy[j]).T)
    b = np.hypot(*(real[i] - real[j]).T)
    return np.corrcoef(a, b)[0, 1]

for weight in (None, "length"):
    lay = ag.layout.stress(campus, weight=weight)
    r = distance_correlation(lay)
    print(f"weight={weight!r:9} stress={lay.meta['stress']:.1f}  r={r:.3f}")

fig = ag.draw(campus, layout="stress", layout_options={"weight": "length"},
              node_color="kind", title="Stress layout", subtitle='weight="length"',
              width=760)
fig.save("campus_stress.svg")
```

```text
weight=None      stress=20.6  r=0.785
weight='length'  stress=13.5  r=0.957
```

Switch between the tabs to compare the stress layout with the real positions:

::::{tab-set}

:::{tab-item} Stress layout
```{figure} ../_static/generated/guide_layout_algorithms/campus_stress.png
:alt: Campus buildings drawn with the stress layout, using the link lengths in meters as edge lengths
Stress layout with `weight="length"`.
```
:::

:::{tab-item} Real positions
```{figure} ../_static/generated/guide_layout_algorithms/campus_positions.png
:alt: Campus buildings drawn at their real positions
Real building positions (© OpenStreetMap contributors, ODbL). The code is under [Using your own coordinates](#using-your-own-coordinates).
```
:::
::::

## ForceAtlas2

{py:func}`aryagraph.layout.force_atlas2` implements ForceAtlas2 (Jacomy et al., 2014): degree-weighted repulsion, attraction along edges, gravity toward the center and the adaptive speed from the paper. Options:

| option | default | effect |
|---|---|---|
| `iterations` | 500 | number of steps |
| `scaling` | 2.0 | repulsion strength; larger values spread the graph |
| `gravity`, `strong_gravity` | 1.0, False | pull toward the center (constant, or growing with distance) |
| `lin_log` | False | logarithmic attraction: tighter, better separated clusters |
| `dissuade_hubs` | False | pushes high-degree nodes to the periphery |
| `prevent_overlap` | False | anti-collision using node radii from `sizes` |
| `barnes_hut`, `theta` | None, 1.2 | quadtree approximation; `None` turns it on above 1,500 nodes |
| `weight` | `"weight"` | edge attribute multiplying the attraction |
| `init` | None | starting positions |

The Les Misérables network below is drawn with the defaults and with `lin_log=True`. Its edges carry a `weight` (chapters shared), which ForceAtlas2 uses by default.

```python
lesmis = ag.gen.les_miserables()
for name, options in (("default", {}), ("lin_log", {"lin_log": True})):
    fig = ag.draw(
        lesmis,
        layout="forceatlas2",
        layout_options=options,
        node_size=lesmis.degree(),
        labels=False,
        legend=False,
        title="ForceAtlas2",
        subtitle="lin_log=True" if options else "default options",
        width=520,
        height=520,
    )
    fig.save(f"lesmis_fa2_{name}.svg")
```

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/generated/guide_layout_algorithms/lesmis_fa2_default.png
:alt: Les Miserables network with the default ForceAtlas2 options
Default options.
```
:::

:::{grid-item}
```{figure} ../_static/generated/guide_layout_algorithms/lesmis_fa2_lin_log.png
:alt: Les Miserables network with ForceAtlas2 and lin_log enabled; clusters are tighter and further apart
`lin_log=True`.
```
:::
::::

## Fruchterman–Reingold

{py:func}`aryagraph.layout.fruchterman_reingold` is the classic spring embedder with a cooling schedule. Its options are `k` (ideal edge length), `iterations` (default 300), `weight` (default `"weight"`, scales attraction), `gravity` (pull toward the centroid, default 0) and two that make it useful for partial layouts:

- `init`: starting positions for some or all nodes;
- `fixed`: nodes that do not move. With `fixed`, the result stays in the coordinate frame of `init`.

This pins four buildings at their real positions and lets the others settle around them:

```python
anchors = ["MSC", "TFDL", "OO", "SH"]
lay = ag.layout.fruchterman_reingold(
    campus, init={n: pos[n] for n in anchors}, fixed=anchors
)
print(lay["MSC"], pos["MSC"])
```

```text
[-22.3 -32.7] (-22.3, -32.7)
```

## Spectral layout

{py:func}`aryagraph.layout.spectral` uses the eigenvectors of the two smallest non-zero eigenvalues of the graph Laplacian as x and y. With `normalized=True` (default) it solves the degree-normalized problem; `weight` gives edge weights. The eigenvalues are stored in `meta`, and the first one is the normalized algebraic connectivity:

```python
lay = ag.layout.spectral(campus)
print(round(lay.meta["eigenvalues"][0], 6))
print(round(ag.alg.algebraic_connectivity(campus, normalized=True), 6))

print(ag.layout.spectral(ag.gen.complete_graph(6)).meta)
```

```text
0.016718
0.016718
{'fallback': 'circular', 'reason': 'degenerate spectrum'}
```

When the spectrum is degenerate (a complete graph, where every eigenvector is equally valid), when the graph has fewer than three nodes, or when it has no edges, spectral falls back to the circular layout and says so in `meta`.

## Geometric layouts

These place nodes on simple shapes. They are abstract layouts with neighboring slots about one unit apart.

| method | options | arrangement |
|---|---|---|
| `circular` | `order`, `start_angle` | evenly spaced on a circle, clockwise from 12 o'clock |
| `arc` | `order` | on one line, for arc diagrams (`meta["arc"]` is True) |
| `shell` | `shells` | concentric rings; by default rings by distance from a central node |
| `grid` | `columns` | row by row in graph order |
| `spiral` | `turns` | along an Archimedean spiral, from the center outwards |
| `bipartite` | `top`, `align` | two parallel lines, sides ordered to reduce crossings |
| `random` | `seed` | uniform in the unit square |

### Node order on a circle

For `circular` and `arc`, `order=None` keeps graph order, a list fixes the order, and `order="auto"` searches for an order with few chord crossings. It builds two candidate orders per component, a depth-first order from a peripheral node and a greedy "most placed neighbors" order, and keeps the one with fewer crossings. For trees and cycles, the depth-first order is free of crossings:

```python
def chord_crossings(g, order):
    """Pairs of edges that cross when drawn as chords of a circle."""
    at = {v: i for i, v in enumerate(order)}
    chords = [tuple(sorted((at[u], at[v]))) for u, v in g.edges]
    return sum(a < c < b < d for a, b in chords for c, d in chords)

tree = ag.gen.random_tree(30, seed=2)
for order in (None, "auto"):
    lay = ag.layout.circular(tree, order=order)
    print(f"order={order!r:7} crossings={chord_crossings(tree, lay.meta['order'])}")
    fig = ag.draw(tree, layout=lay, labels=False, width=380, height=380)
    fig.save(f"tree_circular_{order or 'graph'}.svg")
```

```text
order=None    crossings=142
order='auto'  crossings=0
```

::::{grid} 2 2 2 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/generated/guide_layout_algorithms/tree_circular_graph.png
:alt: Random tree on a circle in graph order, with many crossing chords
Graph order: 142 crossings.
```
:::

:::{grid-item}
```{figure} ../_static/generated/guide_layout_algorithms/tree_circular_auto.png
:alt: The same random tree on a circle with the automatic order and no crossings
`order="auto"`: no crossings.
```
:::
::::

### Two-mode networks

`bipartite` draws two columns (or two rows with `align="horizontal"`). Pass one side as `top`; otherwise each component is 2-colored by breadth-first search. The sides are then reordered by alternating barycenter sweeps to remove crossings:

```python
davis = ag.gen.davis_southern_women()       # 18 women, 14 events
women = [n for n, d in davis.nodes.data() if d["bipartite"] == 0]
fig = ag.draw(davis, layout="bipartite", layout_options={"top": women},
              node_color="bipartite", legend=False, title="Davis Southern Women",
              subtitle="bipartite layout: 18 women (left), 14 events (right)")
fig.save("davis_bipartite.svg")
```

```{figure} ../_static/generated/guide_layout_algorithms/davis_bipartite.png
:alt: Davis Southern Women network with women in the left column and events in the right column
:width: 70%

The Davis Southern Women network: women on the left, the events they attended on the right.
```

## Using your own coordinates

Any `{node: (x, y)}` mapping works as a layout. The campus dataset stores building positions in meters in the `pos` attribute (from OpenStreetMap, © OpenStreetMap contributors, ODbL), already in screen orientation:

```python
fig = ag.draw(
    campus,
    layout=pos,
    node_color="kind",
    title="Real positions",
    subtitle="Positions © OpenStreetMap contributors (ODbL)",
    width=760,
)
fig.save("campus_positions.svg")

own = ag.Layout.from_positions(pos)
print(own)
```

```text
<Layout 'custom': 56 nodes, abstract, 0 routed edges>
```

Keep in mind:

- AryaGraph uses screen coordinates (y down). For latitude and longitude, negate the latitude, for example `(d["lon"], -d["lat"])`, or project to meters first.
- Every node needs a position; a missing one raises an error that names it.
- When drawing, positions are scaled and nodes that would overlap are nudged apart (see [Overlap removal](#overlap-removal)). Pass `avoid_overlap=False` to keep the exact geometry.

## Transforming a layout

{py:class}`aryagraph.layout.base.Layout` methods return new layouts and move edge routes along with the nodes:

| method | result |
|---|---|
| `rotated(degrees, center=None)` | rotated clockwise on screen about the centroid |
| `flipped("x")` / `flipped("y")` | mirrored horizontally / vertically |
| `scaled(sx, sy=None)`, `translated(dx, dy)` | scaled about the origin, shifted |
| `fit(width, height, padding=0)` | uniformly scaled and centered into a box |
| `normalized()` | centered on the origin, larger half-extent 1 |
| `aligned()` | principal axis of the node cloud turned horizontal |
| `subset(nodes)` | only the given nodes |
| `bounds()`, `centroid()`, `positions`, `to_dict()` | inspection and export |

```python
lay = ag.layout.stress(campus)
print([round(v, 2) for v in lay.bounds()])
print([round(v, 1) for v in lay.fit(800, 600, padding=40).bounds()])
print([round(v, 2) for v in lay.normalized().bounds()])

upright = lay.rotated(90).flipped("x")
fig = ag.draw(campus, layout=upright)
```

```text
[-7.34, -3.99, 7.56, 5.3]
[40.0, 75.6, 760.0, 524.4]
[-1.0, -0.62, 1.0, 0.62]
```

## Disconnected graphs

By default, {py:func}`aryagraph.layout.compute` lays out each connected component of a `stress`, `fruchterman_reingold`, `forceatlas2` or `spectral` drawing on its own, rescales every piece to a median edge length of 1, and packs the pieces on shelves (largest first, aiming at a 1.4 aspect ratio, one edge length apart). Every piece then has the same scale, and the pieces sit in rows next to each other.

The indoor-only campus network (tunnels, pedways and attached buildings) splits into 19 components: one of 38 buildings and 18 buildings with no indoor link.

```python
indoor = ag.gen.ucalgary_campus(indoor_only=True)
print(len(ag.alg.connected_components(indoor)), "components")

packed = ag.layout.compute(indoor, "stress")
print(packed.meta["components"])

fig = ag.draw(indoor, layout="stress", node_color="kind", width=760,
              title="Indoor-only campus network", subtitle="19 components, packed")
fig.save("indoor_packed.svg")
```

```text
19 components
19
```

```{figure} ../_static/generated/guide_layout_algorithms/indoor_packed.png
:alt: Indoor campus network with its large component and 18 isolated buildings packed around it

The indoor network: the 38-building component and 18 isolated buildings, packed. Colors show the building kind.
```

`components=None` lays out the whole graph at once for `fruchterman_reingold` and `forceatlas2`, whose gravity then keeps the pieces together. The `stress` and `spectral` engines pack components themselves in either case. `hierarchical`, `tree` and `radial` place components side by side on their own. To combine layouts you computed separately, use {py:func}`aryagraph.layout.base.pack_components`.

## Overlap removal

A force-directed or stress layout can put two nodes closer than their drawn size. When drawing, AryaGraph scales abstract layouts to pixels and then, for graphs of 2 to 3,000 nodes, runs {py:func}`aryagraph.layout.remove_overlaps` with the rendered node sizes. It moves nodes as little as possible until no two boxes overlap. Metric layouts (hierarchical, tree, radial) already respect node sizes and are left as they are. Pass `avoid_overlap=False` to `draw()` to switch this off.

You can also call it directly, in the units of your layout:

```python
scattered = ag.layout.random(campus, seed=1).scaled(400)
clean = ag.layout.remove_overlaps(scattered, sizes=(36, 36), padding=4)

moved = sum(not np.allclose(scattered[n], clean[n]) for n in campus.nodes)
print(moved, "nodes moved, total", round(clean.meta["overlap_displacement"]), "px")
```

```text
51 nodes moved, total 720 px
```

## Reproducibility

Every engine takes a `seed` (default 0) and returns the same coordinates for the same graph, options and seed:

```python
a = ag.layout.force_atlas2(campus, seed=3)
b = ag.layout.force_atlas2(campus, seed=3)
c = ag.layout.force_atlas2(campus, seed=4)
print(np.array_equal(a.xy, b.xy), np.allclose(a.xy, c.xy))
```

```text
True False
```

Nodes are returned in graph order, so adding nodes in the same order gives the same drawing. The [Reproducibility and performance](reproducibility.md) page covers seeds across the library.

## Choosing an engine

- **A DAG, workflow or dependency graph:** `hierarchical`. Use `orientation="LR"` for long chains and `ranks=` when layers have a meaning of their own.
- **A tree:** `tree`, or `radial` when it is wide and shallow.
- **A general graph up to a few thousand nodes:** `stress`, with `weight=` when edges have lengths.
- **Communities, or more than 3,000 nodes:** `forceatlas2`, with `lin_log=True` for clearer clusters.
- **Real geography:** pass the coordinates.
- **A fixed, comparable arrangement:** `circular` with `order="auto"`, `shell`, `bipartite` or `arc`.

For drawing options such as edge styles and labels, see [Drawing](drawing.md); for the algorithms used above, see [Algorithms](algorithms.md).
