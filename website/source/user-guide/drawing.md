# Drawing

{py:func}`ag.draw() <aryagraph.render.draw>` turns a graph into a {py:class}`~aryagraph.render.figure.Figure` in one call. You describe what each visual channel shows (color, size, shape, labels, edge width and so on) and AryaGraph chooses the scales, places the labels, routes the edges and writes the legends.

## A first drawing

The campus dataset stores each building's position (derived from OpenStreetMap) in the `pos` attribute, so the drawing can follow the real map. Coloring by the `kind` attribute takes one argument:

```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
on_map = {n: d["pos"] for n, d in campus.nodes.data()}

fig = ag.draw(
    campus,
    layout=on_map,
    node_color="kind",
    title="University of Calgary main campus",
    subtitle="Buildings colored by kind",
)
print(fig)   # <Figure 1195×895: 56 nodes, 83 edges, custom layout, theme 'light'>
fig.save("campus.svg")
```

```{figure} ../_static/generated/guide_core/drawing_campus_kind.png
:alt: Map-like drawing of 56 campus buildings as colored dots with building codes, and a legend of building kinds with counts.

Buildings colored by kind. The dataset has nine kinds, one more than the palette's eight slots, so the two rarest kinds share the gray "Other (2)" entry. Positions © OpenStreetMap contributors (ODbL).
```

`draw()` computes the whole geometry (layout, node sizes, label placement, edge routes and legends) when you call it and returns the figure; nothing is written until you call `save()` or display the figure. The format follows the file extension (`.svg`, `.html`, `.png` or `.pdf`); see [Exporting](exporting.md) and [Interactive HTML](interactive.md). Every `Graph` also has a `g.draw(...)` shortcut that forwards to the same function.

## What a channel accepts

Every data-driven channel (`node_color`, `node_size`, `node_shape`, `node_opacity`, `labels`, `edge_color`, `edge_width`, `edge_label`) accepts the same six forms:

| Form | Example | Meaning |
|---|---|---|
| constant | `node_color="#e34948"`, `node_size=12` | the same value everywhere |
| attribute name | `node_color="kind"` | read each node's (or edge's) attribute |
| mapping | `node_size=ag.alg.pagerank(g)` | `{node: value}` or `{(u, v): value}`, such as any algorithm result |
| callable | `node_color=lambda n, d: d["kind"] == "library"` | called per node as `f(node)` or `f(node, attrs)`; per edge as `f(u, v)` or `f(u, v, attrs)` |
| sequence | `node_size=[...]` | one value per node (or edge) in graph order |
| `ag.by(...)` | `ag.by("pagerank", kind="log")` | any of the above plus explicit scale options |

A string is treated as an attribute name when at least one node (or edge) has that attribute; otherwise it must be a valid color. A name that is neither raises a `ValueError` that says so, which catches typos early. For colors, nodes or edges that lack the attribute are drawn in the theme's neutral gray, and categorical legends count them under "Missing".

## Node channels

`node_color`
: Fill color. Strings, booleans and small integer codes become categories; other numbers get a continuous color scale (see [Scales](#encodings-and-scales)).

`node_size`
: Marker diameter in pixels when constant. Data values map to the marker's *area*, between 0.55 and 2.6 times the base size, so a value twice as large does not look four times as large. The base size depends on the node count: 18 px up to 30 nodes, 14 px up to 100, 10 px up to 300, then 7, 5 and 3.5 px; `base_size=` overrides it.

`node_shape`
: A shape name for all nodes, or data. When the data are shape names, they are used as given; any other values are treated as categories and mapped to a fixed cycle of eight shapes (circle, square, diamond, triangle, hexagon, triangle_down, pentagon, octagon) with a shape legend.

`node_opacity`
: A number from 0 to 1, or data giving one per node.

`labels`
: `"auto"` (the default) shows every label up to 80 nodes and the 30 most prominent above that. `True` and `False` switch all labels on or off, an integer `k` shows the top `k`, and an attribute, mapping or callable supplies the text. A node's `label` attribute, when present, replaces its id as the default text.

`label_position`, `label_size`, `label_max_width`, `label_collisions`
: Where labels go (`"auto"`, `"center"`, `"right"`, `"left"`, `"above"`, `"below"` or a diagonal such as `"below-right"`), their font size, the width at which text inside a box wraps (160 px by default), and whether labels that cannot be placed are hidden (`"hide"`) or drawn anyway (`"show"`).

`tooltip`
: Which node attributes appear in tooltips and in the interactive details panel: the first ten by default, `False` for none, one name, or a list of names.

## Edge channels

`edge_color`
: A color, data (categorical or continuous, like nodes), or `"source"` / `"target"` to reuse the color of an endpoint.

`edge_width`
: Pixels when constant; data map linearly to 0.75–5 px, with a width legend.

`edge_opacity`
: One number for all edges. By default, data-colored edges are drawn at 0.9 and neutral edges at the theme's edge opacity, reduced for graphs with more than 400 edges so that dense drawings stay readable.

`edge_style`
: `"auto"`, `"straight"`, `"curved"`, `"flow"`, `"orthogonal"` or `"spline"`; see [Edge styles](#edge-styles).

`edge_label`
: Text drawn at the midpoint of each edge, on a small halo in the surface color. Numbers are formatted compactly. Edges whose value is `None` get no label, which lets a mapping label only some edges.

`arrows`, `arrow_size`, `curvature`
: Arrowheads are on for directed graphs and off for undirected ones unless you set `arrows=`. Their length grows with the edge width unless you fix `arrow_size=`. `curvature` (0.16 by default) sets how far curved edges bend.

For example, labeling a route's edges with their lengths:

```python
route = ag.alg.shortest_path(campus, "OO", "SH", weight="length")
legs = {(u, v): f"{campus.edges[u, v]['length']:.0f} m" for u, v in zip(route, route[1:])}
print(len(route) - 1, "legs, first:", legs[route[0], route[1]])   # 10 legs, first: 170 m

fig = ag.draw(campus, layout=on_map, highlight_path=route, edge_label=legs)
fig.save("route_legs.svg")
```

## Emphasis: highlight and highlight_path

`highlight=` takes nodes and edges to bring forward; `highlight_path=` takes a node sequence and emphasizes it together with the edges between consecutive nodes. Emphasized nodes get a ring in the highlight color, emphasized edges are widened and drawn on top, and everything else is dimmed. Labels of emphasized nodes are placed first and stay visible even where they collide. `highlight_color=` overrides the theme's highlight color.

```python
length = sum(campus.edges[u, v]["length"] for u, v in zip(route, route[1:]))
print(route)
# ['OO', 'KNB', 'KNA', 'IH', 'RC', 'RT', 'CH', 'MFH', 'PF', 'EDT', 'SH']
print(f"{length:,.0f} m")   # 1,002 m

fig = ag.draw(
    campus,
    layout=on_map,
    node_color="kind",
    highlight_path=route,
    title="Olympic Oval to Scurfield Hall",
    subtitle=f"Shortest walk, {length:,.0f} m",
)
fig.save("route.svg")
```

```{figure} ../_static/generated/guide_core/drawing_route.png
:alt: Campus map with most buildings faded and an 11-building route from the Olympic Oval to Scurfield Hall drawn in blue with ringed nodes.

`highlight_path` with the shortest walk by length. The ten edges of the route are widened and drawn on top; every other mark is dimmed. Positions © OpenStreetMap contributors (ODbL).
```

## Titles, size and canvas

`title`, `subtitle`, `caption`
: Text above the drawing (title and subtitle) and in small print below it (caption).

`width`, `height`
: The display size in pixels. The drawing is fitted into it; give one of them to keep the natural aspect ratio. Without them, the size follows from the layout and the number of labels.

`scale`, `padding`, `background`
: `scale` fixes the number of pixels per layout unit instead of choosing it automatically. `padding` is the margin (24 px in the built-in themes). `background` is a color, or `"transparent"` for no background.

`legend`
: `"auto"` places legends on the right, or below the drawing when it is wider than about 1.35 times its height and more than 520 px wide. `"right"` and `"bottom"` force a side; `False` removes all legends.

`theme`
: `"light"` (the default), `"dark"`, `"paper"`, `"blueprint"` or a `Theme` object; see [Styling](styling.md).

## Layout argument forms

The `layout` argument decides where nodes go. It accepts:

- `"auto"` (the default): the layered `hierarchical` layout for directed acyclic graphs with at least one edge and up to 2,000 nodes, `stress` for other graphs of up to 3,000 nodes, and `forceatlas2` beyond that;
- a method name such as `"stress"`, `"radial"` or `"circular"`, with engine options in `layout_options=` and the random seed in `seed=` (0 by default);
- a {py:class}`~aryagraph.layout.base.Layout` computed earlier, for example with {py:func}`aryagraph.layout.compute`, so several figures share one layout;
- a `{node: (x, y)}` mapping, such as real coordinates (x to the right, y downward);
- a callable `f(g)` returning a `Layout` or a mapping.

```python
build = ag.gen.software_build()
shared = ag.layout.compute(build, "hierarchical", orientation="LR")

a = ag.draw(build, layout=shared, node_color="team")
b = ag.draw(build, layout=shared, node_color="kind")
print(a.scene.meta["layout"], b.scene.meta["orientation"])   # hierarchical LR
```

For layouts in abstract units (force-directed, stress, geometric and your own coordinates) of up to 3,000 nodes, `draw()` also removes node overlaps in pixel space, which can shift crowded nodes by a few pixels; `avoid_overlap=False` keeps positions as given. Layered, tree and radial layouts are already sized in pixels and skip this step. The [Layouts](layouts.md) page describes every engine and its options.

## Encodings and scales

AryaGraph picks a scale from the data, and {py:func}`ag.by() <aryagraph.style.scales.by>` lets you override any part of it.

### How the scale is chosen

| Data | Scale | Legend |
|---|---|---|
| strings, booleans, mixed values | categorical | swatches with counts |
| integers with at most two distinct values | categorical | swatches with counts |
| integers in an attribute whose name suggests groups (`community`, `group`, `cluster`, `class`, `type`, `kind`, `team`, `level`, `stage`, `role` and similar), up to 64 distinct values | categorical | swatches with counts |
| other numbers | sequential | colorbar |
| CSS color strings (`"#2a78d6"`, `"teal"`, `"rgb(…)"`) | identity: used as given | none |

### Categories and the "Other" slot

Categories take the palette's slots in natural sort order ("n2" before "n10"), or in the order you pass as `domain=`. The default palette has eight slots, and slots are not reused: with more than eight categories, the seven most frequent keep their colors and the rest share one gray "Other" entry, whose label says how many categories it holds. In the campus figure above, the nine building kinds became seven colors plus "Other (2)", covering the administration and library buildings (3 buildings in total):

```python
fig = ag.draw(campus, layout=on_map, node_color="kind")
legend = fig.scene.legends[0][0]
print([(e.label, e.count) for e in legend.entries][-2:])
# [('student-life', 4), ('Other (2)', 3)]
```

Reusing colors would make two categories look identical, which is why the tail folds instead. To keep more categories apart, pass a longer palette (every color in a custom list gets a slot) or map values to colors yourself:

```python
kind_colors = {"academic": "#2a78d6", "residence": "#e87ba4", "library": "#e34948"}
fig = ag.draw(campus, layout=on_map, node_color=ag.by("kind", palette=kind_colors))
print(fig.scene.legends[0][0].entries[-1].label)   # Other (6)
```

Values missing from a `{value: color}` palette fold into "Other" in the same way. `ag.by(..., counts=False)` drops the counts from the legend, and `max_categories=` lowers the number of slots used from the default palette.

### Sequential, log and square-root scales

Numbers map onto a sequential colormap (the theme's blue ramp by default), interpolated in the OKLab color space so that lightness changes smoothly from low to high. `kind="log"` spaces the colors by orders of magnitude, which suits heavy-tailed measures such as PageRank or degree; zero and negative values are drawn in the "Other" gray. `kind="sqrt"` sits between the two. The same kinds apply to sizes and widths:

```python
web = ag.gen.barabasi_albert(300, 2, seed=7)
rank = ag.alg.pagerank(web)
print(f"{rank.describe()['min']:.5f} to {rank.describe()['max']:.4f}")   # 0.00173 to 0.0276

fig = ag.draw(
    web,
    node_color=ag.by(rank, kind="log", palette="viridis", title="PageRank (log)"),
    node_size=ag.by(web.degree(), kind="sqrt", title="degree"),
    labels=False,
    title="Preferential attachment, 300 nodes",
    subtitle="Color: PageRank on a log scale; size: degree",
)
fig.save("pagerank.svg")
```

```{figure} ../_static/generated/guide_core/drawing_log.png
:alt: A dense 300-node network in a stress layout; a few large yellow-green hub nodes stand out among many small dark purple nodes; legends show a viridis colorbar and three size circles.
:width: 75%

PageRank spans a factor of about 16 in this graph. On a log scale the hubs and the periphery both keep visible differences in color.
```

### Diverging scales

`kind="diverging"` uses two hues that meet at a neutral midpoint. The midpoint is 0 when the data span zero, the median otherwise, or the value you pass as `midpoint=`; the scale is symmetric around it, so equal distances from the midpoint get equally strong colors. Here each building is compared by its network distance (the summed `length` of the shortest route, in meters) to the student center (MSC) and to the library (TFDL):

```python
to_msc = ag.alg.shortest_path_length(campus, "MSC", weight="length")
to_tfdl = ag.alg.shortest_path_length(campus, "TFDL", weight="length")
closer = {n: to_msc[n] - to_tfdl[n] for n in campus}
print(min(closer.values()), max(closer.values()))   # -670.0 670.0
print(to_msc["TFDL"])                               # 670.0

fig = ag.draw(
    campus,
    layout=on_map,
    node_color=ag.by(closer, kind="diverging", title="MSC minus TFDL (m)"),
    title="Nearer by the network: student center or library?",
    subtitle="Network distance to MSC minus network distance to TFDL (m)",
)
fig.save("nearer.svg")
```

```{figure} ../_static/generated/guide_core/drawing_diverging.png
:alt: Campus map with buildings in the north colored blue, buildings in the south-east colored red, and pale buildings in between; a blue-to-red colorbar from -500 to 500.

Blue buildings are nearer to the student center, red ones nearer to the library, and pale ones about equally far from both. Positions © OpenStreetMap contributors (ODbL).
```

The values stop at ±670 m because no building can be farther from one hub than from the other by more than the network distance between the two hubs, which is 670 m.

### Identity scales

When the values already are colors or pixel sizes, `kind="identity"` uses them as given and draws no legend. Color strings are detected automatically; for sizes and widths you ask for it, since a column of numbers could also be data:

```python
indoor_kinds = {"tunnel", "pedway", "attached"}
fig = ag.draw(
    campus,
    layout=on_map,
    edge_color=lambda u, v, d: "#52514e" if d["kind"] in indoor_kinds else "#c3c2b7",
    edge_width=ag.by(lambda u, v, d: 2.0 if d["kind"] in indoor_kinds else 1.0, kind="identity"),
)
print(len(fig.scene.legends))   # 0
```

### ag.by options

| Option | Effect |
|---|---|
| `kind` | `"categorical"`, `"sequential"`, `"diverging"`, `"log"`, `"sqrt"` or `"identity"` |
| `palette` | a list of colors, a `{value: color}` dict, or a colormap name or object |
| `domain` | category order, or the `(low, high)` range of a continuous scale; values outside it are clamped |
| `range` | `(min_px, max_px)` for sizes and widths |
| `midpoint` | center of a diverging scale |
| `title` | legend title (defaults to the attribute or result name) |
| `legend` | `False` to leave this channel out of the legend |
| `max_categories`, `counts` | slot limit for the default palette; category counts on or off |

Colormap names and custom palettes are described in [Styling](styling.md#palettes-and-colormaps).

## Legends

Each data-driven color, size, width and shape channel adds its own legend block: swatches with counts for categories, a colorbar with rounded ticks for continuous colors, two or three sample circles for sizes and sample strokes for widths. The legend title is the attribute name, the result's name (a PageRank `NodeMap` is titled `pagerank`), or the `title=` you pass to `ag.by`.

Constant channels and identity scales have nothing to explain and get no legend. Use `legend=False` to drop every legend, or `ag.by(..., legend=False)` to drop one.

## Label placement

Side labels are placed one node at a time, in order of importance: larger nodes first, then higher degree, then graph order, with emphasized nodes ahead of all others. Each label tries eight positions (right, left, above, below, then the four diagonals) and takes the first one that does not overlap a node, an edge label, a self-loop or a label placed earlier. A label with no free position is hidden instead of drawn over something else. Hidden labels are still in the SVG (marked hidden), in the node's tooltip and in the interactive table view, and the interactive toolbar can show them all.

Longer labels need more room, so the drawing grows to fit them. With full building names instead of codes, the campus figure becomes larger, and one label that still finds no free position is hidden:

```python
fig = ag.draw(campus, layout=on_map, labels="name")
shown = [m.node for m in fig.scene.nodes if m.label and m.label.visible]
hidden = [m.node for m in fig.scene.nodes if m.label and not m.label.visible]
print(round(fig.width), round(fig.height), len(shown), hidden)   # 1851 1379 55 ['ICT']
```

In layered and tree layouts of up to 150 nodes, nodes with labels longer than three characters become boxes with the text inside, and each box is sized to its text, wrapping at `label_max_width` (up to three lines). Short labels of up to three characters go inside circles in small graphs. Inside placement is all-or-nothing for one drawing, so the look stays uniform; `label_position="center"` requests it explicitly.

## Node shapes

Fourteen shapes are available: `circle`, `ellipse`, `square`, `rect`, `box`, `roundrect`, `pill`, `diamond`, `triangle`, `triangle_down`, `hexagon`, `octagon`, `pentagon` and `cylinder`. Edges are clipped to each node's actual outline, so arrowheads touch a diamond at its edge and a pill on its curve rather than stopping at a bounding circle.

```python
import math

names = list(ag.style.SHAPES)
spokes = ag.DiGraph([("hub", s) for s in names])
ring = {"hub": (0.0, 0.0)}
for i, s in enumerate(names):
    angle = 2 * math.pi * i / len(names) - math.pi / 2
    ring[s] = (1.35 * math.cos(angle), math.sin(angle))

fig = ag.draw(
    spokes,
    layout=ring,
    scale=200,
    node_shape=lambda n: "circle" if n == "hub" else n,
    labels=lambda n: None if n == "hub" else n,
    label_position="center",
    node_color=lambda n: "#898781" if n == "hub" else "#2a78d6",
    title="Node shapes",
    subtitle="Labels set inside; arrowheads end on each outline",
)
fig.save("shapes.svg")
print(len(names))   # 14
```

```{figure} ../_static/generated/guide_core/drawing_shapes.png
:alt: Fourteen node shapes arranged in a ring around a gray hub, each labeled with its shape name, with an arrow from the hub ending on each outline.
:width: 80%

Every shape with its name inside. Shapes that frame a label (boxes, pills, polygons, the cylinder) are drawn as light cards with a colored outline; circles, squares and ellipses stay solid with the label in a contrasting color. Triangles and pentagons suit markers better than text.
```

## Edge styles

| Style | Drawing |
|---|---|
| `straight` | straight segments; long edges in layered layouts bend at their routing points |
| `curved` | arcs whose bend is set by `curvature` |
| `flow` | smooth S-curves along the layer direction, with edge ends spread across the node's side |
| `orthogonal` | right-angle routing with rounded corners |
| `spline` | straight for short edges, a smooth curve through the routing points of long edges |
| `auto` | `flow` for layered and tree layouts, `straight` otherwise |

With straight edges, a pair of opposite arcs in a directed graph is drawn as two arcs bending apart so both stay visible. Self-loops are drawn in the widest free angle around their node, in every style.

```python
steps = ag.DAG([
    ("fetch", "parse"), ("fetch", "lint"), ("parse", "compile"), ("parse", "docs"),
    ("compile", "test"), ("lint", "test"), ("test", "release"), ("docs", "release"),
    ("fetch", "release"),
])
for style in ("straight", "curved", "flow", "orthogonal"):
    ag.draw(steps, edge_style=style, title=style).save(f"style_{style}.svg")
```

::::{grid} 2 2 4 4
:gutter: 2

:::{grid-item}
```{image} ../_static/generated/guide_core/drawing_style_straight.png
:alt: Layered DAG of seven boxes with straight edges; the long edge from fetch to release bends at two routing points.
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_core/drawing_style_curved.png
:alt: The same DAG with curved edges.
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_core/drawing_style_flow.png
:alt: The same DAG with S-shaped flow edges that leave and enter boxes vertically.
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_core/drawing_style_orthogonal.png
:alt: The same DAG with right-angled edges with rounded corners.
```
:::
::::

<p class="ag-caption">The same seven-step DAG in four edge styles. Its default, <code>auto</code>, is <code>flow</code> because the layout is layered.</p>

## Next steps

- [Styling](styling.md): themes, palettes and colormaps.
- [Interactive HTML](interactive.md): what the `.html` output adds.
- [Exporting](exporting.md): sizes, formats and the PNG/PDF backends.
- [Layouts](layouts.md): every layout engine and its options.
