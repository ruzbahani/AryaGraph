# Campus routing

In this tutorial you plan walking routes across the University of Calgary main campus. You find the shortest route in meters, a route that stays indoors, ranked alternatives, the buildings you can reach without going outside, and the crossroads that many routes depend on.

**Goal:** answer practical routing questions on a real spatial network, and draw the answers on the campus map.

**Prerequisites:**

- [Your first network](first-network.md), or familiarity with `ag.Graph`, node and edge attributes and `ag.draw`.
- AryaGraph installed; nothing else is needed for this page.

```{note}
The campus dataset is a schematic model compiled from public sources. Each `length` is the straight-line distance between two building positions (from OpenStreetMap), not a surveyed walking distance, and the `outdoor` edges are modeled walking links: each building is joined to its two nearest neighbors within 250 m, plus the links needed to connect the campus. The numbers below describe this model. See {py:func}`~aryagraph.generators.datasets.ucalgary_campus` for the full sources.
```

## Step 1: Load the campus twice

`ag.gen.ucalgary_campus()` returns the full network. With `indoor_only=True` it keeps only the tunnel, pedway and attached links (adjoining buildings with an internal door): the network you can use without stepping outside. Both graphs have the same 56 buildings.

```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
indoor = ag.gen.ucalgary_campus(indoor_only=True)
print(campus)
print(indoor)
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
<Graph 'University of Calgary main campus': 56 nodes, 40 edges>
```

Routes are lists of building codes. Two small helpers add up a route's total length and the part of it spent outdoors, and a third prints a one-line summary:

```python
def route_length(graph, path):
    return sum(graph.edges[u, v]["length"] for u, v in zip(path, path[1:]))


def outdoor_length(graph, path):
    return sum(
        graph.edges[u, v]["length"]
        for u, v in zip(path, path[1:])
        if graph.edges[u, v]["kind"] == "outdoor"
    )


def show(graph, path):
    print(" > ".join(path))
    print(f"{len(path) - 1} links, {route_length(graph, path):,.0f} m, "
          f"{outdoor_length(graph, path):,.0f} m of it outdoors")
```

## Step 2: Find the shortest route by length

Walk from the Olympic Oval (OO) on the west side of campus to Scurfield Hall (SH) on the east side. Passing `weight="length"` makes {py:func}`~aryagraph.algorithms.paths.shortest_path` minimize meters with Dijkstra's algorithm instead of counting links:

```python
route = ag.alg.shortest_path(campus, "OO", "SH", weight="length")
show(campus, route)
```

```text
OO > KNB > KNA > IH > RC > RT > CH > MFH > PF > EDT > SH
10 links, 1,002 m, 266 m of it outdoors
```

The route passes through the two Kinesiology blocks (KNB, KNA), crosses the residence area outdoors to International House (IH) and the Rozsa Centre (RC), then continues through the arts buildings, Murray Fraser Hall (MFH), Professional Faculties (PF) and the Education Tower (EDT). Three of its ten links are outdoors, 266 m in total, about a quarter of the distance.

## Step 3: Stay indoors

On a cold day you may prefer a route with no outdoor segments. Run the same query on the indoor-only graph:

```python
inside = ag.alg.shortest_path(indoor, "OO", "SH", weight="length")
show(indoor, inside)
print(f"{route_length(indoor, inside) - route_length(campus, route):.0f} m longer")
```

```text
OO > KNB > MSC > MH > SB > SA > SS > AD > PF > EDC > EDT > SH
11 links, 1,132 m, 0 m of it outdoors
130 m longer
```

The indoor route heads north instead: through MacEwan Student Centre (MSC) and MacEwan Hall (MH), the tunnel to Science B (SB), the science buildings, Social Sciences (SS) and the Administration Building (AD), then back south through Professional Faculties and the Education buildings. It is 130 m (13%) longer and has no outdoor segment. Because it is the shortest route in the indoor graph, no indoor-only route between these two buildings is shorter.

## Step 4: Draw both routes on the map

Every building carries `pos`, its position in meters (x east, y south), so the drawing is to scale and north-up. To show two routes at once, classify each edge with a small function and pass it to `edge_color` through {py:func}`ag.by <aryagraph.style.scales.by>`. The `ag.by` spec also sets the legend title, the category order (`domain`) and the colors (`palette`). With `kind="identity"`, `edge_width` takes the function's values as pixel widths instead of scaling them:

```python
shortest_links = {frozenset(e) for e in zip(route, route[1:])}
indoor_links = {frozenset(e) for e in zip(inside, inside[1:])}


def which_route(u, v):
    e = frozenset((u, v))
    if e in shortest_links and e in indoor_links:
        return "both routes"
    if e in shortest_links:
        return "shortest route"
    if e in indoor_links:
        return "indoor route"
    return "other links"


on_a_route = set(route) | set(inside)


def building_group(n):
    return "on a route" if n in on_a_route else "elsewhere"


def link_width(u, v):
    return 1.0 if which_route(u, v) == "other links" else 3.5


building_colors = {"on a route": "#52514e", "elsewhere": "#c3c2b7"}
link_colors = {
    "shortest route": "#2a78d6",
    "indoor route": "#eb6834",
    "both routes": "#4a3aa7",
    "other links": "#d6d5ce",
}
fig = ag.draw(
    campus,
    layout={n: d["pos"] for n, d in campus.nodes.data()},
    node_color=ag.by(building_group, title="building",
                     domain=list(building_colors), palette=building_colors),
    labels={n: n for n in on_a_route},
    edge_color=ag.by(which_route, title="link",
                     domain=list(link_colors), palette=link_colors),
    edge_width=ag.by(link_width, kind="identity", legend=False),
    title="Olympic Oval to Scurfield Hall",
    subtitle="Shortest route 1,002 m (266 m outdoors) · indoor route 1,132 m",
)
fig.save("oo-to-sh.html")
```

```{figure} ../_static/generated/tutorials_a/routing_routes.png
:alt: Campus map with two routes from the Olympic Oval in the west to Scurfield Hall in the east; the shortest route runs south through the residences, the indoor route runs north through MacEwan Student Centre and the science buildings.
:width: 100%

The shortest route (blue) and the indoor-only route (orange) share their first and last links (violet). Building positions © OpenStreetMap contributors (ODbL).
```

<iframe class="ag-embed" src="../_static/generated/tutorials_a/routing_routes.html" height="710" loading="lazy" title="Interactive campus map with the shortest and indoor routes from the Olympic Oval to Scurfield Hall"></iframe>
<p class="ag-embed-note">Interactive version: hover a building to see its name and attributes, or search for a building code. Building positions © OpenStreetMap contributors (ODbL). <a href="../_static/generated/tutorials_a/routing_routes.html">Open full screen</a></p>

## Step 5: Compare route lengths across trips

One trip is an anecdote. Compare a handful with {py:func}`~aryagraph.algorithms.paths.shortest_path_length`, which returns only the distance:

```python
trips = [("OO", "SH"), ("MSC", "ICT"), ("TFDL", "SS"),
         ("OO", "TFDL"), ("ES", "SH"), ("TFDL", "ICT")]
print(f"{'trip':<11}{'any route':>10}{'indoors':>10}{'extra':>7}")
for s, t in trips:
    mixed = ag.alg.shortest_path_length(campus, s, t, weight="length")
    warm = ag.alg.shortest_path_length(indoor, s, t, weight="length")
    extra = (warm - mixed) / mixed
    print(f"{s + '-' + t:<11}{mixed:>8,.0f} m{warm:>8,.0f} m{extra:>7.0%}")
```

```text
trip        any route   indoors  extra
OO-SH         1,002 m   1,132 m    13%
MSC-ICT         393 m     393 m     0%
TFDL-SS         434 m     501 m    15%
OO-TFDL         694 m   1,210 m    74%
ES-SH           571 m     630 m    10%
TFDL-ICT        729 m     796 m     9%
```

Some trips cost nothing extra: the shortest route from MSC to the ICT building already runs through MacEwan Hall, the tunnel and Earth Sciences. Others cost a lot. From the Olympic Oval to the Taylor Family Digital Library (TFDL), the shortest route ends with an outdoor link from Craigie Hall; the indoor route has to go around through MacEwan and the science buildings and is 74% longer.

## Step 6: Rank alternative routes

The shortest route is not the only reasonable one. {py:func}`~aryagraph.algorithms.paths.k_shortest_paths` (Yen's algorithm) returns the *k* shortest loopless routes, shortest first. Listing the outdoor meters next to each one turns them into a menu:

```python
alternatives = ag.alg.k_shortest_paths(campus, "OO", "SH", 4, weight="length")
for path in alternatives:
    print(f"{route_length(campus, path):6,.0f} m total, "
          f"{outdoor_length(campus, path):4,.0f} m outdoors: {' > '.join(path)}")
```

```text
 1,002 m total,  266 m outdoors: OO > KNB > KNA > IH > RC > RT > CH > MFH > PF > EDT > SH
 1,060 m total,  198 m outdoors: OO > KNB > KNA > IH > RC > RT > CH > MFH > PF > EDC > EDT > SH
 1,073 m total,   68 m outdoors: OO > KNB > MSC > MH > SB > SA > SS > AD > PF > EDT > SH
 1,074 m total,  372 m outdoors: OO > KNB > KNA > IH > RC > RT > CH > MFH > PF > EDT > MTH > SH
```

All four routes are within 73 m of each other, but they differ a lot in exposure. The third one is 71 m (7%) longer than the shortest and spends only 68 m outdoors, on the link from Professional Faculties to the Education Tower. It follows the indoor route of Step 3 and skips the detour through Education Classrooms (EDC). A route planner could offer this trade-off: 71 m more walking for about 200 m less outdoors.

## Step 7: Find every building reachable indoors

Which buildings can you reach from MacEwan Student Centre (MSC) without going outside? That is the connected component of MSC in the indoor graph:

```python
reachable = ag.alg.node_connected_component(indoor, "MSC")
print(len(reachable) - 1, "buildings reachable indoors from MSC")
print(sorted(set(campus) - reachable))
```

```text
37 buildings reachable indoors from MSC
['CC', 'CD', 'CDC', 'CR', 'CSSH', 'EEEL', 'GL', 'GR', 'HP', 'KA', 'MEB', 'OL', 'OVC', 'PP', 'RU', 'TI', 'TRB', 'YA']
```

{py:func}`~aryagraph.algorithms.connectivity.node_connected_component` returns a set, so the code sorts it before printing to get a stable order. The indoor network is one large component of 38 buildings (MSC and the 37 others) plus 18 buildings with no indoor link at all. Seven of those 18 are residence halls, such as Cascade Hall (CD) and Rundle Hall (RU). The Taylor Institute (TI) stands next to MSC, but the dataset joins them only by an outdoor link.

Without a target, `shortest_path_length` returns the distance to every reachable building as a `NodeMap`, so the farthest ones are one call away:

```python
from_msc = ag.alg.shortest_path_length(indoor, "MSC", weight="length")
for building, meters in from_msc.top(3):
    print(f"{building:<5}{campus.nodes[building]['name']:<32}{meters:,.0f} m")
```

```text
AB   Art Building & Art Parkade      950 m
RC   Rozsa Centre                    939 m
TFDL Taylor Family Digital Library   895 m
```

Drawing the indoor graph with node color set to that `NodeMap` gives a map of indoor walking distance. Buildings missing from the mapping have no value, so they are drawn in a neutral color:

```python
fig = ag.draw(
    indoor,
    layout={n: d["pos"] for n, d in indoor.nodes.data()},
    node_color=ag.by(from_msc, title="meters from MSC, indoors"),
    title="How far can you walk indoors from MacEwan Student Centre?",
    subtitle="Tunnels, pedways and attached buildings only · gray: no indoor connection",
)
fig.save("indoor-from-msc.svg")
```

```{figure} ../_static/generated/tutorials_a/routing_indoor_reach.png
:alt: Campus map of the indoor network; buildings connected to MacEwan Student Centre are shaded from light to dark blue by indoor walking distance, and the 18 unconnected buildings are gray.
:width: 100%

Indoor walking distance from MSC. Only indoor links are drawn; the 18 gray buildings have no indoor connection to MSC. Building positions © OpenStreetMap contributors (ODbL).
```

## Step 8: Measure what staying inside costs

Steps 3 and 5 compared a few trips. {py:func}`~aryagraph.algorithms.paths.all_pairs_shortest_path_length` computes every distance at once, so you can check all pairs of buildings that are connected indoors:

```python
any_route = ag.alg.all_pairs_shortest_path_length(campus, weight="length")
indoors = ag.alg.all_pairs_shortest_path_length(indoor, weight="length")

connected = sorted(reachable)
pairs = [(u, v) for i, u in enumerate(connected) for v in connected[i + 1:]]
ratios = sorted(indoors[u][v] / any_route[u][v] for u, v in pairs)
same = sum(1 for r in ratios if r < 1 + 1e-9)
worst = max(pairs, key=lambda p: indoors[p[0]][p[1]] / any_route[p[0]][p[1]])

print(len(pairs), "pairs;", same, "with no extra distance")
print(f"median ratio {ratios[len(ratios) // 2]:.3f}; {sum(r > 2 for r in ratios)} pairs more than 2x")
print(worst, f"{any_route[worst[0]][worst[1]]:,.0f} m vs {indoors[worst[0]][worst[1]]:,.0f} m indoors")
```

```text
703 pairs; 320 with no extra distance
median ratio 1.036; 56 pairs more than 2x
('IH', 'RC') 94 m vs 1,395 m indoors
```

For 320 of the 703 pairs (46%), staying inside adds no distance, and the median pair walks 3.6% farther indoors. The tail is long, though: 56 pairs more than double their distance. The extreme case is International House to the Rozsa Centre, 94 m apart by their outdoor link but 1,395 m apart indoors, because the indoor route has to loop through Kinesiology, MacEwan, the science buildings and Professional Faculties.

## Step 9: Find the chokepoints

Some buildings carry a large share of the traffic. Length-weighted betweenness centrality measures the share of shortest routes between other buildings that pass through each one:

```python
bc = ag.alg.betweenness_centrality(campus, weight="length")
for building, score in bc.top(5):
    print(f"{building:<4}{campus.nodes[building]['name']:<50}{score:.3f}")
```

```text
PF  Professional Faculties                            0.307
KNB Dr. Roger Jackson Kinesiology Complex (Block B)   0.301
IH  International House                               0.218
RC  Rozsa Centre                                      0.209
MFH Murray Fraser Hall                                0.207
```

Professional Faculties and Kinesiology Block B each lie on about 30% of the shortest routes between other buildings. All five lie on the Olympic Oval to Scurfield Hall route from Step 2: in this model, that chain of buildings is the main west-east corridor.

A high score says a building is busy, not that it is irreplaceable. For that, ask which nodes are *articulation points* (removing one disconnects the network) and which edges are *bridges*:

```python
print(ag.alg.articulation_points(campus))
print(ag.alg.bridges(campus))
```

```text
['CR', 'EDT', 'HP', 'MTH', 'PF', 'PP']
[('CDC', 'PP'), ('CR', 'GR'), ('HP', 'PP'), ('MTH', 'OVC')]
```

Most of these sit at the edge of the network. For example, the Olympic Volunteer Centre (OVC) hangs off Mathison Hall (MTH) by a single 626 m link, so that link is a bridge and MTH is an articulation point. Professional Faculties stands out: it is the busiest building and also an articulation point.

Test it: close Professional Faculties (PF) in a copy of the graph and try to route again.

```python
closed = campus.copy()
closed.remove_node("PF")
print(ag.alg.has_path(closed, "OO", "SH"))
cut_off = [c for c in ag.alg.connected_components(closed) if "SH" in c][0]
print(sorted(cut_off))
try:
    ag.alg.shortest_path(closed, "OO", "SH", weight="length")
except ag.NoPath as err:
    print(type(err).__name__, err)
```

```text
False
['CC', 'EDC', 'EDT', 'MTH', 'OVC', 'SH']
NoPath no path from 'OO' to 'SH'
```

Without PF, six buildings on the east side of campus (Scurfield Hall among them) lose every link to the rest of campus, so the query raises {py:class}`~aryagraph.core.exceptions.NoPath` instead of returning a partial route. Treat this as a finding about the model: its outdoor links join each building only to its nearest neighbors, so a real closure would likely leave walking paths that the model does not include. Removing nodes from a copy like this is a quick way to test how robust any network is.

The figure below combines both views. Node size is betweenness and color marks the articulation points:

```python
cut_nodes = ag.alg.articulation_points(campus)
role_colors = {"articulation point": "#e34948", "other building": "#2a78d6"}


def role(n):
    return "articulation point" if n in cut_nodes else "other building"


fig = ag.draw(
    campus,
    layout={n: d["pos"] for n, d in campus.nodes.data()},
    node_size=ag.by(bc, title="betweenness (by length)"),
    node_color=ag.by(role, title="building", palette=role_colors),
    labels={n: n for n in cut_nodes + [b for b, _ in bc.top(5)]},
    label_position="below",
    title="Where campus routes concentrate",
    subtitle="Size: betweenness by length · red: articulation points",
)
fig.save("chokepoints.svg")
```

```{figure} ../_static/generated/tutorials_a/routing_chokepoints.png
:alt: Campus map with node size showing length-weighted betweenness; Professional Faculties and Kinesiology Block B are the largest nodes, and the six articulation points are drawn in red.
:width: 100%

Chokepoints: node size shows length-weighted betweenness and red marks the six articulation points; labels name those and the five buildings with the highest betweenness. Building positions © OpenStreetMap contributors (ODbL).
```

## Recap

You have:

- routed by meters with `shortest_path(..., weight="length")` and restricted routes to indoor links by querying the `indoor_only=True` graph;
- drawn two routes on real geometry with data-driven edge colors from a Python function;
- compared trips with `shortest_path_length` and ranked alternatives with `k_shortest_paths`;
- found the indoor component of MSC and the indoor distance to every building in it;
- summarized the cost of staying inside over all pairs with `all_pairs_shortest_path_length`;
- located chokepoints with betweenness, articulation points and bridges, and tested a closure on a copy of the graph.

## Next steps

- [Pipelines as DAGs](pipeline-dag.md): directed graphs, critical paths and scheduling.
- [Publication figures](publication-figures.md): prepare a drawing like the ones above for a paper.
- Case study: [How connected is the University of Calgary's indoor network?](../case-studies/ucalgary-campus.md)
- User guide: [Algorithms](../user-guide/algorithms.md), [Drawing](../user-guide/drawing.md) and [Styling](../user-guide/styling.md).
