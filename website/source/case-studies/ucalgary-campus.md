# How connected is the University of Calgary's indoor network?

Calgary winters make indoor routes between buildings valuable. This case study uses AryaGraph to measure how much of the University of Calgary's main campus you can cross without going outside, which links and buildings the indoor network depends on, and what a single closed pedway or tunnel does to indoor routes.

```{important}
The data is a **schematic model** compiled from public sources: building codes and names from the university's campus map and building directory, indoor links from its *Indoor Building Routes* map cross-checked against OpenStreetMap, and positions from OpenStreetMap (© OpenStreetMap contributors, ODbL). It is not an official or surveyed dataset and **not an accessibility map**: it does not record stairs, elevators, doors, opening hours or step-free routes. Use the university's own maps for real wayfinding.
```

## The question

Three questions frame the analysis:

1. **Coverage.** How many buildings belong to one connected indoor network, and which have no indoor link at all?
2. **Fragility.** Which indoor links and buildings are single points of failure, and which buildings carry the most indoor through-traffic?
3. **Impact.** If one indoor link closes, how many building pairs lose their indoor route, and how long are the detours when an alternative exists?

## Data

{py:func}`aryagraph.generators.datasets.ucalgary_campus` returns 56 main-campus buildings as nodes, keyed by their official codes (`MSC`, `TFDL`, `ICT`, …). Each edge has a `kind` and a `length` in meters. Three kinds are indoor connections: `tunnel`, `pedway` and `attached` (adjoining buildings with an internal door). The fourth kind, `outdoor`, marks modeled walking links that keep the campus graph connected. With `indoor_only=True` the function keeps all 56 buildings but only the indoor edges.

```python
import aryagraph as ag

full = ag.gen.ucalgary_campus()
indoor = ag.gen.ucalgary_campus(indoor_only=True)

kinds = {}
for _, _, kind in full.edges.data("kind"):
    kinds[kind] = kinds.get(kind, 0) + 1
print(full)
print(indoor)
print(kinds)
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
<Graph 'University of Calgary main campus': 56 nodes, 40 edges>
{'pedway': 11, 'outdoor': 43, 'attached': 26, 'tunnel': 3}
```

So 40 of the 83 links are indoor: 3 tunnels, 11 pedways and 26 attached pairs. Two properties of the data shape every number below:

- A `length` is the straight-line distance between two building centers (the centers of their OpenStreetMap bounding boxes), not a measured walking distance. An indoor route's length is therefore a consistent *index* of distance, not a step count.
- The Olympic Volunteer Centre (`OVC`) stands 626 m south of the main grounds, at McMahon Stadium. It has no indoor link, so the maps below leave it out to keep the rest of the campus legible.

```{figure} ../_static/generated/cases_a/campus_links.png
:alt: Map of 55 University of Calgary buildings at their real positions, with tunnels, pedways and attached-building links drawn in color and outdoor walking links in light gray.
:width: 100%

Every modeled link on the main grounds, drawn at the buildings' real positions (north up). Tunnels, pedways and attached pairs form the indoor network; light gray lines are modeled outdoor links. Positions © OpenStreetMap contributors (ODbL).
```

## Method

Each question maps to a standard graph measure. All of them run on the indoor-only graph, with `length` as the edge weight:

| Question | Measure | AryaGraph function |
|---|---|---|
| Coverage | connected components | {py:func}`~aryagraph.algorithms.connectivity.connected_components` |
| Fragile links | bridges: edges whose removal disconnects the graph | {py:func}`~aryagraph.algorithms.connectivity.bridges` |
| Fragile buildings | articulation points (cut vertices) | {py:func}`~aryagraph.algorithms.connectivity.articulation_points` |
| Through-traffic | length-weighted betweenness centrality | {py:func}`~aryagraph.algorithms.centrality.betweenness_centrality` |
| Closure impact | all-pairs shortest paths before and after removing one edge | {py:func}`~aryagraph.algorithms.paths.all_pairs_shortest_path_length` |
| Longest indoor walk | the largest shortest-route distance in the network | {py:func}`~aryagraph.algorithms.paths.shortest_path` |

Bridges and articulation points answer a yes-or-no question (does a closure disconnect anything?), while the closure experiment measures *how much* each closure costs. Running both is a useful cross-check: every bridge has to cut some pairs off, and no other link can.

## Results

### One campus, nineteen indoor islands

With outdoor links included, the campus is one connected graph. With indoor links only, it splits into 19 components: one network of 38 buildings and 18 buildings on their own.

```python
print("full campus connected:", ag.alg.is_connected(full))
comps = ag.alg.connected_components(indoor)
print(len(comps), "indoor components, sizes", [len(c) for c in comps])

# comps[0] is a set; the subgraph lists its buildings in graph order.
main = indoor.subgraph(comps[0])
print("indoor network:", main)
```

```text
full campus connected: True
19 indoor components, sizes [38, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
indoor network: <Graph 'University of Calgary main campus': 38 nodes, 40 edges>
```

```{figure} ../_static/generated/cases_a/indoor_islands.png
:alt: Map of the campus with indoor links only. 38 buildings in blue form one connected network; 17 buildings in orange have no indoor link.
:width: 100%

Indoor links only. Blue buildings belong to the 38-building indoor network; orange buildings have no indoor link (the eighteenth, the Olympic Volunteer Centre, is off the map). Positions © OpenStreetMap contributors (ODbL).
```

### Buildings with no indoor link

The 18 isolated buildings follow a clear pattern: they are residences, service buildings at the campus edge, and a few detached academic and research buildings.

```python
from collections import Counter

islands = [n for n in indoor if indoor.degree(n) == 0]
for n in islands:
    print(f"{n:<5} {full.nodes[n]['kind']:<15} {full.nodes[n]['name']}")

total = Counter(kind for _, kind in full.nodes.data("kind"))
alone = Counter(full.nodes[n]["kind"] for n in islands)
print({kind: f"{alone[kind]} of {total[kind]}" for kind in alone})
```

```text
CC    services        Child Care Centre
CD    residence       Cascade Hall
CDC   research        Child Development Centre
CR    residence       Crowsnest Hall
CSSH  research        Cenovus Spo'pi Solar House
EEEL  academic        Energy Environment Experiential Learning
GL    residence       Glacier Hall
GR    services        Grounds Building
HP    services        Central Heating and Cooling Plant
KA    residence       Kananaskis Hall
MEB   academic        Mechanical Engineering Building
OL    residence       Olympus Hall
OVC   services        Olympic Volunteer Centre
PP    services        Physical Plant
RU    residence       Rundle Hall
TI    academic        Taylor Institute for Teaching and Learning
TRB   academic        Trailer B (Mathematical Science)
YA    residence       Yamnuska Hall
```

```text
{'services': '5 of 5', 'residence': '7 of 9', 'research': '2 of 3', 'academic': '4 of 25'}
```

The contrast between building types is sharp. All five service buildings and seven of the nine residences have no indoor link; the two connected residences are Aurora Hall and International House, which reach the network through the Dining Centre. Academic buildings are the opposite case: 21 of 25 are connected indoors.

### A tree-like network: bridges and cut buildings

The 38-building network has 40 links. A tree on 38 nodes has 37, so the network contains only three independent loops, and most of it is a single chain of links between any two buildings.

```python
bridges = ag.alg.bridges(indoor)
cut = ag.alg.articulation_points(indoor)
print(f"{len(bridges)} of {len(main.edges)} indoor links are bridges")
print(f"{len(cut)} of {len(main)} connected buildings are articulation points")
print("independent loops:", len(main.edges) - len(main) + 1)
for block in ag.alg.biconnected_components(main):
    if len(block) > 2:
        print("loop block:", sorted(block))
```

```text
30 of 40 indoor links are bridges
25 of 38 connected buildings are articulation points
independent loops: 3
loop block: ['ES', 'MS', 'SA', 'SB', 'SS', 'ST']
loop block: ['END', 'ENE', 'ENG']
```

Three quarters of the indoor links (30 of 40) are bridges: closing any one of them splits the network. The redundancy that exists sits in two places, the science block around Earth Sciences, Mathematical Sciences, Science A and B, Science Theatres and Social Sciences, and a triangle of three engineering blocks. Twenty-five of the 38 connected buildings are articulation points, so closing any one of them (for example for renovation) also disconnects the network.

```{figure} ../_static/generated/cases_a/single_points.png
:alt: The 38-building indoor network at real positions. Most links are drawn in red as bridges; two small groups of blue links form loops. 25 buildings are marked orange as cut buildings.
:width: 100%

Single points of failure. Red links are bridges (closing one disconnects the network); blue links lie on a loop. Orange buildings are articulation points: closing one splits the network. Positions © OpenStreetMap contributors (ODbL).
```

### Where indoor routes converge

Betweenness counts how many shortest indoor routes between *other* buildings pass through a building. On a tree-like network it measures both through-traffic and exposure: a building with high betweenness is one whose closure would cut many routes.

```python
bc = ag.alg.betweenness_centrality(main, weight="length")
others = (len(main) - 1) * (len(main) - 2) // 2      # pairs not involving the building itself
for n, value in bc.top(6):
    print(f"{n:<4} {value:.3f}  {value * others:4.0f} of {others} pairs   {full.nodes[n]['name']}")
```

```text
PF   0.498   332 of 666 pairs   Professional Faculties
SS   0.483   322 of 666 pairs   Social Sciences
AD   0.468   312 of 666 pairs   Administration Building
ES   0.392   261 of 666 pairs   Earth Sciences
SB   0.363   242 of 666 pairs   Science B
ICT  0.348   232 of 666 pairs   Information and Communications Technology
```

Professional Faculties, Social Sciences and the Administration Building each lie on roughly half of all indoor routes between other buildings (312 to 332 of 666 pairs). The three form a chain, `PF`–`AD`–`SS`, that the next section identifies as the network's spine.

```{figure} ../_static/generated/cases_a/betweenness.png
:alt: The indoor network with node size and color showing betweenness. PF, SS and AD are the largest and darkest nodes.
:width: 100%

Length-weighted betweenness on the indoor network: larger, darker buildings sit on more shortest indoor routes. Positions © OpenStreetMap contributors (ODbL).
```

### What one closure does

The closure experiment removes one indoor link at a time, recomputes the shortest indoor route for every pair of buildings, and compares the result with the intact network. A pair is *cut off* when it had an indoor route before and has none after; otherwise the change in route length is its *detour*.

```python
nodes = list(indoor)
pairs = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]]
before = ag.alg.all_pairs_shortest_path_length(indoor, weight="length")
print(sum(b in before[a] for a, b in pairs), "of", len(pairs), "building pairs are connected indoors")

closures = []
for u, v, kind in indoor.edges.data("kind"):
    g = indoor.copy()
    g.remove_edge(u, v)
    after = ag.alg.all_pairs_shortest_path_length(g, weight="length")
    lost, detour, pair = 0, 0.0, None
    for a, b in pairs:
        if b not in before[a]:
            continue                                  # no indoor route even before the closure
        if b not in after[a]:
            lost += 1                                 # this closure cuts the pair off
        elif after[a][b] - before[a][b] > detour + 1e-9:
            detour, pair = after[a][b] - before[a][b], (a, b)
    closures.append({"link": f"{u}–{v}", "kind": kind, "lost": lost, "detour": detour, "pair": pair})

cutting = [r for r in closures if r["lost"] > 0]
print(len(cutting), "closures cut pairs off; the six largest:")
for r in sorted(cutting, key=lambda r: -r["lost"])[:6]:
    print(f"  {r['link']:<8} {r['kind']:<9} {r['lost']:>4} pairs")
print("same links as the bridges:", {r["link"] for r in cutting} == {f"{u}–{v}" for u, v in bridges})
```

```text
703 of 1540 building pairs are connected indoors
30 closures cut pairs off; the six largest:
  AD–SS    pedway     336 pairs
  AD–PF    attached   325 pairs
  ES–ICT   attached   261 pairs
  ENA–ICT  attached   240 pairs
  MFH–PF   pedway     240 pairs
  MH–SB    tunnel     240 pairs
same links as the bridges: True
```

The cross-check holds: the 30 closures that cut pairs off are the 30 bridges. Before any closure, only 703 of the 1,540 building pairs (46%) have an indoor route at all, because every pair involving one of the 18 islands has none. Closing the Administration–Social Sciences pedway alone cuts 336 of those 703 pairs off, nearly half. Which buildings end up on each side? Removing that pedway and listing the components shows the split:

```python
g = indoor.copy()
g.remove_edge("AD", "SS")
sides = [c for c in ag.alg.connected_components(g) if len(c) > 1]
print("sizes:", [len(c) for c in sides])
print("smaller side:", sorted(sides[1]))
```

```text
sizes: [24, 14]
smaller side: ['AB', 'AD', 'CH', 'EDC', 'EDT', 'HNSC', 'MFH', 'MT', 'MTH', 'PF', 'RC', 'RT', 'SH', 'TFDL']
```

The pedway is the only indoor connection between a south-east group of 14 buildings (among them the Taylor Family Digital Library, MacKimmie Tower, Murray Fraser Hall, Craigie Hall, the Education buildings and Scurfield Hall) and the other 24, which include the science and engineering blocks, MacEwan Student Centre and the Kinesiology Complex. The attached AD–PF link is the second half of the same chain, which is why closing it costs almost as much (325 pairs).

```{figure} ../_static/generated/cases_a/closures.png
:alt: Bar chart of the twelve indoor links whose closure cuts off the most building pairs, led by the AD–SS pedway with 336 pairs and the AD–PF attached link with 325.
:width: 90%

Building pairs that lose their indoor route when one link closes, for the twelve most consequential links.
```

The other ten links lie on loops, so closing one of them cuts no pair off; the cost is a longer indoor route. For each, the output lists the pair with the largest detour:

```python
for r in sorted((r for r in closures if r["lost"] == 0), key=lambda r: -r["detour"]):
    a, b = r["pair"]
    print(f"{r['link']:<8} {r['kind']:<9} +{r['detour']:3.0f} m   {a}–{b}: "
          f"{before[a][b]:,.0f} m → {before[a][b] + r['detour']:,.0f} m")
```

```text
MS–ST    attached  +300 m   BI–MS: 174 m → 474 m
ES–SB    attached  +243 m   AU–CCIT: 1,141 m → 1,384 m
ES–MS    pedway    +235 m   CCIT–MS: 409 m → 645 m
SA–SB    attached  +231 m   AU–SA: 828 m → 1,059 m
SS–ST    attached  +112 m   AB–BI: 730 m → 843 m
ENE–ENG  attached  + 69 m   AB–ENE: 1,013 m → 1,083 m
SA–SS    attached  + 65 m   AB–AU: 1,466 m → 1,531 m
END–ENG  attached  + 55 m   AB–CCIT: 1,082 m → 1,137 m
SA–ST    attached  + 52 m   AU–BI: 1,032 m → 1,085 m
END–ENE  attached  + 35 m   CCIT–ENE: 124 m → 159 m
```

Where a loop exists, it keeps detours short: the worst case adds 300 m, when the Mathematical Sciences–Science Theatres connection closes and the indoor route from Biological Sciences to Mathematical Sciences grows from 174 m to 474 m. Nine of the ten loop links are attached pairs; the only pedway among them is Earth Sciences–Mathematical Sciences.

### The longest indoor walk

The longest shortest route inside the network (its weighted diameter) runs from one end of the tree to the other.

```python
dist = ag.alg.all_pairs_shortest_path_length(main, weight="length")
a, b, meters = max(((s, t, d) for s, row in dist.items() for t, d in row.items()), key=lambda x: x[2])
route = ag.alg.shortest_path(main, a, b, weight="length")
print(f"{full.nodes[a]['name']} to {full.nodes[b]['name']}: {meters:,.0f} m through {len(route)} buildings")
print(" → ".join(route))
```

```text
Art Building & Art Parkade to Aurora Hall: 1,466 m through 14 buildings
AB → CH → MFH → PF → AD → SS → SA → SB → MH → MSC → KNB → KNA → DC → AU
```

The walk crosses the whole spine: from the Art Building through Craigie Hall and Murray Fraser Hall, over `PF`–`AD`–`SS`, through the science block and MacEwan Hall, and out through the Kinesiology Complex and the Dining Centre to Aurora Hall. Eleven of its 13 links are bridges. The two exceptions, Social Sciences to Science A and Science A to Science B, lie on the science-block loop; closing the first lengthens this walk by 65 m (the `SA–SS` row above).

```{figure} ../_static/generated/cases_a/longest_walk.png
:alt: Campus map with the indoor route from the Art Building to Aurora Hall highlighted through 14 buildings.
:width: 100%

The longest indoor walk in the model, highlighted over the indoor network. Positions © OpenStreetMap contributors (ODbL).
```

### Explore the network

The interactive view shows the indoor network with building names in tooltips. You can pan, zoom, search for a building code, and click the legend to filter building groups.

<iframe class="ag-embed" src="../_static/generated/cases_a/indoor_network.html" height="620" loading="lazy" title="Interactive map of the University of Calgary indoor network"></iframe>
<p class="ag-embed-note">Interactive: hover a building for its name, drag to pan, scroll to zoom. Positions © OpenStreetMap contributors (ODbL). <a href="../_static/generated/cases_a/indoor_network.html">Open full screen</a></p>

## Interpretation

In this model, the answer to the title question is: well covered, thinly connected.

- **Coverage is broad.** 38 of 56 buildings (68%), including 21 of the 25 academic buildings, share one indoor network. The gaps are the residences and service buildings: in the model, seven of the nine residence halls have no indoor link.
- **Redundancy is thin.** With 40 links for 38 buildings, the network is close to a tree. 30 links and 25 buildings are single points of failure, and the only loops are in the science block and the engineering triangle.
- **One chain carries the most weight.** The `PF`–`AD`–`SS` chain links the south-east group of 14 buildings to the other 24. Its two links are the costliest closures (336 and 325 of 703 connected pairs cut off), and its three buildings top the betweenness ranking.
- **Loops are effective where they exist.** The largest detour caused by closing a loop link is 300 m.

In graph terms, the largest gain in redundancy would come from any new link that closes a loop around the `PF`–`AD`–`SS` chain, joining the south-east group to the rest a second time. The model identifies where such a link would matter; it cannot say whether one is physically or financially feasible.

## Limitations

- **Schematic data.** Links come from a published routes map and OpenStreetMap, not a survey. A missing or extra link changes the bridge and component results directly, so treat individual findings as statements about the model.
- **Distances are center to center.** Route lengths add straight lines between building centers. They rank routes consistently but underestimate real walking distance, especially through large buildings such as the Olympic Oval.
- **Buildings are points.** Each building is one node, so the model assumes you can walk between any two of its indoor links. Real floor plans, locked wings and opening hours can break that assumption.
- **No accessibility information.** Stairs, ramps, elevators and door widths are not in the data, so the analysis says nothing about step-free routes.
- **One closure at a time.** The experiment removes single links. Closing a building (all its links at once) or several links together works the same way: remove them with {py:meth}`~aryagraph.core.graph.Graph.remove_node` or {py:meth}`~aryagraph.core.graph.Graph.remove_edge` before recomputing.

## Reproducibility

Every code block on this page runs top to bottom, in order, with AryaGraph {{ version }} and no random numbers; the printed results above are the actual output. The figures come from the same computations in the site's asset script, [`website/scripts/assets/cases_a.py`](https://github.com/ruzbahani/AryaGraph/blob/main/website/scripts/assets/cases_a.py). To draw a map yourself, pass the stored positions as the layout:

```python
fig = ag.draw(
    main,
    layout={n: d["pos"] for n, d in main.nodes.data()},   # meters, x east, y south: north is up
    node_color=ag.by(bc, kind="sequential", title="betweenness"),
    highlight_path=route,
    title="Indoor network and its longest walk",
    subtitle="Positions © OpenStreetMap contributors (ODbL)",
)
fig.save("indoor_network.svg")    # also .html (interactive), .png, .pdf
```

Related pages: [Campus routing](../tutorials/campus-routing.md), [Datasets and generators](../user-guide/datasets.md), [Algorithms](../user-guide/algorithms.md).
