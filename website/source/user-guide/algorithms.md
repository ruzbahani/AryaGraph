# Algorithms

`ag.alg` collects AryaGraph's graph algorithms: paths, connectivity, DAG scheduling, centrality, structure, communities, flow, spanning trees, matching, coloring, link prediction and matrices. This page tours each area with short examples on the bundled datasets; the [API reference](../reference/algorithms.rst) lists every function and parameter.

## Conventions

Algorithms take the graph as their first argument and return plain Python data, a result map, or a small dataclass. Functions live in topic modules such as `aryagraph.algorithms.paths`, and each one is also available as `ag.alg.<name>`.

Per-node results are {py:class}`aryagraph.core.results.NodeMap` objects: dictionaries from node to value with helpers for ranking and statistics. Per-edge results are {py:class}`aryagraph.core.results.EdgeMap` objects with the same helpers.

```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()      # 56 buildings, 83 links
bc = ag.alg.betweenness_centrality(campus, weight="length")
print(bc)
```

```text
NodeMap('betweenness_centrality', 56 nodes; top: {'PF': 0.3071, 'KNB': 0.301, 'IH': 0.2175, …})
```

Algorithms that have a networkx equivalent are tested against networkx on seeded random graphs and follow its definitions (normalization, handling of dangling nodes, and so on), so results can be cross-checked. The others are tested on hand-checked cases and invariants.

### Weights

A `weight` argument takes one of three forms:

- `None`: every edge counts as 1 (hop counts);
- an edge-attribute name such as `"length"`: edges without the attribute count as 1;
- a callable `f(u, v, attrs)` returning a number. In path algorithms it may return `None` to hide an edge.

The default depends on what the weight means to the algorithm. Distance-based measures such as `shortest_path`, betweenness, closeness, eccentricity and diameter default to `None`, so they count hops unless you ask for lengths. Algorithms that need weights to be meaningful (Dijkstra, Bellman–Ford, A*, *k* shortest paths, spanning trees, weighted matching) default to `"weight"`, and so do PageRank, HITS, Louvain and modularity, which read a weight as a connection *strength*. When in doubt, pass `weight=` explicitly:

```python
import inspect

functions = (
    ag.alg.shortest_path, ag.alg.betweenness_centrality, ag.alg.diameter,
    ag.alg.dijkstra, ag.alg.minimum_spanning_tree,
    ag.alg.pagerank, ag.alg.louvain_communities,
)
for fn in functions:
    default = inspect.signature(fn).parameters["weight"].default
    print(f"{fn.__name__:24} weight={default!r}")
```

```text
shortest_path            weight=None
betweenness_centrality   weight=None
diameter                 weight=None
dijkstra                 weight='weight'
minimum_spanning_tree    weight='weight'
pagerank                 weight='weight'
louvain_communities      weight='weight'
```

```{important}
The campus edges carry both `length` and a `weight` attribute equal to the length in meters. Functions that default to `weight="weight"` therefore read meters on this graph. That suits Dijkstra and spanning trees, but PageRank and Louvain treat weights as strengths, so a long link would count as a strong tie. Pass `weight=None` to those functions on the campus graph.
```

## Traversal

Breadth-first and depth-first searches return orders, edges, layers or trees. `bfs_layers` groups nodes by hop distance:

```python
layers = ag.alg.bfs_layers(campus, "MSC")
print([len(layer) for layer in layers])
print(layers[1])
```

```text
[1, 3, 5, 9, 16, 7, 5, 4, 4, 2]
['KNB', 'MH', 'TI']
```

The other traversal functions are `bfs_order`, `bfs_edges`, `bfs_tree`, `dfs_preorder`, `dfs_postorder`, `dfs_edges`, `dfs_tree`, and `ancestors` / `descendants` for directed graphs. The breadth-first tree above is drawn as a tidy tree on the [Layouts](layouts.md) page.

## Shortest paths

{py:func}`aryagraph.algorithms.paths.shortest_path` picks the method from the weights: breadth-first search when `weight=None`, Dijkstra when every weight is non-negative, and Bellman–Ford otherwise. You can also force one with `method="bfs"`, `"dijkstra"` or `"bellman-ford"`.

```python
route = ag.alg.shortest_path(campus, "OO", "SH", weight="length")
print(" → ".join(route))
meters = ag.alg.shortest_path_length(campus, "OO", "SH", weight="length")
print(round(meters, 1), "m")
print(ag.alg.shortest_path_length(campus, "OO", "SH"), "links")
```

```text
OO → KNB → KNA → IH → RC → RT → CH → MFH → PF → EDT → SH
1001.8 m
10 links
```

With `target=None`, `shortest_path` returns a path to every reachable node, and `shortest_path_length` a `NodeMap` of distances. `all_shortest_paths` yields every tied shortest path, and `has_path` checks reachability.

### Several routes: *k* shortest paths

{py:func}`aryagraph.algorithms.paths.k_shortest_paths` (Yen's algorithm) returns the *k* shortest loopless paths, shortest first. It shows how much a detour costs:

```python
def walk(path):
    return sum(campus.edges[u, v]["length"] for u, v in zip(path, path[1:]))

for path in ag.alg.k_shortest_paths(campus, "OO", "SH", 3, weight="length"):
    print(f"{walk(path):7.1f} m  {len(path)} buildings  via {path[2]}")
```

```text
 1001.8 m  11 buildings  via KNA
 1060.5 m  12 buildings  via KNA
 1073.0 m  11 buildings  via MSC
```

### Hiding edges with a callable weight

A callable weight can return `None` to leave an edge out. This route between the MacEwan Student Centre and the Taylor Family Digital Library stays indoors, using only tunnels, pedways and attached buildings:

```python
def indoors(u, v, d):
    return None if d["kind"] == "outdoor" else d["length"]

inside = ag.alg.shortest_path(campus, "MSC", "TFDL", weight=indoors)
print(" → ".join(inside))
best = ag.alg.shortest_path_length(campus, "MSC", "TFDL", weight="length")
print(round(walk(inside), 1), "m indoors, against", round(best, 1), "m by any route")
```

```text
MSC → MH → SB → SA → SS → AD → PF → MFH → MT → HNSC → TFDL
894.7 m indoors, against 670.0 m by any route
```

```{figure} ../_static/generated/guide_layout_algorithms/campus_indoor_route.png
:alt: Campus map with the indoor route from MSC through MH, SB, SA, SS, AD, PF, MFH, MT and HNSC to TFDL highlighted

The 894.7 m indoor route from MSC to TFDL; the other buildings are dimmed. Positions © OpenStreetMap contributors (ODbL).
```

### A* and bounded searches

{py:func}`aryagraph.algorithms.paths.astar_path` takes a `heuristic(node, target)` that estimates the remaining distance. On the campus graph, edge lengths match the straight-line distances between building positions to within 0.3% (the first line below), so the straight-line distance is a close estimate:

```python
import math

pos = {n: d["pos"] for n, d in campus.nodes.data()}

def straight_line(a, b):
    return math.dist(pos[a], pos[b])

ratios = [d["length"] / straight_line(u, v) for u, v, d in campus.edges.data()]
print(round(min(ratios), 4), round(max(ratios), 4))

found = ag.alg.astar_path(campus, "OO", "SH", heuristic=straight_line, weight="length")
print(found == route)
```

```text
0.9986 1.0026
True
```

{py:func}`aryagraph.algorithms.paths.dijkstra` returns distances and *all* shortest-path predecessors, and accepts a `cutoff`. These are the buildings within 300 m of MSC:

```python
dist, pred = ag.alg.dijkstra(campus, "MSC", weight="length", cutoff=300)
print({n: round(d) for n, d in dist.items()})
```

```text
{'MSC': 0, 'MH': 97, 'TI': 108, 'KNB': 146, 'ENF': 206, 'SB': 219, 'KNA': 260, 'ENE': 267, 'CSSH': 290, 'CR': 300}
```

### Negative weights and negative cycles

Dijkstra refuses negative weights with a `NegativeWeightError`; `shortest_path` switches to Bellman–Ford when it sees one. When a negative cycle makes distances unbounded, Bellman–Ford raises {py:class}`aryagraph.core.exceptions.NegativeCycleError`, and `err.cycle` holds the cycle itself:

```python
signed = ag.DiGraph()
signed.add_edge("a", "b", weight=4)
signed.add_edge("a", "c", weight=1)
signed.add_edge("c", "b", weight=-2)
signed.add_edge("b", "d", weight=1)
print(ag.alg.shortest_path(signed, "a", "d", weight="weight"),
      ag.alg.shortest_path_length(signed, "a", "d", weight="weight"))

signed.add_edge("b", "c", weight=1)     # c → b → c now weighs -1
try:
    ag.alg.bellman_ford(signed, "a")
except ag.NegativeCycleError as err:
    print(err)
    print(err.cycle)
```

```text
['a', 'c', 'b', 'd'] 0
negative cycle 'b' → 'c' → 'b' (total weight -1)
['b', 'c', 'b']
```

On an undirected graph, a single negative edge is already a negative cycle, because it can be walked back and forth.

### All pairs

`floyd_warshall` returns a numpy distance matrix and its node order; `all_pairs_shortest_path_length` returns `{u: NodeMap}` and runs breadth-first search, Dijkstra, or Johnson's reweighting when some weight is negative.

```python
D, order = ag.alg.floyd_warshall(campus, weight="length")
print(D.shape, round(D.max(), 1), "m between the two farthest buildings")
```

```text
(56, 56) 2153.1 m between the two farthest buildings
```

## Connectivity

```python
print(ag.alg.is_connected(campus))
print("cut buildings:", ag.alg.articulation_points(campus))
print("bridges:", ag.alg.bridges(campus))

indoor_net = ag.gen.ucalgary_campus(indoor_only=True)
parts = ag.alg.connected_components(indoor_net)
print(len(parts), "indoor components; the largest has", len(parts[0]), "buildings")
```

```text
True
cut buildings: ['CR', 'EDT', 'HP', 'MTH', 'PF', 'PP']
bridges: [('CDC', 'PP'), ('CR', 'GR'), ('HP', 'PP'), ('MTH', 'OVC')]
19 indoor components; the largest has 38 buildings
```

Removing an articulation point, or a bridge, disconnects the graph. `biconnected_components` returns the pieces that survive the loss of any single building. Components are listed largest first.

For directed graphs, use `strongly_connected_components`, `weakly_connected_components` and `condensation`, which contracts each strongly connected component into one node of a DAG:

```python
site = ag.DiGraph([
    ("home", "about"), ("about", "team"), ("team", "home"),
    ("home", "blog"), ("blog", "post"), ("post", "blog"), ("post", "contact"),
])
print([sorted(c) for c in ag.alg.strongly_connected_components(site)])
dag = ag.alg.condensation(site)
print(dag, list(dag.edges))
```

```text
[['about', 'home', 'team'], ['blog', 'post'], ['contact']]
<DAG: 3 nodes, 2 edges> [(0, 1), (1, 2)]
```

## DAGs and the critical path

The DAG functions cover ordering, longest paths, reachability and scheduling. {py:func}`aryagraph.algorithms.dag.critical_path` runs the critical-path method with durations on the nodes and returns earliest and latest start and finish times and the slack of every task:

```python
plan = ag.gen.project_plan()            # 18 tasks, durations in working hours
cp = ag.alg.critical_path(plan)
print(cp.length, "h;", len(cp.path), "tasks on the critical path")
print("slack:", {task: s for task, s in cp.slack.items() if s > 0})
```

```text
572.0 h; 12 tasks on the critical path
slack: {'windows & doors': 16.0, 'HVAC install': 8.0, 'cabinets & fixtures': 8.0, 'landscaping': 88.0}
```

The plan's arcs also carry a `lag` (mandatory waits such as concrete curing), which `critical_path` does not read. The [DAG workflows](dags.md) page shows how to include those waits as tasks.

```python
print("width:", ag.alg.dag_width(plan), sorted(ag.alg.maximum_antichain(plan)))
lca = ag.alg.lowest_common_ancestors(plan, "plumbing rough-in", "electrical rough-in")
print("common ancestor:", lca)

etl = ag.gen.data_warehouse_etl()
reduced = ag.alg.transitive_reduction(etl)
implied = etl.num_edges - reduced.num_edges
print(implied, "of", etl.num_edges, "ETL dependencies are implied by others")
```

```text
width: 5 ['HVAC install', 'electrical rough-in', 'landscaping', 'plumbing rough-in', 'windows & doors']
common ancestor: {'framing'}
10 of 41 ETL dependencies are implied by others
```

The width is the size of the largest set of tasks in which no task depends on another (a maximum antichain, related to chain covers by Dilworth's theorem): with enough crews, all of them could run at once. Other functions: `topological_sort` (with an optional sort `key`), `all_topological_sorts`, `dag_longest_path`, `dag_levels`, `transitive_closure`, `minimum_chain_partition`, and `is_dag`, `find_cycle` and `simple_cycles` for any directed graph. The [DAG workflows](dags.md) page covers the `DAG` class, scheduling and Monte Carlo planning.

## Centrality

| function | measures |
|---|---|
| `degree_centrality`, `in_degree_centrality`, `out_degree_centrality` | share of possible neighbors |
| `closeness_centrality`, `harmonic_centrality` | how near a node is to all others |
| `betweenness_centrality`, `edge_betweenness_centrality` | how many shortest paths pass through a node or edge |
| `eigenvector_centrality`, `katz_centrality`, `pagerank` | importance inherited from important neighbors |
| `hits` | hub and authority scores of a directed graph |
| `centralities` | several of the above in one call |

Betweenness with `weight="length"` measures which buildings lie on the most shortest routes, with link lengths in meters as distances. Counting hops instead gives a different ranking:

```python
by_meters = ag.alg.betweenness_centrality(campus, weight="length")
by_hops = ag.alg.betweenness_centrality(campus)
print([n for n, _ in by_meters.top(5)])
print([n for n, _ in by_hops.top(5)])

fig = ag.draw(campus, layout=pos,
              node_size=ag.by(by_meters, title="betweenness"),
              node_color=ag.by(by_meters, title="betweenness"),
              title="Betweenness by link length",
              subtitle="Positions © OpenStreetMap contributors (ODbL)",
              width=760)
fig.save("campus_betweenness.svg")
```

```text
['PF', 'KNB', 'IH', 'RC', 'MFH']
['KNB', 'PF', 'SB', 'SS', 'MH']
```

```{figure} ../_static/generated/guide_layout_algorithms/campus_betweenness.png
:alt: Campus map with node size and color showing betweenness centrality by link length; PF and KNB are the largest

Betweenness by link length: larger, darker buildings lie on more shortest routes. Positions © OpenStreetMap contributors (ODbL).
```

Exact betweenness costs one search per node. For large graphs, `k=` samples that many source nodes; pass a `seed` to make the estimate reproducible. With 20 of the 56 buildings as sources, the two leaders stay the same but the rest of the top five changes:

```python
approx = ag.alg.betweenness_centrality(campus, weight="length", k=20, seed=1)
print([n for n, _ in approx.top(5)])
```

```text
['PF', 'KNB', 'EDT', 'SS', 'AD']
```

PageRank treats weights as strengths. In Les Misérables, edge weights count shared chapters, and using them changes who ranks second:

```python
lesmis = ag.gen.les_miserables()
for weight in ("weight", None):
    top = ag.alg.pagerank(lesmis, weight=weight).top(3)
    print(f"weight={weight!r:9}", [(name, round(score, 4)) for name, score in top])

# degree, betweenness, closeness, pagerank and eigenvector in one call
scores = ag.alg.centralities(lesmis)
print({kind: nm.argmax() for kind, nm in scores.items()})
```

```text
weight='weight'  [('Valjean', 0.0996), ('Marius', 0.0517), ('Myriel', 0.0393)]
weight=None      [('Valjean', 0.0754), ('Myriel', 0.0428), ('Gavroche', 0.0358)]
{'degree': 'Valjean', 'betweenness': 'Valjean', 'closeness': 'Valjean', 'pagerank': 'Valjean', 'eigenvector': 'Gavroche'}
```

## Structure

`ag.alg.summary` collects cheap headline statistics and does not raise errors. The other structural measures cover distances, clustering, mixing and cores:

```python
summary = ag.alg.summary(campus)
print({k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()})
meters = ag.alg.diameter(campus, weight="length")
print("diameter:", ag.alg.diameter(campus), "links,", round(meters, 1), "m")
print("center:", ag.alg.center(campus), "periphery:", ag.alg.periphery(campus))
print("transitivity:", round(ag.alg.transitivity(campus), 3))
print("same-kind mixing:", round(ag.alg.attribute_assortativity(campus, "kind"), 3))
```

```text
{'n': 56, 'm': 83, 'directed': False, 'density': 0.054, 'self_loops': 0, 'avg_degree': 2.964, 'max_degree': 5, 'isolates': 0, 'components': 1, 'largest_component': 56, 'avg_clustering': 0.322, 'assortativity': -0.171}
diameter: 15 links, 2153.1 m
center: ['IH', 'MH', 'SB'] periphery: ['CC', 'CDC', 'OVC']
transitivity: 0.293
same-kind mixing: 0.447
```

An attribute assortativity of 0.447 means buildings link to buildings of the same kind (residence to residence, academic to academic) more often than chance would give. Also available: degree histograms and distributions, triangles and square clustering, eccentricity and radius, average shortest path length, Wiener index, global and local efficiency, `core_number`, `k_core`, `onion_layers`, the rich-club coefficient, and `is_tree`, `is_forest`, `is_regular`.

## Communities

Community detection returns a list of node sets, largest first. {py:func}`aryagraph.algorithms.community.louvain_communities` optimizes modularity with node aggregation and a refinement step; `seed` fixes the node order it visits.

```python
communities = ag.alg.louvain_communities(lesmis, seed=0)
print(len(communities), [len(c) for c in communities])
print(round(ag.alg.modularity(lesmis, communities), 4))
print(ag.alg.partition_quality(lesmis, communities))

label = ag.alg.community_labels(communities)      # NodeMap: node -> community index
print(label["Valjean"], label["Javert"], label["Myriel"])

fig = ag.draw(lesmis,
              node_color=ag.by(label, kind="categorical", title="community"),
              node_size=lesmis.degree(weight="weight"),
              edge_width="weight",
              title="Les Misérables: Louvain communities",
              subtitle="node size: weighted degree · edge width: chapters shared")
fig.save("lesmis_communities.html")
```

```text
6 [22, 17, 11, 11, 10, 6]
0.5667
{'coverage': 0.7637795275590551, 'performance': 0.8622693096377307, 'modularity': 0.566687983343248}
0 0 4
```

<iframe class="ag-embed" src="../_static/generated/guide_layout_algorithms/lesmis_communities.html" height="800" loading="lazy" title="Interactive Les Miserables network colored by its six Louvain communities, node size by weighted degree"></iframe>
<p class="ag-embed-note">The six Louvain communities of Les Misérables, node size by weighted degree. Click a legend entry to show one community, or hover a character to see their neighbors. <a href="../_static/generated/guide_layout_algorithms/lesmis_communities.html">Open full screen</a></p>

Other methods find other partitions. On this graph, none of them reaches the modularity of the Louvain partition (0.5667):

```python
methods = {
    "greedy modularity": ag.alg.greedy_modularity_communities(lesmis),
    "label propagation": ag.alg.label_propagation_communities(lesmis, seed=0),
    "fluid (k=6)": ag.alg.asyn_fluid_communities(lesmis, 6, seed=0),
}
for name, parts in methods.items():
    q = ag.alg.modularity(lesmis, parts)
    print(f"{name:18} {len(parts):2} communities  modularity {q:.4f}")

levels = ag.alg.louvain_hierarchy(lesmis, seed=0)
print([len(p) for p in levels])
```

```text
greedy modularity   5 communities  modularity 0.4729
label propagation  10 communities  modularity 0.4703
fluid (k=6)         6 communities  modularity 0.5268
[10, 6]
```

`louvain_hierarchy` returns the partition after every aggregation level, finest first. `girvan_newman` yields ever finer partitions by removing the edge of highest betweenness.

## Flow and cuts

{py:func}`aryagraph.algorithms.flow.maximum_flow` uses Dinic's algorithm and reads capacities from an edge attribute (default `"capacity"`) or a callable. {py:func}`aryagraph.algorithms.flow.minimum_cut` returns the cut value, both sides and the edges that cross:

```python
net = ag.DiGraph()
pipes = [("s", "a", 10), ("s", "b", 5), ("a", "b", 15), ("a", "t", 10), ("b", "t", 10)]
for u, v, c in pipes:
    net.add_edge(u, v, capacity=c)

flow = ag.alg.maximum_flow(net, "s", "t")
print(flow.value, flow.flow["s"])
print(ag.alg.minimum_cut(net, "s", "t").cut_edges)
```

```text
15 {'a': 10, 'b': 5}
[('s', 'a'), ('s', 'b')]
```

With a capacity of 1 on every edge, the maximum flow counts the paths between two nodes that share no edge. Between the Olympic Oval and Scurfield Hall there are two, and the minimum cut names the two links that separate the Oval's corner of campus from the rest:

```python
cut = ag.alg.minimum_cut(campus, "OO", "SH", capacity=lambda u, v, d: 1)
print(cut.value, cut.cut_edges)
print(len(cut.source_side), "buildings on the Oval side")
```

```text
2 [('KA', 'AU'), ('OO', 'KNB')]
9 buildings on the Oval side
```

```{figure} ../_static/generated/guide_layout_algorithms/campus_mincut.png
:alt: Campus map split into two colors by the minimum cut between OO and SH, with the two cut links highlighted

The minimum cut between OO and SH: two links (red) separate the Olympic Oval, five residence halls and three other buildings from the rest of campus. Positions © OpenStreetMap contributors (ODbL).
```

## Spanning trees

`minimum_spanning_tree` and `maximum_spanning_tree` (Kruskal by default, `algorithm="prim"` also available) return a new graph with the original attributes:

```python
mst = ag.alg.minimum_spanning_tree(campus, weight="length")
print(mst)
print(round(sum(d["length"] for _, _, d in mst.edges.data()), 1), "m of",
      round(sum(d["length"] for _, _, d in campus.edges.data()), 1), "m")
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 55 edges>
5572.3 m of 8855.2 m
```

```{figure} ../_static/generated/guide_layout_algorithms/campus_mst.png
:alt: Minimum spanning tree of the campus drawn on the real building positions

The shortest set of links that still connects all 56 buildings. Positions © OpenStreetMap contributors (ODbL).
```

## Matching

- `hopcroft_karp` (alias `bipartite_maximum_matching`): maximum matching of a bipartite graph, returned as a dict in both directions;
- `max_weight_matching` and `min_weight_matching`: Edmonds' blossom algorithm on general graphs;
- `maximal_matching`: a fast greedy matching that cannot be extended;
- `is_matching`, `is_maximal_matching`, `is_perfect_matching`: checks.

```python
davis = ag.gen.davis_southern_women()
women = [n for n, d in davis.nodes.data() if d["bipartite"] == 0]
match = ag.alg.hopcroft_karp(davis, top_nodes=women)
print(len(match) // 2, "women matched to distinct events")

pairs = ag.alg.min_weight_matching(campus, weight="length")
print(len(pairs), "pairs,", ag.alg.is_perfect_matching(campus, pairs),
      round(sum(campus.edges[u, v]["length"] for u, v in pairs), 1), "m")
print(len(ag.alg.maximal_matching(campus)), "pairs in a greedy maximal matching")
```

```text
14 women matched to distinct events
28 pairs, True 3289.0 m
23 pairs in a greedy maximal matching
```

Each of the 14 events is matched to a different woman. On the campus, all 56 buildings can be paired with a neighbor (`is_perfect_matching` returns `True`), and the lightest such pairing uses 3,289 m of links. The greedy maximal matching stops at 23 pairs: no edge can be added to it, but it is not a maximum matching.

## Coloring

{py:func}`aryagraph.algorithms.coloring.greedy_color` gives each node the smallest color not used by its neighbors. The order in which nodes are visited matters, and seven strategies are available:

```python
g = ag.gen.gnm_random_graph(100, 400, seed=1)
for strategy in ("largest_first", "smallest_last", "saturation_largest_first",
                 "connected_sequential_bfs", "random_sequential"):
    colors = ag.alg.greedy_color(g, strategy=strategy, seed=0)
    print(f"{strategy:26} {max(colors.values()) + 1} colors")

sides = ag.alg.bipartite_sets(davis)
print(ag.alg.is_bipartite(davis), [len(side) for side in sides])
```

```text
largest_first              6 colors
smallest_last              5 colors
saturation_largest_first   5 colors
connected_sequential_bfs   6 colors
random_sequential          7 colors
True [18, 14]
```

The two remaining strategies are `"independent_set"` and `"connected_sequential_dfs"`; a callable that returns a node order also works.

## Link prediction

Link-prediction scores rate pairs of non-adjacent nodes by their neighborhoods. `predict_links` returns the highest-scoring missing links as `(u, v, score)`:

```python
for u, v, score in ag.alg.predict_links(lesmis, method="adamic_adar", k=3):
    print(f"{u} – {v}: {score:.2f}")

print(ag.alg.jaccard_coefficient(lesmis, [("Cosette", "Fantine")]))
print(ag.alg.common_neighbors(lesmis, "Cosette", "Fantine"))
```

```text
Eponine – Gavroche: 3.34
Gavroche – Claquesous: 3.20
Marius – Prouvaire: 3.11
[('Cosette', 'Fantine', 0.23809523809523808)]
['Valjean', 'Tholomyes', 'MmeThenardier', 'Thenardier', 'Javert']
```

`jaccard_coefficient`, `adamic_adar_index`, `resource_allocation_index` and `preferential_attachment` score the pairs you pass, or every non-adjacent pair when `pairs` is omitted. `common_neighbors(g, u, v)` lists the shared neighbors of one pair. `predict_links` takes `method="adamic_adar"` (default), `"resource_allocation"`, `"jaccard"`, `"common_neighbors"` or `"preferential_attachment"`.

## Matrices and spectra

Matrix functions return dense numpy arrays, in graph node order unless you pass `nodes=`:

```python
import numpy as np

A = ag.alg.adjacency_matrix(campus, weight=None)
L = ag.alg.laplacian_matrix(campus)
B = ag.alg.incidence_matrix(campus)
print(A.shape, int(A.sum()), B.shape)
print(np.round(ag.alg.laplacian_spectrum(campus)[:3], 4) + 0.0)
print(round(ag.alg.algebraic_connectivity(campus), 4),
      ag.alg.algebraic_connectivity(indoor_net))
```

```text
(56, 56) 166 (56, 83)
[0.     0.0477 0.0661]
0.0477 0.0
```

The algebraic connectivity (second-smallest Laplacian eigenvalue) is zero for a disconnected graph, such as the indoor network, and small for a connected graph with bottlenecks. `adjacency_spectrum`, `degree_vector` and `normalized=True` Laplacians are also available; the spectral layout on the [Layouts](layouts.md) page uses the same eigenvectors.

## Working with results

`NodeMap` and `EdgeMap` are dictionaries, so anything that accepts a dict accepts them, including every visual channel of {py:func}`aryagraph.render.draw` (for example `node_size=bc`). They add:

```python
print(bc.top(3))                     # highest first
print(bc.bottom(2))
print(bc.argmax(), bc.rank()["MSC"])  # dense rank, 1 = highest
print({k: round(v, 4) for k, v in bc.describe().items()})
print(bc.normalized().top(2))         # "max" (default), "sum", "minmax" or "zscore"
print(bc.to_array(["MSC", "TFDL"]))   # aligned with any node order
print(bc.filter(lambda n, v: v > 0.3))
```

```text
[('PF', 0.3070707070707071), ('KNB', 0.301010101010101), ('IH', 0.2175084175084175)]
[('AB', 0.0), ('BI', 0.0)]
PF 22
{'count': 56, 'mean': 0.0937, 'std': 0.0822, 'min': 0.0, '25%': 0.019, '50%': 0.0741, '75%': 0.151, 'max': 0.3071}
[('PF', 1.0), ('KNB', 0.9802631578947367)]
[0.10841751 0.03097643]
NodeMap('betweenness_centrality', 2 nodes; top: {'PF': 0.3071, 'KNB': 0.301})
```

`map(fn)` transforms values and `to_pandas()` returns a pandas Series (pandas is optional).

## Errors

Every error AryaGraph raises on purpose derives from {py:class}`aryagraph.core.exceptions.AryaGraphError`, and many also inherit from the matching builtin (`KeyError` for a missing node, `ValueError` for a cycle, `TypeError` for the wrong kind of graph). Errors carry their evidence:

| error | raised when | carries |
|---|---|---|
| `NodeNotFound` | a node is not in the graph | `.node` |
| `NoPath` | the target cannot be reached | `.source`, `.target` |
| `NotConnected` | a measure needs a connected graph (diameter, radius) | |
| `GraphTypeError` | the algorithm needs a directed or undirected graph | |
| `CycleError` | an operation needs an acyclic graph | `.cycle` |
| `NegativeWeightError`, `NegativeCycleError` | negative weights where they are not allowed | `.edge`, `.cycle` |
| `ConvergenceError` | an iterative method ran out of iterations | |
| `UnboundedFlowError` | an infinite-capacity path joins source and sink | |

```python
for attempt in (
    lambda: ag.alg.shortest_path(indoor_net, "OO", "GR"),
    lambda: ag.alg.diameter(indoor_net),
    lambda: ag.alg.strongly_connected_components(campus),
    lambda: ag.alg.dijkstra(signed, "a"),
    lambda: ag.DAG([("a", "b"), ("b", "c")]).add_edge("c", "a"),
):
    try:
        attempt()
    except ag.AryaGraphError as err:
        print(f"{type(err).__name__}: {err}")
```

```text
NoPath: no path from 'OO' to 'GR'
NotConnected: eccentricity is undefined: the graph is not connected (some distances are infinite)
GraphTypeError: strongly_connected_components requires a directed graph
NegativeWeightError: negative weight -2 on edge ('c', 'b'); use bellman_ford
CycleError: edge 'c' → 'a' would close the cycle 'c' → 'a' → 'b' → 'c'
```

To apply many of these measures at once and get a report, see [Analysis reports](analysis.md).
