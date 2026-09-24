# Graphs

AryaGraph stores networks in three classes: {py:class}`~aryagraph.core.graph.Graph` for undirected graphs, {py:class}`~aryagraph.core.graph.DiGraph` for directed graphs and {py:class}`~aryagraph.core.dag.DAG` for directed graphs that must stay acyclic. This page shows how to build them, read and edit their contents, derive new graphs from them and handle the errors they raise.

## The three graph classes

| Class | Edges | Self-loops | Typical use |
|---|---|---|---|
| `ag.Graph` | undirected, at most one edge per pair | allowed | social, road and campus networks |
| `ag.DiGraph` | directed, at most one arc per ordered pair | allowed | citations, web links, flows |
| `ag.DAG` | directed, acyclic by construction | rejected | pipelines, build systems, schedules, prerequisites |

`DAG` is a subclass of `DiGraph`, which is a subclass of `Graph`, so a `DAG` has every method of a `DiGraph` and a `DiGraph` every method of a `Graph`. Nodes can be any hashable value except `None`: strings, integers, tuples or your own objects.

Three properties hold for all of them:

- **Insertion order.** Nodes iterate in the order they were added, and so do each node's neighbors. Algorithms, layouts and drawings inherit this order, so the same input produces the same output.
- **One attribute dict per edge.** An undirected edge has a single attribute dict that you reach from either endpoint.
- **Constant-time sizes.** `len(g)`, `g.num_nodes` and `g.num_edges` are stored counters.

## Building a graph

The constructor accepts the common ways graph data arrives.

### Edge lists

A list of `(u, v)` pairs creates the endpoints as needed:

```python
import aryagraph as ag

g = ag.Graph([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
print(g)             # <Graph: 4 nodes, 4 edges>
print(list(g.edges)) # [('a', 'b'), ('a', 'c'), ('b', 'c'), ('c', 'd')]
```

A number as the third element becomes the edge's `weight` attribute, which is the attribute weighted algorithms read by default:

```python
roads = ag.Graph([("A", "B", 4.0), ("B", "C", 2.5), ("A", "C", 7.0)])
print(roads.edges["B", "C"])   # {'weight': 2.5}
```

A dict as the third element sets any attributes you like:

```python
links = ag.Graph([
    ("MSC", "MH", {"kind": "pedway", "length": 95.0}),
    ("MH", "TI", {"kind": "attached", "length": 40.0}),
])
print(links.edges["MH", "TI"])   # {'kind': 'attached', 'length': 40.0}
```

### Adjacency dicts

A mapping from each node to its neighbors works too, either as lists or as `{neighbor: attrs}` dicts:

```python
steps = ag.DiGraph({"fetch": ["build", "lint"], "build": ["test"], "lint": ["test"]})
print(steps)   # <DiGraph: 4 nodes, 4 edges>

timed = ag.DiGraph({"fetch": {"build": {"minutes": 3}}, "build": {"test": {"minutes": 8}}})
print(timed.edges["build", "test"])   # {'minutes': 8}
```

### Node order, names and graph attributes

Pass `nodes=` to fix the order in which nodes are stored, laid out and listed; those nodes are added before the edges. `name=` and any extra keyword arguments become graph-level attributes in `g.attrs`:

```python
ordered = ag.Graph([("b", "c"), ("a", "b")], nodes=["a", "b", "c"], name="ordered", source="survey")
print(list(ordered.nodes))   # ['a', 'b', 'c']
print(ordered.attrs)         # {'source': 'survey', 'name': 'ordered'}
```

`nodes=` items can also be `(node, attrs)` pairs, which is how you attach attributes to nodes that have no edges yet.

### networkx graphs

A networkx `Graph` or `DiGraph` can be passed straight to the constructor. Node, edge and graph attributes are copied, and networkx itself is not imported by AryaGraph for this. Multigraphs raise a `TypeError`, because AryaGraph graphs hold at most one edge per pair:

```python
import networkx as nx

club = ag.Graph(nx.karate_club_graph())
print(club)            # <Graph "Zachary's Karate Club": 34 nodes, 78 edges>
print(club.nodes[0])   # {'club': 'Mr. Hi'}
```

The [File formats and interoperability](io.md) page covers the converters for networkx, pandas, numpy and SciPy, and readers for JSON, CSV, GraphML, GEXF, DOT and Mermaid.

### Growing a graph step by step

`add_node` and `add_edge` create an item if it is missing and update its attributes if it exists. Their plural forms take iterables and accept keyword attributes that apply to every item, with per-item values taking precedence:

```python
g = ag.Graph(name="demo")
g.add_node("hub", role="center")
g.add_nodes(["x", ("y", {"role": "leaf"})], role="leaf")
g.add_edge("hub", "x", weight=2.0)
g.add_edges([("hub", "y"), ("x", "y", {"weight": 5.0})], weight=1.0)
print(g.edges(data=True))
# [('hub', 'x', {'weight': 2.0}), ('hub', 'y', {'weight': 1.0}), ('x', 'y', {'weight': 5.0})]
```

`remove_node` (which also drops the incident edges), `remove_edge`, their plural forms and `clear()` complete the set.

## Nodes, edges and neighbors

The examples below use the University of Calgary campus network from the [datasets](datasets.md): 56 buildings joined by 83 walking links, with building codes as node ids.

```python
campus = ag.gen.ucalgary_campus()
print(campus)                  # <Graph 'University of Calgary main campus': 56 nodes, 83 edges>
print(campus.nodes["TFDL"]["name"], campus.nodes["TFDL"]["kind"])
# Taylor Family Digital Library library
print(campus.edges["TFDL", "MT"])
# {'kind': 'outdoor', 'length': 76.7, 'weight': 76.7}
```

The graph exposes live, read-only views:

| View | What it is |
|---|---|
| `g.nodes` | iterate nodes; `g.nodes[n]` is the node's attribute dict; `n in g.nodes` tests membership |
| `g.edges` | iterate edges as `(u, v)`; `g.edges[u, v]` is the edge's attribute dict |
| `g.adj`, `g.succ`, `g.pred` | `{node: {neighbor: attrs}}` mappings; for undirected graphs all three are the same |
| `g[n]` | the neighbor mapping of `n` (successors for directed graphs) |

For undirected graphs each edge is listed once, and lookups work in both orientations; both orientations return the same dict:

```python
print(("MT", "TFDL") in campus.edges)                             # True
print(campus.edges["MT", "TFDL"] is campus.edges["TFDL", "MT"])   # True
print(list(campus.neighbors("TFDL")))                             # ['CH', 'HNSC', 'MT']
```

`data()` pairs every item with its attributes, or with one attribute when you name it. Calling the view with `data=` gives the same result as a list, in the style networkx users know:

```python
print(list(campus.nodes.data("kind"))[:3])
# [('AB', 'arts'), ('AD', 'administration'), ('AU', 'residence')]
lengths = {(u, v): m for u, v, m in campus.edges.data("length")}
print(len(lengths), campus.edges(data="kind")[0])   # 83 ('AB', 'CH', 'pedway')
```

Directed graphs add `successors`, `predecessors`, `all_neighbors`, `in_edges` and `out_edges`; for a `DiGraph`, `neighbors(n)` returns the successors.

## Attributes

Attribute dicts are ordinary Python dicts owned by the graph, so you edit them in place:

```python
campus.nodes["TFDL"]["floors"] = 5
campus.edges["TFDL", "MT"]["covered"] = False
campus.attrs["season"] = "winter"
print(campus.nodes["TFDL"]["floors"], campus.edges["MT", "TFDL"]["covered"])   # 5 False
```

Two methods read edge attributes: `get_edge_data(u, v, default)` returns `default` for a missing edge, and `edge_attrs(u, v)` raises {py:class}`~aryagraph.core.exceptions.EdgeNotFound`.

Views do not let you add or remove nodes and edges; structural changes go through the graph's methods, which keep the adjacency and the counters consistent.

## Degrees and result maps

`degree(n)` returns one node's degree; `degree()` returns a {py:class}`~aryagraph.core.results.NodeMap` for every node. With `weight=`, degrees sum that edge attribute instead of counting edges (a missing value counts as 1), and a self-loop contributes 2 to its node's degree.

```python
print(campus.degree("TFDL"), campus.degree("TFDL", weight="length"))   # 3 247.6

deg = campus.degree()
print(deg.top(3))                  # [('ES', 5), ('CD', 4), ('CH', 4)]
print(deg.describe()["mean"])      # 2.9642857142857144
```

A `NodeMap` is a `dict` subclass, so anything that takes a dict takes it, including every channel of [`draw()`](drawing.md). It adds `top(k)`, `bottom(k)`, `rank()`, `argmax()`, `normalized()`, `describe()`, `to_array()` and `to_pandas()`. Every per-node algorithm result in `ag.alg` is a `NodeMap`; per-edge results are `EdgeMap`s with the same helpers. Directed graphs also provide `in_degree()` and `out_degree()`.

## Derived graphs

Methods that derive a graph return a new object and leave the original untouched. Attribute dicts are copied, so editing the new graph does not change the old one (the attribute values themselves are shared, as with `dict.copy()`).

| Method | Result |
|---|---|
| `subgraph(nodes)` | the nodes and every edge among them |
| `edge_subgraph(edges)` | the given edges and their endpoints |
| `copy()` | same nodes, edges and attributes |
| `relabel(mapping)` | nodes renamed by a dict or a function; unmapped nodes keep their names |
| `compose(other)` | union of both graphs; `other`'s attributes win on overlap |
| `to_directed()` | each undirected edge becomes two opposite arcs |
| `to_undirected()` | arcs lose their direction; with `reciprocal=True`, only pairs linked both ways are kept |
| `reverse()` | directed graphs only: every arc flipped |

For example, the campus's indoor network is the set of links that are not outdoor walks:

```python
indoor = campus.edge_subgraph(
    (u, v) for u, v, kind in campus.edges.data("kind") if kind != "outdoor"
)
print(indoor)   # <Graph 'University of Calgary main campus': 38 nodes, 40 edges>

engineering = campus.subgraph(n for n, name in campus.nodes.data("name") if "Engineering" in name)
print(list(engineering.nodes))   # ['ENA', 'ENB', 'ENC', 'END', 'ENE', 'ENF', 'ENG', 'MEB']
```

```python
fig = ag.draw(
    indoor,
    layout={n: campus.nodes[n]["pos"] for n in indoor},
    node_color="#72716b",
    edge_color="kind",
    edge_width=2,
    title="Indoor walking network",
    subtitle=f"{indoor.num_nodes} buildings joined by tunnels, pedways and attached walls",
)
fig.save("indoor.svg")
```

```{figure} ../_static/generated/guide_core/graphs_indoor.png
:alt: Map-like drawing of 38 University of Calgary buildings connected by colored lines: blue for attached buildings, orange for pedways and green for tunnels.
:width: 90%

The indoor subgraph drawn on the buildings' real positions, with edges colored by kind. Positions © OpenStreetMap contributors (ODbL).
```

`edge_subgraph` keeps only the endpoints of the chosen edges. The dataset also offers `ag.gen.ucalgary_campus(indoor_only=True)`, which keeps all 56 buildings and the same 40 indoor links, so buildings without an indoor connection remain as isolated nodes.

Relabeling must be one-to-one; a mapping that sends two nodes to the same name raises `ValueError` instead of silently merging them:

```python
named = campus.relabel(lambda code: campus.nodes[code]["name"])
print(list(named.nodes)[:2])   # ['Art Building & Art Parkade', 'Administration Building']

flows = ag.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("a", "c")])
print(list(flows.reverse().edges))                      # [('a', 'c'), ('b', 'a'), ('c', 'a'), ('c', 'b')]
print(list(flows.to_undirected(reciprocal=True).edges)) # [('a', 'c')]
```

## DAGs

A `DAG` checks every change that could close a directed cycle. When a change would create one, it raises {py:class}`~aryagraph.core.exceptions.CycleError`, names the cycle in the message and in the `cycle` attribute, and leaves the graph as it was.

```python
dag = ag.DAG([("extract", "clean"), ("clean", "train"), ("clean", "report")])
print(dag.topological_order())   # ['extract', 'clean', 'train', 'report']

try:
    dag.add_edge("report", "extract")
except ag.CycleError as err:
    print(err)
    print(err.cycle)
# edge 'report' → 'extract' would close the cycle 'report' → 'extract' → 'clean' → 'report'
# ['report', 'extract', 'clean', 'report']

print(dag)   # <DAG: 4 nodes, 3 edges>  (unchanged)
```

Self-loops are cycles of length one and are rejected in the same way.

### Atomic bulk inserts

`add_edges` on a `DAG` is all-or-nothing. It inserts the whole batch, validates the result once with Kahn's algorithm (linear in the size of the graph) and, if a cycle appears, rolls back every node, edge and attribute change the batch made:

```python
try:
    dag.add_edges([("report", "publish"), ("publish", "archive"), ("archive", "clean")])
except ag.CycleError as err:
    print(len(err.cycle) - 1, sorted(err.cycle[:-1]))
# 4 ['archive', 'clean', 'publish', 'report']

print(dag, "publish" in dag)   # <DAG: 4 nodes, 3 edges> False
```

The message reads "these edges would create the cycle …" followed by the four arcs. For a batch, the reported cycle starts at the first of its nodes in graph order, so the same batch gives the same message in every run.

Validating once per batch keeps bulk loading linear, instead of running a reachability search for every arc. Constructing a `DAG` from data uses the same path, so `ag.DAG(edges)` either returns a valid DAG or raises with the offending cycle. A `DiGraph` can test itself with `is_dag()` and convert with `to_dag()`:

```python
loop = ag.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
print(loop.is_dag())   # False
try:
    loop.to_dag()
except ag.CycleError as err:
    print(sorted(set(err.cycle)))   # ['a', 'b', 'c']
```

### Order, levels and generations

The bundled software-build DAG has 20 tasks and 33 dependencies:

```python
build = ag.gen.software_build()
print(build.sources(), build.sinks())   # ['fetch dependencies'] ['publish release']
print(build.topological_order()[:3])    # ['fetch dependencies', 'generate protobuf stubs', 'compile utils']

levels = build.levels()
print(levels["compile core"], levels["publish release"])   # 2 8

stages = build.generations()
print(len(stages), [len(s) for s in stages])   # 9 [1, 6, 1, 4, 1, 3, 2, 1, 1]
print(build.depth())                           # 9
print(len(build.ancestors("link cli")), len(build.descendants("compile core")))   # 8 12
```

- {py:meth}`~aryagraph.core.dag.DAG.topological_order` uses Kahn's algorithm and breaks ties by insertion order, so the order is reproducible. The result is cached until the graph changes.
- {py:meth}`~aryagraph.core.dag.DAG.levels` gives each node its longest-path level: sources are 0 and every arc goes to a higher level.
- {py:meth}`~aryagraph.core.dag.DAG.generations` groups nodes by level. Nodes in one generation have no dependencies on each other, so each generation is a set of tasks that can run in parallel once the previous generations are done.
- `depth()` is the number of generations, which is the number of nodes on the longest path.

Passing the levels to the layered layout as fixed ranks draws each generation in its own column:

```python
fig = ag.draw(
    build,
    layout="hierarchical",
    layout_options={"ranks": levels, "orientation": "LR"},
    node_color="team",
    title="Software build in generations",
    subtitle="Each column is one generation: tasks that can run in parallel",
)
fig.save("build_generations.svg")
```

```{figure} ../_static/generated/guide_core/graphs_generations.png
:alt: Layered left-to-right drawing of a 20-task software build, with boxes colored by team and arranged in nine columns.

The software-build DAG with one column per generation. Nine columns hold 1, 6, 1, 4, 1, 3, 2, 1 and 1 tasks.
```

Longest paths, critical-path scheduling, transitive reduction and closure, and the other DAG algorithms are covered in [DAG workflows](dags.md).

## Errors

Every error AryaGraph raises on purpose derives from {py:class}`~aryagraph.core.exceptions.AryaGraphError`, so `except ag.AryaGraphError` catches the library's failures without hiding unrelated bugs. Where a built-in exception is the natural fit, the AryaGraph error also inherits from it, so existing `except KeyError` or `except ValueError` code keeps working.

| Error | Also a | Raised when | Carries |
|---|---|---|---|
| `NodeNotFound` | `KeyError` | a node is looked up but absent | `.node` |
| `EdgeNotFound` | `KeyError` | an edge is looked up but absent | `.edge` |
| `CycleError` | `ValueError` | a DAG change or an acyclic-only algorithm meets a cycle | `.cycle` |
| `NegativeCycleError` | `ValueError` | a shortest-path search meets a negative cycle | `.cycle` |
| `NegativeWeightError` | `ValueError` | a method that needs non-negative weights meets a negative one | `.edge`, `.weight` |
| `NoPath` | | no path joins the requested nodes | `.source`, `.target` |
| `NotConnected` | `ValueError` | a measure needs a connected graph | |
| `GraphTypeError` | `TypeError` | an operation does not support this kind of graph | |
| `ConvergenceError` | `RuntimeError` | an iterative method runs out of iterations | |
| `UnboundedFlowError` | `ValueError` | a maximum flow is infinite | |
| `DependencyError` | `ImportError` | an optional package is missing | `.package` |

```python
try:
    campus.nodes["XYZ"]
except ag.NodeNotFound as err:
    print(err, "|", err.node)          # node 'XYZ' is not in the graph | XYZ

islands = ag.Graph([("a", "b"), ("c", "d")])
try:
    ag.alg.shortest_path(islands, "a", "d")
except ag.NoPath as err:
    print(err)                         # no path from 'a' to 'd'

try:
    ag.Graph([("a", "b")]).compose(ag.DiGraph([("b", "c")]))
except ag.GraphTypeError as err:
    print(err)                         # cannot compose a directed graph with an undirected one
```

## Next steps

- [Drawing](drawing.md) turns any of these graphs into a figure.
- [Algorithms](algorithms.md) lists the analyses that run on them.
- [DAG workflows](dags.md) covers critical paths, scheduling and the other DAG tools.
