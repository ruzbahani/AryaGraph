# File formats and interoperability

AryaGraph reads and writes seven file formats and converts to and from networkx, pandas, numpy and SciPy. This page shows which format to choose, what each one preserves, and how to move graphs between tools without losing information.

## Reading and writing by extension

{py:func}`ag.read <aryagraph.io.read>` and {py:func}`ag.write <aryagraph.io.write>` pick the format from the file extension; any other keyword argument goes to the format's own reader or writer.

| Extension | Format | Reader / writer |
|---|---|---|
| `.json` | node-link JSON (networkx- and d3-compatible) | {py:func}`~aryagraph.io.jsonio.read_json`, {py:func}`~aryagraph.io.jsonio.write_json` |
| `.csv`, `.tsv` | edge list with a header, comma- or tab-separated | {py:func}`~aryagraph.io.edgelist.read_edgelist`, {py:func}`~aryagraph.io.edgelist.write_edgelist` |
| `.txt`, `.edgelist` | edge list separated by blanks | the same, with whitespace delimiters |
| `.adjlist` | adjacency list: a node followed by its neighbors | {py:func}`~aryagraph.io.edgelist.read_adjacency_list`, {py:func}`~aryagraph.io.edgelist.write_adjacency_list` |
| `.graphml` | GraphML (yEd, Gephi, Cytoscape, networkx) | {py:func}`~aryagraph.io.graphml.read_graphml`, {py:func}`~aryagraph.io.graphml.write_graphml` |
| `.gexf` | GEXF 1.3 (Gephi) | {py:func}`~aryagraph.io.gexf.read_gexf`, {py:func}`~aryagraph.io.gexf.write_gexf` |
| `.dot`, `.gv` | Graphviz DOT | {py:func}`~aryagraph.io.dot.read_dot`, {py:func}`~aryagraph.io.dot.write_dot` |
| `.mmd`, `.mermaid` | Mermaid flowchart | {py:func}`~aryagraph.io.mermaid.read_mermaid`, {py:func}`~aryagraph.io.mermaid.write_mermaid` |

```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
ag.write(campus, "campus.graphml")
same = ag.read("campus.graphml")
print(same)

ag.write(campus, "campus.data", format="json")   # an unusual extension needs format=
print(ag.read("campus.data", format="json"))

try:
    ag.read("campus.data")
except ValueError as err:
    print(err)
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
cannot infer the format of 'campus.data' from its extension; use one of ['.adjlist', '.csv', '.dot', '.edgelist', '.gexf', '.graphml', '.gv', '.json', '.mermaid', '.mmd', '.tsv', '.txt'] or pass format=
```

## What survives a round trip

Formats differ in what they can store. The check below writes the campus graph in every format, reads it back and compares:

```python
import os

def compare(original, back):
    nodes = set(back) == set(original)
    edges = {frozenset(e) for e in back.edges} == {frozenset(e) for e in original.edges}
    node_attrs = nodes and all(back.nodes[n] == original.nodes[n] for n in original)
    edge_attrs = edges and all(back.edges[u, v] == d for u, v, d in original.edges.data())
    return nodes, edges, node_attrs, edge_attrs

print(f"{'format':9} {'bytes':>7}  nodes  edges  node attrs  edge attrs  pos read back as")
for ext in ("json", "graphml", "gexf", "csv", "dot", "mmd", "adjlist"):
    path = f"campus.{ext}"
    ag.write(campus, path)
    back = ag.read(path, node_names="label") if ext == "mmd" else ag.read(path)
    n, e, na, ea = compare(campus, back)
    pos = back.nodes["MSC"].get("pos", "(absent)")
    print(f"{ext:9} {os.path.getsize(path):7,}  {n!s:5}  {e!s:5}  {na!s:10}  {ea!s:10}  {pos!r}")
```

```text
format      bytes  nodes  edges  node attrs  edge attrs  pos read back as
json       24,274  True   True   False       True        [-22.3, -32.7]
graphml    29,731  True   True   False       True        '[-22.3, -32.7]'
gexf       37,873  True   True   True        True        (-22.3, -32.7)
csv         2,192  True   True   False       True        '(absent)'
dot        11,785  True   True   False       True        '-22.3,-32.7'
mmd         1,688  True   True   False       False       '(absent)'
adjlist       486  True   True   False       False       '(absent)'
```

Every format keeps the nodes and edges. The differences are in the attributes:

* **JSON** keeps every attribute, but JSON has no tuples, so the `pos` tuple comes back as a list. Node ids that are tuples are restored as tuples.
* **GraphML** declares typed attributes (boolean, int, long, double, string). Values that fit no GraphML type, such as the `pos` tuple, are written as JSON text and read back as strings.
* **GEXF** maps `pos` onto Gephi's `<viz:position>` (with y negated, because Gephi's y axis points up), and `size`, `color`, `label` and `weight` onto their GEXF fields, so here every attribute survives with its type.
* **CSV** and other edge lists store edges and edge attributes only; node attributes are not written.
* **DOT** keeps attributes as DOT values: numbers stay numbers, and other values become strings such as `"-22.3,-32.7"`.
* **Mermaid** and **adjacency lists** store structure only (plus labels for Mermaid). A node name that is not a safe Mermaid id is written as `n0`, `n1`, … with the name as its label; here the building code `END` (Engineering Block D) collides with Mermaid's `end` keyword, which is why the check reads the file with `node_names="label"` to restore the names.

The table compares node and edge attributes only. The graph's own attributes fare differently:

```python
for ext in ("json", "gexf"):
    print(ext, sorted(ag.read(f"campus.{ext}").attrs))
```

```text
json ['attribution', 'description', 'name', 'sources']
gexf ['name']
```

None of these formats stores every graph without change. JSON comes closest: it keeps every node, edge and graph attribute, and tuples come back as lists. GEXF restored every node and edge attribute of the campus with its type because the only tuple, `pos`, maps onto a Gephi field; other lists and tuples come back as JSON text, and of the graph attributes only the name survives. To archive an AryaGraph graph, use JSON; for exchange with Gephi use GEXF, with yEd or Cytoscape GraphML, with Graphviz DOT, and for documentation pages Mermaid.

Directed graphs and DAGs keep their type in JSON, GraphML and GEXF, which record a DAG with an extra marker that other tools ignore. DOT has no notion of a DAG, so a DOT file reads back as a `DiGraph`; call `.to_dag()` on it, or pass `dag=True` to the edge-list, DOT and XML readers.

```python
ml = ag.gen.ml_pipeline()
for ext in ("json", "graphml", "gexf", "dot", "csv"):
    ag.write(ml, f"ml.{ext}")
    print(ext, type(ag.read(f"ml.{ext}")).__name__)
print(type(ag.read("ml.csv", dag=True)).__name__, type(ag.read("ml.dot").to_dag()).__name__)
```

```text
json DAG
graphml DAG
gexf DAG
dot DiGraph
csv Graph
DAG DAG
```

## JSON node-link

The JSON writer produces the node-link layout that networkx's `node_link_graph` and d3 read directly:

```python
small = ag.Graph(name="demo")
small.add_edge("a", "b", weight=2.5)
small.add_node("c", pos=(1.0, 2.0))
print(ag.io.to_json(small, indent=None))
```

```text
{"directed": false, "dag": false, "multigraph": false, "graph": {"name": "demo"}, "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c", "pos": [1.0, 2.0]}], "edges": [{"source": "a", "target": "b", "weight": 2.5}]}
```

On input, networkx's variant (`links` instead of `edges`) and d3's older index form (nodes without ids, edges pointing at list positions) are accepted as well. {py:func}`~aryagraph.io.jsonio.to_dict` and {py:func}`~aryagraph.io.jsonio.from_dict` work with the same structure as Python objects, which is convenient for web APIs.

```python
d3 = '{"nodes": [{"name": "a"}, {"name": "b"}], "links": [{"source": 0, "target": 1, "value": 3}]}'
g = ag.io.from_json(d3)
print(list(g.nodes.data()), list(g.edges.data()))
```

```text
[(0, {'name': 'a'}), (1, {'name': 'b'})] [(0, 1, {'value': 3})]
```

Values JSON cannot hold are converted on output: sets become sorted lists, numpy scalars and arrays become Python numbers and lists, dates become ISO-8601 strings. NaN and infinity are written as `NaN` and `Infinity` (the dialect Python and networkx use), which strict JSON parsers reject.

## CSV, TSV and edge lists

Edge lists are the lingua franca of spreadsheets and databases: one edge per line, with extra columns as edge attributes.

```python
with open("flights.csv", "w", encoding="utf-8") as fh:
    fh.write("origin,dest,minutes,carrier\n"
             "YYC,YVR,85,WS\n"
             "YVR,YYZ,265,AC\n"
             "YYC,YYZ,230,WS\n")
flights = ag.read("flights.csv", source="origin", target="dest", directed=True)
print(flights, list(flights.edges.data())[0])
```

```text
<DiGraph: 3 nodes, 3 edges> ('YYC', 'YVR', {'minutes': 85, 'carrier': 'WS'})
```

The delimiter is detected from the first lines (comma, tab, semicolon or `|`), and a header is recognized automatically when its fields are text and either one of them is a usual column name (`source`, `target`, `from`, `to`, `weight`, …) or a column that is numeric below it is not numeric in the first line.

Values are typed by one predictable rule. A quoted field is read as a string; an unquoted field becomes an `int` when it is a canonical integer, a `float` when it has a decimal point or exponent, and, in attribute columns, a boolean for `true`/`false`. Anything else stays text, so a code like `007` is not turned into the number 7:

```python
with open("codes.csv", "w", encoding="utf-8") as fh:
    fh.write('source,target,weight\n007,"42",1.5\n42,x,true\n')
codes = ag.read("codes.csv")
print(list(codes.nodes), list(codes.edges.data()))
print(list(ag.read("codes.csv", types={"weight": str}).edges.data()))
```

```text
['007', '42', 42, 'x'] [('007', '42', {'weight': 1.5}), (42, 'x', {'weight': True})]
[('007', '42', {'weight': '1.5'}), (42, 'x', {'weight': 'true'})]
```

The node `"42"` (quoted, a string) and the node `42` (unquoted, an integer) are different nodes. `types=` overrides the rule per column, or for every column with a single callable such as `types=str`. The writers quote the strings that would otherwise be misread (numeric-looking or empty text, and text containing the delimiter, quotes or line breaks), so AryaGraph's own files round-trip; isolated nodes are written as lines with an empty target, which `isolates=False` suppresses for tools that do not expect them.

## GraphML and GEXF

Both are XML formats for exchanging attributed graphs with desktop tools. The GraphML writer chooses the narrowest type that holds every value of an attribute; the reader accepts GraphML from any tool, applies key defaults, flattens nested graphs and skips yFiles graphics keys. The GEXF writer targets Gephi's visual fields:

```python
les = ag.gen.les_miserables()
groups = ag.alg.community_labels(ag.alg.louvain_communities(les, seed=0))
fig = ag.draw(les, node_color=ag.by(groups, kind="categorical"))
for mark in fig.scene.nodes:                      # the drawn position and color of every node
    les.nodes[mark.node]["pos"] = (round(mark.x, 1), round(mark.y, 1))
    les.nodes[mark.node]["color"] = mark.fill
    les.nodes[mark.node]["size"] = round(mark.w / 2, 1)
ag.write(les, "lesmis.gexf")
print(ag.read("lesmis.gexf").nodes["Valjean"])
```

```text
{'color': '#2a78d6', 'pos': (44.3, -11.9), 'size': 7.0}
```

Opened in Gephi, `lesmis.gexf` shows the AryaGraph layout and colors, ready for further editing. Non-string node ids (integers, tuples) are marked with an `aryagraph:type` attribute in both formats so they come back with their type; other tools ignore the marker and see string ids, as they would for any GraphML file.

## Graphviz DOT

The DOT reader implements the grammar published by Graphviz: `graph`/`digraph`, `strict`, edge chains, subgraphs as edge endpoints, attribute defaults with scoping, ports, comments, HTML labels and `+` string concatenation. Subgraphs are flattened; nodes inside a *cluster* (a subgraph whose name starts with `cluster`) get a `cluster` attribute, and each cluster's own attributes are kept in `g.attrs["clusters"]`.

```python
dot = """
digraph pipeline {
  rankdir=LR;
  node [shape=box, team="data"];
  subgraph cluster_ingest { label="Ingest"; events; labels; }
  subgraph cluster_model {
    label="Model";
    node [team="ml"];
    train -> evaluate [artifact="model"];
  }
  events -> clean -> features -> train;   // an edge chain
  labels -> features;
  {train evaluate} -> report;             // a subgraph as an endpoint
}
"""
flow = ag.io.from_dot(dot)
print(flow)
print(flow.nodes["events"], flow.nodes["train"], flow.nodes["report"])
print(flow.edges["train", "evaluate"], flow.has_edge("evaluate", "report"))
print(flow.attrs["clusters"], flow.attrs["rankdir"])
```

```text
<DiGraph 'pipeline': 7 nodes, 7 edges>
{'shape': 'box', 'team': 'data', 'cluster': 'cluster_ingest'} {'shape': 'box', 'team': 'ml', 'cluster': 'cluster_model'} {'shape': 'box', 'team': 'data'}
{'artifact': 'model'} True
{'cluster_ingest': {'label': 'Ingest'}, 'cluster_model': {'label': 'Model'}} LR
```

The chain `events -> clean -> features -> train` produced three edges, `{train evaluate} -> report` two, and the `node [...]` defaults applied to nodes created after them, inside their scope. Syntax errors are reported with their position:

```python
try:
    ag.io.from_dot("digraph { a -> ; }")
except ag.io.DotSyntaxError as err:
    print(err)
```

```text
DOT syntax error at line 1, column 16: expected a node or subgraph after '->' but found ';'
```

{py:func}`~aryagraph.io.dot.to_dot` writes DOT for Graphviz; `node_attrs` and `edge_attrs` choose which attributes to include (`True`, `False` or a list of names) and `rankdir` sets the direction:

```python
print(ag.io.to_dot(ag.DAG([("extract", "clean"), ("clean", "train")], name="etl"), rankdir="LR"))
```

```text
digraph etl {
  graph [rankdir=LR];
  extract;
  clean;
  train;
  extract -> clean;
  clean -> train;
}
```

## Mermaid

Mermaid flowcharts render in GitHub and GitLab Markdown and in many documentation tools, which makes them a convenient way to put a small DAG in a README. {py:func}`~aryagraph.io.mermaid.to_mermaid` replaces names that are not safe Mermaid ids with `n0`, `n1`, … and shows the original name as the label, so any text works, including spaces, quotes and Persian script:

```python
print(ag.io.to_mermaid(ag.DAG([("extract data", "clean"), ("clean", "train model")]), direction="LR"))
```

```text
flowchart LR
    n0["extract data"]
    clean
    n2["train model"]
    n0 --> clean
    clean --> n2
```

The reader understands node shapes (stored as `label` and `shape`), all link styles and labels, chains, `&` groups and subgraphs:

```python
chart = ag.io.from_mermaid("""
flowchart LR
  A[Collect data] --> B(Clean)
  B --> C{Valid?}
  C -->|yes| D[Train]
  C -.->|no| A
""")
print(chart, list(chart.nodes.data())[2], list(chart.edges.data())[-1])
```

```text
<DiGraph: 4 nodes, 4 edges> ('C', {'label': 'Valid?', 'shape': 'diamond'}) ('C', 'A', {'style': 'dotted', 'label': 'no'})
```

## networkx, pandas, numpy and SciPy

The converters live in `ag.io` and import the other library only when called; a missing package raises a {py:class}`~aryagraph.core.exceptions.DependencyError` that names the extra to install. From a clone of the repository, run `pip install ".[interop]"` (see [Optional extras](../getting-started/installation.md#optional-extras)).

| Library | To AryaGraph | From AryaGraph |
|---|---|---|
| networkx | {py:func}`~aryagraph.io.interop.from_networkx`, or `ag.Graph(G)` | {py:func}`~aryagraph.io.interop.to_networkx` |
| pandas | {py:func}`~aryagraph.io.interop.from_pandas_edgelist`, {py:func}`~aryagraph.io.interop.from_pandas` | {py:func}`~aryagraph.io.interop.to_pandas` |
| numpy | {py:func}`~aryagraph.io.interop.from_numpy` | {py:func}`~aryagraph.io.interop.to_numpy` |
| SciPy | {py:func}`~aryagraph.io.interop.from_scipy_sparse` | {py:func}`~aryagraph.io.interop.to_scipy_sparse` |

```python
import networkx as nx

karate = ag.Graph(nx.karate_club_graph())        # a networkx graph can be passed directly
print(karate, karate.nodes[0])
back = ag.io.to_networkx(campus)
print(type(back).__name__, back.number_of_nodes(), back.number_of_edges(), back.nodes["MSC"]["name"])
```

```text
<Graph "Zachary's Karate Club": 34 nodes, 78 edges> {'club': 'Mr. Hi'}
Graph 56 83 MacEwan Student Centre
```

`from_networkx(G, as_dag=True)` builds a `DAG` (raising `CycleError` on a cycle), and `collapse=True` merges the parallel edges of a multigraph.

With pandas, a graph is two tables: one row per node and one row per edge. {py:func}`~aryagraph.io.interop.to_pandas` and {py:func}`~aryagraph.io.interop.from_pandas` convert in both directions, and the round trip reproduces the campus attributes:

```python
nodes_df, edges_df = ag.io.to_pandas(campus)
print(nodes_df.shape, edges_df.shape)
print(edges_df.head(3))
rebuilt = ag.io.from_pandas(nodes_df, edges_df)
print(rebuilt.nodes["MSC"] == campus.nodes["MSC"], rebuilt.edges["AB", "CH"] == campus.edges["AB", "CH"])
```

```text
(56, 6) (83, 5)
  source target      kind  length  weight
0     AB     CH    pedway   141.2   141.2
1     AB     RT   outdoor   104.3   104.3
2     AD     PF  attached    85.7    85.7
True True
```

Matrices follow networkx's conventions: rows and columns in graph order (or the order you pass as `nodes=`), an edge without the weight attribute weighs 1, and an undirected graph gives a symmetric matrix. A symmetric matrix reads back as a `Graph` and an asymmetric one as a `DiGraph`, unless you pass `directed=`:

```python
import numpy as np

A = ag.io.to_numpy(campus, weight="length")
S = ag.io.to_scipy_sparse(campus, weight="length")
print(A.shape, S.nnz, np.allclose(A, S.toarray()))
g = ag.io.from_scipy_sparse(S, nodes=list(campus))
print(g, g.edges["AB", "CH"])
print(ag.io.from_numpy(np.array([[0, 1, 0], [0, 0, 2], [0, 0, 0]])))
```

```text
(56, 56) 166 True
<Graph: 56 nodes, 83 edges> {'weight': 141.2}
<DiGraph: 3 nodes, 2 edges>
```

`from_scipy_sparse` needs `nodes=` to restore labels (a matrix only knows positions) and stores the matrix entries under `weight`; other attributes, such as the campus `kind`, need the pandas or file routes.

For a worked example of moving data between these libraries, see the [interoperability tutorial](../tutorials/interop.md); to generate graphs to read and write, see [Datasets and generators](datasets.md).
