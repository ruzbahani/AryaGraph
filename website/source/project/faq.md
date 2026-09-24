# Frequently asked questions

Short, precise answers to the questions new users ask most often. Each answer links to the page that covers the topic
in depth; code on this page runs as shown with AryaGraph {{ version }}.

## Installation and requirements

### How do I install AryaGraph?

AryaGraph {{ version }} installs from its source repository on GitHub. With pip:

```bash
pip install "git+https://github.com/ruzbahani/AryaGraph.git"
```

or from a local clone, which is convenient if you also want the examples and tests:

```bash
git clone https://github.com/ruzbahani/AryaGraph.git
cd AryaGraph
pip install .
```

Check the installation with `python -c "import aryagraph; print(aryagraph.__version__)"`. The
[Installation](../getting-started/installation.md) page covers virtual environments and development installs.

### Which Python versions are supported?

Python 3.10 or newer. The continuous-integration workflow runs the full test suite on Python 3.10, 3.11, 3.12 and 3.13,
each on Ubuntu, Windows and macOS, for every push and pull request (12 combinations).

### Why is numpy the only dependency?

AryaGraph contains its own graph data structures, algorithms, layout engines, SVG renderer, interactive HTML runtime
and simulators, so it needs nothing beyond numpy to run. This keeps installation simple (numpy publishes prebuilt
wheels for Linux, Windows and macOS), keeps saved HTML files free of third-party scripts, and limits the packages
whose updates can change your results to numpy alone.

Everything else is optional and loaded only when you call a feature that needs it:

| Extra | Installs | Enables |
|---|---|---|
| `interop` | networkx, pandas, SciPy | the converters in `ag.io` (`from_networkx`, `from_pandas_edgelist`, `to_scipy_sparse`, …) |
| `png` | cairosvg | PNG and PDF export without a browser |
| `test` | pytest, networkx, pandas, SciPy | running the test suite |

Install an extra from a clone with `pip install ".[interop]"`, or from GitHub with
`pip install "aryagraph[interop] @ git+https://github.com/ruzbahani/AryaGraph.git"`. If an optional package is
missing, the call raises {py:class}`~aryagraph.core.exceptions.DependencyError`, which names the package to install.

## Drawing and exporting

### Does it work in Jupyter?

Yes. A {py:class}`~aryagraph.render.figure.Figure` left as the last expression of a cell renders inline as the
interactive view (pan, zoom, hover, search, table). Charts and analysis reports display inline the same way. Outside
a notebook, `fig.show()` opens the interactive page in your default browser.

```python
import aryagraph as ag

g = ag.gen.florentine_families()
fig = ag.draw(g, node_color=ag.alg.betweenness_centrality(g))
fig  # in a notebook, this line displays the interactive figure
```

See [Interactive HTML](../user-guide/interactive.md) for what the interactive view offers.

### How do I export PNG or PDF without Chrome?

SVG and HTML are written directly by AryaGraph and need no other software. PNG and PDF need a rasterizer, and
AryaGraph looks for one in this order:

1. **cairosvg**, if it is installed: `pip install ".[png]"` from a clone, or `pip install cairosvg`;
2. an installed Chromium-family browser (Chrome, Edge, Chromium or Brave), run headless.

If your browser is in an unusual location, set the `ARYAGRAPH_BROWSER` environment variable to its executable. With
neither option available, `fig.save("figure.png")` raises `DependencyError`. See
[Exporting](../user-guide/exporting.md).

### Can I use my own node positions?

Yes. `layout=` accepts any `{node: (x, y)}` mapping, which is how the campus figures use real building positions:

```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
fig = ag.draw(campus, layout={n: d["pos"] for n, d in campus.nodes.data()}, node_color="kind")
fig.save("campus.svg")
```

AryaGraph uses screen coordinates in every layout: x grows to the right and **y grows downward**. If your coordinates
have y pointing up (latitude, or a mathematical plot), negate y before drawing, or the picture comes out upside down.
The campus dataset's `pos` attribute is already in this convention (meters east and south of the campus center).

### Can I label nodes in Persian, Arabic or Hebrew?

Yes. Labels, titles, legends and tooltips accept right-to-left text as well as CJK text. AryaGraph estimates text
width per script to size boxes and place labels, and marks right-to-left text with its direction so it is written
the correct way:

```python
import aryagraph as ag

courses = ag.DAG([("ریاضی ۱", "ریاضی ۲"), ("ریاضی ۲", "آمار و احتمال")])
print(ag.style.is_rtl("ریاضی ۱"), ag.style.is_rtl("MATH101"))
fig = ag.draw(courses, theme=ag.get_theme("light").with_(font="Vazirmatn, sans-serif"))
fig.save("courses.svg")
```

```text
True False
```

Joining Persian and Arabic letters (glyph shaping) is done by whatever renders the SVG. Browsers do it, and PNG or PDF
export through a Chromium browser renders text the way the interactive view does. If you export with cairosvg
instead, check the output for joined letters. The default font list already includes Vazirmatn and Noto Sans, which
are used when installed on the viewer's device; the `with_(font=...)` call above puts a font first explicitly. See [Languages](../user-guide/languages.md) and the Persian
example in the [Gallery](../gallery/index.md#right-to-left-labels).

### How do I change colors and fonts for every figure?

Derive a theme from a built-in one with {py:meth}`~aryagraph.style.themes.Theme.with_` and register it under a name;
every `theme=` argument then accepts that name:

```python
import aryagraph as ag

brand = ag.get_theme("paper").with_(name="brand", font_size=13, edge="#6b7280")
ag.register_theme(brand)
fig = ag.draw(ag.gen.les_miserables(), theme="brand")
print(fig)
```

```text
<Figure 823×786: 77 nodes, 254 edges, stress layout, theme 'brand'>
```

The four built-in themes are `light`, `dark`, `paper` and `blueprint`; the [Gallery](../gallery/index.md#themes)
shows them side by side, and [Styling](../user-guide/styling.md) lists every theme field.

## Working with other tools

### Can I use AryaGraph with networkx?

Yes, in both directions. {py:func}`~aryagraph.io.interop.from_networkx` copies a networkx graph with all node, edge
and graph attributes, and {py:func}`~aryagraph.io.interop.to_networkx` converts back. `ag.Graph(G)` also accepts a
networkx graph directly.

```python
import networkx as nx
import aryagraph as ag

G = nx.karate_club_graph()
g = ag.io.from_networkx(G)
print(g)
print(ag.alg.pagerank(g).top(3))
ours, theirs = ag.alg.pagerank(g), nx.pagerank(G)
print(f"largest difference: {max(abs(ours[n] - theirs[n]) for n in G):.1e}")
back = ag.io.to_networkx(g)
```

```text
<Graph "Zachary's Karate Club": 34 nodes, 78 edges>
[(33, 0.09698041880501741), (0, 0.08850807396280014), (32, 0.07592643687005647)]
largest difference: 1.4e-17
```

Algorithms follow networkx's definitions where networkx has an established one (for example, betweenness
normalization and PageRank's handling of dangling nodes), so results can be compared directly. The last line compares
the two PageRank results: over the 34 nodes, the largest gap between `ag.alg.pagerank(g)` and `nx.pagerank(G)` is
1.4e-17. Multigraphs need `collapse=True`, which merges parallel edges. See
[File formats and interoperability](../user-guide/io.md).

### Can I use it with pandas?

Yes. Build a graph from an edge table with {py:func}`~aryagraph.io.interop.from_pandas_edgelist`, and turn results
back into pandas objects: every per-node result has `.to_pandas()`, and {py:func}`~aryagraph.io.interop.to_pandas`
returns a node table and an edge table.

```python
import pandas as pd
import aryagraph as ag

df = pd.DataFrame({"source": ["a", "a", "b"], "target": ["b", "c", "c"], "weight": [2.0, 1.0, 3.0]})
g = ag.io.from_pandas_edgelist(df, "source", "target", edge_attr="weight")
print(ag.alg.pagerank(g, weight="weight").to_pandas())
nodes, edges = ag.io.to_pandas(g)
```

```text
a    0.259464
b    0.408623
c    0.331913
Name: pagerank, dtype: float64
```

Simulation results export too: `run.to_pandas()` gives one row per node and frame.

### Which file formats can I read and write?

{py:func}`ag.read <aryagraph.io.read>` and {py:func}`ag.write <aryagraph.io.write>` choose the format from the file
extension: node-link JSON (`.json`, compatible with networkx and d3), CSV/TSV edge lists, adjacency lists, GraphML,
GEXF (Gephi), Graphviz DOT (`.dot`, `.gv`) and Mermaid flowcharts (`.mmd`). The same formats work from the command
line, without writing Python:

```bash
aryagraph draw pipeline.dot -o pipeline.html
aryagraph analyze network.graphml -o report.html
aryagraph convert graph.gv graph.graphml
```

See [File formats and interoperability](../user-guide/io.md) and [Command line](../user-guide/cli.md).

## Performance and reproducibility

### How large a graph can AryaGraph handle?

AryaGraph is written in Python and numpy, and its defaults change with graph size:

- `layout="auto"` uses the hierarchical layout for DAGs of up to 2,000 nodes, stress for graphs of up to 3,000 nodes
  and ForceAtlas2 (with Barnes–Hut approximation above 1,500 nodes) beyond that;
- overlap removal runs for drawings of up to 3,000 nodes;
- `labels="auto"` shows every label for up to 80 nodes and the 30 most prominent above that;
- {py:func}`~aryagraph.analysis.report.analyze` uses exact all-pairs measures up to 3,000 nodes, then skips distance
  measures and estimates betweenness from sampled sources, recording a note for each;
- animations keep at most 600 frames by default and pack each frame as one byte per node.

The site's build script measured these times for Barabási–Albert graphs (`barabasi_albert(n, 2, seed=1)`) on a
Windows 11 machine with Python 3.13 and numpy 2.4, rounded to the nearest second. "Draw" includes the layout and the
interactive HTML page.

| Nodes | Edges | Layout chosen by `auto` | Draw | HTML file |
|---:|---:|---|---:|---:|
| 1,000 | 1,996 | stress | 2 s | 1.1 MB |
| 3,000 | 5,996 | stress | 17 s | 3.3 MB |
| 10,000 | 19,996 | forceatlas2 | 19 s | 11.0 MB |

`ag.analyze` took about 3 s at 1,000 nodes and 6 s at 5,000 nodes (with distances skipped and betweenness sampled
from 500 sources). Stress needs all-pairs graph distances, so its cost grows with the square of the node count; for graphs
of a few thousand nodes, `layout="forceatlas2"` is faster. For graphs of tens of thousands of nodes, pass
`labels=False`, prefer SVG or PNG to HTML, and switch off heavy report sections with
`ag.analyze(g, communities=False, centrality=False)`. Timings on your machine will differ.

### Are results reproducible?

Yes, for the same graph, parameters, seed and AryaGraph version. Every source of randomness takes a `seed`, randomized
layouts default to `seed=0`, and nodes and edges keep their insertion order everywhere, so the same call draws the
same picture:

```python
import aryagraph as ag

les = ag.gen.les_miserables()
first = ag.draw(les, layout="forceatlas2", seed=3).to_svg()
again = ag.draw(les, layout="forceatlas2", seed=3).to_svg()
other = ag.draw(les, layout="forceatlas2", seed=4).to_svg()
print(first == again, first == other)
```

```text
True False
```

Simulations work the same way, and {py:func}`~aryagraph.sim.run_ensemble` records the seed of every run in `ens.seeds`
so any single run can be replayed. To reproduce results in another environment, pin the versions of AryaGraph and
numpy. See [Reproducibility and performance](../user-guide/reproducibility.md).

### How are the algorithms tested?

AryaGraph has 2,457 automated tests; 1,564 of them are in the eight algorithm modules. Wherever networkx offers
an equivalent function, the tests run both on several seeded random graphs and compare the answers. Functions
without an equivalent are checked on hand-computed cases and invariants. Each area also covers edge cases: the empty
graph, a single node, disconnected graphs, self-loops, and directed versus undirected input. The suite runs on
Python 3.10 to 3.13 on Linux, Windows and macOS for every push. To run it yourself from a clone:

```bash
pip install -e ".[test]"
python -m pytest
```

### Why did adding an edge to a DAG raise `CycleError`?

A {py:class}`~aryagraph.core.dag.DAG` stays acyclic: an edge that would close a cycle is rejected, and the error names
the cycle it would create.

```python
import aryagraph as ag

dag = ag.DAG([("extract", "clean"), ("clean", "train"), ("clean", "report")])
try:
    dag.add_edge("report", "extract")
except ag.CycleError as err:
    print(err)
    print(err.cycle)
```

```text
edge 'report' → 'extract' would close the cycle 'report' → 'extract' → 'clean' → 'report'
['report', 'extract', 'clean', 'report']
```

Use {py:class}`~aryagraph.core.graph.DiGraph` if your directed graph may contain cycles. See
[DAG workflows](../user-guide/dags.md).

## Publishing figures

### Can I host interactive figures on an ordinary website, without a Python server?

Yes. Python runs only on your own computer, to create the files. A figure saved with `fig.save("figure.html")`, a
simulation player from `run.animate().save(...)` and a dashboard from `report.save("report.html")` are each a single
self-contained HTML file: the drawing, its data and the small JavaScript runtime are inside the file, and it loads
nothing from a CDN or any other server. Any host that serves static files can publish it, including shared web
hosting, GitHub Pages and university web space. This website works that way: it is plain HTML on ordinary hosting,
and its interactive examples are files that AryaGraph wrote.

To publish a figure:

1. save it: `fig.save("campus.html")`;
2. upload the file with your host's file manager or FTP, for example to `public_html/figures/`;
3. link to it (`https://example.com/figures/campus.html`) or embed it in a page with an iframe:

```html
<iframe src="/figures/campus.html" width="100%" height="640" loading="lazy"
        title="University of Calgary campus network" style="border:0"></iframe>
```

For a static picture, upload the SVG and use `<img src="/figures/campus.svg" alt="...">`. File sizes depend on the
graph: the [Gallery](../gallery/index.md) embeds a 400-node simulation player of 785 KB and a 77-node analysis
dashboard of 236 KB, and a 10,000-node interactive figure is about 11 MB (see the performance answer above).

### Do figures need an internet connection?

No. Saved HTML files work offline, including when opened straight from disk, because they carry their own scripts and
styles. The bundled datasets (the campus, *Les Misérables*, the Florentine families, Davis's Southern Women and the
example DAGs) ship inside the package, so `ag.gen` needs no download either.

## Licensing, citing and support

### What license does AryaGraph use?

The MIT License. You may use, copy, modify, merge, publish, distribute, sublicense and sell AryaGraph, in open-source
and commercial projects alike, provided the copyright notice and license text are included in copies or substantial
portions of the software. Interactive HTML files embed AryaGraph's JavaScript runtime, and that runtime carries its
copyright line and MIT identifier inside the file. See [License](license.md).

### Can I use the University of Calgary campus data?

Yes, with attribution. The building positions come from OpenStreetMap: © OpenStreetMap contributors, available under
the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/). When you publish figures that show
the campus geometry, credit "© OpenStreetMap contributors". If you publicly share a database derived from the
positions, the ODbL's share-alike terms apply to it. Building codes and names come from the university's public campus
map and building directory. The dataset is a schematic model for demonstrations: edge lengths are straight-line
distances between buildings, not surveyed walking routes.

### How do I cite AryaGraph?

The repository contains a `CITATION.cff` file, so GitHub's **Cite this repository** button produces APA and BibTeX
entries. A BibTeX entry and details are on [Citing AryaGraph](citing.md). Please cite the version you used;
`aryagraph.__version__` reports it.

### How do I report a bug or request a feature?

Open an issue at [github.com/ruzbahani/AryaGraph/issues](https://github.com/ruzbahani/AryaGraph/issues). A report that
can be reproduced gets fixed fastest, so include:

- the AryaGraph, Python and numpy versions, and your operating system:
  `python -c "import sys, numpy, aryagraph; print(aryagraph.__version__, sys.version, numpy.__version__)"`;
- a short, complete script that shows the problem, using a bundled dataset or generator with a fixed seed if you can;
- the full traceback, or for a drawing problem the saved `.svg` file and a description of what you expected.

To contribute code or documentation, see [Contributing](contributing.md).
