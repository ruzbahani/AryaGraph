# Command line

The `aryagraph` command draws, analyzes, inspects and converts graph files without writing any Python. It is installed with the package, and `python -m aryagraph` runs the same program where the scripts directory is not on your `PATH`.

```bash
aryagraph draw    graph.json -o graph.svg     # render to .svg, .html, .png or .pdf
aryagraph analyze graph.json -o report.html   # analytical report or dashboard
aryagraph info    graph.json                  # headline statistics
aryagraph convert graph.gv graph.graphml      # change the file format
```

Every subcommand reads any format that {py:func}`ag.read <aryagraph.io.read>` understands and chooses it from the file extension: `.json`, `.csv`, `.tsv`, `.txt`, `.edgelist`, `.adjlist`, `.graphml`, `.gexf`, `.dot`, `.gv`, `.mmd` and `.mermaid` (see [File formats and interoperability](io.md)). Output formats also follow the extension.

## Example files

The examples on this page use three bundled datasets saved as files. Create them with a few lines of Python:

```python
import aryagraph as ag

ag.write(ag.gen.ucalgary_campus(), "campus.json")
ag.write(ag.gen.ml_pipeline(), "pipeline.dot")
ag.write(ag.gen.les_miserables(), "lesmis.graphml")
with open("flights.csv", "w", encoding="utf-8") as fh:
    fh.write("origin,dest,minutes\nYYC,YVR,85\nYVR,YYZ,265\nYYC,YYZ,230\nYYZ,YUL,75\n")
```

## `aryagraph info`

`info` prints the graph and its headline statistics from {py:func}`~aryagraph.algorithms.structure.summary`:

```bash
aryagraph info campus.json
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
  n                  56
  m                  83
  directed           False
  density            0.05389610389610389
  self_loops         0
  avg_degree         2.9642857142857144
  max_degree         5
  isolates           0
  components         1
  largest_component  56
  avg_clustering     0.3220238095238095
  assortativity      -0.17090656799259943
```

Directed graphs add `reciprocity` and `is_dag`.

## `aryagraph draw`

`draw` renders a graph with {py:func}`ag.draw <aryagraph.render.draw>` and saves it. Without `-o`, it writes an `.svg` named after the input into the current directory.

```bash
aryagraph draw campus.json -o campus.png --color kind --size betweenness --title "UCalgary campus"
```

```text
wrote campus.png  (854×784, 56 nodes, 83 edges, stress layout)
```

```{figure} ../_static/generated/guide_sim_data/cli_campus.png
:alt: The UCalgary campus graph in a stress layout, nodes colored by building kind and sized by betweenness centrality, produced by the draw command.
:width: 85%

The file written by the command above. `--color kind` uses the node attribute; `--size betweenness` computes betweenness centrality because no node has an attribute of that name.
```

| Option | Meaning |
|---|---|
| `-o`, `--output FILE` | output file; `.svg`, `.html` (interactive), `.png` or `.pdf` |
| `--layout NAME` | `auto` (the default: hierarchical for DAGs up to 2,000 nodes, stress for other graphs up to 3,000 nodes, ForceAtlas2 beyond), or any layout name such as `stress`, `hierarchical`, `tree`, `radial`, `forceatlas2`, `circular`, `spectral` |
| `--color NAME`, `--size NAME` | a node attribute, or a metric computed on the fly: `pagerank`, `betweenness`, `degree`, `community` (Louvain with seed 0) |
| `--edge-style STYLE` | `auto`, `straight`, `curved`, `flow` or `orthogonal` |
| `--theme NAME` | `light` (default), `dark`, `paper` or `blueprint` |
| `--title`, `--subtitle` | text above the drawing |
| `--no-labels` | hide node labels |
| `--seed N` | seed for randomized layouts (default 0, so repeated runs give the same drawing) |
| `--width`, `--height` | canvas size in pixels |
| `--directed` | read `.csv`, `.tsv`, `.txt` and `.edgelist` files as directed |

A name given to `--color` or `--size` is looked up as a node attribute first, so an attribute called `pagerank` in your file wins over the computed metric.

More examples, each run as shown:

```bash
aryagraph draw pipeline.dot -o pipeline.html --color team
aryagraph draw pipeline.dot -o pipeline.svg --edge-style orthogonal --width 900
aryagraph draw lesmis.graphml -o lesmis.png --color community --size pagerank --no-labels
aryagraph draw lesmis.graphml --layout forceatlas2 --seed 3
aryagraph draw campus.json -o campus_dark.png --color kind --theme dark --no-labels
aryagraph draw flights.csv --directed
```

```text
wrote pipeline.html  (759×1004, 16 nodes, 19 edges, hierarchical layout)
wrote pipeline.svg  (900×1480, 16 nodes, 19 edges, hierarchical layout)
wrote lesmis.png  (578×436, 77 nodes, 254 edges, stress layout)
wrote lesmis.svg  (1067×648, 77 nodes, 254 edges, forceatlas2 layout)
wrote campus_dark.png  (835×747, 56 nodes, 83 edges, stress layout)
wrote flights.svg  (118×338, 4 nodes, 4 edges, hierarchical layout)
```

The DOT file of the pipeline reads as a directed acyclic graph, so `auto` picks the layered layout; the flight list read with `--directed` is acyclic too. The `.html` output is a self-contained interactive page (pan, zoom, search, a table view) that needs no server.

```{note}
The command line places nodes with a layout algorithm. To draw at stored coordinates, such as the campus `pos` attribute, or to set layout options like a left-to-right orientation, use the Python API: `ag.draw(g, layout={n: d["pos"] for n, d in g.nodes.data()})` or `ag.draw(dag, layout_options={"orientation": "LR"})`.
```

## `aryagraph analyze`

`analyze` runs {py:func}`ag.analyze <aryagraph.analysis.report.analyze>`: headline statistics, components, distances, clustering, centrality rankings, communities and, for DAGs, depth, width and the critical path. Without `-o` the text report goes to the terminal:

```bash
aryagraph analyze lesmis.graphml
```

```text
Graph report: Les Misérables
────────────────────────────────────────
Graph                   Undirected · 77 nodes · 254 edges
Density                 0.0868
Average degree          6.6
Max degree              36
Components              1 (largest: 77 nodes, 100%)
Diameter                5
Radius                  3
Avg path length         2.64
Avg clustering          0.573
Transitivity            0.499
Degree assortativity    -0.165
Algebraic connectivity  0.205
Communities             6 (modularity 0.567)
Top degree              Valjean (36), Gavroche (22), Marius (19)
Top pagerank            Valjean (0.0754), Myriel (0.0428), Gavroche (0.0358)
Top betweenness         Valjean (0.57), Myriel (0.177), Gavroche (0.165)
Top closeness           Valjean (0.644), Marius (0.531), Thenardier (0.517)
Top eigenvector         Gavroche (0.318), Valjean (0.268), Enjolras (0.267)
Bridges / cut nodes     18 / 8
Max k-core              9
```

With `-o`, the extension chooses the form: `.html` saves an interactive dashboard with stat tiles, the explorable graph, distributions and rankings; `.json` saves the report's data; `.txt` saves the text above.

```bash
aryagraph analyze lesmis.graphml -o report.html
aryagraph analyze lesmis.graphml -o report.json
aryagraph analyze pipeline.dot -o pipeline.txt
```

```text
wrote report.html
wrote report.json
wrote pipeline.txt
```

<p class="ag-embed-note">The dashboard written by the first command: <a href="../_static/generated/guide_sim_data/cli_report.html">open report.html</a> (a single static file of about 236 KB).</p>

For the pipeline, the text report adds `DAG depth / width  12 levels / 3 parallel`, `Sources / sinks  2 / 1`, the critical path with its length (58.8) and `Redundant edges  1`.

## `aryagraph convert`

`convert` reads one format and writes another, choosing both from the extensions:

```bash
aryagraph convert pipeline.dot pipeline.graphml
aryagraph convert campus.json campus.gexf
aryagraph convert flights.csv flights.json --directed
```

```text
wrote pipeline.graphml  (16 nodes, 19 edges)
wrote campus.gexf  (56 nodes, 83 edges)
wrote flights.json  (4 nodes, 4 edges)
```

What survives depends on the target format; the table in [What survives a round trip](io.md#what-survives-a-round-trip) lists it. `--directed` applies to edge-list inputs, whose files cannot say whether they are directed; the other formats record it.

## Errors and exit codes

The command exits with status 0 on success. A problem with the input or the output prints a one-line message instead of a traceback and exits with status 1; invalid arguments print the usage and exit with status 2, as usual for command-line tools.

```bash
aryagraph info missing.json
aryagraph draw campus.json -o campus.bmp
aryagraph draw campus.json --layout spring-ish
```

```text
aryagraph: error: [Errno 2] No such file or directory: 'missing.json'
aryagraph: error: unsupported file extension '.bmp'; use .svg, .html, .png or .pdf
aryagraph: error: unknown layout method 'spring-ish'; valid names: auto, arc, bipartite, circular, force, force_atlas2, forceatlas2, fruchterman_reingold, grid, hierarchical, kamada_kawai, layered, radial, random, shell, spectral, spiral, spring, stress, sugiyama, tree
```

`aryagraph --help` lists the subcommands, `aryagraph draw --help` (and likewise for the others) lists every option, and `aryagraph --version` prints `AryaGraph 0.1.0`. PNG and PDF output use `cairosvg` when it is installed and otherwise a Chrome, Edge or Chromium browser found on the system; see [Exporting](exporting.md).
