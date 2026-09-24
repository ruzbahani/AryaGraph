# Interactive HTML

Saving a figure as `.html` produces a single self-contained page that you can explore in any modern browser: pan and zoom, inspect nodes, drag them around, search, filter by category and read every value in a table. This page lists what the view does, how to use it in Jupyter, and how to publish it on a website.

## Creating an interactive page

Any figure becomes interactive when you save it with the `.html` extension:

```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
fig = ag.draw(
    campus,
    layout={n: d["pos"] for n, d in campus.nodes.data()},
    node_color="kind",
    tooltip=["name", "kind"],
    title="University of Calgary main campus",
    subtitle="Hover, click, drag, search; the legend filters by kind",
)
fig.save("campus.html")
```

Try it here:

<iframe class="ag-embed" src="../_static/generated/guide_core/interactive_campus.html" height="720" loading="lazy" title="Interactive map of University of Calgary campus buildings colored by kind"></iframe>
<p class="ag-embed-note">The campus network in the interactive view. Hover a building to see its neighborhood and name, click it to pin the details panel, or type a code such as TFDL in the search box. Positions © OpenStreetMap contributors (ODbL). <a href="../_static/generated/guide_core/interactive_campus.html">Open full screen</a></p>

The page contains everything it needs: the SVG drawing, a JSON description of the nodes and edges, the stylesheet and a small JavaScript runtime with no dependencies. It makes no network requests, so it works offline, from a local file, in an email attachment and on any static web host.

```python
html = fig.to_html()
print(round(len(html.encode("utf-8")) / 1024), "KiB")           # 122 KiB
print("<script src" in html, "<link" in html, "@import" in html)   # False False False
```

## What you can do in the view

### Navigate

- **Zoom** with the mouse wheel or a trackpad pinch, with a two-finger pinch on touch screens, or with the **+** and **−** toolbar buttons. The wheel zooms around the pointer, and the zoom range is 0.1× to 40×.
- **Pan** by dragging the background.
- **Reset** with the **Fit** button or by double-clicking the background.

### Inspect

- **Hover a node** to highlight it, its neighbors and the edges between them, and to see a tooltip with its degree (in- and out-degree for directed graphs) and attributes. By default the tooltip lists up to ten attributes plus the values of any data-driven color or size channel; `tooltip=` in `draw()` chooses which attributes appear.
- **Hover an edge** to see its endpoints and attributes.
- **Click a node** to pin the highlight and open the details panel: the node's label and id, its attributes and a list of its neighbors. Clicking a neighbor in the list selects it and centers the view on it. Click the node again, click the background or press **Esc** to close the panel.

### Rearrange

**Drag a node** to move it. Its edges are re-routed while you drag, in the same style as the static drawing (straight, curved, flow or orthogonal), still clipped to the node outlines and with arrowheads recomputed; edge labels follow their edges. Dragging changes only the page in the browser, not your graph or the saved file.

```python
build = ag.gen.software_build()
fig = ag.draw(build, node_color="team", title="Software build", subtitle="Drag a task to re-route its edges")
fig.save("build.html")
print(fig.scene.meta["edge_style"])   # flow
```

<iframe class="ag-embed" src="../_static/generated/guide_core/interactive_build.html" height="700" loading="lazy" title="Interactive layered drawing of a software build DAG with flow-style edges"></iframe>
<p class="ag-embed-note">The software-build DAG from the <a href="graphs.html">Graphs</a> page. Drag a task sideways to see its flow edges re-route. <a href="../_static/generated/guide_core/interactive_build.html">Open full screen</a></p>

### Search

Type in the search box to find nodes whose label or id contains the text (case does not matter). Up to eight matches are listed: those that start with the text come first, then those with more connections. Use **↑** and **↓** to move through the list, **Enter** to select a match (the view centers and zooms on it), and **Esc** to clear the search.

### Filter by category

When nodes are colored by category, the legend is interactive. Hovering a legend entry previews that category; clicking it keeps the category highlighted and dims everything else, and clicking it again clears the filter.

### Labels, table and downloads

- The **Labels** button cycles through three modes: *auto* shows the labels placed without collisions, *all* also shows the labels that were hidden to avoid overlaps, and *off* hides every label. Hovering or selecting a node also reveals the hidden labels in its neighborhood.
- The **Table** button opens a table with one row per node: the label, the degree (or in- and out-degree) and up to ten attribute columns. Click a column header to sort by it (again to reverse), and click a row to select that node in the drawing. The table lists up to 2,000 rows.
- The **SVG** and **PNG** buttons download the drawing as it currently looks, including your zoom and any nodes you have moved. PNG downloads are rendered in the browser at twice the drawing's size.

### Keyboard shortcuts

Click the drawing (or move to it with **Tab**) so that it has focus, then:

| Key | Action |
|---|---|
| **+** or **=** | zoom in |
| **-** or **_** | zoom out |
| **0** | reset the view |
| arrow keys | pan by 40 px |
| **Esc** | clear the selection |
| **Space** | play or pause (animated simulations) |

## Jupyter notebooks

In Jupyter, a figure that ends a cell renders inline as the interactive view:

```pycon
>>> fig = ag.draw(campus, node_color="kind")
>>> fig          # displays the interactive view below the cell
```

The page is placed in a sandboxed `<iframe>` that may run scripts and start downloads, so the drawing cannot affect the rest of the notebook. `fig.show()` displays the figure inline in a notebook; outside a notebook it writes the page to a temporary file and opens it in your default browser.

## Publishing on a website

Because the page is a single static file, publishing it needs only ordinary web hosting: no Python on the server, no database and no build step. Upload the `.html` file next to your page and embed it with an `<iframe>`, as this website does:

```html
<iframe src="figures/campus.html" width="100%" height="720" loading="lazy"
        title="Interactive map of campus buildings" style="border:0"></iframe>
<p><a href="figures/campus.html">Open full screen</a></p>
```

Some practical points:

- **Size.** The drawing scales to the width of the frame, keeps its aspect ratio and is capped at 82% of the frame's height; set the iframe `height` to leave room for the toolbar and the footer below the drawing.
- **Isolation.** An iframe keeps the page's styles and scripts separate from your site's, so neither can break the other.
- **Documentation sites.** In Sphinx, put the file under a static folder (for example `_static/`) and use a raw HTML `<iframe>` in your page; MkDocs and other static site generators work the same way.
- **Linking.** A plain link to the `.html` file opens the view full screen, which is often better on phones.

To place the drawing without any script, for example where JavaScript is not allowed, save a static HTML page instead; see [Exporting](exporting.md#non-interactive-html).

The runtime initializes every AryaGraph app on a page, which is how the [analysis dashboards](analysis.md) combine several interactive views in one file.

## Next steps

- [Exporting](exporting.md): SVG, PNG and PDF, sizes and backends.
- [Drawing](drawing.md): the channels whose values appear in tooltips, the table and the legend.
- [Simulation](simulation.md): animated simulations use the same view with a timeline player.
