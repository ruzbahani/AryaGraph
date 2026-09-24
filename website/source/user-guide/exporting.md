# Exporting

A {py:class}`~aryagraph.render.figure.Figure` saves to SVG, HTML, PNG and PDF, and it can hand you the SVG markup or the positioned marks directly. This page explains the formats, how sizes work, which backend produces PNG and PDF files, and how charts export.

## One method, four formats

{py:meth}`Figure.save() <aryagraph.render.figure.Figure.save>` picks the format from the file extension, creates missing parent folders and returns the path it wrote:

```python
import os
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
on_map = {n: d["pos"] for n, d in campus.nodes.data()}
fig = ag.draw(campus, layout=on_map, node_color="kind", title="University of Calgary main campus")

for ext in ("svg", "html", "png", "pdf"):
    path = fig.save(f"figures/campus.{ext}")
    print(path.name, f"{os.path.getsize(path) / 1024:,.0f} KiB")
```

| Extension | What you get | Needs |
|---|---|---|
| `.svg` | a standalone vector file with an XML declaration | nothing beyond AryaGraph |
| `.html` | the [interactive view](interactive.md), or a static page with `interactive=False` | nothing beyond AryaGraph |
| `.png` | a raster image at `scale` device pixels per CSS pixel (2 by default) | cairosvg, or a Chromium-family browser |
| `.pdf` | a vector PDF whose page is the size of the figure | cairosvg, or a Chromium-family browser |

Any other extension raises a `ValueError` that lists the supported ones. SVG and HTML are written by AryaGraph itself and take milliseconds; PNG and PDF are rendered by an external engine, described [below](#png-and-pdf-backends).

## Sizes

A drawing has a natural size in CSS pixels, computed from the layout, the node sizes, the labels, the legends and the title. `fig.width` and `fig.height` report it:

```python
print(fig)   # <Figure 1195×877: 56 nodes, 83 edges, custom layout, theme 'light'>
```

Three separate settings change the output size:

`width=` / `height=` in `draw()`
: Fit the drawing into a display size. The SVG's `width` and `height` attributes are set to what you asked for. For layouts in abstract units (force-directed, stress, geometric and your own coordinates, such as the campus map) the layout is also rescaled to fill the space; layered, tree and radial layouts are already sized in pixels, so the whole image is scaled instead, text included. Give one of the two to keep the natural aspect ratio.

`scale=` in `draw()`
: Pixels per layout unit, which fixes the spacing of nodes instead of choosing it automatically. For layered, tree and radial layouts it multiplies their pixel coordinates, so `scale=0.6` pulls the nodes closer together. Node and label sizes do not change, so a larger `scale` spreads the same marks farther apart.

`scale=` in `save()` (PNG only)
: Device pixels per CSS pixel of the raster image. The default of 2 gives sharp results on high-density screens; use 1 for the natural size, or 3 to 4 for large prints.

```python
import struct

def png_size(path):
    with open(path, "rb") as f:
        return struct.unpack(">II", f.read(24)[16:24])   # width and height from the PNG header

print(png_size("figures/campus.png"))                        # (2390, 1754)
print(png_size(fig.save("figures/campus_1x.png", scale=1)))  # (1195, 877)

column = ag.draw(campus, layout=on_map, node_color="kind", width=640)
print(round(column.width), round(column.height))            # 640 680
```

SVG and PDF output treat one CSS pixel as 1/96 inch. For a journal column, set `width=` in `draw()` to the column width in those units (a 3.5-inch column is 336 px) and save as PDF or SVG, which stay sharp at any zoom.

Node and label sizes are set in pixels on the drawing's canvas. When the legend, labels and title need more room than the width you ask for, or the layout is sized in pixels, the canvas (`fig.scene.width`) differs from the requested width and the whole image is scaled to `fig.width`, text included. The ratio of the two gives the printed text size:

```python
narrow = ag.draw(campus, layout=on_map, node_color="kind", width=336)
shrink = narrow.width / narrow.scene.width
label_pt = narrow.scene.theme.label_size * shrink * 0.75   # 1 px = 0.75 pt
print(f"canvas {narrow.scene.width:.0f} px, scaled by {shrink:.2f}, labels {label_pt:.1f} pt")
# canvas 493 px, scaled by 0.68, labels 5.9 pt
```

At 336 px, the legend beside the map takes a large share of the width, so the 11.5 px labels print at 5.9 pt. With `legend=False` the factor rises to 0.93 and labels print at 8.0 pt. Raising `label_size` also widens the canvas: `label_size=16` scales by 0.65, which gives 7.8 pt labels. The [Publication figures](../tutorials/publication-figures.md) tutorial runs this check on a finished figure. For slides and web pages, a PNG at `scale=2` is usually enough.

## PNG and PDF backends

AryaGraph writes SVG itself and turns it into PNG or PDF with the first of these that is available:

1. **cairosvg**, if the package is installed (`pip install ".[png]"` from a source checkout). It runs inside Python and needs no browser.
2. **A Chromium-family browser** (Chrome, Edge, Chromium or Brave) run in headless mode. AryaGraph looks for it on the `PATH` and in the usual install locations on Windows and macOS; set the `ARYAGRAPH_BROWSER` environment variable to a browser's executable to choose one yourself. The browser renders text with the same engine as the interactive view, including the shaping of Persian and Arabic script.

If neither is found, saving a PNG or PDF raises {py:class}`~aryagraph.core.exceptions.DependencyError` with installation hints. {py:func}`~aryagraph.render.export.find_browser` tells you which browser would be used:

```python
from aryagraph.render.export import find_browser

browser = find_browser()
print("browser found" if browser else "no browser; install cairosvg for PNG and PDF")
```

Both backends produce a PNG at the requested scale and a single-page vector PDF sized to the figure. Fonts come from the machine that renders the file, so a figure that uses a specific typeface should be exported on a machine where that font is installed.

## Non-interactive HTML

`interactive=False` writes a page with the drawing and its stylesheet but without the JavaScript runtime: no toolbar, no tooltips beyond the browser's own, no scripts. Use it where scripts are not allowed, or when you want the HTML wrapper around a static picture:

```python
fig.save("figures/campus_static.html", interactive=False)
static = fig.to_html(interactive=False)
print("<script>" in static, "<script>" in fig.to_html())   # False True
```

`to_html()` returns the page as a string, and `title=` sets the browser tab's title (the figure title by default).

## Working with the SVG directly

`fig.to_svg()` returns the SVG markup as a string, ready to embed inline in HTML, write into a template or post-process. The saved `.svg` file is the same markup with an XML declaration in front.

```python
svg = fig.to_svg()
print(svg[:40])                                                # <svg xmlns="http://www.w3.org/2000/svg"
print(svg.count('class="ag-n"'), svg.count('class="ag-e"'))    # 56 83
```

The document has a stable structure, so you can style or script it with CSS selectors:

| Element | Contents |
|---|---|
| `svg.ag-root` | the document, with `<title>`, `<desc>` and `role="img"` |
| `g.ag-header` | title and subtitle |
| `g.ag-edges > g.ag-e` | one group per edge (`data-k`, `data-u`, `data-v`), holding `path.ag-edge` and the arrowhead `path.ag-arrow` |
| `g.ag-edge-labels` | edge labels |
| `g.ag-nodes > g.ag-n` | one group per node (`data-k`), holding `path.ag-shape`, any inside label and a `<title>` tooltip |
| `g.ag-labels > g.ag-l` | side labels |
| `g.ag-legend` | legend blocks |
| `text.ag-caption` | the caption |

Emphasized nodes and edges also carry the class `ag-hl`. The `<desc>` element summarizes the drawing for screen readers:

```python
import xml.etree.ElementTree as ET

desc = ET.fromstring(svg).find("{http://www.w3.org/2000/svg}desc").text
print(desc)   # Undirected graph with 56 nodes and 83 edges, custom layout.
```

For programmatic access to the geometry, `fig.scene` holds every mark in pixel coordinates: nodes with their position, size, shape and colors, edges with their SVG path data, labels, legends and the canvas size. `fig.layout` is the pixel-space {py:class}`~aryagraph.layout.base.Layout` the drawing used.

```python
node = fig.scene.nodes[0]
print(node.node, node.shape, node.fill, round(node.w, 1))   # AB circle #eb6834 14.0
```

## Transparent backgrounds

`background="transparent"` leaves out the background rectangle, so the drawing sits directly on whatever is behind it: a slide, a colored page or a PDF layout. Choose a theme whose ink colors suit that background; `paper` suits white pages, `dark` and `blueprint` suit dark ones.

```python
clear = ag.draw(campus, layout=on_map, node_color="kind", background="transparent", theme="paper")
print('class="ag-bg"' in clear.to_svg())   # False
```

## Exporting charts

The charts in `ag.charts` (line charts, bar charts, histograms and Gantt charts) return a {py:class}`~aryagraph.charts.base.Chart` with the same `save()` method and extensions. The HTML version adds a crosshair tooltip that reads out values under the pointer.

```python
lengths = [m for _, _, m in campus.edges.data("length")]
chart = ag.charts.histogram(
    lengths,
    bins=12,
    title="Walking links by length",
    subtitle=f"{len(lengths)} edges of the campus graph",
    x_label="length (m)",
)
print(chart)   # <Chart histogram 560×260>
chart.save("figures/lengths.svg")
chart.save("figures/lengths.html")
```

```{figure} ../_static/generated/guide_core/exporting_histogram.png
:alt: Column histogram of campus walking-link lengths in meters; 80 of the 83 links are shorter than 200 m and the longest is about 626 m.
:width: 80%

A histogram of the campus graph's 83 edge lengths, exported as PNG. Charts use the same themes as graph drawings.
```

Like `Figure.save()`, `Chart.save()` creates missing folders. `chart.to_svg()` and `chart.to_html()` return the markup as strings. The [Charts](charts.md) page covers every chart type.

## Next steps

- [Interactive HTML](interactive.md): publishing the `.html` output on a website.
- [Styling](styling.md): the `paper` theme and custom fonts for publications.
- [Publication figures](../tutorials/publication-figures.md): a tutorial on preparing figures for papers.
