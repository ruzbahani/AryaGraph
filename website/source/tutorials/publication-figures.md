# Publication figures

In this tutorial you turn a network drawing into a figure ready for a journal article: a layout chosen for the question, encodings that carry data, legends a reader can decode, the `paper` theme, a physical size that matches the page, and PNG, PDF and SVG exports. The example is the co-appearance network of *Les Misérables*.

**Goal:** produce a two-column figure (7 inches wide) whose text size, labels and resolution you have checked rather than assumed.

**Prerequisites:**

- [Your first network](first-network.md), or familiarity with `ag.draw` and its encodings.
- For PNG and PDF export: `cairosvg` or an installed Chrome, Edge or Chromium. SVG needs nothing extra.

## Step 1: Load the data and decide what the figure must show

{py:func}`ag.gen.les_miserables <aryagraph.generators.datasets.les_miserables>` returns the 77 characters of Victor Hugo's novel; two characters are linked when they appear in the same chapter, and the edge `weight` counts those chapters. The graph carries its source:

```python
import aryagraph as ag

les = ag.gen.les_miserables()
print(les)
print(les.attrs["citation"])
```

```text
<Graph 'Les Misérables': 77 nodes, 254 edges>
Character co-occurrence in Victor Hugo's novel; D. E. Knuth, The Stanford GraphBase (1993).
```

A figure should answer one question. Here it is: *which groups of characters belong together, and who holds each group together?* That calls for community detection (groups), a centrality measure (who matters) and link strength (how often characters meet).

```python
communities = ag.alg.louvain_communities(les, seed=0)
strength = les.degree(weight="weight")   # chapters shared, summed over all partners

group, leaders = {}, []
for members in communities:
    leader = max(sorted(members), key=strength.__getitem__)
    leaders.append(leader)
    for m in members:
        group[m] = leader
    print(f"{leader:<13}{len(members):>3} characters")
print(round(ag.alg.modularity(les, communities), 3))
```

```text
Valjean       22 characters
Enjolras      17 characters
Fantine       11 characters
Thenardier    11 characters
Myriel        10 characters
Champmathieu   6 characters
0.567
```

Louvain community detection finds six communities with a modularity of 0.567; `seed=0` makes the result reproducible. Naming each community after its most connected member gives the legend words instead of numbers, and the `leaders` list keeps the communities in the order Louvain returned them, largest first. Iterating over `sorted(members)` makes that choice deterministic when two members tie, as Champmathieu and the Judge do (14 shared chapters each).

## Step 2: Choose a layout

The layout decides what the eye sees first, so compare a few before styling anything. Draw the same encoding with three layouts and a fixed seed:

```python
for method in ("stress", "forceatlas2", "circular"):
    fig = ag.draw(les, layout=method, seed=0, node_color=ag.by(group, domain=leaders),
                  node_size=strength, labels=False, legend=False, theme="paper", title=method)
    fig.save(f"layout-{method}.svg")
```

::::{tab-set}

:::{tab-item} stress
```{image} ../_static/generated/tutorials_a/pub_layout_stress.png
:alt: Les Misérables network in a stress layout; the six communities form compact, mostly separate regions.
:width: 480px
```
:::

:::{tab-item} forceatlas2
```{image} ../_static/generated/tutorials_a/pub_layout_forceatlas2.png
:alt: Les Misérables network in a ForceAtlas2 layout; communities are pushed further apart and the periphery spreads out.
:width: 480px
```
:::

:::{tab-item} circular
```{image} ../_static/generated/tutorials_a/pub_layout_circular.png
:alt: Les Misérables network in a circular layout; edges cross the circle and the communities are hard to see.
:width: 480px
```
:::

::::

Stress majorization, the default for general graphs, places nodes so that distances in the drawing follow distances in the graph: the communities come out as compact regions, and their relative positions carry meaning. ForceAtlas2 pushes groups further apart, which separates clusters more clearly but distorts the distances between them. The circular layout buries the structure in edge crossings. This figure uses stress.

Record the layout method and seed in your methods section: layouts are reproducible for the same graph, parameters and seed, and `ag.draw` uses `seed=0` unless you pass another.

## Step 3: Encode the data

Map each part of the question to one visual channel, and give every channel a legend title a reader can understand without the text:

- color: the community, named after its most connected member instead of a number, with `domain=leaders` so the legend lists the largest community first and gives it the first palette color;
- node size: `strength`, the chapters a character shares with all others (sizes map to area, not diameter, between a minimum and a maximum node size);
- edge width: the chapters two characters share;
- labels: the 12 characters with the highest strength.

```python
top12 = [name for name, _ in strength.top(12)]
fig = ag.draw(
    les,
    node_color=ag.by(group, title="community", domain=leaders),
    node_size=ag.by(strength, title="chapters shared"),
    edge_width=ag.by("weight", title="chapters together"),
    labels={name: name for name in top12},
    theme="paper",
    width=672,
)
print(fig)
```

```text
<Figure 672×544: 77 nodes, 254 edges, stress layout, theme 'paper'>
```

`theme="paper"` uses a white background and black text, and keeps the default categorical palette, whose colors are ordered so that neighboring colors stay distinguishable under simulated color-vision deficiency. With more than eight categories, the seven most frequent keep their colors and the rest fold into "Other"; six communities fit without folding.

The figure has no title on purpose: in a paper, the caption does that job, and a title inside the image would repeat it.

## Step 4: Set the physical size and check the text

CSS defines a pixel as 1/96 inch, and AryaGraph's PDF export follows that definition, so `width=672` means 7 inches, a common width for a figure that spans two columns. AryaGraph fits the drawing into that width and, when the finished canvas with its legend comes out wider, scales the whole image down, text included. Check the result:

```python
shrink = fig.width / fig.scene.width
label_pt = fig.scene.theme.label_size * shrink * 0.75    # 1 px = 0.75 pt
legend_pt = fig.scene.theme.font_size * shrink * 0.75
print(f"{fig.width / 96:.2f} x {fig.height / 96:.2f} in; text scaled by {shrink:.3f}")
print(f"labels {label_pt:.1f} pt, legend text {legend_pt:.1f} pt")
```

```text
7.00 x 5.67 in; text scaled by 0.934
labels 8.1 pt, legend text 8.4 pt
```

At 7 inches the text is scaled by 0.934, so labels print at 8.1 pt and legend text at 8.4 pt. If your journal asks for larger text, raise `label_size`, or keep legend titles short: a longer title widens the legend, and the whole image shrinks to make room. With the legend title "community (most connected member)" instead of "community", the same call scales the text by 0.833 and the labels drop to 7.2 pt.

Labels that would overlap a node or another label are hidden rather than drawn on top of each other. In an interactive view that is harmless, because hidden labels stay available on hover; on paper they are gone. List them:

```python
def hidden_labels(figure):
    return [m.node for m in figure.scene.nodes
            if m.label is not None and not m.label.visible]


print(hidden_labels(fig))
```

```text
['Combeferre', 'Bossuet']
```

Two of the twelve labels, Combeferre and Bossuet, do not fit beside their nodes in the crowded Enjolras community. You can remove them from the label set and say in the caption that the most connected characters are labeled where space allows, or pass `label_collisions="show"` and check the overlaps by eye. The figure below keeps them hidden, so ten names appear.

Why not a single column? `width` fits the *layout* into the width you ask for, while node and text sizes stay the same, so a much narrower figure gets crowded. Try 3.5 inches:

```python
narrow = ag.draw(
    les,
    node_color=ag.by(group, title="community", domain=leaders),
    node_size=ag.by(strength, title="chapters shared"),
    edge_width=ag.by("weight", title="chapters together"),
    labels={name: name for name in top12},
    theme="paper",
    width=336,
)
print(len(hidden_labels(narrow)), hidden_labels(narrow))
```

```text
8 ['Valjean', 'Fantine', 'Thenardier', 'Cosette', 'Javert', 'Marius', 'Enjolras', 'Combeferre']
```

Eight of the twelve labels disappear, Valjean's among them. For a single column, label fewer nodes, draw a smaller graph, or let the figure span both columns as this one does.

## Step 5: Export PNG, PDF and SVG

```python
fig.save("lesmis.svg")                  # vector, editable in vector-graphics software
fig.save("lesmis.pdf")                  # vector, 7 in wide
fig.save("lesmis.png", scale=300 / 96)  # raster, 300 pixels per inch at 7 in wide
```

The PNG `scale` multiplies the display size, so pixels per inch equal 96 × `scale` when the figure is printed at its intended width. The default `scale=2.0` gives 192 per inch, suited to screens and slides; `300 / 96` gives 300. Check the files:

```python
import re
import struct

with open("lesmis.png", "rb") as f:
    width_px, height_px = struct.unpack(">II", f.read(24)[16:24])
print(width_px, height_px, "pixels ->", round(width_px / (fig.width / 96)), "per inch")

pdf = open("lesmis.pdf", "rb").read()
print(re.search(rb"/MediaBox\s*\[([^\]]*)\]", pdf).group(1).decode(), "pt")
```

```text
2100 1700 pixels -> 300 per inch
0 0 504 408 pt
```

The PNG is 2,100 pixels wide, 300 per inch at 7 inches. The PDF page is 504 × 408 points (72 points per inch), which is 7 × 5.67 inches. Both follow from `fig.width` and `fig.height`, so the sizes you checked in Step 4 are the sizes you get.

```{figure} ../_static/generated/tutorials_a/pub_lesmis.png
:alt: Les Misérables co-appearance network in the paper theme, colored by six communities named Valjean, Enjolras, Fantine, Thenardier, Myriel and Champmathieu, with node size showing chapters shared and edge width showing chapters together; ten characters are labeled.
:width: 672px

The finished figure, 7 inches wide. A caption for a paper could read: "Co-appearance network of *Les Misérables* (77 characters; Knuth, 1993). Colors: Louvain communities (seed 0), each named after its most connected member. Node area: chapters shared with all other characters. Edge width: chapters in which two characters appear together. Layout: stress majorization, seed 0."
```

<p class="ag-embed-note">Files from this page: <a href="../_static/generated/tutorials_a/pub_lesmis.pdf">PDF</a> · <a href="../_static/generated/tutorials_a/pub_lesmis.svg">SVG</a> · <a href="../_static/generated/tutorials_a/pub_lesmis_300dpi.png">PNG at 300 per inch</a></p>

## Checklist

Before you submit, go through the figure once more:

- [ ] **One question.** The figure answers a single question, and the caption states it.
- [ ] **Layout.** You compared layouts and chose one for the question; the method and seed appear in the caption or methods section, so anyone can reproduce it.
- [ ] **Encodings.** Every data-driven channel has a legend with a meaningful title; categories have names, not numbers.
- [ ] **Sizes.** Size maps to area, not diameter. When size carries data, labels sit outside the nodes (a label drawn inside a node enlarges it to fit the text).
- [ ] **Text.** Labels and legend text meet your journal's minimum size at the final printed width; compute it from `fig.width / fig.scene.width`.
- [ ] **Labels.** `hidden_labels(fig)` is empty, or the caption says that only some nodes are labeled.
- [ ] **Theme.** `theme="paper"` for print; no title inside the image when the caption carries it.
- [ ] **Formats.** PDF or SVG where the journal accepts vector files; otherwise PNG with `scale` set for the required resolution.
- [ ] **Sources.** The data source is cited (here `les.attrs["citation"]`; for the campus dataset, "© OpenStreetMap contributors (ODbL)").
- [ ] **Script.** The code that made the figure is saved next to it, so you can regenerate it after review.

## Recap

You have:

- turned a question into encodings: communities to color, strength to size, co-appearances to edge width;
- compared layouts at a fixed seed and picked one;
- set a physical width, computed the resulting text size in points and found hidden labels through {py:class}`~aryagraph.render.scene.Scene`;
- exported SVG, PDF and a 300-per-inch PNG, and verified their sizes.

## Next steps

- [Styling](../user-guide/styling.md): themes, palettes and custom {py:class}`~aryagraph.style.themes.Theme` objects.
- [Exporting](../user-guide/exporting.md): file formats and export options.
- [Layouts](../user-guide/layouts.md): the layout engines and their options.
- Tutorial: [Communities in Les Misérables](communities.md).
