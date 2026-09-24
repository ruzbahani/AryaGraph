# Styling

A theme decides every color, font and size in a drawing, from the background to the arrowheads. This page covers the four built-in themes, how to derive and register your own, the palettes and colormaps behind data encodings, and the accessibility choices built into all of them.

## Built-in themes

Pass a theme name to `draw()` with `theme=`:

| Theme | Background | Suited to |
|---|---|---|
| `light` (default) | warm off-white | screens, notebooks, slides with light backgrounds |
| `dark` | near-black | dark-mode pages and dashboards |
| `paper` | pure white, black text, slightly darker edges | print, PDF figures in papers and reports |
| `blueprint` | deep navy | dark slides and posters |

The same drawing in each theme:

```python
import aryagraph as ag

flo = ag.gen.florentine_families()
community = ag.alg.community_labels(ag.alg.louvain_communities(flo, seed=0))
between = ag.alg.betweenness_centrality(flo)

for name in ("light", "dark", "paper", "blueprint"):
    fig = ag.draw(
        flo,
        node_color=ag.by(community, kind="categorical", title="community"),
        node_size=ag.by(between, title="betweenness"),
        theme=name,
        title=f'theme="{name}"',
        subtitle="Florentine families",
    )
    fig.save(f"florentine_{name}.svg")
```

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item}
```{image} ../_static/generated/guide_core/styling_theme_light.png
:alt: Florentine families network on a light background, nodes colored by four communities and sized by betweenness.
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_core/styling_theme_dark.png
:alt: The same network on a near-black background with slightly adjusted community colors.
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_core/styling_theme_paper.png
:alt: The same network on pure white with black labels.
```
:::

:::{grid-item}
```{image} ../_static/generated/guide_core/styling_theme_blueprint.png
:alt: The same network on a deep navy background with pale blue labels and edges.
```
:::
::::

<p class="ag-caption">Florentine families (15 nodes), colored by Louvain community and sized by betweenness centrality, in the four built-in themes.</p>

Dark themes are tuned separately rather than computed by inverting the light theme. The categorical colors, for example, use their own dark-mode steps, and sequential colormaps run from dark to light so that low values recede into the background in either mode.

## What a theme controls

A {py:class}`~aryagraph.style.themes.Theme` is a frozen dataclass of named roles. Renderers ask the theme for a role (`surface`, `ink`, `edge`) instead of using fixed colors, so one theme restyles drawings, legends, charts and the interactive page together.

| Group | Fields |
|---|---|
| surfaces and text | `background` (page), `surface` (drawing area), `ink`, `ink_secondary`, `ink_muted`, `grid`, `axis`, `border` |
| graph marks | `node_fill` (default node color), `node_ring` (the thin ring that separates touching nodes; match it to `surface`), `node_ring_width`, `edge`, `edge_opacity`, `edge_highlight`, `dim_opacity` |
| data colors | `categorical` (palette slots), `other` (the "Other" and missing-value gray), `status`, `sequential` (colormap name), `diverging_pair` |
| typography | `font`, `mono`, `font_size`, `label_size`, `title_size`, `subtitle_size`, `label_weight`, `title_weight` |
| geometry | `node_size`, `edge_width`, `arrow_size`, `corner_radius`, `padding` |

{py:func}`ag.get_theme() <aryagraph.style.themes.get_theme>` returns a theme object by name, which is a convenient way to inspect the values:

```python
light = ag.get_theme("light")
print(light.surface, light.ink, light.edge, light.edge_highlight)
# #fcfcfb #0b0b0b #9d9b93 #2a78d6
print(light.font_size, light.label_size, light.title_size, light.padding)
# 12.0 11.5 17.0 24.0
```

## Custom themes

Themes are immutable. `with_()` returns a copy with some fields replaced, and {py:func}`ag.register_theme() <aryagraph.style.themes.register_theme>` makes it available by name to every `theme=` argument, including those of [charts](charts.md):

```python
harbor = ag.get_theme("light").with_(
    name="harbor",
    background="#eef3f7",
    surface="#f7fafc",
    node_ring="#f7fafc",
    ink="#10243a",
    ink_secondary="#3d5670",
    edge="#8fa3b8",
    edge_highlight="#d9480f",
    categorical=("#1c5d99", "#d9480f", "#2b8a3e", "#e8a100", "#862e9c", "#0b7285"),
    font='Georgia, "Times New Roman", serif',
    title_size=20,
    label_size=12.5,
)
ag.register_theme(harbor)

fig = ag.draw(
    flo,
    node_color=ag.by(community, kind="categorical", title="community"),
    node_size=ag.by(between, title="betweenness"),
    highlight_path=ag.alg.shortest_path(flo, "Acciaiuoli", "Strozzi"),
    theme="harbor",
    title="A custom theme",
    subtitle='theme="harbor", registered with ag.register_theme',
)
fig.save("harbor.svg")
print(fig.scene.theme.name)   # harbor
```

```{figure} ../_static/generated/guide_core/styling_custom_theme.png
:alt: The Florentine families network on a pale blue-gray background in a serif font, with a highlighted orange path from Acciaiuoli through Medici and Ridolfi to Strozzi.
:width: 80%

A theme derived from `light` with its own surfaces, ink, edge and highlight colors, a six-color palette and a serif font.
```

You can also pass the `Theme` object directly (`theme=harbor`) without registering it. Registering replaces any theme of the same name, so use a name of your own rather than one of the four built-ins.

A few practical notes:

- When you change `surface`, set `node_ring` to the same color, since the ring is meant to blend into the background between touching nodes.
- A custom `categorical` palette has as many slots as it has colors; more categories than that fold into "Other".
- The `font` value is a CSS font list. SVG and HTML output use the first font installed on the viewer's device; PNG and PDF export use the fonts of the machine that renders them.

## Palettes and colormaps

### The categorical palette

The default palette has eight colors, with separate steps for light and dark themes:

```python
print(ag.style.CATEGORICAL_LIGHT)
# ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
print(ag.style.CATEGORICAL_DARK)
# ('#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767')
print(light.other)   # #a3a199
```

Slots are assigned in order (blue, orange, aqua, yellow, magenta, green, violet, red) and are not reused, as described in [Drawing](drawing.md#categories-and-the-other-slot). Four status colors (`good`, `warning`, `serious`, `critical`) are kept apart from the palette in `theme.status`, so a state such as "critical" does not look like an ordinary category.

To use your own colors for one drawing, pass a list or a `{value: color}` dict as the palette:

```python
fig = ag.draw(
    flo,
    node_color=ag.by(community, kind="categorical",
                     palette={0: "#1c5d99", 1: "#d9480f", 2: "#2b8a3e", 3: "#e8a100"}),
)
print(sorted(set(m.fill for m in fig.scene.nodes)))
# ['#1c5d99', '#2b8a3e', '#d9480f', '#e8a100']
```

### Colormaps

Continuous scales use a colormap, named by `palette=` in `ag.by()` or by the theme's `sequential` field (`"blue"` in all built-in themes):

```{figure} ../_static/generated/guide_core/styling_colormaps.png
:alt: Horizontal color strips for fourteen colormaps: seven single-hue ramps, five perceptual scientific maps and two diverging maps.
:width: 85%

Colormaps as they render in light mode, sampled at 48 steps from low (left) to high (right).
```

| Family | Names | Notes |
|---|---|---|
| single-hue ramps | `blue`, `orange`, `aqua`, `green`, `violet`, `red`, `magenta` | light to dark in light mode, dark to light in dark mode; the steps closest to the surface color are left out so low values stay visible |
| scientific | `viridis`, `magma`, `inferno`, `plasma`, `cividis` | fixed in both modes |
| diverging | `blue_red` (also `"diverging"`), `blue_orange` | two hue arms through a neutral midpoint, dark steps in light mode and light steps in dark mode |

Append `_r` to any name to reverse it (`"viridis_r"`). {py:func}`ag.style.colormap() <aryagraph.style.palettes.colormap>` returns the {py:class}`~aryagraph.style.colors.Colormap` object for a name and mode, and {py:func}`ag.style.diverging() <aryagraph.style.palettes.diverging>` builds a diverging map from any two hue names. You can also define a colormap from your own stops; colors between stops are interpolated in OKLab:

```python
heat = ag.style.Colormap("heat", ("#fff5eb", "#fd8d3c", "#7f2704"))
print(heat(0.0), heat(0.5), heat(1.0))   # #fff5eb #fd8d3c #7f2704

web = ag.gen.barabasi_albert(120, 2, seed=3)
fig = ag.draw(web, node_color=ag.by(ag.alg.pagerank(web), palette=heat), labels=False)
fig.save("heat.svg")
```

## Accessibility

AryaGraph's defaults aim to keep drawings readable for people with color-vision deficiency (CVD), on projectors and in print, and for screen-reader users.

- **Color is ordered for CVD.** The eight categorical slots are ordered so that neighboring slots stay distinguishable under simulated protanopia, deuteranopia and tritanopia. Because slots are filled in order and not reused, a drawing with few categories uses the most distinct colors first, and a ninth category folds into "Other" instead of repeating a color.
- **Text is drawn in ink colors.** Labels, legend text, titles and tooltips use the theme's ink roles rather than data colors; the data color sits in the swatch beside the text. Side labels get a thin halo in the surface color so they stay legible over edges.
- **Labels inside colored nodes pick their own contrast.** White text is kept on fills where it reaches a contrast ratio of at least 3.8:1, and otherwise the higher-contrast of white and near-black is used.
- **Sequential ramps leave out the steps closest to the surface color** (the palest in light mode, the darkest in dark mode), so the lowest values remain visible.
- **Drawings carry text alternatives.** Every SVG has a `<title>` (the figure title, or the graph's name) and a `<desc>` stating the graph type, size and layout, each node carries a `<title>` with its label and attributes, and the interactive page adds a table of every node and its values.

The ink roles of the built-in themes have these contrast ratios against their drawing surface (WCAG 2 formula):

```python
for name in ("light", "dark", "paper", "blueprint"):
    t = ag.get_theme(name)
    ratios = [ag.style.contrast_ratio(ink, t.surface) for ink in (t.ink, t.ink_secondary, t.ink_muted)]
    print(f"{name:<10}" + "  ".join(f"{r:4.1f}" for r in ratios))
# light     19.2   7.7   3.5
# dark      17.4   9.7   4.8
# paper     21.0  10.9   3.6
# blueprint 14.2   9.2   5.2
```

`ink` and `ink_secondary` exceed the 4.5:1 ratio WCAG asks of body text in every theme. `ink_muted`, used for counts, tick labels and dimmed labels, is meant for secondary information and falls below 4.5:1 in the light and paper themes.

To choose readable text for your own colors, use {py:func}`~aryagraph.style.colors.contrast_ratio`, {py:func}`~aryagraph.style.colors.readable_on` and {py:func}`~aryagraph.style.colors.label_on`:

```python
print(ag.style.readable_on("#eda100"))                         # #0b0b0b
print(round(ag.style.contrast_ratio("#ffffff", "#2a78d6"), 2))  # 4.42
```

## Next steps

- [Drawing](drawing.md): the channels and scales that use these palettes.
- [Exporting](exporting.md): the `paper` theme and transparent backgrounds for print.
- [Languages](languages.md): fonts for Persian, Arabic, Hebrew and CJK labels.
