# User guide

The user guide explains every part of AryaGraph with runnable examples. Each page stands on its own, so you can read
it front to back or jump to the topic you need. For signatures and parameter details, see the
[API reference](../reference/index.rst).

## Graphs and figures

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Graphs
:link: graphs
:link-type: doc

Building graphs, attributes and views, and how the `DAG` class rejects edges that would close a cycle.
:::

:::{grid-item-card} Drawing
:link: drawing
:link-type: doc

`draw()` and its channels: colors, sizes, shapes, labels, edge styles, highlighting and legends.
:::

:::{grid-item-card} Styling
:link: styling
:link-type: doc

Themes, palettes and colormaps, custom themes and accessible color choices.
:::

:::{grid-item-card} Interactive HTML
:link: interactive
:link-type: doc

The HTML view: zoom, neighborhoods, dragging, search, legend filters and the table view.
:::

:::{grid-item-card} Exporting
:link: exporting
:link-type: doc

SVG, HTML, PNG and PDF output, sizes, and the tools each format needs.
:::

:::{grid-item-card} Languages
:link: languages
:link-type: doc

Persian and other right-to-left scripts, and CJK labels.
:::
::::

## Layouts, algorithms and analysis

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Layouts
:link: layouts
:link-type: doc

Every layout engine, when to use it, and how to pass your own coordinates.
:::

:::{grid-item-card} Algorithms
:link: algorithms
:link-type: doc

A guided tour of `ag.alg`: paths, connectivity, centrality, communities and more.
:::

:::{grid-item-card} Analysis reports
:link: analysis
:link-type: doc

`analyze()` as text, data and an interactive dashboard.
:::

:::{grid-item-card} Charts
:link: charts
:link-type: doc

Line, bar, histogram and Gantt charts in the same visual language.
:::
::::

## Simulation, data and tools

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Simulation
:link: simulation
:link-type: doc

Epidemics, cascades, walks, opinion dynamics and diffusion, with animated playback.
:::

:::{grid-item-card} DAG workflows
:link: dags
:link-type: doc

Critical paths, scheduling on workers, Monte Carlo risk and drawing pipelines.
:::

:::{grid-item-card} Datasets and generators
:link: datasets
:link-type: doc

The University of Calgary campus, classic networks, example DAGs and graph generators.
:::

:::{grid-item-card} File formats and interoperability
:link: io
:link-type: doc

JSON, CSV, GraphML, GEXF, DOT and Mermaid; networkx, pandas, numpy and SciPy.
:::

:::{grid-item-card} Command line
:link: cli
:link-type: doc

`aryagraph draw`, `analyze`, `info` and `convert`.
:::

:::{grid-item-card} Reproducibility and performance
:link: reproducibility
:link-type: doc

Seeds, ordering, what is deterministic, and practical size limits.
:::
::::

```{toctree}
:hidden:
:caption: Graphs and figures

graphs
drawing
styling
interactive
exporting
languages
```

```{toctree}
:hidden:
:caption: Layouts, algorithms and analysis

layouts
algorithms
analysis
charts
```

```{toctree}
:hidden:
:caption: Simulation, data and tools

simulation
dags
datasets
io
cli
reproducibility
```
