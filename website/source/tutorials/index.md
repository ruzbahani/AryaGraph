# Tutorials

Tutorials are step-by-step projects. Each one has a clear goal, runs from top to bottom, and ends with suggestions for
what to try next. Every code block on these pages is executed when the site is built, so what you see is what you
get.

## Foundations

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Your first network
:link: first-network
:link-type: doc

Build a small graph, then load the University of Calgary campus; inspect, measure, draw and save.
:::

:::{grid-item-card} Campus routing
:link: campus-routing
:link-type: doc

Shortest paths by length, indoor-only routes, alternative routes and the buildings that act as chokepoints.
:::

:::{grid-item-card} Pipelines as DAGs
:link: pipeline-dag
:link-type: doc

Model a machine-learning pipeline, find its critical path and schedule it on workers.
:::

:::{grid-item-card} Publication figures
:link: publication-figures
:link-type: doc

Choose a layout, encodings and theme, then export a figure ready for a paper.
:::
::::

## Dynamics and data

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Epidemics on a small-world network
:link: epidemics
:link-type: doc

SIR and SEIR outbreaks, discrete versus exact simulation, ensembles and animation.
:::

:::{grid-item-card} Communities in Les Misérables
:link: communities
:link-type: doc

Compare community detection methods on Les Misérables and draw the result.
:::

:::{grid-item-card} Influence and cascades
:link: influence
:link-type: doc

Estimate how far information spreads and choose the most influential seed nodes.
:::

:::{grid-item-card} Moving data in and out
:link: interop
:link-type: doc

Move data between AryaGraph, pandas, networkx, numpy, SciPy and common file formats.
:::
::::

```{tip}
Each page has an **Edit on GitHub** link. Corrections and new examples are welcome through pull requests on
[github.com/ruzbahani/AryaGraph](https://github.com/ruzbahani/AryaGraph).
```

```{toctree}
:hidden:
:caption: Foundations

first-network
campus-routing
pipeline-dag
publication-figures
```

```{toctree}
:hidden:
:caption: Dynamics and data

epidemics
communities
influence
interop
```
