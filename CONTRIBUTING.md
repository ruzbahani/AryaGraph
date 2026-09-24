# Contributing to AryaGraph

AryaGraph is a self-contained graph library: its own data structures, algorithms,
layout engines, renderer and simulators. **The only runtime dependency is
numpy.** networkx is used in the test-suite as an *oracle* (to check that our
answers are right), never by the library itself. pandas / scipy / networkx
interop lives behind lazy imports that raise `DependencyError` when missing.

## Layout of the source tree

```
src/aryagraph/
  core/         Graph, DiGraph, DAG, views, NodeMap/EdgeMap, exceptions, utils
  algorithms/   traversal, paths, connectivity, dag, centrality, structure,
                community, flow, spanning, matching, coloring, link prediction,
                matrix; _core.py holds shared primitives (components, BFS/Dijkstra distances)
  layout/       Layout (base.py) + force, stress, spectral, geometric, tree,
                hierarchical (Sugiyama), overlap removal, and the compute() dispatcher
  style/        colors, palettes, themes, scales, text metrics, node shapes
  render/       scene builder, SVG backend, interactive HTML backend, export
  charts/       small SVG charts (line, bar, histogram, gantt) used by reports & simulations
  sim/          compartmental epidemics, cascades, walks, opinion dynamics, diffusion, DAG scheduling
  generators/   classic & random graphs, example DAGs, bundled datasets
  io/           JSON, edge lists, GraphML, DOT, Mermaid, networkx/pandas/scipy interop
  analysis/     the analyze() report
```

## Rules of the code base

1. **Python ≥ 3.10**, `from __future__ import annotations` at the top of every module,
   full type hints on public functions.
2. **Docstrings** in numpy style, concise: one-line summary, what matters about
   parameters/returns, complexity when non-obvious, a short example where it helps.
   Comments explain *why*, not *what*. Match the tone of `core/graph.py`.
3. **Determinism.** Iterate in graph order (insertion order). Every source of
   randomness takes a `seed` argument and goes through `aryagraph.core.utils.make_rng(seed)`.
   Layouts default to `seed=0` so the same graph always draws the same way.
4. **Results.** Per-node results are `NodeMap(dict, name=...)`, per-edge results are
   `EdgeMap`. Structured results are frozen-ish `@dataclass`es with a helpful `__repr__`.
5. **Errors** come from `aryagraph.core.exceptions` (`NodeNotFound`, `CycleError`,
   `GraphTypeError`, `NoPath`, `NegativeCycleError`, `ConvergenceError`, …). Never
   return sentinel values for failure.
6. **Weights.** A `weight` argument is `None` (unweighted), an edge-attribute name
   (missing attribute ⇒ 1), or a callable `f(u, v, attrs) -> float`; normalise it with
   `aryagraph.core.utils.weight_fn`.
7. **Graph access.** Public API: `g.nodes`, `g.edges`, `g.adj`, `g.succ`, `g.pred`,
   `g.directed`, `g.degree()`, … Inside the library, hot loops may read the internal
   dicts `g._node`, `g._succ`, `g._pred` directly (read-only!). For undirected graphs
   `g._pred is g._succ`.
8. **Semantics follow networkx where it has an established definition** (normalisation
   of betweenness, PageRank dangling handling, …) so results can be cross-checked, and
   any deliberate difference is documented in the docstring.
9. **Tests** live in `tests/test_<area>.py`, use pytest, and compare against networkx on
   several seeded random graphs plus hand-checked small cases and edge cases (empty
   graph, single node, disconnected, self-loops, directed vs undirected).
   Run with `python -m pytest` from the project root (`pyproject.toml` puts `src` on the path).

## Coordinate convention

Screen coordinates everywhere: x to the right, **y downward**. See `layout/base.py`.
