# Changelog

All notable changes to AryaGraph are recorded here. The project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed

- PNG and PDF export fall back to a headless browser when cairosvg is installed but the Cairo library is missing, instead of stopping with an `OSError`.
- The message for a missing optional package suggests `pip install <package>`, which works without a PyPI release of AryaGraph.
- The cycle in a `CycleError` raised by `DAG.add_edges`, `ag.DAG(edges)` or a DAG algorithm given a cyclic `DiGraph` no longer depends on Python's hash seed: it starts from the first node, in graph order, that could not be sorted.
- `Graph.subgraph()` given a set lists the nodes in graph order, so the result is the same in every process.
- The `gray` colormap is gray; it used to come out as a saturated olive.
- `Chart.save()` creates missing folders, like `Figure.save()`.
- `analyze()` adds a note when it skips closeness (above 3,000 nodes) or algebraic connectivity (above 1,500 nodes), and its dashboard breaks ties between community members in graph order.
- `draw(tooltip=...)` names the accepted values when given a callable or a mapping.

### Documentation

- Precise docstrings for the categorical palette, `layout.compute(components=None)`, node-size scaling and `project_plan()` lags.

## 0.1.0 · Initial release (2026-09-23)

The first public version of AryaGraph.

- **Graph model:** `Graph`, `DiGraph` and a cycle-safe `DAG` with live node/edge views, attribute dicts, deterministic insertion order, and `NodeMap` / `EdgeMap` results.
- **Visualization:** `draw()` with data-driven encodings, automatic legends, overlap-avoiding label placement, edge clipping to node outlines, arrowheads, curved / flow / orthogonal edge styles, highlighting, four themes; SVG, interactive HTML, PNG and PDF output.
- **Interactive HTML:** pan & zoom, neighborhood highlighting, details panel, node dragging with live edge re-routing, search, legend filtering, table view, SVG/PNG download.
- **Layouts:** Sugiyama hierarchical (network simplex, Brandes–Köpf), tidy and radial trees, stress majorization, ForceAtlas2 with Barnes–Hut, Fruchterman–Reingold, spectral and geometric layouts, overlap removal, component packing.
- **Algorithms:** 130+ functions covering paths, connectivity, DAG and critical path, centrality, structure, communities, flow, spanning trees, matching, coloring, link prediction, spectra.
- **Analysis:** `analyze()` report as text, JSON or interactive dashboard.
- **Simulation:** compartmental epidemics (discrete and Gillespie), cascades and influence maximization, random walks, opinion dynamics, heat diffusion, Kuramoto, DAG scheduling with failures and retries, Monte-Carlo PERT, ensembles, and animated playback.
- **Charts:** line, bar, histogram and Gantt charts.
- **Data & I/O:** generators, example DAGs, the University of Calgary campus dataset and classic network datasets; JSON, CSV/TSV, GraphML, GEXF, DOT, Mermaid; networkx, pandas, numpy and SciPy converters.
- **Command line:** `aryagraph draw | analyze | info | convert`.
- **Languages:** labels in right-to-left scripts (Persian, Arabic, Hebrew) and in CJK are measured per script for layout and label placement, and right-to-left text is written in its own direction.
