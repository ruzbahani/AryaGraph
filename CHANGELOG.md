# Changelog

All notable changes to AryaGraph are recorded here. The project follows [Semantic Versioning](https://semver.org/).

## 0.1.0 · Initial release (2026-09-23)

The first public version of AryaGraph.

- **Graph model:** `Graph`, `DiGraph` and a cycle-safe `DAG` with live node/edge views, attribute dicts, deterministic insertion order, and `NodeMap` / `EdgeMap` results.
- **Visualization:** `draw()` with data-driven encodings, automatic legends, overlap-avoiding label placement, edge clipping to node outlines, arrowheads, curved / flow / orthogonal edge styles, highlighting, four themes; SVG, interactive HTML, PNG and PDF output.
- **Interactive HTML:** pan & zoom, neighborhood highlighting, details panel, node dragging with live edge re-routing, search, legend filtering, table view, SVG/PNG download.
- **Layouts:** Sugiyama hierarchical (network simplex, Brandes–Köpf), tidy and radial trees, stress majorization, ForceAtlas2 with Barnes–Hut, Fruchterman–Reingold, spectral and geometric layouts, overlap removal, component packing.
- **Algorithms:** 130+ functions covering paths, connectivity, DAG and critical path, centrality, structure, communities, flow, spanning trees, matching, coloring, link prediction, spectra.
- **Analysis:** `analyze()` report as text, JSON or interactive dashboard.
- **Simulation:** compartmental epidemics (discrete and Gillespie), cascades and influence maximisation, random walks, opinion dynamics, heat diffusion, Kuramoto, DAG scheduling with failures and retries, Monte-Carlo PERT, ensembles, and animated playback.
- **Charts:** line, bar, histogram and Gantt charts.
- **Data & I/O:** generators, example DAGs, the University of Calgary campus dataset and classic network datasets; JSON, CSV/TSV, GraphML, GEXF, DOT, Mermaid; networkx, pandas, numpy and SciPy converters.
- **Command line:** `aryagraph draw | analyze | info | convert`.
- **Languages:** right-to-left (Persian, Arabic, Hebrew) and CJK labels measured and rendered correctly.
