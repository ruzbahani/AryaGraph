# Changelog

All notable changes to AryaGraph are recorded here. The project follows [Semantic Versioning](https://semver.org/).

## 0.1.1 · Bug fixes (2026-09-24)

A bug-fix release. Every public name and signature is unchanged; the one behavior change for existing code is the
`draw(tooltip=...)` entry below.

### Fixed

- PNG and PDF export fall back to a headless browser when cairosvg is installed but cannot load the Cairo library, instead of stopping with an `OSError`. When no browser is available either, the `DependencyError` names the Cairo problem.
- `DependencyError` suggests `pip install <package>` (for example `pip install cairosvg`) instead of `pip install 'aryagraph[png]'` or `'aryagraph[interop]'`, which needed a PyPI release of AryaGraph. The extra's name is available as the new `extra` attribute, and the headless-browser timeout message suggests `pip install cairosvg` too.
- The cycle in a `CycleError` raised by `DAG.add_edges`, `ag.DAG(edges)`, or a DAG algorithm or schedule simulation given a cyclic `DiGraph`, no longer depends on Python's hash seed: the search for it starts from the first node, in graph order, that could not be sorted.
- `Graph.subgraph()` given a set or frozenset lists the nodes in graph order, so the result is the same in every process. Other iterables keep their own order.
- The `gray` colormap is gray (OKLCH chroma about 0.01 at every step); it used to come out as a saturated olive (chroma up to 0.14). The same fix applies to `diverging()` with a gray arm and to `hue_ramp()` for colors with OKLCH chroma below 0.04.
- `Chart.save()` creates missing folders, like `Figure.save()`.
- `analyze()` adds a note when it skips closeness (above 3,000 nodes) or algebraic connectivity (above 1,500 nodes), and its dashboard breaks ties between community members in graph order.
- `draw(tooltip=...)` raises a `TypeError` that names the accepted values (an attribute name, a list of names, `True` or `False`) when given a callable, a mapping or a number. In 0.1.0 a callable or a number failed with "object is not iterable", and a mapping was read as the list of its keys; pass `list(mapping)` for that result.
- PDFs exported through a headless browser take the SVG's title as their document title, instead of "figure.html": the figure's title (the graph's name, or "Graph" when untitled) or the chart's title (its kind, such as "bar", when untitled). An SVG without a title passed to `svg_to_pdf()` gets the file name.
- The command-line tool writes UTF-8 when its output is redirected to a file or pipe with another encoding. On Windows, `aryagraph analyze` without `-o` used to fail there with a `UnicodeEncodeError`.

### Documentation

- Precise docstrings for the categorical palette, `layout.compute(components=None)`, node-size scaling and `project_plan()` lags.
- Docstrings of 22 functions, the `CompartmentalModel.terminates` property and 4 result classes restructured (Notes and Examples sections; one entry per returned value, exception and attribute), so generated API references no longer list remarks or examples as parameters. `from_mermaid` documents `text` and its return value.
- The `draw()` docstring describes label placement as the README does: the placer avoids overlaps and hides labels that cannot fit.
- The README no longer lists `tooltip` among the data channels (it selects which node attributes the tooltips list), and says that overlap removal applies to every layout in abstract units, including coordinates you supply.
- The package metadata, README and CITATION.cff link the Zenodo concept DOI 10.5281/zenodo.22928604.

### Packaging

- The license is declared as the SPDX expression `MIT` with `license-files`, the form current setuptools requires; building needs setuptools 77 or newer.
- The source distribution includes the changelog, the citation file, the contributing guide and the test helper `tests/nxcompat.py`, so the tests run from an unpacked sdist.

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
