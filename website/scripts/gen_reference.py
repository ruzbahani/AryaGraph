# =============================================================================
#
#      _                     ____                 _
#     / \   _ __ _   _  __ _ / ___|_ __ __ _ _ __ | |__
#    / _ \ | '__| | | |/ _` | |  _| '__/ _` | '_ \| '_ \
#   / ___ \| |  | |_| | (_| | |_| | | | (_| | |_) | | | |
#  /_/   \_\_|   \__, |\__,_|\____|_|  \__,_| .__/|_| |_|
#               |___/                     |_|
#
#          Graph & DAG visualization, analysis and simulation.
#
# -----------------------------------------------------------------------------
#  Copyright (c) 2026 Ali Mohammadi Ruzbahani
#  SPDX-License-Identifier: MIT
#
#  https://ruzbahani.com/aryagraph
# =============================================================================

"""Generate the API reference pages (source/reference/*.rst) from the package's ``__all__``.

Every public function and class gets its own page through autosummary, so the
reference always matches the installed code. Run before ``sphinx-build``
(``build_site.py`` does it for you).
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
ROOT = SITE.parent
OUT = SITE / "source" / "reference"
sys.path.insert(0, str(ROOT / "src"))

# (page slug, title, intro, [(section title, module, names or None for __all__)])
PAGES = [
    (
        "top-level",
        "Top-level API",
        "The functions most programs start from, importable directly from ``aryagraph``. Each is documented once, "
        "in its home module; this table links there.",
        [],
    ),
    (
        "core",
        "Graphs and results",
        "The graph classes, the result maps returned by algorithms, and the exception hierarchy (``aryagraph.core``).",
        [("Graph classes", "aryagraph.core.graph", ["Graph", "DiGraph"]),
         ("DAG", "aryagraph.core.dag", ["DAG"]),
         ("Result maps", "aryagraph.core.results", ["NodeMap", "EdgeMap"]),
         ("Exceptions", "aryagraph.core.exceptions", None),
         ("Utilities", "aryagraph.core.utils", ["make_rng", "weight_fn"])],
    ),
    (
        "algorithms",
        "Algorithms",
        "Every algorithm is also available as ``ag.alg.<name>``.",
        [(title, f"aryagraph.algorithms.{mod}", None) for title, mod in (
            ("Traversal", "traversal"), ("Shortest paths", "paths"), ("Connectivity", "connectivity"),
            ("DAGs and cycles", "dag"), ("Centrality", "centrality"), ("Structure", "structure"),
            ("Communities", "community"), ("Flow", "flow"), ("Spanning trees", "spanning"),
            ("Matching", "matching"), ("Coloring", "coloring"), ("Link prediction", "link_prediction"),
            ("Matrices and spectra", "matrix"))],
    ),
    (
        "layout",
        "Layouts",
        "Layout engines return a :class:`~aryagraph.layout.base.Layout`; ``compute`` dispatches by name.",
        [("Dispatcher", "aryagraph.layout", ["compute", "auto_method"]),
         ("Layout result", "aryagraph.layout.base", ["Layout", "pack_components"]),
         ("Hierarchical and trees", "aryagraph.layout", ["hierarchical", "tree", "radial"]),
         ("Force-directed and spectral", "aryagraph.layout", ["stress", "fruchterman_reingold", "force_atlas2", "spectral"]),
         ("Geometric", "aryagraph.layout", ["circular", "shell", "grid", "random", "spiral", "bipartite", "arc"]),
         ("Post-processing", "aryagraph.layout", ["remove_overlaps"])],
    ),
    (
        "render",
        "Rendering",
        "Scenes, figures, animation and export (``aryagraph.render``).",
        [("Drawing", "aryagraph.render", ["draw", "build_scene"]),
         ("Figure and scene", "aryagraph.render.figure", ["Figure"]),
         ("Scene model", "aryagraph.render.scene", ["Scene", "NodeMark", "EdgeMark", "LabelMark"]),
         ("Animation", "aryagraph.render.animate", ["animate"]),
         ("Export", "aryagraph.render.export", ["find_browser", "svg_to_png", "svg_to_pdf"])],
    ),
    (
        "style",
        "Styling",
        "Themes, colors, colormaps, encodings and node shapes (``aryagraph.style``).",
        [("Themes", "aryagraph.style.themes", ["Theme", "get_theme", "register_theme"]),
         ("Encodings", "aryagraph.style.scales", ["By", "Legend", "LegendEntry"]),
         ("Encoding shorthand", "aryagraph.style.scales", ["!by"]),
         ("Colors and colormaps", "aryagraph.style.colors", ["Colormap", "parse", "to_hex", "mix", "lighten", "darken", "with_alpha", "contrast_ratio", "readable_on", "label_on"]),
         ("Palettes", "aryagraph.style.palettes", ["colormap", "diverging", "available_colormaps"]),
         ("Shapes", "aryagraph.style.shapes", ["Shape", "get_shape"]),
         ("Text", "aryagraph.style.text", ["text_width", "wrap", "truncate", "is_rtl"])],
    ),
    (
        "charts",
        "Charts",
        "Small charts in the AryaGraph visual language (``aryagraph.charts``).",
        [("Charts", "aryagraph.charts", ["line_chart", "bar_chart", "histogram", "gantt"]),
         ("Chart object", "aryagraph.charts.base", ["Chart"])],
    ),
    (
        "sim",
        "Simulation",
        "Every simulator returns a :class:`~aryagraph.sim.base.SimulationResult` (``ag.sim``).",
        [("Results", "aryagraph.sim.base", ["SimulationResult"]),
         ("Result types", "aryagraph.sim.ensemble", ["EnsembleResult"]),
         ("Schedule results", "aryagraph.sim.scheduling", ["ScheduleResult", "MonteCarloResult", "TaskRun"]),
         ("Compartmental models", "aryagraph.sim.compartmental", ["CompartmentalModel", "SI", "SIS", "SIR", "SEIR", "SIRS", "SEIRD"]),
         ("One-call shortcuts", "aryagraph.sim.compartmental", ["!si", "!sis", "!sir", "!seir", "!sirs", "!seird"]),
         ("Cascades and influence", "aryagraph.sim", ["independent_cascade", "linear_threshold", "influence_spread", "greedy_influence_maximization"]),
         ("Walks, opinions and diffusion", "aryagraph.sim", ["random_walk", "stationary_distribution", "transition_matrix", "voter_model", "majority_rule", "degroot", "bounded_confidence", "heat_diffusion", "kuramoto"]),
         ("Scheduling and ensembles", "aryagraph.sim", ["simulate_schedule", "monte_carlo_schedule", "run_ensemble"])],
    ),
    (
        "analysis",
        "Analysis reports",
        "One-call analysis with text, data and dashboard outputs.",
        [("Reports", "aryagraph.analysis.report", ["analyze", "GraphReport"])],
    ),
    (
        "generators",
        "Generators and datasets",
        "Classic and random graphs, example DAGs and bundled datasets (``ag.gen``).",
        [(title, f"aryagraph.generators.{mod}", None) for title, mod in (
            ("Classic graphs", "classic"), ("Random graphs", "random"), ("Example DAGs", "dags"), ("Datasets", "datasets"))],
    ),
    (
        "io",
        "Files and interoperability",
        "Readers, writers and converters (``ag.io``); ``ag.read`` and ``ag.write`` pick the format from the extension.",
        [("By extension", "aryagraph.io", ["read", "write"]),
         ("JSON", "aryagraph.io.jsonio", None),
         ("Edge and adjacency lists", "aryagraph.io.edgelist", None),
         ("GraphML", "aryagraph.io.graphml", None),
         ("GEXF", "aryagraph.io.gexf", None),
         ("DOT", "aryagraph.io.dot", None),
         ("Mermaid", "aryagraph.io.mermaid", None),
         ("networkx, pandas, numpy and SciPy", "aryagraph.io.interop", None)],
    ),
]


TOP_LEVEL = [
    ("draw", "aryagraph.render.draw", "func", "Draw a graph and return a Figure."),
    ("animate", "aryagraph.render.animate.animate", "func", "Interactive playback of a simulation result."),
    ("analyze", "aryagraph.analysis.report.analyze", "func", "One-call analytical report."),
    ("read", "aryagraph.io.read", "func", "Read a graph file; the format follows the extension."),
    ("write", "aryagraph.io.write", "func", "Write a graph file; the format follows the extension."),
    ("by", "aryagraph.style.scales.by", "func", "Explicit encoding spec for a visual channel."),
    ("get_theme", "aryagraph.style.themes.get_theme", "func", "Look up a theme by name."),
    ("register_theme", "aryagraph.style.themes.register_theme", "func", "Make a custom theme available by name."),
    ("Figure", "aryagraph.render.figure.Figure", "class", "A rendered graph: export, display, inspect."),
    ("Layout", "aryagraph.layout.base.Layout", "class", "Node positions, edge routes and provenance."),
    ("GraphReport", "aryagraph.analysis.report.GraphReport", "class", "Result of analyze()."),
    ("Theme", "aryagraph.style.themes.Theme", "class", "A complete visual vocabulary."),
]
TOP_LEVEL_TABLE = [
    ".. list-table::",
    "   :header-rows: 1",
    "   :widths: 25 75",
    "",
    "   * - Name",
    "     - Purpose",
    *[line for name, target, role, what in TOP_LEVEL for line in (f"   * - :py:{role}:`ag.{name} <{target}>`", f"     - {what}")],
    "",
    "Namespaces: ``ag.alg`` (:doc:`algorithms`), ``ag.layout`` (:doc:`layout`), ``ag.sim`` (:doc:`sim`), "
    "``ag.gen`` (:doc:`generators`), ``ag.io`` (:doc:`io`), ``ag.style`` (:doc:`style`), ``ag.charts`` (:doc:`charts`).",
    "",
]


def public_callables(module: str, names: list[str] | None) -> list[str]:
    mod = importlib.import_module(module)
    names = list(names) if names is not None else list(getattr(mod, "__all__", []))
    out = []
    for name in names:
        obj = getattr(mod, name.lstrip("!"), None)
        if obj is None:
            raise SystemExit(f"{module}.{name} does not exist")
        if inspect.isfunction(obj) or inspect.isclass(obj):
            out.append(name)
    return out


def underline(text: str, char: str) -> str:
    return f"{text}\n{char * len(text)}\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    toc = []
    for slug, title, intro, sections in PAGES:
        parts = [underline(title, "="), intro, ""]
        if slug == "top-level":
            parts += TOP_LEVEL_TABLE
        for sec_title, module, names in sections:
            items = public_callables(module, names)
            if not items:
                continue
            total += len(items)
            parts += [underline(sec_title, "-"), f".. currentmodule:: {module}", "", ".. autosummary::", "   :toctree: generated", "   :nosignatures:", ""]
            listed = [n for n in items if not n.startswith("!")]
            inline = [n[1:] for n in items if n.startswith("!")]
            if listed:
                parts += [f"   {name}" for name in listed]
                parts.append("")
            else:
                parts = parts[: -4]  # drop the empty autosummary header, keep the blank line
            for name in inline:
                parts += [f".. autofunction:: {name}", ""]
        (OUT / f"{slug}.rst").write_text("\n".join(parts), encoding="utf-8", newline="\n")
        toc.append(slug)
    index = [
        underline("API reference", "="),
        "The reference is generated from the docstrings of AryaGraph "
        f"{__import__('aryagraph').__version__}. Every public function and class has its own page.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
        *[f"   {slug}" for slug in toc],
        "",
    ]
    (OUT / "index.rst").write_text("\n".join(index), encoding="utf-8", newline="\n")
    print(f"reference: {len(toc)} pages, {total} documented objects")


if __name__ == "__main__":
    main()
