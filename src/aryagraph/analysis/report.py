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

"""``analyze(g)``: a complete analytical report of a graph.

The report computes what is well-defined and affordable for the graph at hand
(exact algorithms up to a few thousand nodes, sampling beyond), records why
anything was skipped instead of failing, and presents the result three ways:
a text summary (``print(report)``), plain data (``report.to_dict()``), and a
self-contained interactive dashboard (``report.save("report.html")``).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Callable, Hashable

from ..core.results import NodeMap
from ..style.numbers import fmt_compact, fmt_number
from ..style.themes import Theme, get_theme

Node = Hashable

EXACT_LIMIT = 3000  # exact all-pairs measures up to this many nodes


@dataclass
class GraphReport:
    """Everything :func:`analyze` found. Metrics are plain values / :class:`NodeMap` objects."""

    graph: Any
    summary: dict[str, Any] = field(default_factory=dict)
    centrality: dict[str, NodeMap] = field(default_factory=dict)
    communities: list[set] | None = None
    community_of: NodeMap | None = None
    modularity: float | None = None
    structure: dict[str, Any] = field(default_factory=dict)
    dag: dict[str, Any] | None = None
    degree_histogram: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    theme: Theme | None = None

    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self.graph.name or ("DAG" if self.dag else "graph")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the report (NodeMaps become ``{str(node): value}``)."""

        def conv(v: Any) -> Any:
            if isinstance(v, dict):
                return {str(k): conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple, set, frozenset)):
                return [conv(x) for x in (sorted(v, key=str) if isinstance(v, (set, frozenset)) else v)]
            if isinstance(v, float) and not math.isfinite(v):
                return None
            if hasattr(v, "item"):
                return v.item()
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            return str(v)

        return conv(
            {
                "name": self.name,
                "summary": self.summary,
                "centrality": {k: dict(v) for k, v in self.centrality.items()},
                "communities": [sorted(c, key=str) for c in self.communities] if self.communities else None,
                "modularity": self.modularity,
                "structure": self.structure,
                "dag": self.dag,
                "degree_histogram": self.degree_histogram,
                "notes": self.notes,
                "timings": self.timings,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        s = self.summary
        rows: list[tuple[str, str]] = []
        kind = "Directed" if s.get("directed") else "Undirected"
        if self.dag is not None:
            kind = "Directed acyclic"
        rows.append(("Graph", f"{kind} · {s.get('n', 0):,} nodes · {s.get('m', 0):,} edges"))
        for key, label in (("density", "Density"), ("avg_degree", "Average degree"), ("max_degree", "Max degree")):
            if key in s:
                rows.append((label, fmt_number(s[key])))
        if "components" in s:
            lcc = s.get("largest_component", 0)
            share = lcc / s["n"] if s.get("n") else 0
            rows.append(("Components", f"{s['components']:,} (largest: {lcc:,} nodes, {share:.0%})"))
        if "strong_components" in s:
            rows.append(("Strong components", f"{s['strong_components']:,}"))
        for key, label in (
            ("diameter", "Diameter"),
            ("radius", "Radius"),
            ("avg_path_length", "Avg path length"),
            ("avg_clustering", "Avg clustering"),
            ("transitivity", "Transitivity"),
            ("assortativity", "Degree assortativity"),
            ("reciprocity", "Reciprocity"),
            ("algebraic_connectivity", "Algebraic connectivity"),
        ):
            if s.get(key) is not None:
                suffix = " (largest component)" if key in ("diameter", "radius", "avg_path_length") and s.get("components", 1) > 1 else ""
                rows.append((label, fmt_number(s[key]) + suffix))
        if self.communities is not None:
            rows.append(("Communities", f"{len(self.communities)} (modularity {fmt_number(self.modularity or 0.0)})"))
        for name, nm in self.centrality.items():
            top = ", ".join(f"{k} ({fmt_number(v)})" for k, v in nm.top(3))
            rows.append((f"Top {name}", top))
        st = self.structure
        if "bridges" in st:
            rows.append(("Bridges / cut nodes", f"{len(st['bridges'])} / {len(st['articulation_points'])}"))
        if "max_core" in st:
            rows.append(("Max k-core", str(st["max_core"])))
        if self.dag is not None:
            d = self.dag
            rows.append(("DAG depth / width", f"{d.get('depth')} levels / {d.get('width', 'n/a')} parallel"))
            rows.append(("Sources / sinks", f"{len(d.get('sources', []))} / {len(d.get('sinks', []))}"))
            if d.get("critical_path"):
                rows.append(("Critical path", " → ".join(map(str, d["critical_path"])) + f"  ({fmt_number(d['critical_length'])})"))
            elif d.get("longest_path"):
                rows.append(("Longest path", " → ".join(map(str, d["longest_path"]))))
            if d.get("redundant_edges") is not None:
                rows.append(("Redundant edges", str(d["redundant_edges"])))
        width = max(len(k) for k, _ in rows) + 2
        title = f"Graph report: {self.name}"
        lines = [title, "─" * max(len(title), 40)]
        lines += [f"{k:<{width}}{v}" for k, v in rows]
        if self.notes:
            lines.append("")
            lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<GraphReport {self.name!r}: {self.summary.get('n', 0)} nodes, {self.summary.get('m', 0)} edges>"

    # ------------------------------------------------------------------ #
    def figure(self, **kwargs: Any):
        """The report's graph drawing (communities colored, PageRank sized, metrics in tooltips)."""
        from ..render import draw
        from ..style.scales import by

        g = self.graph.copy()
        for name, nm in self.centrality.items():
            if name == "degree":  # already shown by every tooltip
                continue
            for n, v in nm.items():
                g.nodes[n][name] = round(float(v), 5) if isinstance(v, float) else v
        if self.community_of is not None:
            for n, c in self.community_of.items():
                g.nodes[n]["community"] = c
        opts: dict[str, Any] = {"theme": self.theme}
        if self.dag is not None:
            if self.dag.get("critical_path"):
                opts["highlight_path"] = self.dag["critical_path"]
        else:
            if self.community_of is not None and len(self.communities or []) > 1:
                opts["node_color"] = by("community", kind="categorical")
            if "pagerank" in self.centrality and len(g) > 1:
                opts["node_size"] = "pagerank"
        opts.update(kwargs)
        return draw(g, **opts)

    def to_html(self, *, title: str | None = None) -> str:
        return _report_html(self, title)

    def save(self, path: str | Path) -> Path:
        """Write ``.html`` (dashboard), ``.json`` (data), ``.txt``/``.md`` (text summary)."""
        path = Path(path)
        ext = path.suffix.lower()
        if ext in (".html", ".htm"):
            path.write_text(self.to_html(), encoding="utf-8")
        elif ext == ".json":
            path.write_text(self.to_json(), encoding="utf-8")
        elif ext in (".txt", ".md"):
            text = str(self)
            path.write_text(f"```\n{text}\n```\n" if ext == ".md" else text + "\n", encoding="utf-8")
        else:
            raise ValueError(f"unsupported extension {ext!r}; use .html, .json, .txt or .md")
        return path

    def _repr_html_(self) -> str:
        page = self.to_html()
        return f'<iframe srcdoc="{escape(page, quote=True)}" style="width:100%;height:900px;border:0" sandbox="allow-scripts allow-downloads"></iframe>'


# --------------------------------------------------------------------------- #
# computation
# --------------------------------------------------------------------------- #
def analyze(
    g: Any,
    *,
    weight: str | None = None,
    duration: str | None = "duration",
    communities: bool = True,
    centrality: bool = True,
    seed: int = 0,
    theme: Any = "light",
) -> GraphReport:
    """Analyse *g* and return a :class:`GraphReport`.

    Parameters
    ----------
    weight:
        Edge attribute used as distance/strength where meaningful (None = unweighted).
    duration:
        Node attribute holding task durations; enables critical-path analysis for DAGs.
    communities, centrality:
        Switch the heavier sections off for very large graphs.
    """
    from ..algorithms import centrality as C
    from ..algorithms import community as CM
    from ..algorithms import connectivity as CN
    from ..algorithms import structure as S
    from ..algorithms._core import components

    rep = GraphReport(graph=g, theme=get_theme(theme))
    n, m = len(g), g.num_edges

    def step(name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as exc:  # the report must never fail as a whole
            rep.notes.append(f"{name} skipped: {exc}")
            return None
        finally:
            rep.timings[name] = round(time.perf_counter() - t0, 4)

    base = step("summary", lambda: S.summary(g)) or {"n": n, "m": m, "directed": g.directed}
    s: dict[str, Any] = {
        "n": n,
        "m": m,
        "directed": g.directed,
        "density": base.get("density"),
        "avg_degree": base.get("avg_degree"),
        "max_degree": base.get("max_degree"),
        "self_loops": base.get("self_loops"),
    }
    comps = step("components", lambda: components(g)) or []
    s["components"] = len(comps)
    lcc_nodes = max(comps, key=len) if comps else []
    s["largest_component"] = len(lcc_nodes)
    und = g.to_undirected() if g.directed else g
    lcc = und.subgraph(lcc_nodes) if len(comps) > 1 else und
    if g.directed:
        s["strong_components"] = len(step("strong components", lambda: CN.strongly_connected_components(g)) or [])
        s["reciprocity"] = step("reciprocity", lambda: S.reciprocity(g)) if m else None
    if n and len(lcc) <= EXACT_LIMIT:
        s["diameter"] = step("diameter", lambda: S.diameter(lcc))
        s["radius"] = step("radius", lambda: S.radius(lcc))
        s["avg_path_length"] = step("average path length", lambda: S.average_shortest_path_length(lcc))
    elif n:
        rep.notes.append(f"distance measures skipped: largest component has {len(lcc):,} nodes (> {EXACT_LIMIT:,})")
    s["avg_clustering"] = step("clustering", lambda: S.average_clustering(und))
    s["transitivity"] = step("transitivity", lambda: S.transitivity(und))
    if m:
        s["assortativity"] = step("assortativity", lambda: S.degree_assortativity(g))
    if 2 < n <= 1500:
        from ..algorithms.matrix import algebraic_connectivity

        s["algebraic_connectivity"] = step("algebraic connectivity", lambda: algebraic_connectivity(und))
    elif n > 1500:
        rep.notes.append(f"algebraic connectivity skipped: {n:,} nodes (> 1,500)")
    rep.degree_histogram = step("degree histogram", lambda: S.degree_histogram(g)) or []
    rep.summary = {k: (float(v) if isinstance(v, float) else v) for k, v in s.items()}

    if centrality and n:
        cent: dict[str, NodeMap] = {}
        cent["degree"] = NodeMap(g.degree(), name="degree")
        pr = step("pagerank", lambda: C.pagerank(g, weight=weight))
        if pr is not None:
            cent["pagerank"] = pr
        if n <= EXACT_LIMIT:
            bt = step("betweenness", lambda: C.betweenness_centrality(g, weight=weight))
        else:
            k = min(500, n)
            bt = step("betweenness (sampled)", lambda: C.betweenness_centrality(g, weight=weight, k=k, seed=seed))
            rep.notes.append(f"betweenness estimated from {k} sampled sources")
        if bt is not None:
            cent["betweenness"] = bt
        if n <= EXACT_LIMIT:
            cl = step("closeness", lambda: C.closeness_centrality(g, weight=weight))
            if cl is not None:
                cent["closeness"] = cl
        else:
            rep.notes.append(f"closeness skipped: {n:,} nodes (> {EXACT_LIMIT:,})")
        if not g.directed or not (g.is_dag() if hasattr(g, "is_dag") else False):
            ev = step("eigenvector", lambda: C.eigenvector_centrality(g, weight=weight))
            if ev is not None:
                cent["eigenvector"] = ev
        for name, nm in cent.items():
            nm.name = name
        rep.centrality = cent

    if communities and n > 1 and m > 0 and not (g.directed and g.is_dag()):
        parts = step("communities", lambda: CM.louvain_communities(und, weight=weight or "weight", seed=seed))
        if parts:
            rep.communities = parts
            rep.community_of = CM.community_labels(parts)
            rep.modularity = step("modularity", lambda: CM.modularity(und, parts, weight=weight or "weight"))

    st: dict[str, Any] = {}
    if n:
        core = step("k-core", lambda: S.core_number(g))
        if core:
            st["max_core"] = max(core.values()) if core else 0
            st["core_number"] = core
        if not g.directed:
            br = step("bridges", lambda: CN.bridges(g))
            ap = step("articulation points", lambda: CN.articulation_points(g))
            if br is not None and ap is not None:
                st["bridges"] = br
                st["articulation_points"] = ap
    rep.structure = st

    if g.directed and n and g.is_dag():
        rep.dag = step("dag", lambda: _dag_section(g, duration)) or {}
    return rep


def _dag_section(g: Any, duration: str | None) -> dict[str, Any]:
    from ..algorithms import dag as D
    from ..core.dag import DAG

    dag = g if isinstance(g, DAG) else DAG(g)
    gens = dag.generations()
    out: dict[str, Any] = {
        "depth": len(gens),
        "level_sizes": [len(x) for x in gens],
        "sources": dag.sources(),
        "sinks": dag.sinks(),
        "longest_path": D.dag_longest_path(dag),
    }
    if len(dag) <= 1500:
        out["width"] = D.dag_width(dag)
    if len(dag) <= 2000:
        red = D.transitive_reduction(dag)
        out["redundant_edges"] = dag.num_edges - red.num_edges
    if duration and any(duration in d for _, d in dag.nodes.data()):
        cp = D.critical_path(dag, duration=duration)
        out["critical_path"] = list(cp.path)
        out["critical_length"] = float(cp.length)
        out["slack"] = dict(cp.slack)
        out["schedule"] = [
            {"node": n, "worker": str(n), "start": float(cp.earliest_start[n]), "end": float(cp.earliest_finish[n])}
            for n in dag.topological_order()
        ]
    return out


# --------------------------------------------------------------------------- #
# HTML dashboard
# --------------------------------------------------------------------------- #
_REPORT_CSS = """
.agr{max-width:1120px;margin:0 auto;padding:28px 20px 40px;font-family:var(--ag-font);color:var(--ag-ink)}
.agr h1{font-size:24px;font-weight:600;letter-spacing:-0.01em;margin:0 0 4px}
.agr .sub{color:var(--ag-ink-2);font-size:13.5px;margin-bottom:22px}
.agr h2{font-size:15px;font-weight:600;margin:30px 0 10px}
.agr .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(122px,1fr));gap:10px}
.agr .tile{background:var(--ag-surface);border:1px solid var(--ag-border);border-radius:12px;padding:12px 14px}
.agr .tile .l{font-size:12px;color:var(--ag-ink-2)}
.agr .tile .v{font-size:24px;font-weight:600;margin-top:4px;letter-spacing:-0.01em}
.agr .tile .a{font-size:11.5px;color:var(--ag-ink-3);margin-top:2px}
.agr .grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:12px;align-items:start}
.agr .card{background:var(--ag-surface);border:1px solid var(--ag-border);border-radius:12px;padding:8px;overflow:hidden}
.agr .card .ag-chart-wrap{display:block}
.agr table.t{border-collapse:collapse;width:100%;font-size:12.5px;background:var(--ag-surface)}
.agr table.t th{text-align:left;color:var(--ag-ink-2);font-weight:600;padding:8px 12px;border-bottom:1px solid var(--ag-grid)}
.agr table.t td{padding:7px 12px;border-bottom:1px solid var(--ag-grid);vertical-align:top}
.agr table.t td.num{text-align:right;font-variant-numeric:tabular-nums}
.agr .sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:-1px}
.agr .notes{font-size:12px;color:var(--ag-ink-3);margin-top:28px;line-height:1.6}
.agr .ag-app{padding:0;background:transparent}
.agr .path{font-size:13px;line-height:1.7}
.agr .path b{font-weight:600}
"""


def _tile(label: str, value: str, aux: str = "") -> str:
    return f'<div class="tile"><div class="l">{escape(label)}</div><div class="v">{escape(value)}</div>' + (f'<div class="a">{escape(aux)}</div>' if aux else "") + "</div>"


def _report_html(rep: GraphReport, title: str | None) -> str:
    from ..charts import bar_chart, gantt, histogram
    from ..charts.runtime import CHART_CSS, CHART_JS, chart_block
    from ..render.html import app_html, page

    th = rep.theme or get_theme("light")
    s = rep.summary
    name = rep.name
    kind = "directed acyclic graph" if rep.dag is not None else ("directed graph" if s.get("directed") else "undirected graph")
    body = ['<div class="agr">']
    body.append(f"<h1>{escape(title or f'Graph report: {name}')}</h1>")
    body.append(f'<div class="sub">{escape(kind.capitalize())} · {s.get("n", 0):,} nodes · {s.get("m", 0):,} edges</div>')

    tiles = [_tile("Nodes", fmt_compact(s.get("n", 0))), _tile("Edges", fmt_compact(s.get("m", 0)))]
    if s.get("density") is not None:
        tiles.append(_tile("Density", f"{s['density']:.3g}"))
    if s.get("avg_degree") is not None:
        tiles.append(_tile("Average degree", f"{s['avg_degree']:.3g}", f"max {s.get('max_degree', 'n/a')}"))
    tiles.append(_tile("Components", f"{s.get('components', 0):,}", f"largest {s.get('largest_component', 0):,} nodes"))
    if s.get("diameter") is not None:
        tiles.append(_tile("Diameter", fmt_number(s["diameter"]), f"avg path {s['avg_path_length']:.3g}" if s.get("avg_path_length") is not None else ""))
    if s.get("avg_clustering") is not None:
        tiles.append(_tile("Clustering", f"{s['avg_clustering']:.3g}", f"transitivity {s['transitivity']:.3g}" if s.get("transitivity") is not None else ""))
    if rep.modularity is not None:
        tiles.append(_tile("Communities", str(len(rep.communities or [])), f"modularity {rep.modularity:.3f}"))
    if rep.dag is not None:
        tiles.append(_tile("Depth", str(rep.dag.get("depth")), "levels"))
        if rep.dag.get("width") is not None:
            tiles.append(_tile("Width", str(rep.dag["width"]), "max parallel tasks"))
        if rep.dag.get("critical_length") is not None:
            tiles.append(_tile("Critical path", fmt_number(rep.dag["critical_length"]), f"{len(rep.dag['critical_path'])} tasks"))
    body.append(f'<div class="tiles">{"".join(tiles)}</div>')

    if s.get("n", 0) and s.get("n", 0) <= 5000:
        body.append("<h2>Structure</h2>")
        fig = rep.figure(width=None)
        body.append(app_html(fig.scene, interactive=True, table=True, footer=False))

    charts: list[str] = []
    if rep.degree_histogram:
        from ..algorithms.structure import degree_histogram  # noqa: F401  (for the docstring link)

        degs = list(rep.graph.degree().values())
        charts.append(chart_block(histogram(degs, title="Degree distribution", x_label="degree", y_label="nodes", theme=th, width=520)))
    for key, label in (("pagerank", "Top 10 by PageRank"), ("betweenness", "Top 10 by betweenness")):
        nm = rep.centrality.get(key)
        if nm:
            top = nm.top(10)
            charts.append(chart_block(bar_chart([str(k) for k, _ in top], [v for _, v in top], title=label, theme=th, width=520)))
    if charts:
        body.append("<h2>Distributions & rankings</h2>")
        body.append('<div class="grid2">' + "".join(f'<div class="card">{c}</div>' for c in charts) + "</div>")

    if rep.communities:
        body.append(f"<h2>Communities <span style=\"font-weight:400;color:var(--ag-ink-3)\">· modularity {rep.modularity:.3f}</span></h2>")
        pal = list(th.categorical)
        pr = rep.centrality.get("pagerank") or rep.centrality.get("degree") or {}
        rows = []
        rank = {v: i for i, v in enumerate(rep.graph)}  # ties follow graph order, not set order
        for i, com in enumerate(rep.communities[:12]):
            members = sorted(com, key=lambda x: (-float(pr.get(x, 0)), rank.get(x, 0)))[:8]
            color = pal[i] if i < 7 or len(rep.communities) <= 8 else th.other
            rows.append(
                f'<tr><td><span class="sw" style="background:{color}"></span>{i}</td><td class="num">{len(com):,}</td>'
                f"<td>{escape(', '.join(map(str, members)))}{' …' if len(com) > 8 else ''}</td></tr>"
            )
        body.append('<div class="card" style="padding:0"><table class="t"><thead><tr><th>Community</th><th style="text-align:right">Size</th><th>Most central members</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")

    if rep.dag is not None:
        d = rep.dag
        body.append("<h2>DAG analysis</h2>")
        items = [
            f"<b>Sources</b> ({len(d.get('sources', []))}): {escape(', '.join(map(str, d.get('sources', [])[:20])))}",
            f"<b>Sinks</b> ({len(d.get('sinks', []))}): {escape(', '.join(map(str, d.get('sinks', [])[:20])))}",
            f"<b>Level sizes</b>: {escape(' · '.join(map(str, d.get('level_sizes', []))))}",
        ]
        if d.get("critical_path"):
            items.append(f"<b>Critical path</b> ({fmt_number(d['critical_length'])}): {escape(' → '.join(map(str, d['critical_path'])))}")
        elif d.get("longest_path"):
            items.append(f"<b>Longest path</b>: {escape(' → '.join(map(str, d['longest_path'])))}")
        if d.get("redundant_edges") is not None:
            items.append(f"<b>Redundant edges</b> (implied by other paths): {d['redundant_edges']}")
        body.append('<div class="card path" style="padding:12px 16px">' + "<br>".join(items) + "</div>")
        if d.get("schedule"):
            ch = gantt(d["schedule"], lanes="task", critical=d.get("critical_path"), title="Earliest-start schedule (unlimited workers)", theme=th, width=900)
            body.append(f'<div class="card" style="margin-top:12px">{chart_block(ch)}</div>')

    if rep.notes:
        body.append('<div class="notes">' + "<br>".join(escape(n) for n in rep.notes) + "</div>")
    total = sum(rep.timings.values())
    from .. import __version__

    body.append(f'<div class="notes">Generated by AryaGraph {escape(__version__)} in {total:.2f} s.</div>')
    body.append("</div>")
    vars_ = ";".join(f"{k}:{v}" for k, v in th.css_vars().items())
    extra = f":root{{{vars_}}}{CHART_CSS}{_REPORT_CSS}"
    html = page("".join(body), th, title or f"Graph report: {name}", interactive=True, extra_css=extra)
    return html.replace("</body>", f"<script>{CHART_JS}</script></body>")


__all__ = ["analyze", "GraphReport"]
