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

"""Command-line interface: ``aryagraph draw | analyze | info | convert``.

Examples::

    aryagraph draw pipeline.dot -o pipeline.svg --theme dark
    aryagraph draw friends.csv -o friends.html --color community --size pagerank
    aryagraph analyze network.graphml -o report.html
    aryagraph info graph.json
    aryagraph convert graph.gv graph.graphml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _read(path: str, directed: bool | None):
    from .io import read

    kwargs = {}
    if directed is not None and Path(path).suffix.lower() in (".csv", ".tsv", ".txt", ".edgelist"):
        kwargs["directed"] = directed
    return read(path, **kwargs)


def _metric(g, name: str | None):
    """Attribute name, or a computed metric when the name is a known algorithm."""
    if not name:
        return None
    if any(name in d for _, d in g.nodes.data()):
        return name
    from .algorithms import centrality as C
    from .algorithms import community as CM

    if name == "pagerank":
        return C.pagerank(g)
    if name == "betweenness":
        return C.betweenness_centrality(g)
    if name == "degree":
        return g.degree()
    if name in ("community", "communities"):
        und = g.to_undirected() if g.directed else g
        from .style.scales import by

        return by(CM.community_labels(CM.louvain_communities(und, seed=0)), kind="categorical", title="community")
    return name


def cmd_draw(a: argparse.Namespace) -> int:
    from .render import draw

    g = _read(a.input, a.directed)
    out = Path(a.output or Path(a.input).with_suffix(".svg").name)
    fig = draw(
        g,
        layout=a.layout,
        theme=a.theme,
        node_color=_metric(g, a.color),
        node_size=_metric(g, a.size),
        labels=False if a.no_labels else "auto",
        title=a.title,
        subtitle=a.subtitle,
        edge_style=a.edge_style,
        seed=a.seed,
        width=a.width,
        height=a.height,
    )
    fig.save(out)
    print(f"wrote {out}  ({fig.width:.0f}×{fig.height:.0f}, {len(g):,} nodes, {g.num_edges:,} edges, {fig.scene.meta['layout']} layout)")
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    from .analysis import analyze

    g = _read(a.input, a.directed)
    rep = analyze(g, theme=a.theme)
    if a.output:
        rep.save(a.output)
        print(f"wrote {a.output}")
    else:
        print(rep)
    return 0


def cmd_info(a: argparse.Namespace) -> int:
    from .algorithms.structure import summary

    g = _read(a.input, a.directed)
    s = summary(g)
    width = max(len(k) for k in s) + 2
    print(repr(g))
    for k, v in s.items():
        print(f"  {k:<{width}}{v}")
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    from .io import write

    g = _read(a.input, a.directed)
    write(g, a.output)
    print(f"wrote {a.output}  ({len(g):,} nodes, {g.num_edges:,} edges)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    p = argparse.ArgumentParser(prog="aryagraph", description="AryaGraph: graph & DAG visualization, analysis and simulation.")
    p.add_argument("--version", action="version", version=f"AryaGraph {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("input", help="graph file (.json .csv .tsv .graphml .gexf .dot .gv .mmd …)")
        sp.add_argument("--directed", action="store_true", default=None, help="treat edge lists as directed")
        sp.add_argument("--theme", default="light", help="light | dark | paper | blueprint")

    d = sub.add_parser("draw", help="render a graph to .svg / .html / .png / .pdf")
    common(d)
    d.add_argument("-o", "--output", help="output file (extension picks the format)")
    d.add_argument("--layout", default="auto", help="auto | stress | hierarchical | tree | radial | forceatlas2 | circular | …")
    d.add_argument("--color", help="node attribute or metric (pagerank, betweenness, degree, community)")
    d.add_argument("--size", help="node attribute or metric for node size")
    d.add_argument("--edge-style", default="auto", help="auto | straight | curved | flow | orthogonal")
    d.add_argument("--title")
    d.add_argument("--subtitle")
    d.add_argument("--no-labels", action="store_true")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--width", type=float)
    d.add_argument("--height", type=float)
    d.set_defaults(func=cmd_draw)

    an = sub.add_parser("analyze", help="full analytical report (.html dashboard, .json, .txt)")
    common(an)
    an.add_argument("-o", "--output")
    an.set_defaults(func=cmd_analyze)

    i = sub.add_parser("info", help="headline statistics")
    common(i)
    i.set_defaults(func=cmd_info)

    c = sub.add_parser("convert", help="convert between graph file formats")
    common(c)
    c.add_argument("output")
    c.set_defaults(func=cmd_convert)
    return p


def _utf8_output() -> None:
    """Write UTF-8 when output goes to a file or pipe with another encoding.

    On Windows that encoding is the ANSI code page (for example cp1252), which
    cannot hold the report's box-drawing rule or non-Latin labels.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_output()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:  # present errors, not tracebacks
        print(f"aryagraph: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
