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

"""Figures and embeds for the first part of the user guide.

Pages: user-guide/graphs.md, drawing.md, styling.md, interactive.md,
exporting.md and languages.md. Each ``draw()`` call below repeats the call
shown on the page it illustrates, so the pictures match the code readers run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import aryagraph as ag

PNG_SCALE = 1.5  # device pixels per CSS pixel: sharp on high-density screens, modest file sizes


def _png(fig, out: Path, name: str) -> None:
    fig.save(out / f"{name}.png", scale=PNG_SCALE)


# --------------------------------------------------------------------------- #
# graphs.md
# --------------------------------------------------------------------------- #
def graphs(out: Path, facts: dict) -> None:
    campus = ag.gen.ucalgary_campus()
    indoor = campus.edge_subgraph(
        (u, v) for u, v, kind in campus.edges.data("kind") if kind != "outdoor"
    )
    fig = ag.draw(
        indoor,
        layout={n: campus.nodes[n]["pos"] for n in indoor},
        node_color="#72716b",
        edge_color="kind",
        edge_width=2,
        title="Indoor walking network",
        subtitle=f"{indoor.num_nodes} buildings joined by tunnels, pedways and attached walls",
    )
    _png(fig, out, "graphs_indoor")
    facts["indoor"] = [indoor.num_nodes, indoor.num_edges]

    build = ag.gen.software_build()
    levels = build.levels()
    fig = ag.draw(
        build,
        layout="hierarchical",
        layout_options={"ranks": levels, "orientation": "LR"},
        node_color="team",
        title="Software build in generations",
        subtitle="Each column is one generation: tasks that can run in parallel",
    )
    _png(fig, out, "graphs_generations")
    facts["generations"] = [len(s) for s in build.generations()]


# --------------------------------------------------------------------------- #
# drawing.md
# --------------------------------------------------------------------------- #
def drawing(out: Path, facts: dict) -> None:
    campus = ag.gen.ucalgary_campus()
    on_map = {n: d["pos"] for n, d in campus.nodes.data()}

    fig = ag.draw(
        campus,
        layout=on_map,
        node_color="kind",
        title="University of Calgary main campus",
        subtitle="Buildings colored by kind",
    )
    _png(fig, out, "drawing_campus_kind")

    route = ag.alg.shortest_path(campus, "OO", "SH", weight="length")
    length = sum(campus.edges[u, v]["length"] for u, v in zip(route, route[1:]))
    fig = ag.draw(
        campus,
        layout=on_map,
        node_color="kind",
        highlight_path=route,
        title="Olympic Oval to Scurfield Hall",
        subtitle=f"Shortest walk, {length:,.0f} m",
    )
    _png(fig, out, "drawing_route")
    facts["route_m"] = round(length)

    web = ag.gen.barabasi_albert(300, 2, seed=7)
    rank = ag.alg.pagerank(web)
    fig = ag.draw(
        web,
        node_color=ag.by(rank, kind="log", palette="viridis", title="PageRank (log)"),
        node_size=ag.by(web.degree(), kind="sqrt", title="degree"),
        labels=False,
        title="Preferential attachment, 300 nodes",
        subtitle="Color: PageRank on a log scale; size: degree",
    )
    _png(fig, out, "drawing_log")

    to_msc = ag.alg.shortest_path_length(campus, "MSC", weight="length")
    to_tfdl = ag.alg.shortest_path_length(campus, "TFDL", weight="length")
    closer = {n: to_msc[n] - to_tfdl[n] for n in campus}
    fig = ag.draw(
        campus,
        layout=on_map,
        node_color=ag.by(closer, kind="diverging", title="MSC minus TFDL (m)"),
        title="Nearer by the network: student center or library?",
        subtitle="Network distance to MSC minus network distance to TFDL (m)",
    )
    _png(fig, out, "drawing_diverging")

    names = list(ag.style.SHAPES)
    spokes = ag.DiGraph([("hub", s) for s in names])
    ring = {"hub": (0.0, 0.0)}
    for i, s in enumerate(names):
        angle = 2 * math.pi * i / len(names) - math.pi / 2
        ring[s] = (1.35 * math.cos(angle), math.sin(angle))
    fig = ag.draw(
        spokes,
        layout=ring,
        scale=200,
        node_shape=lambda n: "circle" if n == "hub" else n,
        labels=lambda n: None if n == "hub" else n,
        label_position="center",
        node_color=lambda n: "#898781" if n == "hub" else "#2a78d6",
        title="Node shapes",
        subtitle="Labels set inside; arrowheads end on each outline",
    )
    _png(fig, out, "drawing_shapes")

    steps = ag.DAG([
        ("fetch", "parse"), ("fetch", "lint"), ("parse", "compile"), ("parse", "docs"),
        ("compile", "test"), ("lint", "test"), ("test", "release"), ("docs", "release"),
        ("fetch", "release"),
    ])
    for style in ("straight", "curved", "flow", "orthogonal"):
        _png(ag.draw(steps, edge_style=style, title=style), out, f"drawing_style_{style}")


# --------------------------------------------------------------------------- #
# styling.md
# --------------------------------------------------------------------------- #
def _colormap_strips(out: Path, name: str, maps: list[str]) -> None:
    """Horizontal strips of each colormap, sampled by AryaGraph and rasterised by its exporter."""
    th = ag.get_theme("light")
    width, row, pad, label_w, steps = 640, 30, 24, 110, 48
    height = pad * 2 + 30 + row * len(maps)
    font = th.font.replace('"', "&quot;")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{font}">',
        f'<rect width="{width}" height="{height}" fill="{th.surface}"/>',
        f'<text x="{pad}" y="{pad + 12}" font-size="15" font-weight="600" fill="{th.ink}">Colormaps (light mode)</text>',
    ]
    cell = (width - 2 * pad - label_w) / steps
    for i, cmap_name in enumerate(maps):
        y = pad + 30 + i * row
        parts.append(
            f'<text x="{pad}" y="{y + 15}" font-size="12.5" fill="{th.ink}" '
            f'font-family="ui-monospace, Consolas, monospace">{cmap_name}</text>'
        )
        for j, color in enumerate(ag.style.colormap(cmap_name, "light").sample(steps)):
            parts.append(
                f'<rect x="{pad + label_w + j * cell:.2f}" y="{y}" width="{cell + 0.6:.2f}" height="20" fill="{color}"/>'
            )
    parts.append("</svg>")
    from aryagraph.render.export import svg_to_png

    svg_to_png("".join(parts), out / f"{name}.png", width, height, scale=PNG_SCALE)


def styling(out: Path, facts: dict) -> None:
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
        _png(fig, out, f"styling_theme_{name}")

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
    _png(fig, out, "styling_custom_theme")

    # "gray" is left out: in this version its ramp takes an olive tint instead of neutral grays
    maps = [m for m in ag.style.available_colormaps() if m != "gray"]
    _colormap_strips(out, "styling_colormaps", maps)
    facts["colormaps"] = maps


# --------------------------------------------------------------------------- #
# interactive.md
# --------------------------------------------------------------------------- #
def interactive(out: Path, facts: dict) -> None:
    campus = ag.gen.ucalgary_campus()
    fig = ag.draw(
        campus,
        layout={n: d["pos"] for n, d in campus.nodes.data()},
        node_color="kind",
        tooltip=["name", "kind"],
        title="University of Calgary main campus",
        subtitle="Hover, click, drag, search; the legend filters by kind",
    )
    fig.save(out / "interactive_campus.html")
    facts["campus_html_kib"] = round(len(fig.to_html().encode("utf-8")) / 1024)

    build = ag.gen.software_build()
    fig = ag.draw(build, node_color="team", title="Software build", subtitle="Drag a task to re-route its edges")
    fig.save(out / "interactive_build.html")


# --------------------------------------------------------------------------- #
# exporting.md
# --------------------------------------------------------------------------- #
def exporting(out: Path, facts: dict) -> None:
    campus = ag.gen.ucalgary_campus()
    lengths = [m for _, _, m in campus.edges.data("length")]
    chart = ag.charts.histogram(
        lengths,
        bins=12,
        title="Walking links by length",
        subtitle=f"{len(lengths)} edges of the campus graph",
        x_label="length (m)",
    )
    chart.save(out / "exporting_histogram.png", scale=PNG_SCALE)


# --------------------------------------------------------------------------- #
# languages.md
# --------------------------------------------------------------------------- #
def languages(out: Path, facts: dict) -> None:
    courses = ag.DAG([
        ("ریاضی ۱", "ریاضی ۲"),
        ("ریاضی ۲", "آمار و احتمال"),
        ("مبانی برنامه‌نویسی", "ساختمان داده"),
        ("ساختمان داده", "طراحی الگوریتم"),
        ("ریاضی گسسته", "طراحی الگوریتم"),
        ("آمار و احتمال", "یادگیری ماشین"),
        ("طراحی الگوریتم", "یادگیری ماشین"),
        ("ساختمان داده", "پایگاه داده"),
    ])
    for course in courses:
        is_math = "ریاضی" in course or "آمار" in course
        courses.nodes[course]["گروه"] = "ریاضی" if is_math else "کامپیوتر"
    fig = ag.draw(
        courses,
        node_color=ag.by("گروه", counts=False),
        layout_options={"orientation": "RL"},
        title="پیش‌نیازهای درسی",
        subtitle="Course prerequisites with Persian labels, flowing right to left",
    )
    _png(fig, out, "languages_persian")

    hello = {
        "English": "Hello", "فارسی": "سلام", "العربية": "مرحبا", "עברית": "שלום",
        "中文": "你好", "日本語": "こんにちは", "한국어": "안녕하세요",
        "Ελληνικά": "Γειά σου", "हिन्दी": "नमस्ते",
    }
    greetings = ag.Graph([("🌐", language) for language in hello])
    fig = ag.draw(
        greetings,
        layout="radial",
        scale=0.6,
        labels=lambda n: n if n == "🌐" else f"{n}: {hello[n]}",
        title="Hello in nine languages",
    )
    _png(fig, out, "languages_hello")


# --------------------------------------------------------------------------- #
def build(out: Path) -> None:
    facts: dict = {}
    for section in (graphs, drawing, styling, interactive, exporting, languages):
        section(out, facts)
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
