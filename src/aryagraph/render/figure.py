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

"""The :class:`Figure` returned by :func:`aryagraph.draw`."""

from __future__ import annotations

import html as _html
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

from .scene import Scene
from .svg import scene_to_svg


class Figure:
    """A rendered graph: export it, display it, or inspect its scene.

    >>> fig = aryagraph.draw(g, node_color="community")
    >>> fig.save("graph.svg")      # also .html (interactive), .png, .pdf
    >>> fig                         # renders inline in Jupyter
    """

    def __init__(self, scene: Scene, graph: Any = None, extras: dict[str, Any] | None = None) -> None:
        self.scene = scene
        self.graph = graph
        self.extras = extras or {}
        self._svg: str | None = None

    # ------------------------------------------------------------------ #
    @property
    def width(self) -> float:
        return (self.scene.display_size or (self.scene.width, self.scene.height))[0]

    @property
    def height(self) -> float:
        return (self.scene.display_size or (self.scene.width, self.scene.height))[1]

    @property
    def layout(self):
        """The pixel-space :class:`~aryagraph.layout.Layout` used for this drawing."""
        return self.scene.layout

    def __repr__(self) -> str:
        m = self.scene.meta
        return (
            f"<Figure {self.width:.0f}×{self.height:.0f}: {m.get('nodes', 0)} nodes, "
            f"{m.get('edges', 0)} edges, {m.get('layout')} layout, theme {self.scene.theme.name!r}>"
        )

    # ------------------------------------------------------------------ #
    def to_svg(self) -> str:
        """Standalone SVG markup."""
        if self._svg is None:
            self._svg = scene_to_svg(self.scene)
        return self._svg

    def to_html(self, *, interactive: bool = True, title: str | None = None) -> str:
        """Self-contained HTML page (interactive: pan/zoom, hover, search, drag, table view)."""
        from .html import figure_to_html

        return figure_to_html(self, interactive=interactive, title=title)

    def save(self, path: str | Path, *, scale: float = 2.0, **kwargs: Any) -> Path:
        """Write to *path*; the format follows the extension (.svg, .html, .png, .pdf)."""
        path = Path(path)
        ext = path.suffix.lower()
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if ext == ".svg":
            path.write_text(scene_to_svg(self.scene, xml_declaration=True), encoding="utf-8")
        elif ext in (".html", ".htm"):
            path.write_text(self.to_html(**kwargs), encoding="utf-8")
        elif ext == ".png":
            from .export import svg_to_png

            svg_to_png(self.to_svg(), path, self.width, self.height, scale=scale)
        elif ext == ".pdf":
            from .export import svg_to_pdf

            svg_to_pdf(self.to_svg(), path, self.width, self.height)
        else:
            raise ValueError(f"unsupported file extension {ext!r}; use .svg, .html, .png or .pdf")
        return path

    def show(self) -> None:
        """Display inline in a notebook, or open the interactive page in a browser."""
        try:
            from IPython import get_ipython  # type: ignore

            shell = get_ipython()
            if shell is not None and getattr(shell, "kernel", None) is not None:
                from IPython.display import HTML, display  # type: ignore

                display(HTML(self._repr_html_()))
                return
        except ImportError:
            pass
        tmp = Path(tempfile.mkdtemp(prefix="aryagraph-")) / "figure.html"
        tmp.write_text(self.to_html(), encoding="utf-8")
        webbrowser.open(tmp.as_uri())

    def _repr_html_(self) -> str:
        page = self.to_html()
        h = int(self.height) + 64
        return (
            f'<iframe srcdoc="{_html.escape(page, quote=True)}" style="width:100%;max-width:{int(self.width) + 40}px;'
            f'height:{h}px;border:0;border-radius:8px" sandbox="allow-scripts allow-downloads"></iframe>'
        )


__all__ = ["Figure"]
