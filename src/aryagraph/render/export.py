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

"""Raster/PDF export.

SVG and HTML are written directly. PNG and PDF are produced by, in order of
preference, ``cairosvg`` (if installed) or any installed Chromium-family
browser (Chrome, Edge, Chromium, Brave) in headless mode, which renders text
exactly as the interactive view does.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..core.exceptions import DependencyError

_BROWSER_NAMES = ("chrome", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "msedge", "microsoft-edge", "brave-browser")
_BROWSER_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def find_browser() -> str | None:
    """Path of a Chromium-family browser usable for headless export, or None."""
    env = os.environ.get("ARYAGRAPH_BROWSER")
    if env and Path(env).exists():
        return env
    for name in _BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    local = os.environ.get("LOCALAPPDATA")
    extra = [os.path.join(local, r"Google\Chrome\Application\chrome.exe")] if local else []
    for p in (*_BROWSER_PATHS, *extra):
        if Path(p).exists():
            return p
    return None


# The <title> that opens an SVG document written by AryaGraph (the figure's title).
_SVG_TITLE = re.compile(r"^\s*(?:<\?xml[^>]*>\s*)?<svg\b[^>]*>\s*<title\b[^>]*>(.*?)</title>", re.DOTALL)


def _page(svg: str, width: float, height: float, title: str = "") -> str:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>"
        f"@page{{size:{width}px {height}px;margin:0}}html,body{{margin:0;padding:0;background:transparent}}"
        "svg{display:block}</style></head><body>" + svg + "</body></html>"
    )


# Flags that keep headless Chrome from waiting on anything interactive: first-run
# and default-browser prompts, extension or component updates, and (on macOS) the
# keychain, which otherwise blocks non-interactive sessions.
HEADLESS_FLAGS = (
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-component-update",
    "--use-mock-keychain",
    "--password-store=basic",
)


def _cairosvg():
    """The cairosvg module, or None when it (or the Cairo library it loads) is unavailable."""
    try:
        import cairosvg  # type: ignore
    except (ImportError, OSError):  # OSError: cairosvg is installed but the Cairo C library is not
        return None
    return cairosvg


def _no_converter(feature: str) -> DependencyError:
    """The error for a PNG/PDF export that has neither a usable cairosvg nor a browser."""
    try:
        import cairosvg  # type: ignore  # noqa: F401
    except OSError as exc:  # cairosvg is installed but the Cairo C library is not
        hint = f"it is installed but cannot load the Cairo library: {exc}; install Cairo for your system"
        return DependencyError("cairosvg", feature, extra="png", hint=hint)
    except ImportError:
        pass
    return DependencyError("cairosvg", feature, extra="png")


def _run_browser(args: list[str], timeout: float = 120) -> None:
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    try:
        subprocess.run(args, check=True, capture_output=True, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"the headless browser did not finish within {timeout:g} s; "
            "install cairosvg (pip install cairosvg) for browser-free export"
        ) from None


def svg_to_png(svg: str, path: str | Path, width: float, height: float, scale: float = 2.0) -> Path:
    """Rasterise *svg* to *path* at *scale* × its CSS pixel size."""
    path = Path(path)
    cairosvg = _cairosvg()
    if cairosvg is not None:
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(path), scale=scale)
        return path
    browser = find_browser()
    if browser is None:
        raise _no_converter("PNG export (or install Chrome/Edge/Chromium)")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "figure.html"
        page.write_text(_page(svg, width, height), encoding="utf-8")
        _run_browser(
            [
                browser,
                *HEADLESS_FLAGS,
                "--hide-scrollbars",
                f"--user-data-dir={Path(tmp) / 'profile'}",
                f"--force-device-scale-factor={scale}",
                f"--window-size={int(round(width))},{int(round(height))}",
                f"--screenshot={path.resolve()}",
                page.resolve().as_uri(),
            ]
        )
    if not path.exists():
        raise RuntimeError(f"headless browser did not produce {path}")
    return path


def svg_to_pdf(svg: str, path: str | Path, width: float, height: float) -> Path:
    """Vector PDF of *svg* at its natural size."""
    path = Path(path)
    cairosvg = _cairosvg()
    if cairosvg is not None:
        cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(path))
        return path
    browser = find_browser()
    if browser is None:
        raise _no_converter("PDF export (or install Chrome/Edge/Chromium)")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "figure.html"
        found = _SVG_TITLE.match(svg)
        title = html.unescape(found.group(1)).strip() if found else ""
        # the browser uses the page title as the PDF's document title
        page.write_text(_page(svg, width, height, title or path.stem), encoding="utf-8")
        _run_browser(
            [
                browser,
                *HEADLESS_FLAGS,
                f"--user-data-dir={Path(tmp) / 'profile'}",
                "--no-pdf-header-footer",
                f"--print-to-pdf={path.resolve()}",
                page.resolve().as_uri(),
            ]
        )
    if not path.exists():
        raise RuntimeError(f"headless browser did not produce {path}")
    return path


__all__ = ["find_browser", "svg_to_png", "svg_to_pdf", "HEADLESS_FLAGS"]
