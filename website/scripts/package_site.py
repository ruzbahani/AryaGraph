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

"""Package the built site for ordinary (static) web hosting.

Creates ``dist/aryagraph/`` (a copy of ``_build/html`` without Sphinx build
leftovers) and ``dist/aryagraph-website.zip``. Upload the folder's contents to
``public_html/aryagraph/`` on the host, or upload the zip and extract it there.
The ``.htaccess`` file assumes that address (see ``BASE``) for its 404 page.
No server-side code is involved: every page, the search and the interactive
figures run in the visitor's browser.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
BUILT = SITE / "_build" / "html"
DIST = SITE / "dist"
SKIP = {".buildinfo", ".doctrees", "webpack-macros.html"}  # the last is a theme template, not a page
BASE = "/aryagraph/"  # where the site is served: https://ruzbahani.com/aryagraph/

HTACCESS = f"""DirectoryIndex index.html
ErrorDocument 404 {BASE}404.html
AddDefaultCharset UTF-8
AddType image/svg+xml .svg
AddType text/css .css
AddType application/javascript .js
AddType application/json .json
"""

# Served for unknown URLs at any depth, so every link is absolute and the
# styles are inline (relative paths would resolve against the missing URL).
NOT_FOUND = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Page not found · AryaGraph</title>
<link rel="icon" href="{BASE}_static/img/favicon.svg" type="image/svg+xml">
<style>
:root {{ --bg: #fcfcfb; --ink: #1d1d1b; --ink-2: #52514e; --line: #e4e3df; --accent: #2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --line: #383835; --accent: #6da7ec; }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 560px; margin: 12vh auto 0; padding: 0 16px; }}
img {{ width: 56px; height: 56px; }}
h1 {{ font-size: 28px; margin: 16px 0 8px; }}
p {{ color: var(--ink-2); margin: 0 0 20px; }}
form {{ display: flex; gap: 8px; margin: 0 0 24px; }}
input {{ flex: 1; min-width: 0; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; background: transparent; color: var(--ink); font: inherit; }}
button {{ padding: 10px 16px; border: 0; border-radius: 8px; background: var(--accent); color: #fff; font: inherit; cursor: pointer; }}
ul {{ list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 8px 20px; }}
a {{ color: var(--accent); }}
</style>
</head>
<body>
<main>
<img src="{BASE}_static/img/mark.svg" alt="AryaGraph">
<h1>Page not found</h1>
<p>The page you asked for does not exist or has moved. Search the documentation, or start from one of these pages.</p>
<form action="{BASE}search.html" method="get" role="search">
<input type="search" name="q" placeholder="Search the documentation" aria-label="Search the documentation">
<button type="submit">Search</button>
</form>
<ul>
<li><a href="{BASE}index.html">Home</a></li>
<li><a href="{BASE}getting-started/index.html">Getting started</a></li>
<li><a href="{BASE}user-guide/index.html">User guide</a></li>
<li><a href="{BASE}tutorials/index.html">Tutorials</a></li>
<li><a href="{BASE}reference/index.html">API reference</a></li>
</ul>
</main>
</body>
</html>
"""


def main() -> int:
    if not (BUILT / "index.html").exists():
        print("build the site first: python scripts/build_site.py")
        return 1
    target = DIST / "aryagraph"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(BUILT, target, ignore=lambda d, names: [n for n in names if n in SKIP])
    # Apache hosts: index pages, UTF-8, and a "not found" page for unknown URLs
    (target / ".htaccess").write_text(HTACCESS, encoding="utf-8")
    (target / "404.html").write_text(NOT_FOUND, encoding="utf-8")
    archive = DIST / "aryagraph-website.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(target).as_posix())
    files = sum(1 for p in target.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"{files} files, {size / 1e6:.1f} MB -> {target}")
    print(f"zip: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
