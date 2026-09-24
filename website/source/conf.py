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

"""Sphinx configuration for the AryaGraph website (https://ruzbahani.com/aryagraph)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import aryagraph  # noqa: E402

# -- project -----------------------------------------------------------------
project = "AryaGraph"
author = "Ali Mohammadi Ruzbahani"
copyright = "2026, Ali Mohammadi Ruzbahani"
release = aryagraph.__version__
version = release

# -- extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "attrs_inline",
    "attrs_block",
    "fieldlist",
    "html_image",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_substitutions = {"version": release}

# API reference from the numpy-style docstrings
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {"members": True, "member-order": "bysource", "show-inheritance": True}
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_class_signature = "mixed"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML --------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "AryaGraph"
html_baseurl = "https://ruzbahani.com/aryagraph/"
html_static_path = ["_static"]
html_css_files = ["css/aryagraph.css"]
html_favicon = "_static/img/favicon.svg"
html_show_sourcelink = False
html_copy_source = False
html_last_updated_fmt = "%Y-%m-%d"

html_theme_options = {
    "logo": {
        "text": "AryaGraph",
        "image_light": "_static/img/mark.svg",
        "image_dark": "_static/img/mark-dark.svg",
        "alt_text": "AryaGraph home",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 7,
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/ruzbahani/AryaGraph", "icon": "fa-brands fa-github"},
    ],
    "use_edit_page_button": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_prev_next": True,
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": ["license-note"],
}

html_context = {
    "github_user": "ruzbahani",
    "github_repo": "AryaGraph",
    "github_version": "main",
    "doc_path": "website/source",
    "default_mode": "auto",
}

# The landing and section overview pages use the full width.
html_sidebars = {
    "index": [],
    "case-studies/index": [],
    "gallery/index": [],
}
