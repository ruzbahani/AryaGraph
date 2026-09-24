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

"""Reading and writing graphs: JSON, edge/adjacency lists, GraphML, GEXF, DOT, Mermaid, and interop.

:func:`read` and :func:`write` pick the format from the file extension:

==================================  ===========================================
extension                           format
==================================  ===========================================
``.json``                           node-link JSON (:func:`read_json`)
``.csv`` ``.tsv`` ``.txt``          edge list (:func:`read_edgelist`); ``.tsv``
``.edgelist``                       uses tabs, ``.txt``/``.edgelist`` blanks
``.adjlist``                        adjacency list (:func:`read_adjacency_list`)
``.graphml``                        GraphML (:func:`read_graphml`)
``.gexf``                           GEXF 1.3 (:func:`read_gexf`)
``.dot`` ``.gv``                    Graphviz DOT (:func:`read_dot`)
``.mmd`` ``.mermaid``               Mermaid flowchart (:func:`read_mermaid`)
==================================  ===========================================

>>> import aryagraph.io as nio
>>> nio.write(g, "graph.graphml")          # doctest: +SKIP
>>> h = nio.read("graph.graphml")          # doctest: +SKIP
"""

from __future__ import annotations

import os
from typing import Any, Callable

from ..core.graph import Graph
from .dot import DotSyntaxError, from_dot, read_dot, to_dot, write_dot
from .edgelist import read_adjacency_list, read_edgelist, write_adjacency_list, write_edgelist
from .gexf import read_gexf, write_gexf
from .graphml import from_graphml, read_graphml, to_graphml, write_graphml
from .interop import (
    from_networkx,
    from_numpy,
    from_pandas,
    from_pandas_edgelist,
    from_scipy_sparse,
    to_networkx,
    to_numpy,
    to_pandas,
    to_scipy_sparse,
)
from .jsonio import from_dict, from_json, read_json, to_dict, to_json, write_json
from .mermaid import from_mermaid, read_mermaid, to_mermaid, write_mermaid

_FORMATS: dict[str, tuple[Callable[..., Graph], Callable[..., None], dict[str, Any], dict[str, Any]]] = {
    # name: (reader, writer, reader defaults, writer defaults)
    "json": (read_json, write_json, {}, {}),
    "csv": (read_edgelist, write_edgelist, {"delimiter": ","}, {"delimiter": ","}),
    "tsv": (read_edgelist, write_edgelist, {"delimiter": "\t"}, {"delimiter": "\t"}),
    "edgelist": (read_edgelist, write_edgelist, {}, {"delimiter": " "}),
    "adjlist": (read_adjacency_list, write_adjacency_list, {}, {}),
    "graphml": (read_graphml, write_graphml, {}, {}),
    "gexf": (read_gexf, write_gexf, {}, {}),
    "dot": (read_dot, write_dot, {}, {}),
    "mermaid": (read_mermaid, write_mermaid, {}, {}),
}
_EXTENSIONS = {
    ".json": "json",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "edgelist",
    ".edgelist": "edgelist",
    ".adjlist": "adjlist",
    ".graphml": "graphml",
    ".gexf": "gexf",
    ".dot": "dot",
    ".gv": "dot",
    ".mmd": "mermaid",
    ".mermaid": "mermaid",
}


def _format_of(path: str | os.PathLike, format: str | None) -> str:
    if format is not None:
        name = format.lower().lstrip(".")
        name = _EXTENSIONS.get("." + name, name)
        if name not in _FORMATS:
            raise ValueError(f"unknown format {format!r}; choose from {sorted(_FORMATS)}")
        return name
    ext = os.path.splitext(os.fspath(path))[1].lower()
    if ext not in _EXTENSIONS:
        raise ValueError(
            f"cannot infer the format of {os.fspath(path)!r} from its extension; "
            f"use one of {sorted(_EXTENSIONS)} or pass format="
        )
    return _EXTENSIONS[ext]


def read(path: str | os.PathLike, *, format: str | None = None, **kwargs: Any) -> Graph:
    """Read a graph, choosing the reader from the extension (or *format*); *kwargs* go to the reader."""
    reader, _, defaults, _ = _FORMATS[_format_of(path, format)]
    return reader(path, **{**defaults, **kwargs})


def write(g: Graph, path: str | os.PathLike, *, format: str | None = None, **kwargs: Any) -> None:
    """Write *g*, choosing the writer from the extension (or *format*); *kwargs* go to the writer."""
    _, writer, _, defaults = _FORMATS[_format_of(path, format)]
    writer(g, path, **{**defaults, **kwargs})


__all__ = [
    "read",
    "write",
    # JSON
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    "write_json",
    "read_json",
    # edge and adjacency lists
    "read_edgelist",
    "write_edgelist",
    "read_adjacency_list",
    "write_adjacency_list",
    # XML formats
    "write_graphml",
    "read_graphml",
    "to_graphml",
    "from_graphml",
    "write_gexf",
    "read_gexf",
    # text diagram languages
    "to_dot",
    "write_dot",
    "from_dot",
    "read_dot",
    "DotSyntaxError",
    "to_mermaid",
    "from_mermaid",
    "write_mermaid",
    "read_mermaid",
    # interop
    "from_networkx",
    "to_networkx",
    "from_pandas_edgelist",
    "from_pandas",
    "to_pandas",
    "from_numpy",
    "to_numpy",
    "from_scipy_sparse",
    "to_scipy_sparse",
]
