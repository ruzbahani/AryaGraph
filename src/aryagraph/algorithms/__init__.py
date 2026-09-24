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

"""Graph algorithms: every function takes a graph first and returns plain data,
:class:`~aryagraph.NodeMap` / :class:`~aryagraph.EdgeMap` results, or small dataclasses.

Modules: traversal, paths, connectivity, dag, centrality, structure, community,
flow, spanning, matching, coloring, link_prediction, matrix. Everything is also
importable from here: ``from aryagraph.algorithms import pagerank`` (or ``ag.alg.pagerank``).
"""

from . import (
    centrality,
    coloring,
    community,
    connectivity,
    dag,
    flow,
    link_prediction,
    matching,
    matrix,
    paths,
    spanning,
    structure,
    traversal,
)
from .centrality import *  # noqa: F401,F403
from .coloring import *  # noqa: F401,F403
from .community import *  # noqa: F401,F403
from .connectivity import *  # noqa: F401,F403
from .dag import *  # noqa: F401,F403
from .flow import *  # noqa: F401,F403
from .link_prediction import *  # noqa: F401,F403
from .matching import *  # noqa: F401,F403
from .matrix import *  # noqa: F401,F403
from .paths import *  # noqa: F401,F403
from .spanning import *  # noqa: F401,F403
from .structure import *  # noqa: F401,F403
from .traversal import *  # noqa: F401,F403

_MODULES = (
    traversal,
    paths,
    connectivity,
    dag,
    centrality,
    structure,
    community,
    flow,
    spanning,
    matching,
    coloring,
    link_prediction,
    matrix,
)

__all__ = sorted({name for mod in _MODULES for name in mod.__all__})
