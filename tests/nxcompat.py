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

"""networkx version guards for oracle comparisons.

AryaGraph follows current networkx semantics. Python 3.10 can only install
networkx up to 3.4, and a few reference behaviours changed after that, so the
comparisons that depend on them are skipped on older networkx (the rest of each
test still runs).
"""

from __future__ import annotations

import networkx as nx
import pytest

NX_VERSION = tuple(int(p) for p in nx.__version__.split(".")[:2])


def needs_networkx(version: tuple[int, int], why: str) -> None:
    """Skip the rest of the current test when networkx is older than *version*."""
    if NX_VERSION < version:
        pytest.skip(f"networkx {nx.__version__} < {version[0]}.{version[1]}: {why}")
