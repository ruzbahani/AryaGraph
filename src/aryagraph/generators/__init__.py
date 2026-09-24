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

"""Graph generators: classic families, random models, example DAGs and bundled datasets.

>>> from aryagraph.generators import layered_dag, project_plan, ucalgary_campus
>>> g = ucalgary_campus()
>>> dag = layered_dag([2, 4, 4, 1], seed=0)
"""

from __future__ import annotations

from .classic import (
    balanced_tree,
    barbell_graph,
    binomial_tree,
    circular_ladder_graph,
    complete_bipartite_graph,
    complete_graph,
    complete_multipartite_graph,
    cycle_graph,
    empty_graph,
    grid_graph,
    hypercube_graph,
    ladder_graph,
    lollipop_graph,
    path_graph,
    petersen_graph,
    star_graph,
    turan_graph,
    wheel_graph,
)
from .dags import course_prerequisites, data_warehouse_etl, ml_pipeline, project_plan, software_build
from .datasets import davis_southern_women, florentine_families, les_miserables, ucalgary_campus
from .random import (
    barabasi_albert,
    configuration_model,
    erdos_renyi,
    gnm_random_graph,
    layered_dag,
    planted_partition,
    powerlaw_cluster,
    random_dag,
    random_geometric,
    random_regular,
    random_task_dag,
    random_tree,
    stochastic_block_model,
    watts_strogatz,
)

__all__ = [
    # classic
    "empty_graph",
    "path_graph",
    "cycle_graph",
    "complete_graph",
    "complete_bipartite_graph",
    "star_graph",
    "wheel_graph",
    "grid_graph",
    "hypercube_graph",
    "balanced_tree",
    "binomial_tree",
    "ladder_graph",
    "circular_ladder_graph",
    "lollipop_graph",
    "barbell_graph",
    "petersen_graph",
    "complete_multipartite_graph",
    "turan_graph",
    # random
    "erdos_renyi",
    "gnm_random_graph",
    "barabasi_albert",
    "watts_strogatz",
    "stochastic_block_model",
    "planted_partition",
    "random_geometric",
    "random_regular",
    "random_tree",
    "configuration_model",
    "powerlaw_cluster",
    "random_dag",
    "layered_dag",
    "random_task_dag",
    # example DAGs
    "ml_pipeline",
    "software_build",
    "project_plan",
    "data_warehouse_etl",
    "course_prerequisites",
    # datasets
    "les_miserables",
    "florentine_families",
    "davis_southern_women",
    "ucalgary_campus",
]
