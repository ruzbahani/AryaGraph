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

"""Simulation on graphs: epidemics, cascades, walks, opinions, diffusion, DAG scheduling.

Every simulator returns a :class:`SimulationResult` (frames of per-node states
or values, ready for ``.animate()`` and ``.plot()``); :func:`run_ensemble`
summarises many independent runs. All randomness goes through a ``seed``.
"""

from __future__ import annotations

from .base import ROLES, SimulationResult, default_roles
from .cascade import (
    InfluenceMaximization,
    SpreadEstimate,
    greedy_influence_maximization,
    independent_cascade,
    influence_spread,
    linear_threshold,
)
from .compartmental import (
    DEFAULT_T_MAX,
    SEIR,
    SEIRD,
    SI,
    SIR,
    SIRS,
    SIS,
    CompartmentalModel,
    Transition,
    seir,
    seird,
    si,
    sir,
    sirs,
    sis,
)
from .diffusion import heat_diffusion, kuramoto
from .ensemble import EnsembleResult, run_ensemble
from .opinion import bounded_confidence, degroot, majority_rule, voter_model
from .scheduling import MonteCarloResult, ScheduleResult, TaskRun, monte_carlo_schedule, simulate_schedule
from .walk import random_walk, stationary_distribution, transition_matrix

__all__ = [
    # results
    "SimulationResult",
    "ROLES",
    "default_roles",
    # compartmental
    "CompartmentalModel",
    "Transition",
    "DEFAULT_T_MAX",
    "SI",
    "SIS",
    "SIR",
    "SEIR",
    "SIRS",
    "SEIRD",
    "si",
    "sis",
    "sir",
    "seir",
    "sirs",
    "seird",
    # cascades
    "independent_cascade",
    "linear_threshold",
    "influence_spread",
    "greedy_influence_maximization",
    "SpreadEstimate",
    "InfluenceMaximization",
    # walks
    "random_walk",
    "stationary_distribution",
    "transition_matrix",
    # opinions
    "voter_model",
    "majority_rule",
    "degroot",
    "bounded_confidence",
    # diffusion
    "heat_diffusion",
    "kuramoto",
    # scheduling
    "simulate_schedule",
    "monte_carlo_schedule",
    "ScheduleResult",
    "MonteCarloResult",
    "TaskRun",
    # ensembles
    "run_ensemble",
    "EnsembleResult",
]
