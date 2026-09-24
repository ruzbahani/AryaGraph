Simulation
==========

Every simulator returns a :class:`~aryagraph.sim.base.SimulationResult` (``ag.sim``).

Results
-------

.. currentmodule:: aryagraph.sim.base

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimulationResult

Result types
------------

.. currentmodule:: aryagraph.sim.ensemble

.. autosummary::
   :toctree: generated
   :nosignatures:

   EnsembleResult

Schedule results
----------------

.. currentmodule:: aryagraph.sim.scheduling

.. autosummary::
   :toctree: generated
   :nosignatures:

   ScheduleResult
   MonteCarloResult
   TaskRun

Compartmental models
--------------------

.. currentmodule:: aryagraph.sim.compartmental

.. autosummary::
   :toctree: generated
   :nosignatures:

   CompartmentalModel
   SI
   SIS
   SIR
   SEIR
   SIRS
   SEIRD

One-call shortcuts
------------------

.. currentmodule:: aryagraph.sim.compartmental

.. autofunction:: si

.. autofunction:: sis

.. autofunction:: sir

.. autofunction:: seir

.. autofunction:: sirs

.. autofunction:: seird

Cascades and influence
----------------------

.. currentmodule:: aryagraph.sim

.. autosummary::
   :toctree: generated
   :nosignatures:

   independent_cascade
   linear_threshold
   influence_spread
   greedy_influence_maximization

Walks, opinions and diffusion
-----------------------------

.. currentmodule:: aryagraph.sim

.. autosummary::
   :toctree: generated
   :nosignatures:

   random_walk
   stationary_distribution
   transition_matrix
   voter_model
   majority_rule
   degroot
   bounded_confidence
   heat_diffusion
   kuramoto

Scheduling and ensembles
------------------------

.. currentmodule:: aryagraph.sim

.. autosummary::
   :toctree: generated
   :nosignatures:

   simulate_schedule
   monte_carlo_schedule
   run_ensemble
