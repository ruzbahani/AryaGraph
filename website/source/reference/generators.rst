Generators and datasets
=======================

Classic and random graphs, example DAGs and bundled datasets (``ag.gen``).

Classic graphs
--------------

.. currentmodule:: aryagraph.generators.classic

.. autosummary::
   :toctree: generated
   :nosignatures:

   empty_graph
   path_graph
   cycle_graph
   complete_graph
   complete_bipartite_graph
   star_graph
   wheel_graph
   grid_graph
   hypercube_graph
   balanced_tree
   binomial_tree
   ladder_graph
   circular_ladder_graph
   lollipop_graph
   barbell_graph
   petersen_graph
   complete_multipartite_graph
   turan_graph

Random graphs
-------------

.. currentmodule:: aryagraph.generators.random

.. autosummary::
   :toctree: generated
   :nosignatures:

   erdos_renyi
   gnm_random_graph
   barabasi_albert
   watts_strogatz
   stochastic_block_model
   planted_partition
   random_geometric
   random_regular
   random_tree
   configuration_model
   powerlaw_cluster
   random_dag
   layered_dag
   random_task_dag

Example DAGs
------------

.. currentmodule:: aryagraph.generators.dags

.. autosummary::
   :toctree: generated
   :nosignatures:

   ml_pipeline
   software_build
   project_plan
   data_warehouse_etl
   course_prerequisites

Datasets
--------

.. currentmodule:: aryagraph.generators.datasets

.. autosummary::
   :toctree: generated
   :nosignatures:

   les_miserables
   florentine_families
   davis_southern_women
   ucalgary_campus
