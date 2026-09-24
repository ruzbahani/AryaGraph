Graphs and results
==================

The graph classes, the result maps returned by algorithms, and the exception hierarchy (``aryagraph.core``).

Graph classes
-------------

.. currentmodule:: aryagraph.core.graph

.. autosummary::
   :toctree: generated
   :nosignatures:

   Graph
   DiGraph

DAG
---

.. currentmodule:: aryagraph.core.dag

.. autosummary::
   :toctree: generated
   :nosignatures:

   DAG

Result maps
-----------

.. currentmodule:: aryagraph.core.results

.. autosummary::
   :toctree: generated
   :nosignatures:

   NodeMap
   EdgeMap

Exceptions
----------

.. currentmodule:: aryagraph.core.exceptions

.. autosummary::
   :toctree: generated
   :nosignatures:

   AryaGraphError
   NodeNotFound
   EdgeNotFound
   CycleError
   NegativeCycleError
   NegativeWeightError
   UnboundedFlowError
   NoPath
   NotConnected
   GraphTypeError
   ConvergenceError
   DependencyError

Utilities
---------

.. currentmodule:: aryagraph.core.utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_rng
   weight_fn
