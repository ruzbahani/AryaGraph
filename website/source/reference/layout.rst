Layouts
=======

Layout engines return a :class:`~aryagraph.layout.base.Layout`; ``compute`` dispatches by name.

Dispatcher
----------

.. currentmodule:: aryagraph.layout

.. autosummary::
   :toctree: generated
   :nosignatures:

   compute
   auto_method

Layout result
-------------

.. currentmodule:: aryagraph.layout.base

.. autosummary::
   :toctree: generated
   :nosignatures:

   Layout
   pack_components

Hierarchical and trees
----------------------

.. currentmodule:: aryagraph.layout

.. autosummary::
   :toctree: generated
   :nosignatures:

   hierarchical
   tree
   radial

Force-directed and spectral
---------------------------

.. currentmodule:: aryagraph.layout

.. autosummary::
   :toctree: generated
   :nosignatures:

   stress
   fruchterman_reingold
   force_atlas2
   spectral

Geometric
---------

.. currentmodule:: aryagraph.layout

.. autosummary::
   :toctree: generated
   :nosignatures:

   circular
   shell
   grid
   random
   spiral
   bipartite
   arc

Post-processing
---------------

.. currentmodule:: aryagraph.layout

.. autosummary::
   :toctree: generated
   :nosignatures:

   remove_overlaps
