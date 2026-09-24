Files and interoperability
==========================

Readers, writers and converters (``ag.io``); ``ag.read`` and ``ag.write`` pick the format from the extension.

By extension
------------

.. currentmodule:: aryagraph.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   read
   write

JSON
----

.. currentmodule:: aryagraph.io.jsonio

.. autosummary::
   :toctree: generated
   :nosignatures:

   to_dict
   from_dict
   to_json
   from_json
   write_json
   read_json

Edge and adjacency lists
------------------------

.. currentmodule:: aryagraph.io.edgelist

.. autosummary::
   :toctree: generated
   :nosignatures:

   read_edgelist
   write_edgelist
   read_adjacency_list
   write_adjacency_list

GraphML
-------

.. currentmodule:: aryagraph.io.graphml

.. autosummary::
   :toctree: generated
   :nosignatures:

   write_graphml
   read_graphml
   to_graphml
   from_graphml

GEXF
----

.. currentmodule:: aryagraph.io.gexf

.. autosummary::
   :toctree: generated
   :nosignatures:

   write_gexf
   read_gexf

DOT
---

.. currentmodule:: aryagraph.io.dot

.. autosummary::
   :toctree: generated
   :nosignatures:

   to_dot
   write_dot
   from_dot
   read_dot
   DotSyntaxError

Mermaid
-------

.. currentmodule:: aryagraph.io.mermaid

.. autosummary::
   :toctree: generated
   :nosignatures:

   to_mermaid
   from_mermaid
   write_mermaid
   read_mermaid

networkx, pandas, numpy and SciPy
---------------------------------

.. currentmodule:: aryagraph.io.interop

.. autosummary::
   :toctree: generated
   :nosignatures:

   from_networkx
   to_networkx
   from_pandas_edgelist
   from_pandas
   to_pandas
   from_numpy
   to_numpy
   from_scipy_sparse
   to_scipy_sparse
