Top-level API
=============

The functions most programs start from, importable directly from ``aryagraph``. Each is documented once, in its home module; this table links there.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Name
     - Purpose
   * - :py:func:`ag.draw <aryagraph.render.draw>`
     - Draw a graph and return a Figure.
   * - :py:func:`ag.animate <aryagraph.render.animate.animate>`
     - Interactive playback of a simulation result.
   * - :py:func:`ag.analyze <aryagraph.analysis.report.analyze>`
     - One-call analytical report.
   * - :py:func:`ag.read <aryagraph.io.read>`
     - Read a graph file; the format follows the extension.
   * - :py:func:`ag.write <aryagraph.io.write>`
     - Write a graph file; the format follows the extension.
   * - :py:func:`ag.by <aryagraph.style.scales.by>`
     - Explicit encoding spec for a visual channel.
   * - :py:func:`ag.get_theme <aryagraph.style.themes.get_theme>`
     - Look up a theme by name.
   * - :py:func:`ag.register_theme <aryagraph.style.themes.register_theme>`
     - Make a custom theme available by name.
   * - :py:class:`ag.Figure <aryagraph.render.figure.Figure>`
     - A rendered graph: export, display, inspect.
   * - :py:class:`ag.Layout <aryagraph.layout.base.Layout>`
     - Node positions, edge routes and provenance.
   * - :py:class:`ag.GraphReport <aryagraph.analysis.report.GraphReport>`
     - Result of analyze().
   * - :py:class:`ag.Theme <aryagraph.style.themes.Theme>`
     - A complete visual vocabulary.

Namespaces: ``ag.alg`` (:doc:`algorithms`), ``ag.layout`` (:doc:`layout`), ``ag.sim`` (:doc:`sim`), ``ag.gen`` (:doc:`generators`), ``ag.io`` (:doc:`io`), ``ag.style`` (:doc:`style`), ``ag.charts`` (:doc:`charts`).
