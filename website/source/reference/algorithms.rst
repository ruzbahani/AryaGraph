Algorithms
==========

Every algorithm is also available as ``ag.alg.<name>``.

Traversal
---------

.. currentmodule:: aryagraph.algorithms.traversal

.. autosummary::
   :toctree: generated
   :nosignatures:

   bfs_order
   bfs_edges
   bfs_layers
   bfs_tree
   dfs_preorder
   dfs_postorder
   dfs_edges
   dfs_tree
   descendants
   ancestors

Shortest paths
--------------

.. currentmodule:: aryagraph.algorithms.paths

.. autosummary::
   :toctree: generated
   :nosignatures:

   shortest_path
   shortest_path_length
   dijkstra
   bellman_ford
   astar_path
   all_shortest_paths
   all_pairs_shortest_path_length
   floyd_warshall
   has_path
   all_simple_paths
   k_shortest_paths

Connectivity
------------

.. currentmodule:: aryagraph.algorithms.connectivity

.. autosummary::
   :toctree: generated
   :nosignatures:

   connected_components
   number_connected_components
   is_connected
   node_connected_component
   strongly_connected_components
   weakly_connected_components
   number_strongly_connected_components
   number_weakly_connected_components
   is_strongly_connected
   is_weakly_connected
   condensation
   articulation_points
   bridges
   biconnected_components
   is_biconnected

DAGs and cycles
---------------

.. currentmodule:: aryagraph.algorithms.dag

.. autosummary::
   :toctree: generated
   :nosignatures:

   is_dag
   topological_sort
   all_topological_sorts
   find_cycle
   simple_cycles
   dag_longest_path
   dag_longest_path_length
   dag_levels
   CriticalPath
   critical_path
   transitive_closure
   transitive_reduction
   lowest_common_ancestors
   dag_width
   maximum_antichain
   minimum_chain_partition

Centrality
----------

.. currentmodule:: aryagraph.algorithms.centrality

.. autosummary::
   :toctree: generated
   :nosignatures:

   degree_centrality
   in_degree_centrality
   out_degree_centrality
   closeness_centrality
   harmonic_centrality
   betweenness_centrality
   edge_betweenness_centrality
   eigenvector_centrality
   katz_centrality
   pagerank
   hits
   centralities

Structure
---------

.. currentmodule:: aryagraph.algorithms.structure

.. autosummary::
   :toctree: generated
   :nosignatures:

   density
   degree_histogram
   degree_distribution
   average_degree
   reciprocity
   triangles
   clustering
   average_clustering
   transitivity
   square_clustering
   degree_assortativity
   attribute_assortativity
   numeric_assortativity
   eccentricity
   diameter
   radius
   center
   periphery
   average_shortest_path_length
   wiener_index
   global_efficiency
   local_efficiency
   core_number
   k_core
   onion_layers
   rich_club_coefficient
   is_tree
   is_forest
   is_regular
   s_metric
   summary

Communities
-----------

.. currentmodule:: aryagraph.algorithms.community

.. autosummary::
   :toctree: generated
   :nosignatures:

   modularity
   partition_quality
   community_labels
   louvain_communities
   louvain_hierarchy
   greedy_modularity_communities
   label_propagation_communities
   asyn_fluid_communities
   girvan_newman

Flow
----

.. currentmodule:: aryagraph.algorithms.flow

.. autosummary::
   :toctree: generated
   :nosignatures:

   FlowResult
   CutResult
   maximum_flow
   minimum_cut

Spanning trees
--------------

.. currentmodule:: aryagraph.algorithms.spanning

.. autosummary::
   :toctree: generated
   :nosignatures:

   minimum_spanning_tree
   maximum_spanning_tree
   minimum_spanning_edges
   maximum_spanning_edges

Matching
--------

.. currentmodule:: aryagraph.algorithms.matching

.. autosummary::
   :toctree: generated
   :nosignatures:

   hopcroft_karp
   bipartite_maximum_matching
   maximal_matching
   is_matching
   is_maximal_matching
   is_perfect_matching
   max_weight_matching
   min_weight_matching

Coloring
--------

.. currentmodule:: aryagraph.algorithms.coloring

.. autosummary::
   :toctree: generated
   :nosignatures:

   greedy_color
   is_bipartite
   bipartite_sets

Link prediction
---------------

.. currentmodule:: aryagraph.algorithms.link_prediction

.. autosummary::
   :toctree: generated
   :nosignatures:

   common_neighbors
   jaccard_coefficient
   adamic_adar_index
   resource_allocation_index
   preferential_attachment
   predict_links

Matrices and spectra
--------------------

.. currentmodule:: aryagraph.algorithms.matrix

.. autosummary::
   :toctree: generated
   :nosignatures:

   adjacency_matrix
   degree_vector
   laplacian_matrix
   incidence_matrix
   adjacency_spectrum
   laplacian_spectrum
   algebraic_connectivity
