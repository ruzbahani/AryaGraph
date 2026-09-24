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

"""Bundled example datasets, stored as JSON (no download, no networkx needed).

* Classic small social networks: exact copies of networkx's versions (same
  nodes in the same order, same edges and attributes), so results can be
  cross-checked. Their ``citation`` graph attribute names the original study.
* :func:`ucalgary_campus`: a real-world spatial network of University of
  Calgary main-campus buildings joined by indoor links and walking links.

Every function returns a fresh graph that callers may mutate freely.
"""

from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from importlib import resources
from typing import Any

from ..core.graph import DiGraph, Graph


@lru_cache(maxsize=None)
def _load(key: str) -> dict[str, Any]:
    ref = resources.files("aryagraph.generators") / "data" / f"{key}.json"
    return json.loads(ref.read_text(encoding="utf-8"))


def _dataset(key: str) -> Graph:
    """Build a graph from the compact column-wise JSON in ``generators/data``.

    Layout: ``nodes`` (labels), ``node_attrs`` (name → list aligned with
    ``nodes``), ``edges`` (pairs of node positions), ``edge_attrs`` (name →
    list aligned with ``edges``); ``null`` marks a missing value.
    """
    data = _load(key)
    cls = DiGraph if data["directed"] else Graph
    g = cls(name=data["name"], citation=data["source"])
    g.attrs.update(copy.deepcopy(data["graph"]))
    nodes = data["nodes"]
    node_attrs = data["node_attrs"]
    for i, n in enumerate(nodes):
        g.add_node(n, **{k: vals[i] for k, vals in node_attrs.items() if vals[i] is not None})
    edge_attrs = data["edge_attrs"]
    for j, (a, b) in enumerate(data["edges"]):
        g.add_edge(nodes[a], nodes[b], **{k: vals[j] for k, vals in edge_attrs.items() if vals[j] is not None})
    return g


def les_miserables() -> Graph:
    """Co-appearance network of the 77 characters of *Les Misérables* (254 edges).

    Edge ``weight`` counts the chapters in which two characters appear
    together. Identical to ``networkx.les_miserables_graph()``.
    """
    return _dataset("les_miserables")


def florentine_families() -> Graph:
    """Marriage alliances among 15 Renaissance Florentine families (20 edges).

    The Medici's central position is the textbook example for betweenness.
    Identical to ``networkx.florentine_families_graph()`` (the isolated Pucci
    family is omitted there too).
    """
    return _dataset("florentine_families")


def davis_southern_women() -> Graph:
    """Davis's Southern Women: 18 women × 14 social events, 89 attendances (bipartite).

    Women carry ``bipartite=0`` and events (``"E1" … "E14"``) ``bipartite=1``;
    the graph attributes ``top`` and ``bottom`` list the two sides. Identical
    to ``networkx.davis_southern_women_graph()``.
    """
    return _dataset("davis_southern_women")


_EARTH_RADIUS = 6_371_008.8  # mean Earth radius in metres (IUGG)
_INDOOR_KINDS = frozenset({"tunnel", "pedway", "attached"})


def ucalgary_campus(*, indoor_only: bool = False) -> Graph:
    """University of Calgary main campus: 56 buildings and the ways between them.

    A real-world spatial network for routing, centrality and layout demos.
    Nodes are official building codes (``"MSC"``, ``"TFDL"``, ``"ICT"``, …);
    edges are undirected, and each edge's ``kind`` is one of:

    * ``"tunnel"``: underground pedestrian tunnel (e.g. MacEwan Hall ↔ Science B);
    * ``"pedway"``: enclosed bridge or link corridor (e.g. Earth Sciences ↔
      Mathematical Sciences);
    * ``"attached"``: adjoining buildings with an internal door;
    * ``"outdoor"``: a modelled walking link. Each building is joined to its
      two nearest neighbours within 250 m, plus the shortest links needed to
      make the campus connected.

    Parameters
    ----------
    indoor_only:
        Keep only the tunnel / pedway / attached edges, which form the "stay
        warm, stay inside" network. All 56 buildings are kept, so buildings
        without an indoor connection become isolated nodes.

    Returns
    -------
    Graph
        Node attributes: ``name``, ``kind`` (``"academic"``, ``"residence"``,
        ``"student-life"``, ``"athletics"``, ``"arts"``, ``"library"``,
        ``"research"``, ``"administration"`` or ``"services"``), ``lat`` and
        ``lon`` (WGS84 degrees) and ``pos = (x, y)`` in metres on a local plane
        centred on the campus: x points east and y points **south** (the screen
        convention, y down), so drawings come out north-up without a layout
        algorithm. Edge attributes: ``kind``, ``length`` (metres) and
        ``weight`` (equal to ``length``). Graph attributes: ``name``,
        ``description``, ``attribution`` and ``sources`` (URLs).

    Notes
    -----
    This is a schematic model compiled from public sources, not an official
    or surveyed campus dataset:

    * building codes and names: UCalgary Main Campus Map (July 2023) and the
      Facilities building directory, updated for later renamings: ``"KNA"``
      and ``"KNB"`` (Kinesiology A and B) have been the Dr. Roger Jackson
      Kinesiology Complex since February 2026;
    * indoor links: UCalgary's *Indoor Building Routes* map (November 2025),
      cross-checked against OpenStreetMap tunnels, bridges and shared walls,
      plus UCalgary, Haskayne, Alberta Major Projects and DIALOG pages. The
      Aurora Hall ↔ Dining Centre tunnel, part of the residence tunnels, comes
      from UCalgary's Aurora Hall page rather than the routes map;
    * coordinates: the centre of each building's OpenStreetMap bounding box
      (© OpenStreetMap contributors, ODbL 1.0).

    Positions are approximate (a bounding-box centre can sit a few metres off
    an irregular building's centroid), and every ``length`` is the
    straight-line (haversine) distance between two positions, not a measured
    walking distance. The ``"outdoor"`` edges are modelled nearest-neighbour
    walking links, not surveyed paths. Craigie Hall's blocks C–G are merged
    into one node, ``"CH"``. One building lies off the main grounds: the
    Olympic Volunteer Centre (``"OVC"``), which UCalgary lists among its
    main-campus buildings, stands at McMahon Stadium south of 24 Avenue NW.
    Its only edge is a modelled 626 m outdoor link to Mathison Hall, the
    longest edge in the graph, and it stretches the north–south extent of
    any north-up drawing by about half.

    >>> campus = ucalgary_campus()
    >>> campus.nodes["TFDL"]["name"]
    'Taylor Family Digital Library'
    >>> sorted({k for _, _, k in campus.edges.data("kind")})
    ['attached', 'outdoor', 'pedway', 'tunnel']
    """
    data = _load("ucalgary_campus")
    g = Graph(
        name=data["name"],
        description=data["description"],
        attribution=data["attribution"],
        sources=list(data["sources"]),
    )
    nodes = data["nodes"]
    lat0 = sum(n["lat"] for n in nodes) / len(nodes)
    lon0 = sum(n["lon"] for n in nodes) / len(nodes)
    east = math.radians(1.0) * _EARTH_RADIUS * math.cos(math.radians(lat0))  # metres per degree of longitude
    north = math.radians(1.0) * _EARTH_RADIUS  # metres per degree of latitude
    for n in nodes:
        pos = (round((n["lon"] - lon0) * east, 1), round(-(n["lat"] - lat0) * north, 1))
        g.add_node(n["id"], name=n["name"], kind=n["kind"], lat=n["lat"], lon=n["lon"], pos=pos)
    for e in data["edges"]:
        if indoor_only and e["kind"] not in _INDOOR_KINDS:
            continue
        g.add_edge(e["source"], e["target"], kind=e["kind"], length=e["length"], weight=e["length"])
    return g


__all__ = ["les_miserables","florentine_families", "davis_southern_women", "ucalgary_campus"]
