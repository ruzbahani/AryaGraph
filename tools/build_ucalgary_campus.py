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

"""Build src/aryagraph/generators/data/ucalgary_campus.json from the merged research notes.

Deterministic: same input -> byte-identical output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "aryagraph" / "generators" / "data" / "ucalgary_campus.json"

R_EARTH = 6_371_008.8  # mean Earth radius (IUGG), metres
K_NEAREST = 2
RADIUS = 250.0

# code, display name, kind  (names from the official map legend / Facilities directory)
BUILDINGS = [
    ("AB", "Art Building & Art Parkade", "arts"),
    ("AD", "Administration Building", "administration"),
    ("AU", "Aurora Hall", "residence"),
    ("BI", "Biological Sciences", "academic"),
    ("CC", "Child Care Centre", "services"),
    ("CCIT", "Calgary Centre for Innovative Technology", "research"),
    ("CD", "Cascade Hall", "residence"),
    ("CDC", "Child Development Centre", "research"),
    ("CH", "Craigie Hall (incl. University Theatre)", "arts"),
    ("CR", "Crowsnest Hall", "residence"),
    ("CSSH", "Cenovus Spo'pi Solar House", "research"),
    ("DC", "Dining Centre", "student-life"),
    ("EDC", "Education Classrooms", "academic"),
    ("EDT", "Education Tower", "academic"),
    ("EEEL", "Energy Environment Experiential Learning", "academic"),
    ("ENA", "Engineering Block A", "academic"),
    ("ENB", "Engineering Block B", "academic"),
    ("ENC", "Engineering Block C", "academic"),
    ("END", "Engineering Block D", "academic"),
    ("ENE", "Engineering Block E", "academic"),
    ("ENF", "Engineering Block F", "academic"),
    ("ENG", "Engineering Block G", "academic"),
    ("ES", "Earth Sciences", "academic"),
    ("GL", "Glacier Hall", "residence"),
    ("GR", "Grounds Building", "services"),
    ("HNSC", "Hunter Student Commons", "student-life"),
    ("HP", "Central Heating and Cooling Plant", "services"),
    ("ICT", "Information and Communications Technology", "academic"),
    ("IH", "International House", "residence"),
    ("KA", "Kananaskis Hall", "residence"),
    ("KNA", "Dr. Roger Jackson Kinesiology Complex (Block A)", "athletics"),
    ("KNB", "Dr. Roger Jackson Kinesiology Complex (Block B)", "athletics"),
    ("MEB", "Mechanical Engineering Building", "academic"),
    ("MFH", "Murray Fraser Hall", "academic"),
    ("MH", "MacEwan Hall", "student-life"),
    ("MS", "Mathematical Sciences", "academic"),
    ("MSC", "MacEwan Student Centre", "student-life"),
    ("MT", "MacKimmie Tower", "administration"),
    ("MTH", "Mathison Hall", "academic"),
    ("OL", "Olympus Hall", "residence"),
    ("OO", "Olympic Oval", "athletics"),
    ("OVC", "Olympic Volunteer Centre", "services"),
    ("PF", "Professional Faculties", "academic"),
    ("PP", "Physical Plant", "services"),
    ("RC", "Rozsa Centre", "arts"),
    ("RT", "Reeve Theatre", "arts"),
    ("RU", "Rundle Hall", "residence"),
    ("SA", "Science A", "academic"),
    ("SB", "Science B", "academic"),
    ("SH", "Scurfield Hall", "academic"),
    ("SS", "Social Sciences", "academic"),
    ("ST", "Science Theatres", "academic"),
    ("TFDL", "Taylor Family Digital Library", "library"),
    ("TI", "Taylor Institute for Teaching and Learning", "academic"),
    ("TRB", "Trailer B (Mathematical Science)", "academic"),
    ("YA", "Yamnuska Hall", "residence"),
]

# OSM building centroids (WGS84). CC is the OSM 'Childcare Centre' way 119065140
# (450 Campus Place NW, the address UCalgary gives for the centre).
COORDS = {
    "AD": (51.0781552, -114.1271352), "AB": (51.075439, -114.1301486), "AU": (51.0748957, -114.1340445),
    "BI": (51.0799355, -114.125624), "CC": (51.0783487, -114.1244905), "CCIT": (51.0802822, -114.1333883),
    "CD": (51.0760344, -114.1374401), "CDC": (51.0747518, -114.1439864), "CH": (51.076698, -114.1298854),
    "CR": (51.0791631, -114.134913), "CSSH": (51.079757, -114.1336589), "DC": (51.0758412, -114.1334087),
    "EDC": (51.076719, -114.1265704), "EDT": (51.0770464, -114.1261475), "EEEL": (51.0810803, -114.1294433),
    "ENA": (51.08026, -114.1314689), "ENB": (51.0806885, -114.1315246), "ENC": (51.0808937, -114.1318533),
    "END": (51.0805397, -114.132609), "ENE": (51.0799823, -114.1325147), "ENF": (51.0794276, -114.1325667),
    "ENG": (51.0802465, -114.1320257), "ES": (51.0801969, -114.1290929), "GL": (51.0756389, -114.135924),
    "GR": (51.0795063, -114.1407015), "HNSC": (51.0779603, -114.129009), "HP": (51.0749855, -114.1387945),
    "ICT": (51.0802016, -114.1303466), "IH": (51.0761521, -114.1328137), "KA": (51.0753335, -114.1348307),
    "KNA": (51.0770887, -114.1329486), "KNB": (51.0779782, -114.1337659), "MEB": (51.0823251, -114.1303243),
    "MFH": (51.0770978, -114.1284669), "MH": (51.0785683, -114.1304506), "MS": (51.0799885, -114.1278401),
    "MSC": (51.0781866, -114.131706), "MT": (51.0774806, -114.1288766), "MTH": (51.0767635, -114.1247034),
    "OL": (51.0750126, -114.1357886), "OO": (51.0770105, -114.1356452), "PF": (51.077394, -114.1269438),
    "PP": (51.0751413, -114.1432734), "RC": (51.0763766, -114.131524), "RT": (51.0762811, -114.1308076),
    "RU": (51.0750549, -114.1327499), "SA": (51.079136, -114.1281934), "SB": (51.0794473, -114.1294246),
    "SH": (51.0773722, -114.1247919), "SS": (51.0791313, -114.1270143), "ST": (51.0796444, -114.1272154),
    "TFDL": (51.0774272, -114.1299715), "TI": (51.079118, -114.131263), "TRB": (51.0803101, -114.1281285),
    "YA": (51.0751478, -114.1372662), "OVC": (51.0713569, -114.1221974),
}

# Indoor links as reported. Craigie Hall blocks CHC-CHF collapse onto the single CH node.
INDOOR_RAW = [
    ("KNA", "KNB", "attached"), ("KNB", "OO", "attached"), ("KNB", "MSC", "pedway"), ("MSC", "MH", "attached"),
    ("MH", "SB", "tunnel"), ("KNA", "DC", "tunnel"), ("DC", "IH", "attached"), ("AU", "DC", "tunnel"),
    ("SB", "ES", "attached"), ("SB", "SA", "attached"), ("ICT", "ES", "attached"), ("ES", "MS", "pedway"),
    ("ENA", "ICT", "attached"), ("ENA", "ENB", "attached"), ("ENA", "ENG", "attached"), ("END", "ENG", "attached"),
    ("CCIT", "END", "attached"), ("ENG", "ENE", "attached"), ("END", "ENE", "attached"), ("ENE", "ENF", "attached"),
    ("ENB", "ENC", "attached"), ("MS", "ST", "attached"), ("SA", "ST", "attached"), ("ST", "SS", "attached"),
    ("SA", "SS", "attached"), ("ST", "BI", "pedway"), ("SS", "AD", "pedway"), ("AD", "PF", "attached"),
    ("PF", "MFH", "pedway"), ("PF", "EDC", "pedway"), ("EDC", "EDT", "attached"), ("EDT", "SH", "pedway"),
    ("SH", "MTH", "pedway"), ("MFH", "MT", "attached"), ("MT", "HNSC", "attached"), ("TFDL", "HNSC", "pedway"),
    ("MFH", "CHC", "pedway"), ("CHC", "CHD", "attached"), ("CHD", "CHE", "attached"), ("CHE", "CHF", "attached"),
    ("CHE", "AB", "pedway"), ("CHF", "RT", "attached"), ("RT", "RC", "attached"),
]
ALIASES = {"CHC": "CH", "CHD": "CH", "CHE": "CH", "CHF": "CH", "CHG": "CH"}

SOURCES = [
    "https://www.ucalgary.ca/sites/default/files/teams/157/UCalgary-Main_Campus_Map-20230724.pdf",
    "https://ucalgary.ca/live-uc-ucalgary-site/sites/default/files/teams/223/Indoor%20Routes%20Map%20-%20s%20-%2020251104.pdf",
    "https://www.ucalgary.ca/facilities/buildings-grounds/buildings",
    "https://www.ucalgary.ca/facilities/buildings-grounds/buildings/kinesiology-complex",
    "https://www.ucalgary.ca/student-services/calendar-scheduling/general-assignment-rooms/classroom-search-tool/room-trb-102",
    "https://www.ucalgary.ca/risk/sites/default/files/teams/8/UofC-Building-Hours.pdf",
    "https://www.ucalgary.ca/child-care",
    "https://ucalgary.ca/sustainability/our-sustainable-campus/buildings/ucalgary-leed-buildings/aurora-hall",
    "https://haskayne.ucalgary.ca/contacts/location-and-spaces/scurfield-hall",
    "https://majorprojects.alberta.ca/details/Mathison-Hall/4079",
    "https://dialogdesign.ca/projects/university-of-calgary-mackimmie-complex-2/",
    "https://www.openstreetmap.org/copyright",
]

DESCRIPTION = (
    "Schematic model of 56 University of Calgary main-campus buildings, compiled from public sources:"
    " codes and names from the official Main Campus Map (2023) and the Facilities directory, updated "
    "for later renamings (Kinesiology A and B became the Dr. Roger Jackson Kinesiology Complex in "
    "February 2026); indoor links (tunnels, enclosed pedways, attached buildings) from UCalgary's "
    "Indoor Building Routes map (Nov 2025) cross-checked with OpenStreetMap, except the Aurora "
    "Hall–Dining Centre residence tunnel, which comes from UCalgary's Aurora Hall page; positions "
    "from OpenStreetMap (the centre of each building's bounding box). Positions are approximate; "
    "'outdoor' edges are modelled nearest-neighbour walking links, not surveyed paths; every length "
    "is the straight-line distance between two positions. The Olympic Volunteer Centre (OVC), which "
    "UCalgary lists among its main-campus buildings, stands at McMahon Stadium south of 24 Avenue NW;"
    " its only link is a modelled 626 m straight line to Mathison Hall. Not an official UCalgary "
    "product."
)
ATTRIBUTION = (
    "Building positions and several links derived from OpenStreetMap data, (c) OpenStreetMap "
    "contributors, available under the Open Database License (ODbL) 1.0."
)


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(h))


def main() -> None:
    codes = sorted(c for c, *_ in BUILDINGS)
    assert len(codes) == len(set(codes)) == 56, len(codes)
    assert set(codes) == set(COORDS), set(codes) ^ set(COORDS)
    info = {c: (n, k) for c, n, k in BUILDINGS}

    def key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    edges: dict[tuple[str, str], str] = {}
    for a, b, kind in INDOOR_RAW:
        a, b = ALIASES.get(a, a), ALIASES.get(b, b)
        if a == b:
            continue  # link between two blocks of the same modelled building
        assert a in COORDS and b in COORDS, (a, b)
        k = key(a, b)
        assert k not in edges, k
        edges[k] = kind
    n_indoor = len(edges)

    dist = {key(a, b): haversine(COORDS[a], COORDS[b]) for i, a in enumerate(codes) for b in codes[i + 1 :]}

    # outdoor: each building to its K nearest others within RADIUS
    for a in codes:
        near = sorted((dist[key(a, b)], b) for b in codes if b != a)
        for d, b in near[:K_NEAREST]:
            if d <= RADIUS and key(a, b) not in edges:
                edges[key(a, b)] = "outdoor"

    # connectivity: Kruskal over all pairs, shortest first, join components only
    parent = {c: c for c in codes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    bridges = []
    for (a, b), d in sorted(dist.items(), key=lambda kv: (kv[1], kv[0])):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            edges[(a, b)] = "outdoor"
            bridges.append((a, b, round(d, 1)))
    assert len({find(c) for c in codes}) == 1

    nodes = []
    for c in codes:
        name, kind = info[c]
        lat, lon = COORDS[c]
        nodes.append({"id": c, "name": name, "kind": kind, "lat": lat, "lon": lon})
    edge_rows = [
        {"source": a, "target": b, "kind": kind, "length": round(dist[(a, b)], 1)}
        for (a, b), kind in sorted(edges.items())
    ]

    header = {
        "name": "University of Calgary main campus",
        "description": DESCRIPTION,
        "attribution": ATTRIBUTION,
        "sources": SOURCES,
    }
    dump = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    lines = ["{"]
    for k, v in header.items():
        lines.append(f"{dump(k)}:{dump(v)},")
    lines.append('"nodes":[')
    lines.append(",\n".join(dump(n) for n in nodes))
    lines.append('],\n"edges":[')
    lines.append(",\n".join(dump(e) for e in edge_rows))
    lines.append("]}")
    text = "\n".join(lines) + "\n"
    json.loads(text)
    OUT.write_text(text, encoding="utf-8", newline="\n")

    from collections import Counter

    print("nodes", len(nodes), "edges", len(edge_rows), "indoor", n_indoor)
    print(Counter(e["kind"] for e in edge_rows))
    print("connectivity edges:", bridges)
    deg = Counter()
    for e in edge_rows:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    print("degree:", sorted(deg.items(), key=lambda kv: kv[1]))
    longest = sorted(edge_rows, key=lambda e: -e["length"])[:8]
    print("longest:", [(e["source"], e["target"], e["kind"], e["length"]) for e in longest])
    print("longest indoor:", sorted(((e["length"], e["source"], e["target"], e["kind"]) for e in edge_rows if e["kind"] != "outdoor"))[-6:])


if __name__ == "__main__":
    main()
