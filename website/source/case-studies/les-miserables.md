# Les Misérables: a character network

Victor Hugo's novel, read as a network, has 77 characters joined by 254 co-appearance ties. This case study ranks the characters with four centrality measures, finds the communities of the story and the characters who bridge them, peels the network into k-cores, and zooms in on Jean Valjean's ego network.

## The data

`ag.gen.les_miserables()` loads the co-appearance network compiled by Donald Knuth for The Stanford GraphBase (1993), the same data as `networkx.les_miserables_graph()`. Two characters are linked when they appear in the same chapter, and the edge attribute `weight` counts those chapters. Node names follow the dataset's spelling, such as `Thenardier` and `MlleGillenormand`.

```python
from collections import Counter

import aryagraph as ag

g = ag.gen.les_miserables()
print(g)
print(g.attrs["citation"])
print("Valjean and Javert:", g.edges["Valjean", "Javert"])
```

```text
<Graph 'Les Misérables': 77 nodes, 254 edges>
Character co-occurrence in Victor Hugo's novel; D. E. Knuth, The Stanford GraphBase (1993).
Valjean and Javert: {'weight': 17}
```

A few summary measures describe the shape of the network:

```python
total = sum(d["weight"] for _, _, d in g.edges.data())
print(f"{total} co-appearances in total")
print(f"density {ag.alg.density(g):.3f}, mean degree {ag.alg.average_degree(g):.2f}")
print(f"average clustering {ag.alg.average_clustering(g):.3f}, transitivity {ag.alg.transitivity(g):.3f}")
print(f"diameter {ag.alg.diameter(g)}, average distance {ag.alg.average_shortest_path_length(g):.2f}")
print(f"degree assortativity {ag.alg.degree_assortativity(g):.3f}")
```

```text
820 co-appearances in total
density 0.087, mean degree 6.60
average clustering 0.573, transitivity 0.499
diameter 5, average distance 2.64
degree assortativity -0.165
```

The network is sparse (8.7% of all possible pairs are linked) yet close-knit: any character reaches any other in at most 5 steps, 2.64 on average, and the neighbors of a character are often linked to each other (average clustering 0.573). The negative degree assortativity means that well-connected characters tend to be linked to poorly connected ones: the protagonists meet many minor characters.

## Who is central?

Centrality has several meanings, and each measure answers a different question:

- **Degree:** how many different characters does this character meet?
- **Weighted degree:** in how many chapters, summed over all ties, does this character share the stage? {py:meth}`Graph.degree <aryagraph.core.graph.Graph.degree>` computes it with `weight="weight"`.
- **Betweenness:** how often does this character lie on the shortest path between two others? {py:func}`~aryagraph.algorithms.centrality.betweenness_centrality` counts hops, so it ignores the weights.
- **PageRank:** how likely is a random walk that follows the ties in proportion to their weight, and now and then jumps to a random character, to be at this character? {py:func}`~aryagraph.algorithms.centrality.pagerank` reads the `weight` attribute by default.

Every measure returns a {py:class}`~aryagraph.core.results.NodeMap`, whose `top(k)` gives the ranking. The next block prints the rows of the table below.

```python
degree = g.degree()
strength = g.degree(weight="weight")          # weighted degree: shared chapters, summed
betweenness = ag.alg.betweenness_centrality(g)
pagerank = ag.alg.pagerank(g)                 # reads the "weight" attribute by default

measures = [degree, strength, betweenness, pagerank]
print("| # | degree | weighted degree | betweenness | PageRank |")
print("|---|---|---|---|---|")
for i, row in enumerate(zip(*(m.top(10) for m in measures)), 1):
    print(f"| {i} | " + " | ".join(f"{v} ({x:.3g})" for v, x in row) + " |")
```

| # | degree | weighted degree | betweenness | PageRank |
|---|---|---|---|---|
| 1 | Valjean (36) | Valjean (158) | Valjean (0.57) | Valjean (0.0996) |
| 2 | Gavroche (22) | Marius (104) | Myriel (0.177) | Marius (0.0517) |
| 3 | Marius (19) | Enjolras (91) | Gavroche (0.165) | Myriel (0.0393) |
| 4 | Javert (17) | Courfeyrac (84) | Marius (0.132) | Cosette (0.0369) |
| 5 | Thenardier (16) | Cosette (68) | Fantine (0.13) | Enjolras (0.0366) |
| 6 | Fantine (15) | Combeferre (68) | Thenardier (0.0749) | Thenardier (0.0357) |
| 7 | Enjolras (15) | Bossuet (66) | Javert (0.0543) | Courfeyrac (0.033) |
| 8 | Courfeyrac (13) | Thenardier (61) | MlleGillenormand (0.0476) | Gavroche (0.0283) |
| 9 | Bossuet (13) | Gavroche (56) | Enjolras (0.0426) | Fantine (0.0272) |
| 10 | Bahorel (12) | Fantine (47) | Tholomyes (0.0406) | Javert (0.0268) |

Valjean leads all four rankings, by a wide margin: averaged over all pairs of other characters, 57% of the shortest paths between them pass through him. Below him, the measures disagree in informative ways:

- **Gavroche** is second by degree (22 characters) but ninth by weighted degree: he crosses paths with many characters, often briefly.
- **The students** Enjolras, Courfeyrac, Combeferre and Bossuet rank high by weighted degree but lower by betweenness: they share many chapters, mostly with each other.
- **Myriel**, the bishop of the opening book, has only 10 ties yet ranks second by betweenness.

Myriel's rank comes from characters at the margin of the network. A cut node ({py:func}`~aryagraph.algorithms.connectivity.articulation_points`) is a character whose removal splits the network, and a bridge ({py:func}`~aryagraph.algorithms.connectivity.bridges`) is a tie whose removal disconnects it. Characters with a single tie hang from the rest of the network by a bridge:

```python
print("cut nodes:", ag.alg.articulation_points(g))
print(len(ag.alg.bridges(g)), "bridge edges")
single = Counter(next(iter(g.neighbors(v))) for v in g if g.degree(v) == 1)
print(sum(single.values()), "characters with a single tie, by their only neighbor:", single.most_common())

# Every shortest path from a single-tie character runs through its neighbor. As a share
# of all pairs of other characters (the betweenness scale), the pairs that include one:
n = g.num_nodes
for v in ("Myriel", "Valjean"):
    m = single[v]
    share = (m * (n - 1 - m) + m * (m - 1) / 2) / ((n - 1) * (n - 2) / 2)
    print(f"{v}: pairs via his {m} single-tie neighbors {share:.4f}, betweenness {betweenness[v]:.4f}")
```

```text
cut nodes: ['Myriel', 'Valjean', 'Thenardier', 'Fauchelevent', 'MmeBurgon', 'Gavroche', 'MlleGillenormand', 'Mabeuf']
18 bridge edges
17 characters with a single tie, by their only neighbor: [('Myriel', 7), ('Valjean', 5), ('Thenardier', 1), ('Fauchelevent', 1), ('MmeBurgon', 1), ('MlleGillenormand', 1), ('Mabeuf', 1)]
Myriel: pairs via his 7 single-tie neighbors 0.1768, betweenness 0.1768
Valjean: pairs via his 5 single-tie neighbors 0.1281, betweenness 0.5700
```

Seventeen of the 18 bridges lead to a character with a single tie; the eighteenth, Gavroche–MmeBurgon, cuts off MmeBurgon and Jondrette together. Seven of the single-tie characters depend on Myriel, and the last two lines show what that means for betweenness. For Myriel, the pairs that include one of the seven match his score to four decimals, so they account for all of it. For Valjean, his five single-tie neighbors account for 0.128 of his 0.570: most of his betweenness comes from linking characters who have other ties.

### Weights as distances

Betweenness above treats every tie as one step. To make frequent co-appearance count as closeness, pass a length function that shrinks as the weight grows:

```python
as_distance = lambda u, v, d: 1 / d["weight"]      # more shared chapters = closer
weighted_betweenness = ag.alg.betweenness_centrality(g, weight=as_distance)
print([(v, round(x, 3)) for v, x in weighted_betweenness.top(6)])
strongest = sorted(g.adj["Marius"].items(), key=lambda item: -item[1]["weight"])[:5]
print("Marius's strongest ties:", [(u, d["weight"]) for u, d in strongest])
```

```text
[('Valjean', 0.795), ('Marius', 0.499), ('Myriel', 0.224), ('Fantine', 0.193), ('Courfeyrac', 0.177), ('Thenardier', 0.172)]
Marius's strongest ties: [('Cosette', 21), ('Valjean', 19), ('Gillenormand', 12), ('Courfeyrac', 9), ('Enjolras', 7)]
```

Shortest paths now prefer strong ties, and Marius rises from 0.132 to 0.499: his strongest ties go to Cosette and Valjean, and he also has strong ties to the students Courfeyrac and Enjolras, so many strong-tie routes between the main plot and the barricade run through him. Whether weights should shorten paths is a modeling choice; this page uses the unweighted version elsewhere.

The figure encodes betweenness as node area and PageRank as color. It uses the default layout, stress majorization with `seed=0`, which every full-network figure on this page shares, so characters keep their positions from figure to figure.

```python
chapters = ag.by("weight", title="shared chapters")
ag.draw(
    g,
    node_size=ag.by(betweenness, title="betweenness"),
    node_color=ag.by(pagerank, kind="log", title="PageRank"),
    edge_width=chapters,
    title="Betweenness and PageRank",
    subtitle="size: betweenness centrality · color: weighted PageRank (log scale)",
).save("lesmis_centrality.svg")
```

```{figure} ../_static/generated/cases_b/lesmis_centrality.png
:alt: The Les Misérables network with node area proportional to betweenness and color by PageRank. Valjean is the largest and darkest node at the center; Myriel, Gavroche, Marius and Fantine are the next largest.

Betweenness (area) and weighted PageRank (color, log scale). Myriel's node is large because seven characters reach the story only through him.
```

## Communities

A community is a group of characters with more ties among themselves than the degrees of its members would lead you to expect. Modularity measures this excess, and {py:func}`~aryagraph.algorithms.community.louvain_communities` searches heuristically for a partition with high modularity, using the weights. {py:func}`~aryagraph.algorithms.community.community_labels` numbers the communities from the largest, and each one is named here after its member with the largest weighted degree.

```python
communities = ag.alg.louvain_communities(g, seed=0)
labels = ag.alg.community_labels(communities)      # node -> 0 (largest community) … 5
names = {i: max((v for v in g if labels[v] == i), key=strength.__getitem__) for i in range(len(communities))}
print({k: round(x, 3) for k, x in ag.alg.partition_quality(g, communities).items()})
for i, c in enumerate(communities):
    members = sorted(c, key=lambda v: (-strength[v], v))
    print(f"{i} {names[i]:<11} {len(c):>2}: {', '.join(members)}")
```

```text
{'coverage': 0.764, 'performance': 0.862, 'modularity': 0.567}
0 Valjean     22: Valjean, Marius, Cosette, Javert, Gillenormand, MlleGillenormand, Fauchelevent, LtGillenormand, Woman2, MotherInnocent, Toussaint, Pontmercy, Woman1, BaronessT, Gribier, MmePontmercy, Gervais, Isabeau, Labarre, MlleVaubois, MmeDeR, Scaufflaire
1 Enjolras    17: Enjolras, Courfeyrac, Combeferre, Bossuet, Gavroche, Joly, Bahorel, Feuilly, Prouvaire, Grantaire, Mabeuf, MmeHucheloup, Child1, Child2, MmeBurgon, MotherPlutarch, Jondrette
2 Fantine     11: Fantine, Favourite, Tholomyes, Blacheville, Dahlia, Fameuil, Listolier, Zephine, Simplice, Marguerite, Perpetue
3 Thenardier  11: Thenardier, MmeThenardier, Babet, Gueulemer, Claquesous, Eponine, Brujon, Montparnasse, Anzelma, Magnon, Boulatruelle
4 Myriel      10: Myriel, MmeMagloire, MlleBaptistine, Count, Champtercier, CountessDeLo, Cravatte, Geborand, Napoleon, OldMan
5 Judge        6: Champmathieu, Judge, Bamatabois, Brevet, Chenildieu, Cochepaille
```

The partition has a modularity of 0.567, and 76.4% of the ties fall inside communities (the coverage). The six communities follow the story arcs of the novel:

| # | Named after | Size | Who is in it |
|---|---|---|---|
| 0 | Valjean | 22 | The main plot: Valjean, Marius, Cosette, Javert, the Gillenormand family, Fauchelevent and the convent |
| 1 | Enjolras | 17 | The students of the Friends of the ABC, with Gavroche, Mabeuf and others around the barricade |
| 2 | Fantine | 11 | Fantine, Tholomyès and his friends and their companions, and the people who care for Fantine at the end of her life |
| 3 | Thenardier | 11 | The Thénardier family and the Patron-Minette gang |
| 4 | Myriel | 10 | Bishop Myriel, his household and the characters of the opening book |
| 5 | Judge | 6 | Champmathieu's trial at Arras. Judge and Champmathieu tie on weighted degree (14 each), and Judge comes first in the dataset |

The figure colors each character by community, with node area for weighted degree and edge width for the number of shared chapters. Saved as HTML, it becomes interactive: hover a character to see its neighborhood, search by name, click legend entries to filter communities, or open a sortable table of every character.

```python
community = ag.by({v: f"{labels[v]} · {names[labels[v]]}" for v in g}, kind="categorical", title="community")
fig = ag.draw(
    g,
    node_color=community,
    node_size=ag.by(strength, title="weighted degree"),
    edge_width=chapters,
    title="Les Misérables: six communities",
    subtitle=f"Louvain (seed 0), modularity {ag.alg.modularity(g, communities):.3f} · size: weighted degree · width: shared chapters",
)
fig.save("lesmis_communities.html")   # interactive; .svg, .png and .pdf work too
```

<iframe class="ag-embed" src="../_static/generated/cases_b/lesmis_communities.html" height="780" loading="lazy" title="Interactive Les Misérables network colored by community"></iframe>
<p class="ag-embed-note">Six Louvain communities; the legend lists the number of characters in each. Scroll to zoom, drag to pan, click a character to pin its neighborhood. <a href="../_static/generated/cases_b/lesmis_communities.html">Open full screen</a></p>

### How stable is the partition?

The [communities tutorial](../tutorials/communities.md) compares Louvain with greedy modularity and label propagation on this network and runs Louvain with ten seeds. Those runs return two partitions: the one above (seed 0) and the one from seed 3. This section checks which characters they disagree on:

```python
alt = ag.alg.louvain_communities(g, seed=3)
for seed, parts in [(0, communities), (3, alt)]:
    print(f"seed {seed}: sizes", [len(c) for c in parts], f"modularity {ag.alg.modularity(g, parts):.4f}")
alt_labels = ag.alg.community_labels(alt)
match = {b: Counter(labels[v] for v in g if alt_labels[v] == b).most_common(1)[0][0] for b in set(alt_labels.values())}
print("assigned differently:", [v for v in g if match[alt_labels[v]] != labels[v]])
```

```text
seed 0: sizes [22, 17, 11, 11, 10, 6] modularity 0.5667
seed 3: sizes [24, 17, 11, 10, 9, 6] modularity 0.5658
assigned differently: ['Perpetue', 'Simplice']
```

The two partitions differ in two characters. Simplice and Perpetue, the nuns who nurse Fantine, move from Fantine's community to Valjean's, and the modularity changes from 0.5667 to 0.5658. For a report, state the method and seed, and mention the characters whose assignment changes.

## Bridges between communities

Betweenness counts shortest paths, but a character can score high by connecting a few isolated characters inside one community. To measure how evenly a character's ties spread across communities, use the participation coefficient $P = 1 - \sum_c (s_c / s)^2$, where $s_c$ is the number of chapters the character shares with members of community $c$ and $s$ is the character's weighted degree. $P$ is 0 when every tie stays in one community and grows as ties spread over more communities. The next block prints the rows of the table below for the ten characters with the highest betweenness.

```python
def participation(v):
    """1 - sum over communities of (share of v's co-appearances with that community)^2."""
    share = Counter()
    for u in g.neighbors(v):
        share[labels[u]] += g.edges[v, u]["weight"] / strength[v]
    return 1 - sum(s**2 for s in share.values()), len(share)

print("| character | community | betweenness | communities reached | participation |")
print("|---|---|---|---|---|")
for v, b in betweenness.top(10):
    p, reached = participation(v)
    print(f"| {v} | {labels[v]} · {names[labels[v]]} | {b:.3f} | {reached} | {p:.2f} |")
```

| character | community | betweenness | communities reached | participation |
|---|---|---|---|---|
| Valjean | 0 · Valjean | 0.570 | 6 | 0.63 |
| Myriel | 4 · Myriel | 0.177 | 2 | 0.27 |
| Gavroche | 1 · Enjolras | 0.165 | 3 | 0.33 |
| Marius | 0 · Valjean | 0.132 | 4 | 0.54 |
| Fantine | 2 · Fantine | 0.130 | 4 | 0.53 |
| Thenardier | 3 · Thenardier | 0.075 | 4 | 0.49 |
| Javert | 0 · Valjean | 0.054 | 5 | 0.69 |
| MlleGillenormand | 0 · Valjean | 0.048 | 1 | 0.00 |
| Enjolras | 1 · Enjolras | 0.043 | 3 | 0.32 |
| Tholomyes | 2 · Fantine | 0.041 | 2 | 0.14 |

- **Valjean** is the bridge of the novel: his ties reach all six communities.
- **Javert** ranks only seventh by betweenness but has the highest participation of the ten (0.69), with ties in five communities: the inspector turns up in almost every story arc.
- **Marius, Fantine and Thenardier** each reach four communities.
- **MlleGillenormand** shows why betweenness alone is not a measure of bridging. Her betweenness comes from MlleVaubois, whose only tie is to her, while all her own ties stay inside community 0, so her participation is 0.

How much of the story happens between communities?

```python
inter = [(u, v, d["weight"]) for u, v, d in g.edges.data() if labels[u] != labels[v]]
print(f"{len(inter)} of {g.num_edges} ties join two communities, "
      f"{sum(w for *_, w in inter)} of {total} co-appearances")
between = Counter()
for u, v, w in inter:
    between[tuple(sorted((labels[u], labels[v])))] += w
print(between.most_common())
crossing = Counter(x for u, v, _ in inter for x in (u, v))
print("crossing ties per character:", crossing.most_common(5))
print(sum(1 for v in g if crossing[v] == 0), "characters have no crossing tie")
```

```text
60 of 254 ties join two communities, 155 of 820 co-appearances
[((0, 3), 48), ((0, 1), 48), ((0, 2), 21), ((0, 5), 15), ((0, 4), 11), ((1, 3), 8), ((2, 3), 3), ((2, 5), 1)]
crossing ties per character: [('Valjean', 21), ('Marius', 12), ('Javert', 11), ('Gavroche', 8), ('Thenardier', 7)]
40 characters have no crossing tie
```

About a quarter of the ties (60 of 254) cross between communities, but they are weaker than average and carry 155 of the 820 co-appearances (19%). Every community is linked to Valjean's, and the strongest links run from his community to the Thénardiers and to the students (48 co-appearances each). Myriel's community is linked to Valjean's alone.

The first figure draws the 60 crossing ties in dark gray over light gray ties inside communities, and sizes each character by the number of crossing ties it has. The community graph collapses each community into one node.

```python
ag.draw(
    g,
    node_color=community,
    node_size=ag.by({v: crossing[v] for v in g}, title="ties to other communities"),
    edge_color=ag.by(
        lambda u, v, d: "between communities" if labels[u] != labels[v] else "within a community",
        kind="categorical",
        palette={"between communities": "#1f1e1c", "within a community": "#d3d1c8"},
        title="tie",
    ),
    edge_width=chapters,
    title="Bridges between communities",
    subtitle=f"dark: the {len(inter)} ties that join two communities · size: ties to other communities",
).save("lesmis_bridges.svg")

cg = ag.Graph(name="Les Misérables communities")
for i, c in enumerate(communities):
    cg.add_node(f"{i} · {names[i]}", members=len(c))
for (a, b), w in between.items():
    cg.add_edge(f"{a} · {names[a]}", f"{b} · {names[b]}", weight=w)
ag.draw(
    cg,
    node_color=ag.by({n: n for n in cg}, kind="categorical", legend=False),
    node_size=ag.by("members", title="characters"),
    edge_width=ag.by("weight", title="shared chapters"),
    edge_label="weight",
    layout="stress",
    title="The six communities as one graph",
    subtitle="size: characters in the community · width and label: co-appearances between communities",
).save("lesmis_community_graph.svg")
```

```{figure} ../_static/generated/cases_b/lesmis_bridges.png
:alt: The Les Misérables network with nodes colored by community and sized by their number of ties to other communities. Dark edges, the ties between communities, fan out mostly from Valjean, Marius and Javert, the three largest nodes; ties inside communities are light gray.

Ties between communities in dark gray, with node area for each character's number of such ties. Valjean is at one end of 21 of them, Marius of 12 and Javert of 11; 40 of the 77 characters have none.
```

```{figure} ../_static/generated/cases_b/lesmis_community_graph.png
:alt: Six community nodes. Valjean's community is in the middle and linked to all five others; the thickest links, labeled 48, go to the Thénardier and Enjolras communities; Myriel's community hangs from Valjean's alone.
:width: 80%

The community graph: node area is the number of characters, and edge labels count co-appearances between communities.
```

## k-cores

The k-core is the largest part of the network in which every character has at least k ties to other members of the part. A character's core number is the largest k for which it belongs to the k-core, so peeling the network core by core separates the dense center from the periphery. {py:func}`~aryagraph.algorithms.structure.core_number` computes every core number, and {py:func}`~aryagraph.algorithms.structure.k_core` returns the innermost core.

```python
core = ag.alg.core_number(g)
print("characters per core number:", sorted(Counter(core.values()).items()))
inner = ag.alg.k_core(g)                      # the innermost core
print(inner, "density", round(ag.alg.density(inner), 3))
print(sorted(inner))
print("communities:", Counter(names[labels[v]] for v in inner))
print("Valjean's core number:", core["Valjean"])
```

```text
characters per core number: [(1, 18), (2, 11), (3, 7), (4, 3), (6, 7), (7, 11), (8, 8), (9, 12)]
<Graph 'Les Misérables': 12 nodes, 62 edges> density 0.939
['Bahorel', 'Bossuet', 'Combeferre', 'Courfeyrac', 'Enjolras', 'Feuilly', 'Gavroche', 'Grantaire', 'Joly', 'Mabeuf', 'Marius', 'Prouvaire']
communities: Counter({'Enjolras': 11, 'Valjean': 1})
Valjean's core number: 8
```

18 characters have core number 1: they hang from the network by a single tie or a chain of single ties, like the seven characters tied only to Myriel. At the other end, the 9-core holds 12 characters with 62 of the 66 possible ties among them, a density of 0.939: eleven members of the students' community and Marius, who joins them at the barricade. Valjean has core number 8, so although he is the most central character by every measure above, he is not part of the densest group.

```python
kmax = max(core.values())
innermost = [v for v, c in core.items() if c == kmax]
ag.draw(
    g,
    node_color=ag.by(core, kind="sequential", title="core number"),
    highlight=innermost,
    title=f"k-cores: the {kmax}-core",
    subtitle=f"color: core number · highlighted: the {len(innermost)} characters of the innermost core",
).save("lesmis_cores.svg")
```

```{figure} ../_static/generated/cases_b/lesmis_cores.png
:alt: The Les Misérables network colored by core number, with the 12 characters of the 9-core highlighted in dark blue at the lower left (Marius, Enjolras, Gavroche, Mabeuf and the students) and every other character dimmed.

The innermost core: Marius and the students of the barricade. The highlight dims the other characters.
```

## Valjean's ego network

An ego network is a character, its neighbors, and the ties among those neighbors. Removing the ego from it shows whether the neighbors know each other without him.

```python
ego = g.subgraph(["Valjean", *g.neighbors("Valjean")])
alters = g.subgraph(g.neighbors("Valjean"))
print(ego)
print(Counter(names[labels[v]] for v in alters))
parts = ag.alg.connected_components(alters)
print(len(parts), "groups without Valjean, sizes", sorted((len(c) for c in parts), reverse=True))
print("small groups:", [sorted(c) for c in parts if len(c) < 5])
strongest = sorted(g.adj["Valjean"].items(), key=lambda item: -item[1]["weight"])[:3]
print("Valjean's strongest ties:", [(u, d["weight"]) for u, d in strongest])
```

```text
<Graph 'Les Misérables': 37 nodes, 112 edges>
Counter({'Valjean': 15, 'Thenardier': 6, 'Judge': 6, 'Myriel': 3, 'Fantine': 3, 'Enjolras': 3})
7 groups without Valjean, sizes [28, 3, 1, 1, 1, 1, 1]
small groups: [['MlleBaptistine', 'MmeMagloire', 'Myriel'], ['Labarre'], ['MmeDeR'], ['Isabeau'], ['Gervais'], ['Scaufflaire']]
Valjean's strongest ties: [('Cosette', 31), ('Marius', 19), ('Javert', 17)]
```

Valjean's 36 neighbors come from all six communities: 15 from his own and 21 from the other five. Without him, 28 of them still form one connected group, but Myriel's household becomes a separate group of 3 and five characters (Labarre, MmeDeR, Isabeau, Gervais and Scaufflaire) are left with no tie at all. For them, Valjean is the only link to the rest of the novel.

```python
ag.draw(
    ego,
    node_color=community,
    node_size=ag.by({v: strength[v] for v in ego}, title="weighted degree"),
    edge_width=chapters,
    title="Valjean's ego network",
    subtitle=f"his {len(ego) - 1} neighbors and the ties among them · color: community in the full network",
).save("lesmis_valjean.svg")
```

```{figure} ../_static/generated/cases_b/lesmis_valjean.png
:alt: Valjean at the center of his 36 neighbors, colored by community. The Thénardier group, the trial group and Myriel's household form separate clusters around him, and five characters on the right connect only to Valjean.

Valjean's ego network, colored by the communities of the full network. Edge width counts shared chapters; the thickest tie, 31 chapters, is to Cosette.
```

## The same analysis in one call

{py:func}`~aryagraph.analysis.report.analyze` computes most of the measures above in one call:

```python
report = ag.analyze(g)
print(report)
```

```text
Graph report: Les Misérables
────────────────────────────────────────
Graph                   Undirected · 77 nodes · 254 edges
Density                 0.0868
Average degree          6.6
Max degree              36
Components              1 (largest: 77 nodes, 100%)
Diameter                5
Radius                  3
Avg path length         2.64
Avg clustering          0.573
Transitivity            0.499
Degree assortativity    -0.165
Algebraic connectivity  0.205
Communities             6 (modularity 0.567)
Top degree              Valjean (36), Gavroche (22), Marius (19)
Top pagerank            Valjean (0.0754), Myriel (0.0428), Gavroche (0.0358)
Top betweenness         Valjean (0.57), Myriel (0.177), Gavroche (0.165)
Top closeness           Valjean (0.644), Marius (0.531), Thenardier (0.517)
Top eigenvector         Gavroche (0.318), Valjean (0.268), Enjolras (0.267)
Bridges / cut nodes     18 / 8
Max k-core              9
```

The report agrees with the sections above on the structure, the betweenness ranking and the six communities. Its PageRank values differ (Valjean 0.0754 against 0.0996) because `analyze` computes centralities without weights unless you pass `weight=...`, while {py:func}`~aryagraph.algorithms.centrality.pagerank` reads the `weight` attribute by default. Communities in the report use the weights in both cases. `report.save("report.html")` writes an interactive dashboard of stat tiles, the explorable graph, distributions and rankings: <a href="../_static/generated/cases_b/lesmis_report.html">open the Les Misérables dashboard</a>.

## What the network shows

- Valjean dominates every centrality measure and is the only character with ties to all six communities.
- The measures disagree below him, and the disagreements are informative: Gavroche meets many characters briefly, the students share many chapters among themselves, and Myriel is a gatekeeper for the characters of the opening book.
- The six communities match the story arcs of the novel, and the partition is stable across seeds except for two characters on a border.
- The densest part of the network is the barricade, not the main plot.

## Where to go next

- The [communities tutorial](../tutorials/communities.md) walks through community detection step by step.
- [Algorithms](../user-guide/algorithms.md) in the user guide covers the centrality, community and structure functions used here.
- [Analysis reports](../user-guide/analysis.md) describes `ag.analyze` and its dashboard.
- [Styling](../user-guide/styling.md) explains the encodings (`ag.by`), themes and legends used in the figures.
