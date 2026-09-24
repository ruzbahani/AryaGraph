# Reproducibility and performance

A figure in a paper, a number in a report or a test in a pipeline is only useful if it comes out the same when you run the code again. This page explains what AryaGraph keeps stable from run to run, which inputs you must control yourself, and how long common operations take, with measurements and size guidance.

## The rules in brief

* **Insertion order is the graph's order.** Nodes and edges are kept in the order you added them, and algorithms, result maps, layouts and drawings follow that order. The exceptions are outputs that come from a Python set, listed under [Sets and Python's hash seed](#sets-and-pythons-hash-seed).
* **Every random process takes a `seed`.** An integer seed reproduces the result; `seed=None` draws fresh entropy.
* **Layouts and drawings default to `seed=0`**, so the same graph, options and version draw the same way, byte for byte.
* **Simulations, random generators and randomized algorithms default to `seed=None`.** Pass a seed whenever you need to repeat a result.

## Insertion order

A graph remembers the order in which nodes and edges were added, like a Python dict. Iteration, `NodeMap` results, tie-breaking in algorithms (topological orders, critical paths, greedy seed selection) and the node order of layouts all follow it. Two graphs with the same nodes and edges in a different order are equal as sets but can produce differently ordered output:

```python
import aryagraph as ag
import numpy as np

les = ag.gen.les_miserables()
shuffled = ag.Graph()
shuffled.add_nodes(reversed(list(les)))
shuffled.add_edges([(u, v, d) for u, v, d in reversed(list(les.edges.data()))])

print(list(les)[:3], list(shuffled)[:3])
pr_a, pr_b = ag.alg.pagerank(les), ag.alg.pagerank(shuffled)
print(max(abs(pr_a[n] - pr_b[n]) for n in les) < 1e-12)

a = ag.layout.compute(les, "stress")
b = ag.layout.compute(shuffled, "stress")
shift = max(np.linalg.norm(a.xy[a.index(n)] - b.xy[b.index(n)]) for n in les)
extent = np.ptp(a.xy, axis=0).max()
print(round(shift / extent, 2), round(a.meta["stress"], 1), round(b.meta["stress"], 1))
```

```text
['Napoleon', 'Myriel', 'MlleBaptistine'] ['MmeHucheloup', 'Brujon', 'Child2']
True
0.4 242.2 241.6
```

The PageRank values agree to within floating-point rounding: summing the same numbers in a different order can change the last bits. The stress layout, however, settles in a different local optimum of almost the same quality (final stress 242.2 against 241.6), and some nodes land in quite different places: the largest displacement is 40% of the drawing's larger dimension. Layout engines are deterministic for a given input, and the insertion order is part of the input. To reproduce a figure, rebuild the graph in the same order, or save it with {py:func}`ag.write <aryagraph.io.write>` as JSON, GraphML, GEXF or DOT: these files list nodes and edges in graph order and read back in that order. Edge lists do not store the node order, because nodes are created as they first appear in an edge.

## Seeds

Randomness flows through one helper, {py:func}`~aryagraph.core.utils.make_rng`, which accepts three kinds of `seed`:

| `seed` | Behavior | Recorded in `result.seed` |
|---|---|---|
| an integer | a fresh `numpy.random.Generator` seeded with it; same seed, same result | the integer |
| a `numpy.random.Generator` | used as is and advanced; two calls sharing a generator give different results | `None` |
| `None` | fresh entropy from the operating system; results differ between runs | `None` |

```python
g = ag.gen.watts_strogatz(200, 4, 0.1, seed=0)
one = ag.sim.sir(g, 0.2, 0.1, method="gillespie", seed=3)
two = ag.sim.sir(g, 0.2, 0.1, method="gillespie", seed=3)
print(np.array_equal(one.values, two.values), one.seed)

rng = np.random.default_rng(3)
first = ag.sim.sir(g, 0.2, 0.1, method="gillespie", seed=rng)
second = ag.sim.sir(g, 0.2, 0.1, method="gillespie", seed=rng)
print(np.array_equal(first.values, one.values), first.seed)
print(int(first.counts()["R"][-1]), int(second.counts()["R"][-1]))
```

```text
True 3
True None
194 185
```

The first call on a generator created with seed 3 reproduces `seed=3`; the second call continues the same random stream and gives a different epidemic (185 nodes recovered instead of 194). A generator is useful when one script runs many steps from a single seed: the whole sequence is reproducible as long as the calls happen in the same order. For many independent runs, {py:func}`~aryagraph.sim.run_ensemble` derives one child seed per run from a master seed and stores them in `ens.seeds`, so any run can be replayed alone (see [Simulation](simulation.md#ensembles)).

### Defaults

| Function family | Default `seed` | Consequence |
|---|---|---|
| layouts ({py:func}`~aryagraph.layout.compute` and every engine), {py:func}`ag.draw <aryagraph.render.draw>`, {py:func}`ag.analyze <aryagraph.analysis.report.analyze>`, `aryagraph draw` | `0` | deterministic unless you ask for another seed |
| random generators (`erdos_renyi`, `watts_strogatz`, `random_task_dag`, …) | `None` | pass `seed=` to get the same graph again |
| simulations (`ag.sim.*`), `run_ensemble` | `None` | pass `seed=` to repeat a run |
| randomized algorithms: `louvain_communities`, `label_propagation_communities`, `asyn_fluid_communities`, sampled `betweenness_centrality(k=...)`, random coloring strategies | `None` | pass `seed=` for a stable partition or estimate |

Community detection shows why the last row matters. Louvain visits nodes in a random order, and different orders can end in partitions of different quality:

```python
for s in range(5):
    parts = ag.alg.louvain_communities(les, seed=s)
    print(s, len(parts), round(ag.alg.modularity(les, parts), 4))
```

```text
0 6 0.5667
1 6 0.5667
2 6 0.5667
3 6 0.5658
4 6 0.5658
```

Without a seed, the partition can change from one run to the next, among these outcomes or others. With `seed=0`, the partition, its modularity and every figure colored by it stay the same.

### Layout seeds

Layouts that start from random positions (ForceAtlas2, Fruchterman–Reingold, random initial orders in the hierarchical engine) use the seed too. Because it defaults to 0, drawings are stable; change it to explore alternatives:

```python
fa_0 = ag.layout.compute(les, "forceatlas2")
fa_0_again = ag.layout.compute(les, "forceatlas2", seed=0)
fa_1 = ag.layout.compute(les, "forceatlas2", seed=1)
print(np.array_equal(fa_0.xy, fa_0_again.xy), np.allclose(fa_0.xy, fa_1.xy))
```

```text
True False
```

## What is and is not deterministic

With the same graph (including its insertion order), the same arguments and seeds, and the same versions of AryaGraph, Python and numpy:

| Output | Reproducible? |
|---|---|
| algorithm results (paths, centralities, CPM, flows, …) | yes |
| the cycle in a `CycleError` | yes |
| a graph built with `subgraph()` | yes: a set gives graph order, any other iterable keeps its own order |
| layouts | yes (seed 0 unless given) |
| `Figure.to_svg()`, interactive `.html`, animation players | yes, byte for byte |
| simulations and generated graphs with an integer seed | yes |
| simulations and generated graphs with `seed=None` | no |
| `analyze(g).timings` | no: wall-clock times |
| the `analyze` dashboard (`.html`) | yes, apart from the analysis time printed at its foot |
| iteration order of **set** results with string nodes | no: see below |
| PNG and PDF files | depends on the rasterizer (cairosvg or the installed browser) and its fonts |
| results on another machine or with other numpy versions | the same up to floating-point rounding; see below |

You can check the byte-for-byte claim yourself:

```python
import hashlib

def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

print(fingerprint(ag.draw(les).to_svg()) == fingerprint(ag.draw(les).to_svg()))
print(fingerprint(ag.draw(les).to_html()) == fingerprint(ag.draw(les).to_html()))
```

```text
True
True
```

Separate Python processes are a stricter test, because Python randomizes the hashing of strings in every new process (see the next section). The check below writes a small script, runs it in three processes with different values of `PYTHONHASHSEED`, and counts how many different fingerprints each output produced:

```python
import json
import os
import subprocess
import sys

probe = '''
import hashlib, json
import aryagraph as ag

def fingerprint(value):
    text = value if isinstance(value, str) else json.dumps(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

les = ag.gen.les_miserables()
ml = ag.gen.ml_pipeline()
checks = {
    "drawing (SVG)": ag.draw(les).to_svg(),
    "stress layout": ag.layout.compute(les, "stress").xy.tolist(),
    "PageRank": list(ag.alg.pagerank(les).items()),
    "betweenness": list(ag.alg.betweenness_centrality(les).items()),
    "Louvain, sorted": sorted(sorted(c) for c in ag.alg.louvain_communities(les, seed=0)),
    "Gillespie SIR": ag.sim.sir(les, 0.2, 0.1, method="gillespie", seed=3).values.tolist(),
    "Monte Carlo schedule": ag.sim.monte_carlo_schedule(ml, runs=500, seed=1).makespans.tolist(),
    "JSON export": ag.io.to_json(les),
    "a set, unsorted": list(ml.descendants("feature engineering")),
    "subgraph() of a set": list(les.subgraph({"Valjean", "Javert", "Cosette", "Marius", "Fantine"})),
}
try:
    ag.DAG([("a", "b"), ("b", "c"), ("c", "a")])
except ag.CycleError as err:
    checks["batch cycle"] = err.cycle
print(json.dumps({name: fingerprint(value) for name, value in checks.items()}))
'''
with open("probe.py", "w", encoding="utf-8") as fh:
    fh.write(probe)

runs = []
for hash_seed in ("1", "2", "3"):
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    done = subprocess.run([sys.executable, "probe.py"], env=env, capture_output=True, text=True, check=True)
    runs.append(json.loads(done.stdout))
for name in runs[0]:
    distinct = len({run[name] for run in runs})
    print(f"{name:26} {'same' if distinct == 1 else f'{distinct} different results'}")
```

```text
drawing (SVG)              same
stress layout              same
PageRank                   same
betweenness                same
Louvain, sorted            same
Gillespie SIR              same
Monte Carlo schedule       same
JSON export                same
a set, unsorted            3 different results
subgraph() of a set        same
batch cycle                same
```

Drawings, layouts, centralities, sorted partitions, seeded simulations, exports, subgraphs and reported cycles agree across the three processes. The one row that differs iterates over a Python set, which the next section explains.

### Sets and Python's hash seed

A few functions return Python sets: the communities of {py:func}`~aryagraph.algorithms.community.louvain_communities` (a list of sets), `CriticalPath.critical`, `maximum_antichain`, `lowest_common_ancestors`, `ancestors` and `descendants`. Their *content* is deterministic, but Python randomizes the hashing of strings in every new process, so the order in which you iterate over a set of string nodes can change from one run of your script to the next. Sort before printing, saving or plotting in order:

```python
parts = ag.alg.louvain_communities(les, seed=0)
print([sorted(c)[:3] for c in sorted(parts, key=len, reverse=True)[:2]])
```

```text
[['BaronessT', 'Cosette', 'Fauchelevent'], ['Bahorel', 'Bossuet', 'Child1']]
```

Set order also reaches two outputs that are not sets themselves:

Where AryaGraph itself works with sets, it restores graph order before anything reaches you:

* **The cycle in a `CycleError`.** `DAG.add_edges`, `ag.DAG(edges)`, and {py:func}`~aryagraph.algorithms.dag.topological_sort` or another DAG algorithm given a cyclic `DiGraph` find cycles with Kahn's algorithm. They trace the reported cycle from the first node, in graph order, that the algorithm could not sort, so the cycle and the error message are the same in every process.
* **A graph built from a set.** {py:meth}`Graph.subgraph <aryagraph.core.graph.Graph.subgraph>` given a set (or frozenset) lists the nodes in graph order. Given any other iterable, such as a list, it keeps that iterable's order.
* **The `analyze` dashboard** lists the members of each community by PageRank and breaks ties in graph order.

```python
chosen = {"Valjean", "Javert", "Cosette", "Marius", "Fantine"}
core = les.subgraph(chosen)
print(list(core))
```

```text
['Valjean', 'Fantine', 'Cosette', 'Javert', 'Marius']
```

Setting the environment variable `PYTHONHASHSEED=0` before starting Python makes set order repeatable as well.

### Across machines and versions

Floating-point results can differ in the last digits between CPUs, operating systems and numpy or BLAS builds, because linear algebra routines (used by the stress and spectral layouts, heat diffusion and matrix functions) may sum in a different order. Such differences are usually far below anything visible, but a layout that stops at a tolerance or an optimizer that compares close alternatives can occasionally take a different branch. For archival results:

* record `ag.__version__`, `numpy.__version__` and the Python version next to your outputs;
* save the graph itself (JSON keeps the attributes and the insertion order; tuples come back as lists), and save computed layouts as node attributes if the precise drawing matters;
* keep the seeds in your code, and read `result.seed` and `result.params` from simulation results.

```python
run = ag.sim.sir(g, 0.2, 0.1, method="gillespie", seed=3)
print(run.model, run.seed, run.params)
import sys
print(ag.__version__, np.__version__, sys.version.split()[0])
```

```text
SIR 3 {'beta': 0.2, 'gamma': 0.1, 'method': 'gillespie', 't_max': inf, 'weight': None}
0.1.0 2.4.6 3.13.13
```

## Performance

AryaGraph is written in Python with numpy for the numerical kernels; its only runtime dependency is numpy. Graphs are stored as dictionaries of dictionaries, which makes attribute access and incremental edits cheap and keeps insertion order, at the cost of memory per edge.

### Measured timings

The table below was measured with the website's asset script (`website/scripts/assets/guide_sim_data.py`, which writes `timings.json`) on one machine: an AMD Ryzen 9 9950X3D, Windows 11, Python 3.13.13, numpy 2.4.6 and AryaGraph 0.1.0. Fast operations are the median of 3 repetitions, slow ones a single run. Random graphs are `gnm_random_graph(n, m, seed=1)` unless noted.

| Operation | Size | Time |
|---|---|---|
| `bfs_order` from one node | 100,000 nodes, 500,000 edges | 0.19 s |
| `connected_components` | 100,000 nodes, 500,000 edges | 0.26 s |
| `pagerank` | 100,000 nodes, 500,000 edges | 0.39 s |
| `dijkstra` from one node | 100,000 nodes, 500,000 edges | 0.57 s |
| generate `gnm_random_graph` | 100,000 nodes, 500,000 edges | 0.88 s |
| `louvain_communities` | 10,000 nodes, 50,000 edges | 2.5 s |
| `betweenness_centrality`, exact | 1,000 nodes, 5,000 edges | 1.3 s |
| `betweenness_centrality`, exact | 3,000 nodes, 15,000 edges | 13 s |
| `betweenness_centrality`, `k=200` sampled sources | 10,000 nodes, 50,000 edges | 4.3 s |
| stress layout | 500 nodes, 1,500 edges | 0.34 s |
| stress layout | 2,000 nodes, 6,000 edges | 7.3 s |
| ForceAtlas2 layout | 2,000 nodes, 6,000 edges | 3.0 s |
| ForceAtlas2 layout | 10,000 nodes, 30,000 edges | 18 s |
| hierarchical layout (`random_task_dag`) | 300 tasks, 570 arcs | 0.90 s |
| hierarchical layout (`random_task_dag`) | 1,000 tasks, 1,895 arcs | 5.0 s |
| `ag.draw` + SVG, layout included | 1,000 nodes, 3,000 edges | 1.9 s |
| `ag.analyze` (`barabasi_albert(1000, 3)`) | 1,000 nodes, 2,991 edges | 2.1 s |
| SIR, Gillespie (`watts_strogatz(10000, 10, 0.05)`) | 10,000 nodes, 50,000 edges | 0.23 s |
| SIR, discrete, `dt = 1` | 10,000 nodes, 50,000 edges | 0.07 s |
| `monte_carlo_schedule`, 10,000 runs, unlimited workers | 200 tasks, 377 arcs | 0.09 s |
| `monte_carlo_schedule`, 1,000 runs, 4 workers | 200 tasks, 377 arcs | 0.62 s |

A graph with 100,000 nodes and 500,000 edges and no attributes occupied 130 MB, about 260 bytes per edge (measured with `tracemalloc`); attributes add to that.

Absolute times depend on the processor, the Python version and the load on the machine, so treat them as orders of magnitude. Two runs of the script a few minutes apart differed by up to 15% for the operations that take more than a second and by up to about a third for the fastest ones. A later run, made while other processes kept the same machine busy, took up to 2.3 times as long. The ratios between rows transfer better than the numbers themselves.

### Practical size limits

The guidance below follows from the algorithms' complexity and the scaling seen in the table, rather than from this machine's speed:

* **Linear-time work scales to large graphs.** Traversals, components, PageRank iterations and single-source shortest paths grow roughly with the number of edges; at 500,000 edges each took under a second here. Memory usually becomes the limit before time does: at about 260 bytes per edge on 64-bit CPython, ten million edges need about 2.6 GB before any attributes.
* **All-pairs measures grow quadratically or faster.** Exact betweenness costs O(*n*·*m*): tripling the nodes and edges multiplied the time by about 10. Beyond a few thousand nodes, use `k=` sampled sources. {py:func}`ag.analyze <aryagraph.analysis.report.analyze>` does this on its own above 3,000 nodes and notes it in the report.
* **Stress layouts are for up to a few thousand nodes.** They need all-pairs distances and O(*n*²) work per iteration; quadrupling the nodes multiplied the time by about 21. `layout="auto"` switches to ForceAtlas2 above 3,000 nodes, and hierarchical layouts are chosen for DAGs up to 2,000 nodes.
* **Drawings are for people.** Beyond a few thousand nodes a static drawing is hard to read whatever the speed. Pixel-space overlap removal runs up to 3,000 nodes; for larger graphs, draw a summary instead (the largest component, a k-core, a community graph) or pass `labels=False`.
* **Simulations scale with events, not frames.** A Gillespie run costs O(log *N*) per event plus the neighbors of the node that changed; discrete runs cost O(*N* + *m*) per step. Monte Carlo CPM without workers or resources is vectorized over the runs, while runs with workers simulate the dispatcher one at a time.

Most of AryaGraph runs in a single Python thread. numpy's linear algebra, which stress and spectral layouts and heat diffusion rely on, may use several cores depending on your numpy build.

### Measuring on your machine

The same measurements take a few lines; `time.perf_counter` and repeated runs give stable figures:

```python
import statistics
import time

def median_time(fn, repeat=3):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)

big = ag.gen.gnm_random_graph(10_000, 50_000, seed=1)
seconds = median_time(lambda: ag.alg.pagerank(big))
print(f"pagerank on {len(big):,} nodes: {seconds * 1000:.1f} ms")
```

The output of this block depends on your machine, so it is not shown here. For timings inside a report, `ag.analyze(g).timings` lists how long each analysis step took.
