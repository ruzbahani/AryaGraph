# Epidemic interventions

Two contact networks with the same number of people and almost the same number of contacts can host very different epidemics. This case study runs stochastic SIR outbreaks on a Watts–Strogatz and a Barabási–Albert network, then compares four ways of immunizing 10% of the nodes, with 200 simulated outbreaks per scenario and uncertainty on every result.

## What this case study compares

The analysis asks two questions:

1. **Structure.** With size and mean degree held fixed, how do the final size and the peak of an outbreak depend on how the contacts are arranged?
2. **Targeting.** If you can immunize 100 of the 1,000 nodes, how much does it matter which 100 you choose?

Everything below runs with {py:class}`~aryagraph.sim.compartmental.CompartmentalModel` and {py:func}`~aryagraph.sim.run_ensemble`; the code on this page took about 35 seconds to run on the machine that built this site. The numbers quoted in the text are the printed outputs of this code with the seeds shown.

## Two networks with the same size and mean degree

A Watts–Strogatz network starts from a ring in which every node links to its three nearest neighbors on each side, then rewires 10% of the edges to random targets. A Barabási–Albert network grows one node at a time, and each new node links to three existing nodes chosen in proportion to their degree. Both have 1,000 nodes and a mean degree of about 6.

```python
import numpy as np
import aryagraph as ag

N = 1000
networks = {
    "WS": ag.gen.watts_strogatz(N, 6, 0.1, seed=1),  # ring of 6 neighbors, 10% of edges rewired
    "BA": ag.gen.barabasi_albert(N, 3, seed=1),      # each new node links to 3 existing ones
}
label = {"WS": "Watts–Strogatz", "BA": "Barabási–Albert"}
for name, g in networks.items():
    k = np.array(list(g.degree().values()), dtype=float)
    print(f"{name}: {g.num_edges} edges, degree {k.min():.0f} to {k.max():.0f} "
          f"(mean {k.mean():.2f}, median {np.median(k):.0f}), mean of k² {np.mean(k**2):.1f}, "
          f"clustering {ag.alg.average_clustering(g):.3f}, mean distance {ag.alg.average_shortest_path_length(g):.2f}")
```

```text
WS: 3000 edges, degree 4 to 9 (mean 6.00, median 6), mean of k² 36.6, clustering 0.448, mean distance 6.17
BA: 2991 edges, degree 3 to 83 (mean 5.98, median 4), mean of k² 83.9, clustering 0.034, mean distance 3.48
```

The two networks differ in almost everything except their size and mean degree. The Watts–Strogatz network is clustered (neighbors of a node tend to know each other), and every node has between 4 and 9 contacts. The Barabási–Albert network has hubs: the median node has 4 contacts, the most connected one has 83, and the mean of the squared degree is more than twice as large as in the Watts–Strogatz network. Hubs also shorten paths, so the mean distance between two nodes is 3.48 steps instead of 6.17.

The charts below count the nodes by degree. For the Barabási–Albert network the rows widen as the degree grows, so the few hubs stay visible on a linear scale next to the hundreds of nodes with degree 3, and every bar prints its count.

```python
first_degree = {"WS": [4, 5, 6, 7, 8, 9], "BA": [3, 4, 5, 6, 10, 20, 50]}  # where each row starts
for name, g in networks.items():
    k = np.array(list(g.degree().values()))
    edges = [*first_degree[name], k.max() + 1]
    rows = [str(a) if b == a + 1 else f"{a}–{b - 1}" for a, b in zip(edges, edges[1:])]
    counts = [int(np.sum((k >= a) & (k < b))) for a, b in zip(edges, edges[1:])]
    print(name, dict(zip(rows, counts)))
    ag.charts.bar_chart(
        rows,
        counts,
        title=f"{label[name]}: nodes by degree",
        subtitle=f"{N:,} nodes · mean degree {k.mean():.2f} · rows: degree · bars: nodes",
        sort=False,
        width=460,
    ).save(f"degrees_{name.lower()}.svg")
```

```text
WS {'4': 24, '5': 173, '6': 612, '7': 163, '8': 26, '9': 2}
BA {'3': 404, '4': 196, '5': 98, '6–9': 199, '10–19': 71, '20–49': 23, '50–83': 9}
```

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/generated/cases_b/degrees_ws.png
:target: ../_static/generated/cases_b/degrees_ws.png
:alt: Horizontal bars counting the nodes of the Watts–Strogatz network by degree. Degrees run from 4 to 9, and 612 of the 1,000 nodes have degree 6.

Watts–Strogatz: degrees stay close to 6.
```
:::

:::{grid-item}
```{figure} ../_static/generated/cases_b/degrees_ba.png
:target: ../_static/generated/cases_b/degrees_ba.png
:alt: Horizontal bars counting the nodes of the Barabási–Albert network by degree, in rows that widen as the degree grows. 404 nodes have degree 3, and 9 hubs have degrees from 50 to 83.

Barabási–Albert: 404 nodes have the minimum degree of 3, and 9 hubs have 50 or more contacts.
```
:::
::::

## An SIR model with an immunized state

In the SIR model, a susceptible node (S) becomes infected (I) at rate β for every infected neighbor, and an infected node recovers (R) at rate γ. Read time in days: with β = 0.15 and γ = 0.25 per day, an infection lasts 4 days on average, and the chance that an infected node passes the infection to a given neighbor before recovering is the transmissibility $T = \beta / (\beta + \gamma)$.

Whether a large outbreak is possible depends on T and on the degree distribution. For a random network with a given degree sequence, the epidemic threshold is $T_c = \langle k \rangle / (\langle k^2 \rangle - \langle k \rangle)$, where $\langle k \rangle$ is the mean degree and $\langle k^2 \rangle$ the mean squared degree. Clustering, which this formula ignores, tends to raise the threshold, so treat the value for the clustered Watts–Strogatz network as a guide.

```python
beta, gamma = 0.15, 0.25
T = beta / (beta + gamma)   # chance that an infected node infects a given neighbor
print(f"T = {T:.3f}")
for name, g in networks.items():
    k = np.array(list(g.degree().values()), dtype=float)
    print(f"{name}: epidemic threshold T_c ≈ {k.mean() / (np.mean(k**2) - k.mean()):.3f}")
```

```text
T = 0.375
WS: epidemic threshold T_c ≈ 0.196
BA: epidemic threshold T_c ≈ 0.077
```

T is above both thresholds, so both networks can sustain a large outbreak. It is about 1.9 times the Watts–Strogatz threshold and about 4.9 times the Barabási–Albert threshold, a difference in margin that helps explain the results below.

Immunization adds a fourth state, V, with no transition in or out: an immunized node can neither catch nor pass on the infection, which has the same effect on the dynamics as removing it from the network. {py:class}`~aryagraph.sim.compartmental.CompartmentalModel` accepts any list of states, so the model is four lines:

```python
model = ag.sim.CompartmentalModel(
    ["S", "I", "R", "V"],               # V: immunized, with no transition in or out
    spontaneous=[("I", "R", gamma)],
    induced=[("S", "I", "I", beta)],
    roles={"V": "accent"},              # draw immunized nodes in the accent color
)
print(model)
```

```text
<CompartmentalModel SIRV: I→R (0.25), S→I (0.15 × I)>
```

Every run below uses `method="gillespie"`, the continuous-time stochastic simulation, and starts with 5 infected nodes chosen at random. The final size of an outbreak is the number of nodes in R when no infected node is left, the 5 initial infections included.

## How structure shapes the outbreak

{py:func}`~aryagraph.sim.run_ensemble` repeats a simulation with independent child seeds drawn from one master seed, and returns an {py:class}`~aryagraph.sim.ensemble.EnsembleResult` with the count of every state in every run on a common time grid. The helper `outbreak` reads three numbers per run from it.

```python
RUNS = 200

def outbreak(ens):
    """Final size, peak prevalence (both as shares of N) and peak day of every run."""
    infected = ens.samples[:, :, ens.states.index("I")]
    return {
        "final": ens.final_sizes["R"] / N,
        "peak": infected.max(axis=1) / N,
        "peak_day": ens.times[infected.argmax(axis=1)],
    }

baseline = {
    name: ag.sim.run_ensemble(model.simulate, runs=RUNS, g=g, initial={"I": 5},
                              method="gillespie", seed=7, n_points=301)
    for name, g in networks.items()
}
for name, ens in baseline.items():
    o = outbreak(ens)
    f5, f95 = np.percentile(o["final"], [5, 95])
    p5, p95 = np.percentile(o["peak"], [5, 95])
    print(f"{name}: final size {o['final'].mean():.1%} ({f5:.1%}–{f95:.1%}), "
          f"peak {o['peak'].mean():.1%} ({p5:.1%}–{p95:.1%}) on day {np.median(o['peak_day']):.1f}, "
          f"{np.mean(o['final'] < 0.05):.1%} of runs stay below 5%")
```

```text
WS: final size 68.5% (46.1%–81.5%), peak 10.1% (5.1%–14.8%) on day 25.4, 2.5% of runs stay below 5%
BA: final size 80.3% (76.2%–85.3%), peak 31.8% (27.2%–36.4%) on day 7.5, 1.5% of runs stay below 5%
```

The ranges in parentheses are the 5th and 95th percentiles over the 200 runs, and the peak day is the median over runs. With the same number of contacts:

- **The Barabási–Albert outbreak is faster and sharper.** On average 31.8% of the nodes are infected at the same time at its peak, about three times the 10.1% of the Watts–Strogatz outbreak, and the median peak comes on day 7.5 instead of day 25.4. Hubs are infected early, because they have many neighbors, and then infect many nodes each.
- **It also reaches more nodes,** 80.3% against 68.5% on average.
- **The Watts–Strogatz outcome is less predictable.** Its final size ranges from 46.1% to 81.5% between the 5th and 95th percentiles, against 76.2% to 85.3%. Infection there spreads mostly along the clustered ring, so the outcome depends more on when it reaches the random shortcuts.

To draw both ensembles on one chart, `infected_on` resamples every run onto the same days, and {py:func}`~aryagraph.charts.line_chart` draws the mean with a 5–95% band.

```python
def infected_on(ens, days):
    """Share infected in every run on a common time grid (counts hold between frames)."""
    idx = np.clip(np.searchsorted(ens.times, days, side="right") - 1, 0, None)
    return ens.samples[:, idx, ens.states.index("I")] / N

days = np.linspace(0, 100, 401)
series, bands = {}, {}
for name, ens in baseline.items():
    share = infected_on(ens, days) * 100
    series[label[name]] = share.mean(axis=0)
    bands[label[name]] = (np.percentile(share, 5, axis=0), np.percentile(share, 95, axis=0))
ag.charts.line_chart(
    days,
    series,
    bands=bands,
    band_label="5–95% of runs",
    title="Share of nodes infected over time",
    subtitle=f"SIR, β = {beta}, γ = {gamma} · {RUNS} runs per network · mean with 5–95% band",
    x_label="day",
    y_label="% infected",
    width=680,
    height=320,
).save("curves_structure.svg")
```

```{figure} ../_static/generated/cases_b/curves_structure.png
:alt: Line chart of the share of nodes infected over 100 days. The Barabási–Albert curve rises steeply to a peak near day 8 and falls by day 40; the Watts–Strogatz curve rises slowly to a lower, broader peak near day 25, with a much wider band.

Mean share of nodes infected, with the band holding the middle 90% of runs at each day. The mean curves peak lower than the average single run because runs peak on different days.
```

## Four ways to immunize 10% of the nodes

Each strategy immunizes K = 100 nodes before the outbreak starts:

- **Random:** 100 nodes drawn uniformly. This needs no knowledge of the network.
- **Acquaintance:** pick a node at random and immunize one of its neighbors, and repeat until 100 distinct nodes are immunized. This needs only local knowledge (people can name a contact), yet it favors well-connected nodes, because a node with many neighbors is named more often.
- **Highest degree:** the 100 nodes with the most contacts. This needs the degree of every node.
- **Highest betweenness:** the 100 nodes that lie on the most shortest paths, from {py:func}`~aryagraph.algorithms.centrality.betweenness_centrality`. This needs the whole network.

The two randomized strategies draw a new set of immunized nodes in every run, so their ensembles average over the choice of nodes as well as over the course of the epidemic. The runner receives the seed from {py:func}`~aryagraph.sim.run_ensemble` and uses one generator both to choose the nodes and to run the simulation.

```python
K = N // 10   # immunize 100 nodes
names = {"none": "no immunization", "random": "random", "acquaintance": "acquaintance",
         "degree": "highest degree", "betweenness": "highest betweenness"}

def random_targets(g, rng):
    nodes = list(g)
    return [nodes[i] for i in rng.choice(len(nodes), size=K, replace=False)]

def acquaintance_targets(g, rng):
    """Pick a node at random and immunize one of its neighbors, until K are immunized."""
    nodes, chosen = list(g), set()
    while len(chosen) < K:
        v = nodes[rng.integers(len(nodes))]
        neighbors = list(g.neighbors(v))
        chosen.add(neighbors[rng.integers(len(neighbors))])
    return list(chosen)

def immunize_and_simulate(g, pick, seed):
    rng = np.random.default_rng(seed)   # a new immunized set in every run
    return model.simulate(g, initial={"V": pick(g, rng), "I": 5}, method="gillespie", seed=rng)
```

The targeted strategies use one fixed list per network. Explicit nodes in `initial` are placed first, and the 5 initial infections are then drawn from the remaining nodes.

```python
ensembles = {}
for name, g in networks.items():
    betweenness = ag.alg.betweenness_centrality(g)
    targets = {
        "degree": [v for v, _ in g.degree().top(K)],
        "betweenness": [v for v, _ in betweenness.top(K)],
    }
    shared = len(set(targets["degree"]) & set(targets["betweenness"]))
    print(f"{name}: {shared} of the {K} highest-degree nodes are also among the {K} highest-betweenness nodes")
    ensembles[name] = {"none": baseline[name]}
    for strategy, pick in [("random", random_targets), ("acquaintance", acquaintance_targets)]:
        ensembles[name][strategy] = ag.sim.run_ensemble(
            immunize_and_simulate, runs=RUNS, g=g, pick=pick, seed=7, n_points=301)
    for strategy, nodes in targets.items():
        ensembles[name][strategy] = ag.sim.run_ensemble(
            model.simulate, runs=RUNS, g=g, initial={"V": nodes, "I": 5},
            method="gillespie", seed=7, n_points=301)
```

```text
WS: 42 of the 100 highest-degree nodes are also among the 100 highest-betweenness nodes
BA: 89 of the 100 highest-degree nodes are also among the 100 highest-betweenness nodes
```

## Results: final size and peak

The next block prints the rows of the table below, where WS stands for Watts–Strogatz and BA for Barabási–Albert. The reduction is the drop in mean final size relative to no immunization, in percentage points. Its ± value is a 95% interval for the difference of two means (1.96 standard errors, treating the two ensembles as independent samples), while the ranges in the final-size column describe how much single runs vary. The column *runs below 5%* counts minor outbreaks, runs in which fewer than 5% of the nodes were ever infected.

```python
print("| network | strategy | final size (5–95% of runs) | mean peak | runs below 5% | reduction, points (95% CI) | relative |")
print("|---|---|---|---|---|---|---|")
for name, by_strategy in ensembles.items():
    base = outbreak(by_strategy["none"])["final"]
    for strategy, ens in by_strategy.items():
        o = outbreak(ens)
        f = o["final"]
        lo, hi = np.percentile(f, [5, 95])
        cells = [name, names[strategy], f"{f.mean():.1%} ({lo:.1%}–{hi:.1%})",
                 f"{o['peak'].mean():.1%}", f"{np.mean(f < 0.05):.1%}"]
        if strategy == "none":
            cells += ["", ""]
        else:
            diff = base.mean() - f.mean()
            half = 1.96 * np.sqrt(base.var(ddof=1) / base.size + f.var(ddof=1) / f.size)
            cells += [f"{100 * diff:.1f} ± {100 * half:.1f}", f"{diff / base.mean():.0%}"]
        print("| " + " | ".join(cells) + " |")
```

| network | strategy | final size (5–95% of runs) | mean peak | runs below 5% | reduction, points (95% CI) | relative |
|---|---|---|---|---|---|---|
| WS | no immunization | 68.5% (46.1%–81.5%) | 10.1% | 2.5% |  |  |
| WS | random | 36.4% (3.6%–58.1%) | 4.9% | 9.0% | 32.1 ± 3.2 | 47% |
| WS | acquaintance | 39.4% (5.1%–60.4%) | 5.5% | 5.0% | 29.1 ± 3.2 | 42% |
| WS | highest degree | 20.7% (2.2%–41.0%) | 3.2% | 13.5% | 47.8 ± 2.8 | 70% |
| WS | highest betweenness | 16.1% (2.1%–39.1%) | 2.7% | 19.5% | 52.4 ± 2.7 | 76% |
| BA | no immunization | 80.3% (76.2%–85.3%) | 31.8% | 1.5% |  |  |
| BA | random | 67.4% (62.8%–72.2%) | 24.7% | 1.0% | 12.8 ± 1.7 | 16% |
| BA | acquaintance | 51.8% (44.1%–61.1%) | 14.5% | 4.0% | 28.5 ± 2.1 | 35% |
| BA | highest degree | 5.9% (0.6%–17.7%) | 1.4% | 59.0% | 74.4 ± 1.6 | 93% |
| BA | highest betweenness | 6.0% (0.7%–16.2%) | 1.4% | 58.0% | 74.2 ± 1.6 | 92% |

On the **Barabási–Albert** network, targeting decides the outcome:

- Immunizing the 100 most connected nodes cut the mean final size from 80.3% to 5.9%, a relative reduction of 93%, and in 59% of the runs fewer than 5% of the nodes were ever infected.
- Immunizing 100 random nodes removed 12.8 ± 1.7 points, 16% of the outbreak. Half of the nodes have at most 4 contacts, so a random pick is usually a poorly connected node, and most of the hubs that drive the epidemic remain unprotected.
- Acquaintance immunization removed 28.5 ± 2.1 points, more than twice the effect of random immunization, without any knowledge of the network beyond who knows whom.

On the **Watts–Strogatz** network, the picture is different:

- Random immunization is relatively more effective: it removed 32.1 ± 3.2 points, 47% of the outbreak. This network runs closer to its epidemic threshold, where the same loss of transmission paths has a larger effect.
- Acquaintance immunization removed 29.1 ± 3.2 points, close to random immunization. When every node has between 4 and 9 contacts, a random neighbor is not much better connected than a random node.
- Targeting still helps most, with relative reductions of 70% by degree and 76% by betweenness, but even then the 95th percentile of the final size stays near 40%.

### Degree or betweenness?

The table suggests that betweenness targeting did slightly better on the Watts–Strogatz network. The difference of the two means, with its own interval, shows whether the gap is larger than the sampling noise:

```python
for name in networks:
    by_degree = outbreak(ensembles[name]["degree"])["final"]
    by_betweenness = outbreak(ensembles[name]["betweenness"])["final"]
    diff = by_degree.mean() - by_betweenness.mean()
    half = 1.96 * np.sqrt(by_degree.var(ddof=1) / RUNS + by_betweenness.var(ddof=1) / RUNS)
    print(f"{name}: degree targeting minus betweenness targeting: {100 * diff:.1f} ± {100 * half:.1f} points")
```

```text
WS: degree targeting minus betweenness targeting: 4.6 ± 2.5 points
BA: degree targeting minus betweenness targeting: -0.1 ± 1.1 points
```

On the Watts–Strogatz network, betweenness targeting left 4.6 ± 2.5 points fewer infections than degree targeting. Degrees vary little there, and the two lists of 100 nodes share only 42 members, so betweenness, which scores nodes by the shortest paths through them, picks a different and more effective set. On the Barabási–Albert network, the lists share 89 nodes and the two strategies are indistinguishable (−0.1 ± 1.1 points). Degree is cheaper to obtain: it needs only a count of each node's contacts, while betweenness needs the whole network.

### Charts of the comparison

{py:func}`~aryagraph.charts.bar_chart` shows the mean final size per strategy, and {py:func}`~aryagraph.charts.line_chart` the mean share infected over time.

```python
horizon = {"WS": 100, "BA": 60}
for name, by_strategy in ensembles.items():
    ag.charts.bar_chart(
        [names[s] for s in by_strategy],
        [outbreak(e)["final"].mean() for e in by_strategy.values()],
        title=f"{label[name]}: final outbreak size",
        subtitle=f"mean share of nodes ever infected, {RUNS} runs each",
        value_format=lambda v: f"{v:.0%}" if v >= 0.1 else f"{v:.1%}",
        sort=False,
        width=460,
    ).save(f"final_{name.lower()}.svg")
    t = np.linspace(0, horizon[name], 301)
    ag.charts.line_chart(
        t,
        {names[s]: infected_on(e, t).mean(axis=0) * 100 for s, e in by_strategy.items()},
        title=f"{label[name]}: mean share infected",
        subtitle=f"10% of nodes immunized ({K} of {N:,}) · {RUNS} runs per strategy",
        x_label="day",
        y_label="% infected",
        width=680,
        height=320,
    ).save(f"curves_{name.lower()}.svg")
```

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/generated/cases_b/final_ws.png
:target: ../_static/generated/cases_b/final_ws.png
:alt: Bar chart of mean final outbreak size on the Watts–Strogatz network by strategy, from 68% with no immunization to 16% with highest-betweenness immunization.

Watts–Strogatz: mean final size per strategy.
```
:::

:::{grid-item}
```{figure} ../_static/generated/cases_b/final_ba.png
:target: ../_static/generated/cases_b/final_ba.png
:alt: Bar chart of mean final outbreak size on the Barabási–Albert network by strategy, from 80% with no immunization to about 6% with highest-degree or highest-betweenness immunization.

Barabási–Albert: mean final size per strategy.
```
:::
::::

::::{tab-set}

:::{tab-item} Barabási–Albert
```{figure} ../_static/generated/cases_b/curves_ba.png
:alt: Line chart of the mean share infected over 60 days on the Barabási–Albert network for five strategies. Without immunization the curve peaks near 29% around day 8; random immunization lowers the peak to about 22%; acquaintance immunization to about 12% around day 13; the two targeted strategies stay near zero.

Barabási–Albert: mean share infected per strategy. The two targeted curves overlap near zero.
```
:::

:::{tab-item} Watts–Strogatz
```{figure} ../_static/generated/cases_b/curves_ws.png
:alt: Line chart of the mean share infected over 100 days on the Watts–Strogatz network for five strategies, with every immunization strategy lowering and flattening the curve and the targeted strategies lowest.

Watts–Strogatz: mean share infected per strategy.
```
:::
::::

## One run on the network

Averages hide what an outbreak looks like. The next block runs the model once on the Barabási–Albert network with random immunization and once with the 100 highest-degree nodes immunized, then saves each run as an interactive player with {py:meth}`~aryagraph.sim.base.SimulationResult.animate`.

```python
ba = networks["BA"]
rng = np.random.default_rng(1)
runs = {
    "random": model.simulate(ba, initial={"V": random_targets(ba, rng), "I": 5}, method="gillespie", seed=rng),
    "degree": model.simulate(ba, initial={"V": [v for v, _ in ba.degree().top(K)], "I": 5},
                             method="gillespie", seed=1),
}
for strategy, run in runs.items():
    infected = run.summary()["R"]["final"]
    print(strategy, run)
    player = run.animate(
        labels=False,
        node_size=ag.by(ba.degree(), title="degree"),
        title=f"Barabási–Albert network, 10% immunized: {names[strategy]}",
        subtitle=f"one run · blue: immunized (V) · green: infected, then recovered (R) · {infected} of {N:,} infected",
        width=620,
    )
    player.save(f"network_ba_{strategy}.html")
```

```text
random <SimulationResult SIRV: 1000 nodes, 101 frames, t=0…47.2; final S=220, I=0, R=680, V=100>
degree <SimulationResult SIRV: 1000 nodes, 101 frames, t=0…19.88; final S=878, I=0, R=22, V=100>
```

Both runs are typical of their ensembles: 680 infections with random immunization, where the ensemble mean is 67.4%, and 22 with degree targeting, where 59% of the runs stay below 50 infections. The still images show the last frame, with node area proportional to degree; click one to open it at full size.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
```{figure} ../_static/generated/cases_b/network_ba_random.png
:target: ../_static/generated/cases_b/network_ba_random.png
:alt: The 1,000-node Barabási–Albert network after an outbreak with random immunization. Blue immunized nodes are scattered over the network, most nodes, including most of the large hubs, are green (infected, then recovered), and a minority are gray (never infected).

Random immunization: the blue nodes are spread over the whole network, most hubs are not immunized, and 680 nodes end up infected.
```
:::

:::{grid-item}
```{figure} ../_static/generated/cases_b/network_ba_degree.png
:target: ../_static/generated/cases_b/network_ba_degree.png
:alt: The same Barabási–Albert network after an outbreak with the 100 highest-degree nodes immunized. The large central hubs are blue, almost every other node is gray (never infected), and a few scattered nodes are green.

Degree targeting: the immunized hubs sit in the core of the network, and the outbreak stops after 22 infections.
```
:::
::::

In the player below you can press play, scrub along the timeline, and watch the transmissions flash along the edges. The chart under the network follows the state counts.

<iframe class="ag-embed" src="../_static/generated/cases_b/network_ba_random.html" height="1100" loading="lazy" title="Animated SIR outbreak on a Barabási–Albert network with 10% random immunization"></iframe>
<p class="ag-embed-note">One run on the Barabási–Albert network with 100 randomly immunized nodes. <a href="../_static/generated/cases_b/network_ba_random.html">Open full screen</a>, or open <a href="../_static/generated/cases_b/network_ba_degree.html">the run with degree targeting</a>.</p>

## What these results do and do not show

The comparison is controlled: both networks have 1,000 nodes and nearly the same number of edges, every scenario uses the same model, 5 initial infections chosen at random and 200 independent runs, and every ensemble reproduces from `seed=7`. Some limits apply before you carry the conclusions to real populations:

- **Two synthetic networks.** Real contact networks combine clustering, hubs, communities and contacts that change over time. The two models used here isolate clustering with homogeneous degrees (Watts–Strogatz) and hubs without clustering (Barabási–Albert).
- **Complete network knowledge.** The degree and betweenness strategies assume you know the network. The acquaintance strategy shows how much of the benefit remains with local information only.
- **One parameter set.** With T = 0.375 both networks are above threshold. The effect of each strategy depends on how far T is above the threshold, so rerun the comparison with the parameters that fit your question.
- **Immunity before the outbreak.** Immunized nodes are protected from time 0, completely and permanently. Reactive vaccination during an outbreak is a different question.
- **Sampling uncertainty.** The ± intervals cover the uncertainty of the mean from 200 runs. They do not cover uncertainty in the model or its parameters.

To explore further, change `K`, `beta` or the network generators and rerun the page's code: every number above is recomputed from them.

## Where to go next

- The [epidemics tutorial](../tutorials/epidemics.md) introduces the SIR family step by step.
- [Simulation](../user-guide/simulation.md) in the user guide covers the solvers, recording options and ensembles in depth.
- [Charts](../user-guide/charts.md) describes the chart functions used for the figures on this page.
- The [influence tutorial](../tutorials/influence.md) looks at the opposite problem: choosing the nodes that spread a message furthest.
