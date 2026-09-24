# Influence and cascades

Suppose a rumor starts with a handful of characters in *Les Misérables*: how far does it travel, and which characters should hear it first to reach the most people? In this tutorial you simulate two classic spreading models, the independent cascade and the linear threshold model, estimate expected spread with standard errors, and choose seed sets greedily, then compare them with the obvious alternative of picking the most connected characters.

**Goal:** estimate how far a spreading process reaches from a seed set, with its standard error, and choose seed sets by expected spread rather than by degree alone.

**Prerequisites:**

- [Your first network](first-network.md), or familiarity with `ag.Graph` and `ag.draw`.
- Helpful but not required: [Communities in Les Misérables](communities.md) introduces the network and the groups mentioned near the end, and [Epidemics on a small-world network](epidemics.md) covers simulation results and ensembles in more depth.
- AryaGraph installed; nothing else is needed for this page.

The code blocks build on each other: run them in order in one Python session or notebook. Every random step takes a seed, so you get the numbers printed on this page. The network is the co-appearance graph from the [communities tutorial](communities.md): 77 characters, linked when they appear in the same chapter.

## Run one independent cascade

In the **independent cascade** (IC) model, a node that becomes active at step *t* gets a single chance to activate each inactive neighbor at step *t* + 1, succeeding with probability *p*. Active nodes stay active, and the cascade stops when a step activates nobody. Start one from Valjean with *p* = 0.15:

```python
import numpy as np
import aryagraph as ag

les = ag.gen.les_miserables()
run = ag.sim.independent_cascade(les, ["Valjean"], p=0.15, seed=1)
print(run)
for step, nodes in enumerate(run.meta["newly_active"]):
    print(step, nodes)
```

```text
<SimulationResult IC: 77 nodes, 4 frames, t=0…3; final inactive=60, active=17>
0 ['Valjean']
1 ['Fantine', 'Fauchelevent', 'Scaufflaire', 'Gavroche', 'Babet']
2 ['Thenardier', 'Javert', 'Enjolras', 'Bahorel', 'Grantaire', 'Gueulemer', 'Montparnasse', 'Brujon']
3 ['Simplice', 'Eponine', 'Prouvaire']
```

Valjean has 36 neighbors, so on average 0.15 × 36 = 5.4 of them are activated in the first step; in this run 5 were. Those five pass the rumor on in step 2, and after step 3 nobody new is reached: 17 characters in total. The result is a {py:class}`~aryagraph.sim.base.SimulationResult` with one frame per step, the same type the [epidemic models](epidemics.md) return, so it can be animated. Press Play to replay this run step by step:

```python
player = run.animate(title="Independent cascade from Valjean", time_format="step {t}")
player.save("cascade_player.html")
```

<iframe class="ag-embed" src="../_static/generated/tutorials_b/cascade_player.html" height="1000" loading="lazy" title="Step-by-step playback of an independent cascade from Valjean on the Les Misérables network"></iframe>
<p class="ag-embed-note">Each step flashes the edges that passed the rumor on. Hover over a character to highlight its neighbors. <a href="../_static/generated/tutorials_b/cascade_player.html">Open full screen</a></p>

The size of a cascade varies a lot from run to run. Ten seeds give ten outcomes:

```python
spreads = [ag.sim.independent_cascade(les, ["Valjean"], p=0.15, seed=s).meta["spread"]
           for s in range(10)]
print(spreads)
```

```text
[17, 17, 19, 27, 17, 21, 27, 29, 23, 15]
```

Between 15 and 29 characters: a single run says little about what to expect.

## Run one linear threshold cascade

The **linear threshold** (LT) model describes peer pressure rather than contagion. Each character *v* draws a random threshold θ between 0 and 1 and activates once the active neighbors carry at least that share of *v*'s total edge weight. With `weight="weight"`, a neighbor you share many chapters with counts for more:

```python
lt = ag.sim.linear_threshold(les, ["Valjean"], weight="weight", seed=4)
print(lt)
print([len(nodes) for nodes in lt.meta["newly_active"]])
print(round(lt.meta["thresholds"]["Javert"], 3))
```

```text
<SimulationResult LT: 77 nodes, 12 frames, t=0…11; final inactive=18, active=59>
[1, 13, 7, 8, 5, 4, 7, 5, 3, 3, 2, 1]
0.041
```

This run reaches 59 characters over 11 steps. Influence accumulates in LT: a character who resists one active neighbor can tip over when a second one joins, so a cascade can keep going in small waves, as this one does. `meta["thresholds"]` holds the thresholds drawn for this run; Javert's is 0.041, so a small share of his co-appearances being active is enough. You can also pass your own thresholds as a number, a node attribute or a mapping.

```{note}
The two models answer different questions, and their numbers are not comparable. In IC, *p* sets how contagious the rumor is. In LT, the random thresholds set how easily each character is persuaded, and there is no *p*.
```

## Estimate the expected spread

To judge a seed set you need the *expected* number of active characters, averaged over the randomness of the model. {py:func}`~aryagraph.sim.influence_spread` estimates it by Monte Carlo and reports the standard error of the estimate:

```python
for runs in (100, 1_000, 10_000):
    est = ag.sim.influence_spread(les, ["Valjean"], model="ic", p=0.15, runs=runs, seed=1)
    lo, hi = est.confidence_interval()
    print(f"{runs:6} runs: {est.mean:.2f} ± {est.stderr:.2f}  "
          f"(95% interval {lo:.2f} to {hi:.2f})")
```

```text
   100 runs: 21.39 ± 1.05  (95% interval 19.33 to 23.45)
  1000 runs: 21.98 ± 0.31  (95% interval 21.36 to 22.59)
 10000 runs: 21.66 ± 0.10  (95% interval 21.47 to 21.86)
```

The standard error shrinks with the square root of the number of runs: ten times more runs, about three times less error. With 10,000 runs, Valjean alone reaches 21.66 ± 0.10 characters on average. Decide how many runs you need from the difference you want to detect: two seed sets whose estimates differ by less than about two combined standard errors are not reliably different.

`influence_spread` does not replay cascades step by step. It samples *live-edge graphs*: for IC it keeps each edge with probability *p* and counts the characters reachable from the seeds. This gives the same distribution of final sizes as simulating the cascade (Kempe, Kleinberg and Tardos, 2003) at a lower cost. You can check that against full simulations with {py:func}`~aryagraph.sim.run_ensemble`, which accepts the cascade functions too:

```python
ens = ag.sim.run_ensemble(ag.sim.independent_cascade, runs=2000, g=les,
                          seeds=["Valjean"], p=0.15, seed=3, keep=True)
print(ens)
```

```text
<EnsembleResult IC: 2000 runs, 77 nodes; final inactive=55.67±10.2, active=21.33±10.2>
```

2,000 simulated cascades end with 21.33 active characters on average. The ± here is the standard deviation of single runs (10.2), so the standard error of that mean is 10.2 / √2000 ≈ 0.23, and the two estimates (21.33 and 21.66) agree within about 1.3 combined standard errors.

## Choose seeds greedily

Now pick five seeds. Checking every set of five is out of reach (77 choose 5 is over 19 million sets), so {py:func}`~aryagraph.sim.greedy_influence_maximization` builds the set one seed at a time, each time adding the character with the largest gain in expected spread:

```python
best = ag.sim.greedy_influence_maximization(les, k=5, model="ic", p=0.15, runs=1000, seed=1)
print(best)
print([round(x, 2) for x in best.spread])
print([round(x, 2) for x in best.gains])
```

```text
InfluenceMaximization(IC: seeds=['Valjean', 'Bahorel', 'Favourite', 'Myriel', 'Bamatabois'], spread≈32.31, runs=1000, evaluations=208)
[21.98, 26.14, 29.0, 30.88, 32.31]
[21.98, 4.16, 2.86, 1.88, 1.43]
```

- `seeds[:i]` is the greedy set of size *i*, `spread[i-1]` its estimated spread and `gains[i-1]` what the *i*-th seed added. The gains shrink (4.16, 2.86, 1.88, 1.43) because later seeds mostly reach characters the earlier ones already cover.
- Expected spread in these models is *submodular*: adding a seed helps less the larger the set already is. Submodularity implies that the greedy set reaches at least 1 − 1/e ≈ 63% of the optimal spread, as measured on the same samples (Kempe, Kleinberg and Tardos, 2003), and it lets the CELF algorithm skip re-evaluating candidates whose gain cannot have grown enough. Here it needed 208 marginal-gain evaluations instead of the 375 that a plain greedy selection makes.
- All evaluations share the same 1,000 sampled live-edge graphs, so the comparisons between candidates are consistent. The flip side: the reported spread (32.31) is measured on the samples the seeds were chosen from and runs slightly high. Re-estimate it with a different seed.

## Compare with the most connected characters

The obvious alternative is to seed the five characters with the most neighbors. Re-estimate both sets on fresh samples:

```python
by_degree = [v for v, _ in les.degree().top(5)]
print(by_degree)
for name, seeds in [("greedy", best.seeds), ("top degree", by_degree)]:
    est = ag.sim.influence_spread(les, seeds, model="ic", p=0.15, runs=10_000, seed=2)
    print(f"{name:10}", est)
```

```text
['Valjean', 'Gavroche', 'Marius', 'Javert', 'Thenardier']
greedy     SpreadEstimate(IC: 31.8 ± 0.07, runs=10000)
top degree SpreadEstimate(IC: 29.08 ± 0.057, runs=10000)
```

The greedy seeds reach 31.8 characters on average, 2.7 more than the top-degree seeds (29.08), a gap of about 30 standard errors. The unbiased estimate of the greedy set, 31.8, is 0.5 below the 32.31 measured on its selection samples, which shows the optimism mentioned above.

The reason is overlap. Gavroche, Marius, Javert and Thénardier are all neighbors of Valjean, so their cascades largely reach the same characters. The greedy seeds come from five different communities of the [Louvain partition](communities.md): Valjean's circle, the students (Bahorel), Fantine's friends (Favourite), Myriel's household and the Champmathieu trial (Bamatabois).

To see how the gap develops, evaluate both strategies for 1 to 8 seeds. Because `seeds[:k]` of a greedy run is itself the greedy set of size *k*, one run with `k=8` covers every size:

```python
greedy8 = ag.sim.greedy_influence_maximization(les, k=8, model="ic", p=0.15, runs=1000, seed=1)
degree8 = [v for v, _ in les.degree().top(8)]
ks = list(range(1, 9))
curve = {"greedy (CELF)": [], "top degree": []}
for k in ks:
    for name, seeds in (("greedy (CELF)", greedy8.seeds), ("top degree", degree8)):
        est = ag.sim.influence_spread(les, seeds[:k], p=0.15, runs=5000, seed=2)
        curve[name].append(est.mean)
print(greedy8.seeds)
print(degree8)
print({name: [round(v, 1) for v in vals] for name, vals in curve.items()})
ag.charts.line_chart(
    ks, curve, title="Expected spread by seed-set size",
    subtitle="Independent cascade on Les Misérables, p = 0.15 · 5,000 runs per point",
    x_label="seeds", y_label="expected active characters",
).save("spread_by_k.svg")
```

```text
['Valjean', 'Bahorel', 'Favourite', 'Myriel', 'Bamatabois', 'Eponine', 'MmePontmercy', 'Zephine']
['Valjean', 'Gavroche', 'Marius', 'Javert', 'Thenardier', 'Fantine', 'Enjolras', 'Courfeyrac']
{'greedy (CELF)': [21.7, 25.9, 28.5, 30.4, 31.8, 33.3, 34.5, 35.6], 'top degree': [21.7, 26.0, 27.3, 28.4, 29.1, 30.8, 31.4, 31.8]}
```

```{figure} ../_static/generated/tutorials_b/spread_by_k.png
:alt: Line chart of expected spread against the number of seeds; the greedy and top-degree curves start together and separate from three seeds on, reaching 35.6 and 31.8 characters at eight seeds.
:width: 100%

Expected spread for 1 to 8 seeds, 5,000 runs per point. Both strategies start with Valjean and are level at two seeds; from three seeds on, the greedy sets pull ahead.
```

At two seeds the strategies are level (25.9 against 26.0, within the Monte Carlo error): Bahorel and Gavroche add about the same. From three seeds on the greedy sets pull ahead, and at eight seeds they reach 35.6 characters against 31.8. Eight top-degree seeds reach about as many characters (31.8) as five greedy ones (31.8).

## See where the rumor goes

Expected spread is one number; a map shows *who* the rumor reaches. Keep every run of an ensemble with `keep=True` and average each character's final state (0 for inactive, 1 for active) to get the probability that the character ends up active:

```python
def activation(seeds, runs=2000):
    e = ag.sim.run_ensemble(ag.sim.independent_cascade, runs=runs, g=les,
                            seeds=seeds, p=0.15, seed=3, keep=True)
    share = np.mean([r.values[-1] for r in e.results], axis=0)
    return dict(zip(e.results[0].nodes, share.tolist()))

reach = activation(best.seeds)
print(sorted(reach.items(), key=lambda kv: -kv[1])[:8])
```

```text
[('Myriel', 1.0), ('Valjean', 1.0), ('Favourite', 1.0), ('Bamatabois', 1.0), ('Bahorel', 1.0), ('Gavroche', 0.802), ('Marius', 0.782), ('Enjolras', 0.7735)]
```

The seeds are active in every run; Gavroche, Marius and Enjolras hear the rumor in about 8 of 10 runs. Draw the probabilities as a sequential color, mark the seeds with a different shape and label only them:

```python
role = {v: ("seed" if v in best.seeds else "other") for v in les}
fig = ag.draw(
    les,
    node_color=ag.by(reach, kind="sequential", domain=(0, 1), title="P(active)"),
    node_shape=ag.by(role, title="role"),
    labels={v: v for v in best.seeds},
    label_collisions="show",
    title="Greedy seeds",
)
fig.save("reach_greedy.svg")
```

`domain=(0, 1)` fixes the color scale, so figures for different seed sets use the same colors for the same probability. The figures below are drawn the same way, with the seed names and the expected spread added to the title. Compare the two seed sets:

::::{tab-set}

:::{tab-item} Greedy seeds
```{figure} ../_static/generated/tutorials_b/reach_greedy.png
:alt: Les Misérables network colored by the probability of being reached from the five greedy seeds, drawn as squares in five separate groups.
:width: 100%

Seeds in five communities: Fantine's circle, for example, is active with an average probability of 0.45 (0.27 with the top-degree seeds). <a href="../_static/generated/tutorials_b/reach_greedy.html">Interactive version</a>.
```
:::

:::{tab-item} Top-degree seeds
```{figure} ../_static/generated/tutorials_b/reach_degree.png
:alt: Les Misérables network colored by the probability of being reached from the five top-degree seeds, drawn as squares clustered around Valjean.
:width: 100%

The top-degree seeds cluster around Valjean; the ten members of Myriel's household are active in 8% of runs on average. <a href="../_static/generated/tutorials_b/reach_degree.html">Interactive version</a>.
```
:::

::::

## Choose seeds for the threshold model

Which seeds work well depends on the model. Run the greedy selection under linear threshold, then evaluate three seed sets under LT:

```python
lt_best = ag.sim.greedy_influence_maximization(les, k=5, model="lt", weight="weight",
                                               runs=1000, seed=1)
print(lt_best.seeds)
sets = {"greedy (LT)": lt_best.seeds, "greedy (IC)": best.seeds, "top degree": by_degree}
for name, seeds in sets.items():
    est = ag.sim.influence_spread(les, seeds, model="lt", weight="weight", runs=10_000, seed=2)
    print(f"{name:12}", est)
```

```text
['Valjean', 'Courfeyrac', 'Myriel', 'Thenardier', 'Fantine']
greedy (LT)  SpreadEstimate(LT: 54.14 ± 0.098, runs=10000)
greedy (IC)  SpreadEstimate(LT: 46.75 ± 0.11, runs=10000)
top degree   SpreadEstimate(LT: 47.98 ± 0.09, runs=10000)
```

Under LT, the greedy seeds reach 54.14 characters, 6.2 more than the top-degree seeds, and the seeds chosen for IC do worse than the top-degree ones (46.75). The reason is how LT weighs influence: a lone active neighbor *u* activates *v* with probability w(*u*, *v*) divided by *v*'s strength. Summing that over *u*'s neighbors gives the number of characters *u* tips over on its own, on average:

```python
strength = les.degree(weight="weight")

def pushes(u):
    return sum(d["weight"] / strength[v] for v, d in les.adj[u].items())

for name, seeds in [("greedy (IC)", best.seeds), ("greedy (LT)", lt_best.seeds)]:
    print(f"{name:12}", {u: round(pushes(u), 2) for u in seeds})
```

```text
greedy (IC)  {'Valjean': 11.74, 'Bahorel': 0.98, 'Favourite': 0.98, 'Myriel': 8.03, 'Bamatabois': 0.61}
greedy (LT)  {'Valjean': 11.74, 'Courfeyrac': 1.8, 'Myriel': 8.03, 'Thenardier': 3.47, 'Fantine': 2.55}
```

Bahorel, Favourite and Bamatabois each tip over one neighbor or fewer: their ties are weak compared with their neighbors' other ties. The LT seeds besides Valjean are the hubs of four communities (Courfeyrac, Myriel, Thénardier and Fantine), and each of them tips over between 1.8 and 8.03 characters alone. Select seeds with the model you believe describes the spreading process, and evaluate them under that model.

## Next steps

- [Simulation](../user-guide/simulation.md) in the user guide places the cascade models among the other simulators. The API reference for {py:func}`~aryagraph.sim.independent_cascade` and {py:func}`~aryagraph.sim.linear_threshold` lists every option, such as edge-specific probabilities (`p` as an edge attribute or a function) and node-specific thresholds.
- [Communities in Les Misérables](communities.md) explains the groups that the greedy seeds spread across.
- [Epidemics on a small-world network](epidemics.md) covers ensembles, bands and the animation player in more depth.
- [Reproducibility and performance](../user-guide/reproducibility.md) covers seeds and repeatable results.
