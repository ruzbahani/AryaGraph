# Epidemics on a small-world network

In this tutorial you simulate an SIR epidemic on a small-world contact network. You compare the discrete-time and the exact stochastic (Gillespie) solvers, read the peak and the final size from a result, summarize 200 runs with uncertainty bands, watch a run in the animation player, and finally extend the model with a latent period and with a compartment of your own.

**Goal:** simulate an outbreak on a contact network, describe its uncertainty over many runs, and adapt the model to a question of your own.

**Prerequisites:**

- [Your first network](first-network.md), or familiarity with `ag.Graph`, nodes and edges.
- AryaGraph installed (see [Installation](../getting-started/installation.md)). One step converts a result to a DataFrame and needs pandas (the `interop` extra); the SVG charts and the HTML player need nothing extra.

No epidemiology is assumed: the page introduces each model before it uses it. The code blocks build on each other: run them in order in one Python session or notebook. Every random step takes a seed, so with the seeds shown you get the numbers printed on this page.

## Build a small-world contact network

Epidemics spread along contacts, so the structure of the contact network shapes the outbreak. A Watts–Strogatz graph is a classic stand-in for social contact: most links are local (your neighbors know each other), and a few random shortcuts connect distant parts of the population.

```python
import math

import numpy as np
import aryagraph as ag

g = ag.gen.watts_strogatz(400, 6, 0.1, seed=3)
lattice = ag.gen.watts_strogatz(400, 6, 0.0, seed=3)
shortcuts = sum(1 for u, v in g.edges if min(abs(u - v), 400 - abs(u - v)) > 3)

print(g)
print("rewired shortcuts:", shortcuts)
c, c0 = (ag.alg.average_clustering(h) for h in (g, lattice))
d, d0 = (ag.alg.average_shortest_path_length(h) for h in (g, lattice))
print(f"clustering: {c:.3f} (ring lattice {c0:.3f})")
print(f"mean distance: {d:.2f} (ring lattice {d0:.2f})")
```

```text
<Graph 'watts_strogatz(400, 6, 0.1)': 400 nodes, 1200 edges>
rewired shortcuts: 122
clustering: 0.443 (ring lattice 0.600)
mean distance: 5.15 (ring lattice 33.75)
```

{py:func}`~aryagraph.generators.random.watts_strogatz` places 400 nodes on a ring, links each one to its 3 nearest neighbors on either side, then rewires the far end of each edge with probability 0.1. The result keeps most of the lattice's local clustering (0.443 against 0.600) while the shortcuts cut the mean distance between two nodes from 33.75 hops to 5.15. That combination is what "small world" means, and it matters for epidemics: local clusters slow the spread down, shortcuts carry it to new regions.

## Define the SIR model

In the SIR model every node is **S**usceptible, **I**nfected or **R**ecovered. {py:func}`~aryagraph.sim.compartmental.SIR` returns a {py:class}`~aryagraph.sim.compartmental.CompartmentalModel` with two transitions:

```python
sir = ag.sim.SIR(beta=0.08, gamma=0.1)
print(sir)
print(f"transmissibility: {0.08 / (0.08 + 0.1):.3f}")
```

```text
<CompartmentalModel SIR: I→R (0.1), S→I (0.08 × I)>
transmissibility: 0.444
```

- `S→I (0.08 × I)` is an *induced* transition: a susceptible node with *k* infected neighbors becomes infected at rate 0.08·*k*.
- `I→R (0.1)` is a *spontaneous* transition: an infected node recovers at rate 0.1, so it stays infectious for 1/0.1 = 10 time units on average.

The ratio β/(β + γ) is the **transmissibility**: the probability that an infected node passes the infection to a given susceptible neighbor before it recovers. Here it is 0.444, so each contact of an infected node is roughly a coin flip. Time has no fixed unit; read it as days if that helps.

## Choose a solver: discrete time or Gillespie

{py:meth}`~aryagraph.sim.compartmental.CompartmentalModel.simulate` runs the model on a graph. `initial={"I": 3}` infects 3 nodes drawn at random (with the run's seed); every other node starts in the first state, `S`.

```python
discrete = sir.simulate(g, initial={"I": 3}, method="discrete", dt=1.0, seed=7)
exact = sir.simulate(g, initial={"I": 3}, method="gillespie", seed=7)
print(discrete)
print(exact)
print(exact.meta["n_events"], "events, absorbed at t =", round(exact.meta["t_end"], 1))
```

```text
<SimulationResult SIR: 400 nodes, 104 frames, t=0…103; final S=22, I=0, R=378>
<SimulationResult SIR: 400 nodes, 101 frames, t=0…110.8; final S=17, I=0, R=383>
763 events, absorbed at t = 110.8
```

The two solvers answer the same question in different ways:

`method="discrete"`
: Updates every node at once, once per step of length `dt`. A node whose transitions have total rate *H* changes state with probability 1 − exp(−*H*·`dt`). The result has one frame per step. This is fast and easy to reason about, and it carries a small bias that shrinks with `dt` (you measure it [below](#how-the-discrete-solver-differs)).

`method="gillespie"`
: Simulates the continuous-time process event by event with Gillespie's direct method, so there is no step size to choose. Here the run took 763 events. By default the result is sampled on a regular grid of 101 frames.

Both runs stop on their own once no transition can fire (the epidemic has died out), because SIR has no way back from R. The two runs use the same seed but different random streams, so their numbers differ by chance: 378 against 383 nodes infected in total.

## Read the results

A {py:class}`~aryagraph.sim.base.SimulationResult` stores the state of every node in every frame. Its methods answer the usual questions directly:

```python
t_peak, n_peak = exact.peak("I")
print(f"peak: {n_peak} infected at t = {t_peak:.1f}")
counts = exact.counts()
print({state: int(c[-1]) for state, c in counts.items()})
print(exact.summary()["I"])
```

```text
peak: 93 infected at t = 27.7
{'S': 17, 'I': 0, 'R': 383}
{'initial': 3, 'final': 0, 'peak': 93, 'peak_time': 27.69933192952744}
```

{py:meth}`~aryagraph.sim.base.SimulationResult.counts` gives one array per state (nodes per frame), {py:meth}`~aryagraph.sim.base.SimulationResult.peak` the largest count and when it happened, and {py:meth}`~aryagraph.sim.base.SimulationResult.summary` both ends and the peak for every state. For this run, 93 nodes were infectious at once at t ≈ 27.7, and 383 of the 400 nodes were infected at some point.

Per-node questions work too. {py:meth}`~aryagraph.sim.base.SimulationResult.first_time` returns a {py:class}`~aryagraph.core.results.NodeMap` with the time each node first appears in a state, and {py:meth}`~aryagraph.sim.base.SimulationResult.history` the state of one node in every frame:

```python
onset = exact.first_time("I")
print({k: round(v, 1) for k, v in onset.describe().items()})
print(exact.history(0)[::10])
```

```text
{'count': 367, 'mean': 29.0, 'std': 18.9, 'min': 0.0, '25%': 15.5, '50%': 24.4, '75%': 37.7, 'max': 95.3}
['S', 'S', 'S', 'S', 'S', 'S', 'I', 'R', 'R', 'R', 'R']
```

Half of the recorded infections happened by t ≈ 24.4, and the last one shows up at t ≈ 95.3. The second line samples every tenth frame of node 0: it stays susceptible through frame 50, is infected in frame 60 and has recovered by frame 70.

Now look at the count: 367, although 383 nodes were infected. The difference is a property of frames, not a bug. A Gillespie result samples the process on a grid, here every 1.1 time units, and 16 nodes caught the infection *and* recovered between two grid points, so no frame shows them as infected. When you need every event, record them all:

```python
events = sir.simulate(g, initial={"I": 3}, method="gillespie", seed=7, record="events")
print(events.T, "frames")
print(events.peak("I"))
print(events.first_time("I").describe()["count"])
```

```text
764 frames
(27.583171142397546, 93)
383
```

With `record="events"` the same seed replays the same run with one frame per event, so the result is the exact piecewise-constant trajectory: the peak is the same 93 nodes, reached at t = 27.6, and all 383 infections are visible. The grid is lighter to store and animate; the event record is the one to use for per-node timing.

For your own analysis, {py:meth}`~aryagraph.sim.base.SimulationResult.to_pandas` returns the result in long format, one row per node and frame (pandas is optional):

```python
df = exact.to_pandas()
print(df.shape)
print(df.head(3))
```

```text
(40400, 4)
   frame  time  node state
0      0   0.0     0     S
1      0   0.0     1     S
2      0   0.0     2     S
```

To see the whole run, plot it. {py:meth}`~aryagraph.sim.base.SimulationResult.plot` returns a {py:class}`~aryagraph.charts.base.Chart` that saves to SVG, PNG or PDF:

```python
chart = exact.plot(title="SIR on a small-world network",
                   subtitle="One Gillespie run, seed 7")
chart.save("sir_counts.svg")
```

```{figure} ../_static/generated/tutorials_b/sir_counts.png
:alt: Line chart of susceptible, infected and recovered node counts over time for one SIR run; infections peak at 93 near t = 28 and 383 nodes end up recovered.
:width: 100%

One Gillespie run. Counts change in steps because every event moves one node from one state to another.
```

## Summarize many runs with bands

One run is one possible outbreak. To describe what the model does, run it many times and summarize. {py:func}`~aryagraph.sim.run_ensemble` calls a simulation function repeatedly with independent child seeds derived from one master seed:

```python
ens = ag.sim.run_ensemble(sir.simulate, runs=200, g=g, initial={"I": 3},
                          method="gillespie", seed=11)
print(ens)
```

```text
<EnsembleResult SIR: 200 runs, 400 nodes; final S=58.08±70.2, I=0±0, R=341.9±70.2>
```

The {py:class}`~aryagraph.sim.ensemble.EnsembleResult` puts every run on a common time grid and stores the mean, the standard deviation and the 5, 25, 50, 75 and 95% quantiles of each state's count, plus the final count of every run:

```python
i = int(np.argmax(ens.mean["I"]))
lo, hi = ens.band("I")
print(f"mean curve peaks at {ens.mean['I'][i]:.1f} infected, t = {ens.times[i]:.1f}")
print(f"5–95% band at that time: {lo[i]:.0f} to {hi[i]:.0f}")
peaks = ens.samples[:, :, ens.states.index("I")].max(axis=1)
print(f"mean of the per-run peaks: {peaks.mean():.1f}")
final = ens.final_sizes["R"]
print("runs that stayed below 40 cases:", int((final < 40).sum()), "of", ens.runs)
major = final[final >= 40]
print(f"final size of the others: mean {major.mean():.1f}, min {major.min()}, max {major.max()}")
```

```text
mean curve peaks at 70.6 infected, t = 37.1
5–95% band at that time: 9 to 105
mean of the per-run peaks: 86.6
runs that stayed below 40 cases: 7 of 200
final size of the others: mean 354.0, min 41, max 391
```

Two things stand out:

- **The peak of the mean is not the mean of the peaks.** Each run peaks at its own time, so averaging the curves flattens them: the mean curve tops out at 70.6 infected, while the runs themselves peak at 86.6 on average. Quote the per-run figure when you mean "how bad does it get".
- **Some outbreaks fizzle out.** In 7 of the 200 runs the infection died out while fewer than 40 nodes had caught it; the other 193 infected between 41 and 391 nodes. Early on, with only a few infected nodes, chance decides whether the outbreak takes off. That is why the mean final size (341.9) sits below the typical major outbreak (354.0).

Every run can be replayed on its own, because `ens.seeds` keeps the integer seed of each one:

```python
replay = sir.simulate(g, initial={"I": 3}, method="gillespie", seed=ens.seeds[0])
print(replay.counts()["R"][-1], ens.final_sizes["R"][0])
```

```text
354 354
```

{py:meth}`~aryagraph.sim.ensemble.EnsembleResult.plot` draws the mean of each state with its 25–75% and 5–95% bands, and {py:func}`~aryagraph.charts.histogram` shows the distribution of final sizes:

```python
ens.plot(title="SIR: 200 Gillespie runs",
         subtitle="Mean with 25–75% and 5–95% bands").save("sir_ensemble.svg")
hist = ag.charts.histogram(final, bins=20, title="Final epidemic size",
                           subtitle="200 Gillespie runs of the same SIR model",
                           x_label="nodes ever infected", y_label="runs")
hist.save("sir_final_sizes.svg")
```

```{figure} ../_static/generated/tutorials_b/sir_ensemble.png
:alt: Mean susceptible, infected and recovered counts over 200 runs with shaded 25–75% and 5–95% bands; the bands are wide around the peak.
:width: 100%

Mean counts of 200 runs with their 25–75% (darker) and 5–95% (lighter) bands. The bands are widest around t ≈ 40, where runs that took off early and runs that took off late differ most.
```

```{figure} ../_static/generated/tutorials_b/sir_final_sizes.png
:alt: Histogram of final epidemic sizes: a small group of runs near zero and a large group between about 300 and 391 nodes.
:width: 100%

Final sizes are bimodal: a few minor outbreaks near zero, and major outbreaks that reach most of the network.
```

## How the discrete solver differs

The single runs above could not tell the solvers apart, because chance dominates one run. Ensembles can. The next block runs the discrete solver with two step sizes and compares the results with the Gillespie ensemble. It also computes the transmissibility each solver implies: in discrete time an infected node stays infectious for a whole number of steps and transmits with probability 1 − exp(−β·`dt`) per step, which works out slightly higher than β/(β + γ).

```python
def transmissibility(beta, gamma, dt=None):
    if dt is None:
        return beta / (beta + gamma)
    p = 1 - math.exp(-gamma * dt)  # chance to recover in one step
    x = math.exp(-beta * dt)       # chance not to transmit in one step
    return 1 - p * x / (1 - (1 - p) * x)

rows = [("gillespie", None, ens)]
for dt in (1.0, 0.25):
    e = ag.sim.run_ensemble(sir.simulate, runs=200, g=g, initial={"I": 3},
                            method="discrete", dt=dt, seed=11)
    rows.append(("discrete", dt, e))
for method, dt, e in rows:
    f = e.final_sizes["R"]
    print(f"{method:9} dt={dt}: T={transmissibility(0.08, 0.1, dt):.3f}  "
          f"mean final {f.mean():5.1f}  major-outbreak final {f[f >= 40].mean():5.1f}")
```

```text
gillespie dt=None: T=0.444  mean final 341.9  major-outbreak final 354.0
discrete  dt=1.0: T=0.467  mean final 366.5  major-outbreak final 370.1
discrete  dt=0.25: T=0.450  mean final 352.6  major-outbreak final 357.9
```

With `dt=1` the transmissibility rises from 0.444 to 0.467, and the average major outbreak grows from 354.0 to 370.1 infected nodes. Shrinking the step to 0.25 brings both back toward the exact values (0.450 and 357.9), at four times the number of steps. Use the discrete solver when your model is defined in steps (daily reporting, for example) or when you want many fast runs; use Gillespie when rates are the natural description and you want results that do not depend on a step size.

## Watch one run

{py:meth}`~aryagraph.sim.base.SimulationResult.animate` turns a result into an interactive player. A circular layout keeps the ring of the Watts–Strogatz construction visible: the infection creeps along the ring through local contacts and jumps across the circle along the shortcuts.

```python
player = exact.animate(
    layout="circular",
    avoid_overlap=False,   # keep every node on the circle
    labels=False,
    node_size=6,
    width=640,
    height=640,
    title="SIR on a small-world network",
    subtitle="Nodes sit on the ring; the chords are the rewired shortcuts",
)
player.save("sir_player.html")
```

<iframe class="ag-embed" src="../_static/generated/tutorials_b/sir_player.html" height="1100" loading="lazy" title="Animated SIR epidemic on a 400-node small-world network in a circular layout, with a synchronized chart of state counts"></iframe>
<p class="ag-embed-note">Press Play, or drag the slider or the chart to scrub through time. Transmissions flash along the edges that carried them. <a href="../_static/generated/tutorials_b/sir_player.html">Open full screen</a></p>

The saved page is self-contained HTML with no external requests, so you can share it as a single file or host it as a static page. `avoid_overlap=False` turns off the overlap removal that would otherwise push the densely packed nodes off the circle.

## Add a latent period: SEIR

Many diseases have a latent period: infected but not yet infectious. {py:func}`~aryagraph.sim.compartmental.SEIR` adds an **E**xposed state that nodes leave at rate σ. With σ = 0.2 the latent period lasts 5 time units on average. Keep β and γ as before and compare:

```python
seir = ag.sim.SEIR(beta=0.08, sigma=0.2, gamma=0.1)
ens_seir = ag.sim.run_ensemble(seir.simulate, runs=200, g=g, initial={"I": 3},
                               method="gillespie", seed=11)
print(seir)

def outbreak(e):
    infected = e.samples[:, :, e.states.index("I")]
    f = e.final_sizes["R"]
    major = f >= 40
    peak_times = e.times[infected.argmax(axis=1)][major]
    return infected.max(axis=1)[major].mean(), np.median(peak_times), f.mean()

for name, e in [("SIR", ens), ("SEIR", ens_seir)]:
    peak, when, size = outbreak(e)
    print(f"{name:4}: typical peak {peak:5.1f} infected, "
          f"median peak time {when:5.1f}, mean final size {size:.1f}")
```

```text
<CompartmentalModel SEIR: E→I (0.2), I→R (0.1), S→E (0.08 × I)>
SIR : typical peak  89.6 infected, median peak time  37.1, mean final size 341.9
SEIR: typical peak  47.1 infected, median peak time  80.9, mean final size 339.5
```

```python
grid = np.linspace(0, 150, 151)
curves = {
    "SIR, infected": np.interp(grid, ens.times, ens.mean["I"]),
    "SEIR, infected": np.interp(grid, ens_seir.times, ens_seir.mean["I"]),
    "SEIR, exposed": np.interp(grid, ens_seir.times, ens_seir.mean["E"]),
}
ag.charts.line_chart(
    grid, curves, title="Adding a latent period",
    subtitle="Same beta and gamma; SEIR adds a mean latent period of 5",
    x_label="time", y_label="nodes (mean of 200 runs)",
).save("sir_vs_seir.svg")
```

```{figure} ../_static/generated/tutorials_b/sir_vs_seir.png
:alt: Mean infected curves of SIR and SEIR ensembles; the SEIR curve peaks later and lower and is much broader, with a separate exposed curve below it.
:width: 100%

Mean curves of the two ensembles. The latent period slows the epidemic down and spreads it out; the total number of infections barely changes.
```

Without a latent period, a transmission that happens comes on average 1/(β + γ) ≈ 5.6 time units after the infector was infected; the latent period adds 5 to that. The epidemic therefore runs about twice as slowly (median peak time 80.9 against 37.1), and fewer nodes are infectious at the same time: in a typical major outbreak the peak drops from about 90 to about 47 infected. The final size barely moves (339.5 against 341.9), and that is expected: an exposed node cannot infect anyone, so the latent period changes *when* transmissions happen but not the probability β/(β + γ) that a given contact is infected.

## Write your own model

The ready-made models are {py:class}`~aryagraph.sim.compartmental.CompartmentalModel` instances, and you can build your own the same way: list the states, the spontaneous transitions `(from, to, rate)` and the induced transitions `(from, to, via, rate)`. As an example, add isolation: infected nodes are detected at rate 0.05 and move to a **Q**uarantined state, where they infect nobody until they recover.

```python
siqr = ag.sim.CompartmentalModel(
    ["S", "I", "Q", "R"],
    spontaneous=[("I", "R", 0.1), ("I", "Q", 0.05), ("Q", "R", 0.1)],
    induced=[("S", "I", "I", 0.08)],
    name="SIQR",
    roles={"Q": "serious"},
)
print(siqr)
print(siqr.terminates)
ens_q = ag.sim.run_ensemble(siqr.simulate, runs=200, g=g, initial={"I": 3},
                            method="gillespie", seed=11)
print(ens_q)
f = ens_q.final_sizes["R"]
print(f"transmissibility {0.08 / (0.08 + 0.1 + 0.05):.3f}; mean final size {f.mean():.1f}; "
      f"runs below 40: {int((f < 40).sum())}")
ens_q.plot(title="SIQR: isolating infected nodes",
           subtitle="200 Gillespie runs").save("siqr_ensemble.svg")
```

```text
<CompartmentalModel SIQR: I→R (0.1), I→Q (0.05), Q→R (0.1), S→I (0.08 × I)>
True
<EnsembleResult SIQR: 200 runs, 400 nodes; final S=202.6±95.3, I=0±0, Q=0±0, R=197.4±95.3>
transmissibility 0.348; mean final size 197.4; runs below 40: 25
```

A few details make custom models work well:

- `roles` gives each state a semantic color role for charts and animations. `S`, `E`, `I` and `R` get theirs automatically; `Q` would otherwise take the next color of a fixed cycle.
- {py:attr}`~aryagraph.sim.compartmental.CompartmentalModel.terminates` is `True` because no chain of transitions leads back to an earlier state, so simulations run until the epidemic dies out. Models that can cycle (SIS, SIRS) run to `t_max`, 100 by default.
- Isolation shortens the infectious period from 1/0.1 = 10 to 1/0.15 ≈ 6.7 time units, which lowers the transmissibility from 0.444 to 0.348. In this network that cuts the mean final size by about 42%, from 341.9 to 197.4 nodes, and raises the number of outbreaks that fizzle out from 7 to 25 of 200.

```{figure} ../_static/generated/tutorials_b/siqr_ensemble.png
:alt: Mean susceptible, infected, quarantined and recovered counts over 200 SIQR runs with uncertainty bands; the recovered band is very wide.
:width: 100%

The SIQR ensemble. The wide bands on the final counts show how variable the outcome becomes when fewer contacts transmit.
```

## Next steps

- [Simulation](../user-guide/simulation.md) in the user guide covers every model family, including SIS, SIRS, SEIRD, opinion dynamics and random walks.
- [Influence and cascades](influence.md) applies the same machinery to information spreading and seed selection.
- [Epidemic interventions](../case-studies/epidemic-interventions.md) is a case study built on these tools.
- [Reproducibility and performance](../user-guide/reproducibility.md) covers seeds and repeatable results.
