# Simulation

The `ag.sim` namespace runs dynamics on graphs: epidemics, information cascades, random walks, opinion dynamics, heat flow and coupled oscillators. Every simulator returns the same result type, so counting, plotting, exporting and animating work the same way whatever the model, and every source of randomness goes through a `seed` argument.

## What is available

| Family | Functions | Result |
|---|---|---|
| Epidemics | {py:class}`~aryagraph.sim.compartmental.CompartmentalModel`; ready-made {py:func}`~aryagraph.sim.compartmental.SI`, {py:func}`~aryagraph.sim.compartmental.SIS`, {py:func}`~aryagraph.sim.compartmental.SIR`, {py:func}`~aryagraph.sim.compartmental.SEIR`, {py:func}`~aryagraph.sim.compartmental.SIRS`, {py:func}`~aryagraph.sim.compartmental.SEIRD` and the one-call shortcuts `si` … `seird` | categorical states |
| Cascades | {py:func}`~aryagraph.sim.independent_cascade`, {py:func}`~aryagraph.sim.linear_threshold`, {py:func}`~aryagraph.sim.influence_spread`, {py:func}`~aryagraph.sim.greedy_influence_maximization` | categorical states, spread estimates |
| Random walks | {py:func}`~aryagraph.sim.random_walk`, {py:func}`~aryagraph.sim.stationary_distribution`, {py:func}`~aryagraph.sim.transition_matrix` | categorical states, exact distribution |
| Opinions | {py:func}`~aryagraph.sim.voter_model`, {py:func}`~aryagraph.sim.majority_rule` (discrete opinions); {py:func}`~aryagraph.sim.degroot`, {py:func}`~aryagraph.sim.bounded_confidence` (real-valued) | categorical or continuous |
| Physics | {py:func}`~aryagraph.sim.heat_diffusion`, {py:func}`~aryagraph.sim.kuramoto` | continuous values |
| Ensembles | {py:func}`~aryagraph.sim.run_ensemble` | mean curves and quantile bands |
| DAG execution | {py:func}`~aryagraph.sim.simulate_schedule`, {py:func}`~aryagraph.sim.monte_carlo_schedule` | covered in [DAG workflows](dags.md) |

A *categorical* result stores a small integer state code per node and frame (`"S"`, `"I"`, `"R"`, `"active"`, …); a *continuous* result stores a float per node and frame (an opinion, an amount of heat, a phase).

## A first epidemic

The example below runs an SIR epidemic on a Watts–Strogatz small world: 300 people on a ring, each linked to their 6 nearest neighbors, with every link rewired to a random person with probability 0.08.

```python
import aryagraph as ag

g = ag.gen.watts_strogatz(300, 6, 0.08, seed=1)
model = ag.sim.SIR(beta=0.1, gamma=0.1)
run = model.simulate(g, initial={"I": 3}, method="gillespie", seed=7)
print(run)
t_peak, n_peak = run.peak("I")
print(f"peak: {n_peak} infectious nodes at t = {t_peak:.1f}")
```

```text
<SimulationResult SIR: 300 nodes, 101 frames, t=0…135.7; final S=15, I=0, R=285>
peak: 70 infectious nodes at t = 27.1
```

`beta` is the infection rate per infected neighbor and `gamma` the recovery rate, so a node stays infectious for 1/γ = 10 time units on average. `initial={"I": 3}` infects 3 nodes drawn at random (from the run's seed) and leaves everybody else in the first state, `"S"`. No horizon is given, and SIR has no way back from `"R"`, so the run continues until no transition can fire. `peak("I")` returns `(time, count)` for the largest number of infectious nodes.

{py:meth}`~aryagraph.sim.base.SimulationResult.plot` turns the result into a chart of the state counts:

```python
chart = run.plot(title="SIR epidemic: nodes per state")
chart.save("sir_curve.svg")      # also .png, .pdf or .html
```

```{figure} ../_static/generated/guide_sim_data/sir_curve.png
:alt: Line chart of susceptible, infected and recovered counts over time; infections peak at 70 nodes near t = 27 and 285 nodes end recovered.
:width: 100%

State counts of the run above. Gillespie runs are drawn as step functions because the counts jump at event times.
```

{py:meth}`~aryagraph.sim.base.SimulationResult.animate` builds an interactive player: the graph is drawn once, each frame recolors the nodes, the edges that transmitted flash, and a chart under the player follows the timeline.

```python
player = run.animate(labels=False, title="SIR epidemic on a small-world network")
player.save("sir_player.html")
```

<iframe class="ag-embed" src="../_static/generated/guide_sim_data/sir_player.html" height="1100" loading="lazy" title="Interactive SIR epidemic player on a 300-node small-world network"></iframe>
<p class="ag-embed-note">The same run in the live player: press play, drag the timeline or the chart, and hover or click a node to see its neighborhood. <a href="../_static/generated/guide_sim_data/sir_player.html">Open full screen</a></p>

The saved page is a single self-contained HTML file with no network requests, so it works offline and on any static web host.

## Reading a result

Every simulator returns a {py:class}`~aryagraph.sim.base.SimulationResult`. Its core is a `(T, N)` array, `values`, with one row per frame and one column per node in graph order. The methods below read it without you having to index the array.

| Member | What it gives you |
|---|---|
| `times`, `T`, `N`, `nodes`, `states` | frame times (starting at 0), frame count, node count, column order, state names |
| `frame(i)`, `at(t)`, `final()` | `{node: state}` of frame *i*, of the frame in force at time *t*, of the last frame |
| `history(node)` | the state of one node in every frame |
| `counts()`, `fractions()` | `{state: array}` of how many nodes (or what share) are in each state per frame |
| `summary()` | initial, final and peak count and the peak time of each state |
| `peak(state)`, `first_time(state)` | largest count and when it happened; per node, the first frame time in *state* |
| `edge_activity` | per frame, the edges that fired since the previous frame (who infected whom) |
| `meta`, `params`, `seed`, `model` | model-specific extras and the provenance of the run |
| `to_records()`, `to_pandas()` | long format: one row per node and frame |

```python
print(run.T, run.N, run.states)
print(run.times[:4].round(3))
final = run.counts()
print({state: int(c[-1]) for state, c in final.items()})
print(run.summary()["I"])
print(run.history(187)[:8])
print(run.edge_activity[1])
print(run.meta)
```

```text
101 300 ['S', 'I', 'R']
[0.    1.357 2.714 4.071]
{'S': 15, 'I': 0, 'R': 285}
{'initial': 3, 'final': 0, 'peak': 70, 'peak_time': 27.14023559276943}
['S', 'I', 'R', 'R', 'R', 'R', 'R', 'R']
[(186, 187)]
{'absorbed': True, 't_end': 135.70117796384716, 'n_events': 567, 'transitions': ['I→R (0.1)', 'S→I (0.1 × I)']}
```

Node 187 is infected by node 186, one of the three initial cases, before the second frame (t ≈ 1.4) and has already recovered by the third (t ≈ 2.7). `edge_activity[1]` records that transmission as `(source, target)`; `meta` records whether the run was absorbed, when it ended, how many events happened, and the transitions of the model.

For analysis in pandas, {py:meth}`~aryagraph.sim.base.SimulationResult.to_pandas` returns the long format, which pivots into whatever shape you need:

```python
df = run.to_pandas()
print(df.shape)
print(df.head(3))
per_state = df.groupby(["time", "state"]).size().unstack(fill_value=0)
print(per_state.iloc[[0, 20, -1]])
```

```text
(30300, 4)
   frame  time  node state
0      0   0.0     0     S
1      0   0.0     1     S
2      0   0.0     2     S
state        I    R    S
time                    
0.000000     3    0  297
27.140236   70  115  115
135.701178   0  285   15
```

### Frames, events and first times

A Gillespie run is a sequence of events at irregular times. By default (`record="frames"`) it is sampled on a regular grid of 101 frames, which keeps results small and animations smooth. `record="events"` keeps one frame per event time instead, which is the exact piecewise-constant trajectory, and `record="final"` keeps only the first and last frames.

The choice matters for per-node timing questions. {py:meth}`~aryagraph.sim.base.SimulationResult.first_time` reports the first *frame* in which a node is seen in a state, so a node that goes from S to I to R between two grid points is not seen in I at all:

```python
import math

grid = run.first_time("I")
exact = model.simulate(g, initial={"I": 3}, method="gillespie", seed=7, record="events")
never_grid = sum(math.isnan(t) for t in grid.values())
never_exact = sum(math.isnan(t) for t in exact.first_time("I").values())
print(exact.T, never_grid, never_exact, int(exact.counts()["S"][-1]))
```

```text
568 31 15 15
```

On the 101-frame grid, 31 nodes are not seen infected in any frame. With one frame per event (568 frames here), only the 15 nodes that escaped infection are, which matches the final susceptible count. Both runs use the same seed and therefore the same events; only the sampling differs. Use `record="events"` whenever you need per-node times, and the default grid for plots and animations.

## Compartmental models

A compartmental model is a list of states and two kinds of transitions:

* **spontaneous** `X → Y` at a constant rate (recovery, loss of immunity, vaccination);
* **induced** `X → Y` driven by neighbors in a *via* state (infection): the rate for node *v* is `rate × Σ w(u, v)` over the neighbors *u* of *v* that are in the via state. With `weight=None` every edge counts 1; pass an edge attribute name to scale contact rates.

The ready-made factories cover the classic models:

| Factory | States | Transitions |
|---|---|---|
| `SI(beta)` | S, I | S → I at β per infected neighbor |
| `SIS(beta, gamma)` | S, I | SI plus I → S at γ |
| `SIR(beta, gamma)` | S, I, R | SI plus I → R at γ |
| `SEIR(beta, sigma, gamma)` | S, E, I, R | S → E at β per infected neighbor, E → I at σ, I → R at γ |
| `SIRS(beta, gamma, xi)` | S, I, R | SIR plus R → S at ξ (waning immunity) |
| `SEIRD(beta, sigma, gamma, mu)` | S, E, I, R, D | SEIR where I → D at μ competes with I → R at γ |

Each factory returns a {py:class}`~aryagraph.sim.compartmental.CompartmentalModel`; its `simulate()` method runs it. The lowercase shortcuts do both in one call: `ag.sim.sir(g, 0.1, 0.1, seed=7)` equals `ag.sim.SIR(0.1, 0.1).simulate(g, seed=7)`.

```python
models = {
    "SI": ag.sim.SI(0.1),
    "SIS": ag.sim.SIS(0.1, 0.1),
    "SIR": ag.sim.SIR(0.1, 0.1),
    "SEIR": ag.sim.SEIR(0.1, 0.2, 0.1),
    "SIRS": ag.sim.SIRS(0.1, 0.1, 0.02),
    "SEIRD": ag.sim.SEIRD(0.1, 0.2, 0.09, 0.01),
}
for name, m in models.items():
    res = m.simulate(g, initial={"I": 3}, method="gillespie", seed=7)
    final = {state: int(c[-1]) for state, c in res.counts().items()}
    print(f"{name:6} terminates={m.terminates!s:5}  t_end={res.times[-1]:6.1f}  {final}")
```

```text
SI     terminates=True   t_end=  36.3  {'S': 0, 'I': 300}
SIS    terminates=False  t_end= 100.0  {'S': 59, 'I': 241}
SIR    terminates=True   t_end= 135.7  {'S': 15, 'I': 0, 'R': 285}
SEIR   terminates=True   t_end= 170.9  {'S': 13, 'E': 0, 'I': 0, 'R': 287}
SIRS   terminates=False  t_end= 100.0  {'S': 148, 'I': 6, 'R': 146}
SEIRD  terminates=True   t_end= 170.9  {'S': 13, 'E': 0, 'I': 0, 'R': 267, 'D': 20}
```

`terminates` is true when the transitions between states contain no cycle, so every run must reach a state where nothing can happen; such runs continue until absorption. SIS and SIRS can cycle forever, so without `t_max` they stop at `ag.sim.DEFAULT_T_MAX` (100 time units), which is why both end at t = 100 above.

### Starting states and horizons

`initial` accepts several forms:

| Form | Meaning |
|---|---|
| `{"I": 3}` | 3 nodes drawn at random (with the run's seed) start in I |
| `{"I": 0.05}` | 5% of the nodes, rounded half up |
| `{"I": ["MSC", "ICT"]}` or `{"I": "MSC"}` | these nodes; explicit nodes are placed before counts and fractions are drawn |
| `{"I": 2, "R": 0.3}` | several states at once |
| `{node: state, ...}` | a complete assignment |
| omitted | one random node in the via state of the first induced transition |

The horizon is `t_max` (a time) or `steps` (a number of `dt` steps, `t_max = steps · dt`). A run also stops early as soon as no transition can fire.

### Custom models

Any model that fits the two transition kinds is one constructor call away. This one adds a vaccination campaign to SIR: susceptible people get vaccinated at rate 0.02 per time unit, which competes with infection.

```python
vaccination = ag.sim.CompartmentalModel(
    ["S", "V", "I", "R"],
    spontaneous=[("S", "V", 0.02), ("I", "R", 0.1)],   # (from, to, rate)
    induced=[("S", "I", "I", 0.1)],                    # (from, to, via, rate)
    name="SIR + vaccination",
)
print(vaccination)
print(vaccination.terminates, vaccination.roles)
res = vaccination.simulate(g, initial={"I": 3}, method="gillespie", seed=7)
print(res)
t_peak, n_peak = res.peak("I")
print(f"peak: {n_peak} infectious nodes at t = {t_peak:.1f}")
```

```text
<CompartmentalModel SIR + vaccination: S→V (0.02), I→R (0.1), S→I (0.1 × I)>
True {'S': 'neutral', 'V': 'good', 'I': 'critical', 'R': 'good'}
<SimulationResult SIR + vaccination: 300 nodes, 101 frames, t=0…399.6; final S=0, V=181, I=0, R=119>
peak: 39 infectious nodes at t = 20.0
```

Compared with plain SIR under the same seed (peak of 70 infectious nodes, 285 recovered), vaccination lowers the peak to 39 and leaves 181 of the 300 nodes vaccinated before the epidemic reached them; 119 were infected at some point. The run lasts longer (t ≈ 400) because it only ends once every susceptible node has been either infected or vaccinated.

```{figure} ../_static/generated/guide_sim_data/vaccination_curve.png
:alt: Line chart of the shares of susceptible, vaccinated, infectious and recovered nodes; the vaccinated share rises steadily to about 60 percent.
:width: 100%

The custom model plotted with `res.plot(fractions=True)`: the y axis shows shares of nodes instead of counts.
```

`roles` controls colors in charts and animations. Conventional names get their usual meaning (`"S"` neutral, `"I"` critical, `"R"` and `"V"` good, `"E"` warning, `"D"` muted); other names take the next role of a fixed cycle, and `roles={"Q": "serious"}` overrides any of them.

## Discrete time or Gillespie

`simulate()` offers two solvers.

`method="gillespie"`
: Exact stochastic simulation of the continuous-time Markov chain (Gillespie's direct method). Each event updates only the node that changed and its neighbors, and the next event is drawn from a binary sum tree in O(log N). Use it when timing matters or when you compare with analytical results.

`method="discrete"` (the default)
: Synchronous updates every `dt`. A node whose total hazard is *H* changes state with probability `1 − exp(−H·dt)`, choosing among competing transitions in proportion to their rates. It is fast and easy to reason about step by step, but it carries an O(dt) bias: for example, a node stays infectious for a whole number of steps, on average `dt / (1 − exp(−γ·dt)) ≈ 1/γ + dt/2`.

The bias is visible in an ensemble of 200 runs per solver:

```python
import numpy as np

for method, extra in [("gillespie", {}), ("discrete", {"dt": 1.0}), ("discrete", {"dt": 0.1})]:
    ens = ag.sim.run_ensemble(ag.sim.sir, runs=200, g=g, beta=0.1, gamma=0.1,
                              initial={"I": 3}, method=method, seed=11, **extra)
    k = int(np.argmax(ens.mean["I"]))
    print(f"{method:9} {str(extra):12} mean final R = {ens.final_sizes['R'].mean():6.1f}"
          f"   mean-curve peak = {ens.mean['I'][k]:5.1f} at t = {ens.times[k]:4.1f}")
```

```text
gillespie {}           mean final R =  276.8   mean-curve peak =  65.8 at t = 29.3
discrete  {'dt': 1.0}  mean final R =  288.8   mean-curve peak =  76.5 at t = 32.0
discrete  {'dt': 0.1}  mean final R =  278.2   mean-curve peak =  69.5 at t = 29.4
```

With `dt = 1` the discrete solver overstates the final size and the peak, because every infectious node keeps transmitting for a full step. At `dt = 0.1` the discrete results move close to the Gillespie ones, at ten times the number of steps.

## Ensembles

A single stochastic run is one sample. {py:func}`~aryagraph.sim.run_ensemble` repeats any simulator that takes a `seed` and summarizes the state counts on a common time grid:

```python
ens = ag.sim.run_ensemble(ag.sim.sir, runs=200, g=g, beta=0.1, gamma=0.1,
                          initial={"I": 3}, method="gillespie", seed=11)
print(ens)
sizes = ens.final_sizes["R"]
print(np.percentile(sizes, [5, 50, 95]))
print((sizes < 30).mean())
lo, hi = ens.band("I")                 # 5th and 95th percentile curves
print(ens.times.shape, lo.shape, sorted(ens.quantiles["I"]))
```

```text
<EnsembleResult SIR: 200 runs, 300 nodes; final S=23.19±25.3, I=0±0, R=276.8±25.3>
[253.85 281.   296.  ]
0.005
(101,) (101,) [5, 25, 50, 75, 95]
```

Across the 200 runs, the final number of recovered nodes ranges from 254 (5th percentile) to 296 (95th percentile) around a median of 281, and 1 run in 200 died out early with fewer than 30 recoveries. The {py:class}`~aryagraph.sim.ensemble.EnsembleResult` holds the mean and standard deviation per state, the requested percentiles (5, 25, 50, 75 and 95 by default), every run's final counts and every run's seed. Child seeds come from `numpy.random.SeedSequence(seed).spawn`, so the ensemble is reproducible from one master seed and each run can be replayed on its own:

```python
k = 17
replay = ag.sim.sir(g, 0.1, 0.1, initial={"I": 3}, method="gillespie", seed=ens.seeds[k])
print(int(replay.counts()["R"][-1]), int(ens.final_sizes["R"][k]))
```

```text
272 272
```

```{figure} ../_static/generated/guide_sim_data/sir_ensemble.png
:alt: Mean susceptible, infected and recovered curves of 200 runs with shaded 25 to 75 percent and 5 to 95 percent bands.
:width: 100%

`ens.plot()`: the mean of each state with the 25–75% band (darker) and the 5–95% band (lighter).
```

`interpolation="step"` (the default) holds each count until the next frame, which matches jump processes; `interpolation="linear"` interpolates between frames. Pass `keep=True` to also keep every individual {py:class}`~aryagraph.sim.base.SimulationResult`.

## Cascades and influence maximization

Cascade models describe how an idea, a product or a rumor spreads once: nodes go from `"inactive"` to `"active"` and stay active.

* **Independent cascade** ({py:func}`~aryagraph.sim.independent_cascade`): a node activated at step *t* gets one chance to activate each inactive neighbor at step *t* + 1, with probability `p` (a number, an edge attribute name or a function of the edge).
* **Linear threshold** ({py:func}`~aryagraph.sim.linear_threshold`): each node has a threshold θ (uniform on (0, 1] unless given) and activates once the share of its in-weight coming from active neighbors reaches θ.

```python
les = ag.gen.les_miserables()
ic = ag.sim.independent_cascade(les, "Valjean", p=0.1, seed=3)
print(ic)
print(ic.meta["newly_active"])
lt = ag.sim.linear_threshold(les, ["Valjean", "Javert"], seed=3)
print(lt.meta["spread"], len(lt.meta["newly_active"]) - 1)
```

```text
<SimulationResult IC: 77 nodes, 2 frames, t=0…1; final inactive=74, active=3>
[['Valjean'], ['MmeDeR', 'Fantine']]
37 8
```

One cascade is a single random draw. {py:func}`~aryagraph.sim.influence_spread` estimates the *expected* final number of active nodes by Monte Carlo, using the live-edge representation (Kempe, Kleinberg and Tardos, 2003): each run samples which edges would transmit and counts the nodes reachable from the seeds, which has the same distribution as a full cascade at a fraction of the cost.

```python
est = ag.sim.influence_spread(les, ["Valjean"], runs=2000, p=0.1, seed=1)
low, high = est.confidence_interval()
print(est, f"95% interval: {low:.2f} to {high:.2f}")
```

```text
SpreadEstimate(IC: 9.796 ± 0.14, runs=2000) 95% interval: 9.52 to 10.07
```

{py:func}`~aryagraph.sim.greedy_influence_maximization` picks the *k* seeds with the largest estimated spread, adding one node at a time by marginal gain. It uses CELF lazy evaluation, which returns the same choice as a plain greedy search with far fewer evaluations. The spread it reports is measured on the samples used for the selection, so it is optimistically biased; re-estimate it with a different seed before you compare strategies:

```python
im = ag.sim.greedy_influence_maximization(les, 3, runs=500, p=0.1, seed=1)
print(im)
greedy = ag.sim.influence_spread(les, im.seeds, runs=2000, p=0.1, seed=2)
degree = les.degree()
top = sorted(degree, key=degree.get, reverse=True)[:3]
by_degree = ag.sim.influence_spread(les, top, runs=2000, p=0.1, seed=2)
print(im.seeds, greedy)
print(top, by_degree)
```

```text
InfluenceMaximization(IC: seeds=['Valjean', 'Gavroche', 'Fantine'], spread≈16.62, runs=500, evaluations=144)
['Valjean', 'Gavroche', 'Fantine'] SpreadEstimate(IC: 17.26 ± 0.13, runs=2000)
['Valjean', 'Gavroche', 'Marius'] SpreadEstimate(IC: 16.82 ± 0.12, runs=2000)
```

The greedy search evaluated 144 marginal gains for 77 candidates and 3 seeds. On independent samples its seed set (Valjean, Gavroche, Fantine) reaches about 17.3 nodes on average, against 16.8 for the three highest-degree characters (Valjean, Gavroche, Marius). Degree ignores overlap: only 5 of Marius's 19 neighbors are not already neighbors of Valjean or Gavroche, against 8 of Fantine's 15. Both estimates use the same seed, so they are evaluated on the same random live-edge samples and the comparison is paired.

Pass `model="lt"` to optimize for the linear threshold model instead, and `candidates=` to restrict which nodes may be chosen.

## Random walks

{py:func}`~aryagraph.sim.random_walk` moves independent walkers along edges, choosing each step in proportion to edge weight. A walker on a node without outgoing weight teleports to a random node, as in PageRank, and `restart` sends it back to its own starting node with the given probability per step. The result marks nodes as `"unvisited"`, `"visited"` or `"current"`, and `meta` keeps the full trajectories and the share of time spent on each node.

For a connected undirected graph, the share of time converges to the stationary distribution, which {py:func}`~aryagraph.sim.stationary_distribution` computes without sampling (here in closed form: degree divided by twice the number of edges):

```python
campus = ag.gen.ucalgary_campus()
walk = ag.sim.random_walk(campus, walkers=200, steps=2000, seed=4)
visits = walk.meta["visits"]
pi = ag.sim.stationary_distribution(campus)
print([(b, round(p, 4)) for b, p in pi.top(3)])
print([(b, round(p, 4)) for b, p in visits.top(3)])
print(round(sum(abs(visits[n] - pi[n]) for n in campus), 4))
```

```text
[('ES', 0.0301), ('CD', 0.0241), ('CH', 0.0241)]
[('ES', 0.0295), ('MTH', 0.0252), ('CH', 0.0249)]
0.025
```

With restarts, the walk stays near its origin; the visit shares then measure proximity to that building, the idea behind personalized PageRank:

```python
local = ag.sim.random_walk(campus, walkers=50, steps=200, start="MSC", restart=0.15, seed=4)
print([(b, round(p, 3)) for b, p in local.meta["visits"].top(4)])
print(ag.sim.transition_matrix(campus).shape)
```

```text
[('MSC', 0.232), ('TI', 0.106), ('MH', 0.104), ('KNB', 0.086)]
(56, 56)
```

## Opinion dynamics

Four models cover the common cases. Influence travels along edges; on a directed graph, `u → v` means that *v* listens to *u*.

| Model | Opinions | Update |
|---|---|---|
| {py:func}`~aryagraph.sim.voter_model` | discrete | a random node copies a random neighbor; *n* updates per sweep; stops at consensus |
| {py:func}`~aryagraph.sim.majority_rule` | discrete | every node adopts the opinion with the most weight among its neighbors; stops at a fixed point |
| {py:func}`~aryagraph.sim.degroot` | real values | every node averages itself and its neighbors: `x(t+1) = W x(t)` |
| {py:func}`~aryagraph.sim.bounded_confidence` | values in [0, 1] | Deffuant–Weisbuch: two neighbors whose opinions differ by less than ε move toward each other |

```python
voter = ag.sim.voter_model(g, opinions=2, steps=2000, seed=5)
print(voter.meta)
majority = ag.sim.majority_rule(g, opinions=2, seed=5)
print(majority, majority.meta)
```

```text
{'consensus': '0', 'consensus_step': 1165}
<SimulationResult majority: 300 nodes, 5 frames, t=0…4; final 0=158, 1=142> {'fixed_point': True}
```

The voter model on 300 nodes reached consensus on opinion `"0"` after 1,165 sweeps, while majority rule reached a fixed point after 4 steps with two local camps of 158 and 142 nodes. DeGroot averaging converges to a weighted consensus, and `meta["influence"]` gives each node's weight in it (its social power):

```python
florence = ag.gen.florentine_families()
rng = np.random.default_rng(0)
start = {family: float(x) for family, x in zip(florence, rng.random(len(florence)))}
avg = ag.sim.degroot(florence, start, steps=50)
print(round(avg.meta["consensus"], 4))
print([(family, round(w, 3)) for family, w in avg.meta["influence"].top(3)])
```

```text
0.5087
[('Medici', 0.127), ('Strozzi', 0.091), ('Guadagni', 0.091)]
```

In the bounded-confidence model the confidence bound ε sets how many opinion clusters survive. On a complete graph of 100 people:

```python
crowd = ag.gen.complete_graph(100)
for eps in (0.1, 0.2, 0.3):
    bc = ag.sim.bounded_confidence(crowd, epsilon=eps, steps=500, seed=5)
    print(eps, bc.meta["settled"], [round(c, 2) for c in bc.meta["clusters"]])
```

```text
0.1 True [0.05, 0.25, 0.45, 0.69, 0.88]
0.2 True [0.09, 0.35, 0.81]
0.3 True [0.52]
```

`meta["clusters"]` groups the sorted final opinions wherever consecutive values are at least ε apart. On sparse graphs, neighborhoods can settle on different values without such gaps appearing in the sorted list, so inspect the final opinions directly there.

## Heat diffusion

{py:func}`~aryagraph.sim.heat_diffusion` solves the heat equation `dx/dt = −rate · L x` on the graph, where *L* is the graph Laplacian. The solution is evaluated in closed form from one eigendecomposition, so there is no time-stepping error, and the total heat is conserved.

```python
heat = ag.sim.heat_diffusion(campus, "MSC", rate=1.0, t_max=5.0)
snapshot = heat.at(1.0)
print({b: f"{snapshot[b]:.1e}" for b in ("MSC", "OO", "SS", "TFDL", "OVC")})
hops = ag.alg.shortest_path_length(campus, "MSC")
print({b: hops[b] for b in ("OO", "SS", "TFDL", "OVC")})
print(round(float(heat.meta["total"][-1]), 12), round(heat.meta["steady_state"]["MSC"], 5))
```

```text
{'MSC': '1.8e-01', 'OO': '3.6e-02', 'SS': '3.8e-03', 'TFDL': '1.5e-05', 'OVC': '2.4e-07'}
{'OO': 2, 'SS': 4, 'TFDL': 7, 'OVC': 9}
1.0 0.01786
```

Heat follows edges, not straight-line distance. The Taylor Family Digital Library (TFDL) is less than 150 m from MacEwan Student Centre as the crow flies but 7 links away in the graph, so after one time unit it holds far less heat than Social Sciences (SS), 4 links away, or the Olympic Oval (OO), 2 links away. As *t* grows every node tends to the average, 1/56 of the unit released.

```{figure} ../_static/generated/guide_sim_data/heat_campus.png
:alt: Map of the UCalgary campus buildings colored by the logarithm of heat one time unit after releasing heat at MacEwan Student Centre; nearby buildings in graph terms are dark, remote ones light.
:width: 100%

Heat on the campus graph one time unit after its release at MSC, colored on a log10 scale. Building positions © OpenStreetMap contributors (ODbL).
```

`normalized=True` uses the random-walk Laplacian instead, whose steady state is proportional to degree. `weight=` turns an edge attribute into conductance.

## Kuramoto oscillators

{py:func}`~aryagraph.sim.kuramoto` couples phase oscillators along the edges: each node has a natural frequency and is pulled toward its neighbors' phases with strength `coupling`. The system is integrated with fourth-order Runge–Kutta, and `meta["order_parameter"]` measures synchrony *r(t)* from 0 (incoherent) to 1 (all in phase).

```python
for K in (0.5, 1.0, 2.0, 4.0):
    osc = ag.sim.kuramoto(g, coupling=K, t_max=20, seed=2)
    r = osc.meta["order_parameter"]
    print(f"K = {K:3}: r(0) = {r[0]:.3f}, r(20) = {r[-1]:.3f}")
```

```text
K = 0.5: r(0) = 0.076, r(20) = 0.467
K = 1.0: r(0) = 0.076, r(20) = 0.750
K = 2.0: r(0) = 0.076, r(20) = 0.949
K = 4.0: r(0) = 0.076, r(20) = 0.988
```

```{figure} ../_static/generated/guide_sim_data/kuramoto.png
:alt: Line chart of the order parameter over time for four coupling strengths; stronger coupling synchronizes faster and more fully.
:width: 100%

Synchrony on the 300-node small world for four coupling strengths, drawn with {py:func}`aryagraph.charts.line_chart`.
```

Phases are stored in [0, 2π), and `meta["frequencies"]` keeps the natural frequencies (standard normal draws from the seed unless you pass them). `frequencies` and `initial` accept one number, a `{node: value}` mapping or values in graph order; `normalize=True` divides the coupling by each node's degree.

## Plotting and animating

Every result has the same two display methods.

`result.plot(...)`
: A {py:class}`~aryagraph.charts.base.Chart` of the state counts over time (categorical results) or of the mean with a min–max band across nodes (continuous results). Options include `fractions=True`, `title`, `subtitle`, `width`, `height` and `theme`. Save it as `.svg`, `.png`, `.pdf` or `.html`.

`result.animate(layout=None, ...)`
: An interactive {py:class}`~aryagraph.render.figure.Figure`. `layout` takes anything {py:func}`aryagraph.render.draw` accepts, including a `{node: (x, y)}` mapping of real positions; every other drawing option (`labels`, `node_size`, `theme`, …) passes through. `max_frames` (default 600) subsamples long runs while merging the edge activity of skipped frames, `static_frame` chooses the frame shown in `.svg` and `.png` exports, `chart=False` removes the synced chart, and `autoplay=True` starts playback on load.

```python
campus_pos = {n: d["pos"] for n, d in campus.nodes.data()}
fig = heat.animate(layout=campus_pos, labels=False, static_frame=20)
fig.save("heat.html")          # interactive player
fig.save("heat.svg")           # frame 20 as a static drawing
print(fig)
```

```text
<Figure 1237×895: 56 nodes, 83 edges, custom layout, theme 'light'>
```

{py:class}`~aryagraph.sim.ensemble.EnsembleResult` and {py:class}`~aryagraph.sim.scheduling.MonteCarloResult` have a `.plot()` of their own, and {py:class}`~aryagraph.sim.scheduling.ScheduleResult` adds `.gantt()`. For a guided example, see the [epidemics tutorial](../tutorials/epidemics.md) and the [influence tutorial](../tutorials/influence.md).
