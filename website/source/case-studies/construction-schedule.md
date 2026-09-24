# Scheduling a house build under uncertainty

A house is a classic project network: permits before excavation, framing before roofing, drywall before paint. This case study takes AryaGraph's bundled construction plan through the critical-path method, a crew-constrained schedule, a PERT Monte Carlo and a rework model, and turns the results into planning guidance you can check line by line.

## The question

A builder planning this house wants four answers:

1. How long does the build take on paper, and which tasks set that length?
2. How many crews are worth assigning?
3. How much should the schedule be padded for uncertain durations, and whose uncertainty matters most?
4. What does rework (a task that has to be redone) do to the finish date?

## Data

{py:func}`aryagraph.generators.dags.project_plan` returns an 18-task plan for a single-family house, from site survey to final inspection. It is an illustrative plan in the style of textbook CPM examples, not a record of a real project. Durations are **working hours** (40 h is one working week); every task has a three-point estimate (`min`, `mode`, `max`) and the planned `duration` equals the `mode`. Two arcs carry a `lag`, a mandatory wait after the predecessor finishes: concrete curing after the foundation and drying after the drywall.

```python
import aryagraph as ag

plan = ag.gen.project_plan()
print(plan)
print(f"{'task':<22} {'min':>5} {'mode':>5} {'max':>5}  kind")
for n, d in plan.nodes.data():
    print(f"{n:<22} {d['min']:>5g} {d['mode']:>5g} {d['max']:>5g}  {d['kind']}")
print("arcs with a lag:", [(u, v, lag) for u, v, lag in plan.edges.data("lag") if lag])
print("total work:", sum(d["duration"] for _, d in plan.nodes.data()), "h")
```

```text
<DAG 'house construction': 18 nodes, 23 edges>
task                     min  mode   max  kind
site survey               12    16    24  planning
architectural design      60    80   120  planning
building permit           80   120   240  permit
excavation                16    24    40  sitework
foundation                32    40    64  structure
framing                   96   120   160  structure
roofing                   32    40    64  envelope
windows & doors           16    24    32  envelope
plumbing rough-in         32    40    56  systems
electrical rough-in       32    40    60  systems
HVAC install              24    32    48  systems
insulation                12    16    24  interior
drywall                   40    48    72  interior
interior painting         24    32    48  interior
flooring                  24    32    48  interior
cabinets & fixtures       16    24    40  interior
landscaping               24    40    64  exterior
final inspection           2     4     8  inspection
arcs with a lag: [('foundation', 'framing', 72.0), ('drywall', 'interior painting', 24.0)]
total work: 772.0 h
```

The building permit has by far the widest estimate: its pessimistic value (240 h) sits 120 h above its most likely one (120 h), three times the next-largest spread (40 h, for architectural design and framing). Final inspection has the same two-to-one ratio, but only 4 h separates its mode from its maximum. Eleven of the other 17 tasks have a pessimistic estimate at most 50% above the mode.

## Method

| Step | What it computes | AryaGraph function |
|---|---|---|
| Critical-path method (CPM) | earliest and latest start of every task, slack, the critical path | {py:func}`~aryagraph.algorithms.dag.critical_path` |
| Crew-limited schedule | makespan when at most *k* tasks run at once, for a dispatching policy | {py:func}`~aryagraph.sim.simulate_schedule` |
| PERT Monte Carlo | distribution of the finish time (P50, P80, P95) and each task's criticality index | {py:func}`~aryagraph.sim.monte_carlo_schedule` |
| Sensitivity | the change in P80 when one task's duration is fixed at its most likely value | {py:func}`~aryagraph.sim.monte_carlo_schedule` on modified copies |
| Rework | the finish-time distribution when each attempt fails with probability *p* and is redone | `failure_rate` and `max_retries` of the same functions |

Two modeling choices need explaining before the results.

**Lags become wait tasks.** {py:func}`~aryagraph.algorithms.dag.critical_path` and the schedulers read durations from the *nodes*; they do not read the `lag` attribute on arcs. The plan's two waits are real constraints, so the helper below replaces each lagged arc `u → v` with `u → wait → v`, where the wait task has a fixed duration equal to the lag.

**Crews are a resource.** A crew here is a generic unit of capacity: every task occupies one crew for its whole duration, and the two waits occupy none (concrete cures without a crew on site). The schedulers express this with `resources={"crew": k}` and a `crew` attribute on each task. Passing `workers=None` removes the separate worker limit, so crews are the only constraint.

## Results

### The plan on paper: 668 working hours

Run CPM on the plan as bundled and you get 572 h, because the two lags are not part of the calculation:

```python
print(ag.alg.critical_path(plan))
```

```text
CriticalPath(length=572, path=['site survey' → 'architectural design' → 'building permit' → 'excavation' → 'foundation' → 'framing' → 'roofing' → 'insulation' → …], critical=14 of 18 activities)
```

With the waits made explicit, the plan grows by the full 96 h of the two lags (572 + 72 + 24 = 668), since both sit on the critical path:

```python
def with_lags(dag):
    """Turn every arc's lag into an explicit wait task (fixed duration, needs no crew)."""
    out = dag.copy()
    for u, v, lag in dag.edges.data("lag"):
        if lag:
            wait = f"wait after {u}"
            out.remove_edge(u, v)
            out.add_node(wait, duration=lag, min=lag, mode=lag, max=lag, kind="wait", team="none")
            out.add_edges([(u, wait), (wait, v)])
    for n in out:
        out.nodes[n]["crew"] = 0 if out.nodes[n]["kind"] == "wait" else 1
    return out


house = with_lags(plan)
cp = ag.alg.critical_path(house)
print(house)
print(cp)
print(f"{cp.length:g} h = {cp.length / 40:.1f} working weeks")
```

```text
<DAG 'house construction': 20 nodes, 25 edges>
CriticalPath(length=668, path=['site survey' → 'architectural design' → 'building permit' → 'excavation' → 'foundation' → 'wait after foundation' → 'framing' → 'roofing' → …], critical=16 of 20 activities)
668 h = 16.7 working weeks
```

Sixteen of the 20 activities have zero slack. The four with slack are the only places where the schedule can absorb a delay for free:

```python
print("critical:", [n for n in house.topological_order() if n in cp.critical])
for n in house.topological_order():
    if cp.slack[n] > 0:
        print(f"{n:<20} earliest start {cp.earliest_start[n]:>4g} h, slack {cp.slack[n]:>4g} h")
```

```text
critical: ['site survey', 'architectural design', 'building permit', 'excavation', 'foundation', 'wait after foundation', 'framing', 'roofing', 'plumbing rough-in', 'electrical rough-in', 'insulation', 'drywall', 'wait after drywall', 'interior painting', 'flooring', 'final inspection']
windows & doors      earliest start  472 h, slack   16 h
HVAC install         earliest start  472 h, slack    8 h
landscaping          earliest start  512 h, slack  112 h
cabinets & fixtures  earliest start  632 h, slack    8 h
```

Roofing, plumbing and electrical rough-in tie at 40 h after framing, so all three are critical; any one of them running late delays insulation. Landscaping can slip 112 h (almost three working weeks) without moving the finish date.

```{figure} ../_static/generated/cases_a/house_plan.png
:alt: Layered diagram of the 20-activity house plan, from site survey at the top to final inspection at the bottom, colored by phase, with the critical path highlighted.
:width: 75%

The plan with its two waits made explicit, colored by phase. The highlighted chain is one critical path (roofing is shown; plumbing and electrical rough-in tie with it).
```

<iframe class="ag-embed" src="../_static/generated/cases_a/house_plan.html" height="710" loading="lazy" title="Interactive house construction plan"></iframe>
<p class="ag-embed-note">Interactive: hover a task for its estimates, drag to pan, scroll to zoom. <a href="../_static/generated/cases_a/house_plan.html">Open full screen</a></p>

### Crews: the second one pays, the fifth reaches the bound

The plan is a single chain until framing; only then can up to five tasks run side by side. The table compares one to five crews under two dispatching policies: `critical_path` (start the ready task with the longest remaining chain first) and `fifo` (first ready, first started).

```python
real = [n for n in house if house.nodes[n]["kind"] != "wait"]
work = sum(house.nodes[n]["duration"] for n in real)
print("crews  critical_path  fifo  crew utilization")
for k in range(1, 6):
    best = ag.sim.simulate_schedule(house, workers=None, resources={"crew": k}, policy="critical_path")
    fifo = ag.sim.simulate_schedule(house, workers=None, resources={"crew": k}, policy="fifo")
    print(f"{k:>5} {best.makespan:>14g} {fifo.makespan:>5g} {work / (k * best.makespan):>17.0%}")
```

```text
crews  critical_path  fifo  crew utilization
    1            844   868               91%
    2            724   724               53%
    3            700   700               37%
    4            684   684               28%
    5            668   668               23%
```

A second crew saves 120 h (three working weeks). The third saves 24 h, and the fourth and fifth 16 h each; five crews reach the 668 h critical-path bound, when every task after framing can start the moment it is ready. Crew utilization (work hours divided by crews times makespan) falls quickly because the long chain before framing keeps extra crews idle.

With one crew, the policy matters. Critical-path-first keeps the crew on the interior sequence and fills the drying wait with landscaping; FIFO does landscaping as soon as the roof is on, so the crew later has nothing to do while the drywall dries:

```python
for policy in ("critical_path", "fifo"):
    one = ag.sim.simulate_schedule(house, workers=None, resources={"crew": 1}, policy=policy)
    start = {t.node: t.start for t in one.tasks}
    print(f"{policy:<13} landscaping starts at {start['landscaping']:g} h; "
          f"drying wait {start['wait after drywall']:g}–{start['wait after drywall'] + 24:g} h; "
          f"finish {one.makespan:g} h")
```

```text
critical_path landscaping starts at 712 h; drying wait 712–736 h; finish 844 h
fifo          landscaping starts at 648 h; drying wait 752–776 h; finish 868 h
```

```{figure} ../_static/generated/cases_a/house_gantt.png
:alt: Gantt chart of the two-crew schedule. Crew 1 carries the pre-construction chain and framing; both crews work in parallel after framing; a separate lane shows the two waits.
:width: 100%

The planned two-crew schedule (724 h). Crew lanes are assigned in start order; the "waiting" lane shows curing and drying, which need no crew. Accent-colored bars are critical activities.
```

### Uncertainty: plan for about 735 hours, not 668

The Monte Carlo samples every task's duration from a PERT (Beta) distribution fitted to its three-point estimate and recomputes the critical path, 5,000 times. The waits have equal `min`, `mode` and `max`, so they stay fixed. With `workers=None` and no resources this is a pure CPM/PERT simulation, which AryaGraph vectorizes over the runs.

```python
mc = ag.sim.monte_carlo_schedule(house, runs=5000, seed=7)
print(mc)
print(f"chance of finishing within the plan's {mc.cpm_length:g} h: {mc.probability(mc.cpm_length):.1%}")
print(f"standard error of the mean: {mc.stderr:.2f} h")
```

```text
<MonteCarloResult: 5000 runs, makespan 705.6 ± 34.6; P50=703.4, P80=734.5, P95=765.4>
chance of finishing within the plan's 668 h: 13.8%
standard error of the mean: 0.49 h
```

The deterministic 668 h is met in only 13.8% of runs. Two effects push the distribution right: the estimates are right-skewed (the permit most of all), and where several paths run in parallel after framing, the finish waits for the slowest of them. The median run takes 703 h, and a date with an 80% chance of being met is 734.5 h, or 18.4 working weeks.

```{figure} ../_static/generated/cases_a/house_montecarlo.png
:alt: Histogram of 5,000 simulated makespans between about 610 and 850 hours, with reference lines at the plan (668), P50 (703.4), P80 (734.5) and P95 (765.4).
:width: 90%

Distribution of the finish time over 5,000 PERT runs with unlimited crews, with the deterministic plan and the P50, P80 and P95 marked.
```

Crews and uncertainty combine: the same simulation with a crew limit runs a full schedule per sample.

```python
for k in range(1, 6):
    m = ag.sim.monte_carlo_schedule(house, runs=1000, resources={"crew": k}, seed=7)
    print(f"{k} crews: " + ", ".join(f"{q} {v:.0f} h" for q, v in m.percentiles.items()))
```

```text
1 crews: P50 883 h, P80 915 h, P95 949 h
2 crews: P50 758 h, P80 791 h, P95 821 h
3 crews: P50 730 h, P80 762 h, P95 792 h
4 crews: P50 715 h, P80 747 h, P95 778 h
5 crews: P50 704 h, P80 738 h, P95 767 h
```

The pattern from the deterministic table holds under uncertainty: going from one crew to two lowers P80 by 124 h, and each crew after that by 29 h or less.

```{figure} ../_static/generated/cases_a/house_crews.png
:alt: Line chart of makespan against crews from 1 to 5, for the deterministic plan and the P50 and P80 of the PERT simulation. All three lines drop steeply from 1 to 2 crews and flatten afterwards.
:width: 90%

Makespan against crews: the deterministic plan and the simulated P50 and P80 (1,000 runs per point).
```

### Which tasks drive the finish date

The **criticality index** is the share of runs in which a task had zero total float, that is, lay on a critical path.

```python
always = [n for n in real if mc.criticality[n] == 1.0]
print(len(always), "tasks critical in every run:", always)
for n, share in sorted(mc.criticality.items(), key=lambda kv: -kv[1]):
    if share < 1.0:
        print(f"  {n:<20} {share:6.1%}")
```

```text
10 tasks critical in every run: ['site survey', 'architectural design', 'building permit', 'excavation', 'foundation', 'framing', 'insulation', 'drywall', 'interior painting', 'final inspection']
  flooring              88.7%
  roofing               38.8%
  electrical rough-in   33.7%
  plumbing rough-in     25.7%
  cabinets & fixtures   11.3%
  HVAC install           1.8%
  windows & doors        0.0%
  landscaping            0.0%
```

Ten tasks (plus the two waits) were critical in every run. Most of them lie on every path through the plan; the only path around insulation, drywall and painting runs through landscaping, and its 112 h of slack was not used up in any run. After framing, the three 40 h tasks share the critical role: roofing 38.8%, electrical 33.7%, plumbing 25.7%, and HVAC 1.8%. These four add up to 100% because in each run the one that finishes last is the critical one. Windows & doors and landscaping were not critical in any of the 5,000 runs.

```{figure} ../_static/generated/cases_a/house_criticality.png
:alt: Bar chart of the criticality index. Ten tasks at 100%, flooring 88.7%, roofing 38.8%, electrical rough-in 33.7%, plumbing rough-in 25.7%, cabinets and fixtures 11.3%, HVAC install 1.8%; windows and doors and landscaping have no bar (0%).
:width: 90%

Criticality index of the 18 real tasks over 5,000 runs. Windows & doors and landscaping have a criticality of 0%.
```

Criticality says *whether* a task can delay the project, not *how much* its uncertainty contributes. To measure that, pin one task at a time to its most likely duration and see how far P80 moves:

```python
ref = ag.sim.monte_carlo_schedule(house, runs=20000, seed=7)
gains = {}
for n in real:
    pinned = house.copy()
    pinned.nodes[n]["min"] = pinned.nodes[n]["max"] = pinned.nodes[n]["mode"]
    m = ag.sim.monte_carlo_schedule(pinned, runs=20000, seed=7)
    gains[n] = (ref.percentiles["P80"] - m.percentiles["P80"], m.std)
print(f"reference: P80 {ref.percentiles['P80']:.1f} h, standard deviation {ref.std:.1f} h")
for n, (gain, sd) in sorted(gains.items(), key=lambda kv: -kv[1][0])[:5]:
    print(f"  pin {n:<20} P80 {-gain:+5.1f} h, standard deviation {sd:4.1f} h")
other_seed = ag.sim.monte_carlo_schedule(house, runs=20000, seed=8)
print(f"P80 with seed 8 instead of 7: {other_seed.percentiles['P80']:.1f} h")
```

```text
reference: P80 734.9 h, standard deviation 34.9 h
  pin building permit      P80 -25.8 h, standard deviation 20.4 h
  pin architectural design P80  -4.8 h, standard deviation 33.0 h
  pin framing              P80  -4.3 h, standard deviation 33.1 h
  pin drywall              P80  -3.2 h, standard deviation 34.6 h
  pin foundation           P80  -3.0 h, standard deviation 34.5 h
```

```text
P80 with seed 8 instead of 7: 735.3 h
```

The permit dominates. Fixing it at 120 h lowers P80 by 25.8 h and the standard deviation from 34.9 h to 20.4 h, which is a reduction in variance of about two thirds. No other single task moves P80 by more than 5 h. Changing the seed moves the reference P80 by 0.4 h, which gives a sense of the Monte Carlo noise in these differences.

```{figure} ../_static/generated/cases_a/house_sensitivity.png
:alt: Bar chart of hours saved at P80 when one task's uncertainty is removed. Building permit 25.8 h; architectural design 4.8 h; framing 4.3 h; drywall 3.2 h; foundation 3.0 h; the rest under 3 h.
:width: 90%

Hours saved at P80 by pinning one task to its most likely duration (20,000 runs each, the eight largest shown).
```

### Rework: the cost of doing a task twice

In the scheduler, each attempt of a task fails with probability *p*; the failure shows at the end of the attempt, and the task is queued again while it has retries left. Here every real task gets the same *p* (the waits cannot fail), two crews do the work, and up to 10 retries make it practically certain that every task is eventually completed.

```python
for p in (0.0, 0.05, 0.10, 0.15, 0.20):
    m = ag.sim.monte_carlo_schedule(
        house, runs=2000, resources={"crew": 2}, failure_rate={n: p for n in real}, max_retries=10, seed=7
    )
    print(f"p = {p:4.0%}: " + ", ".join(f"{q} {v:.0f} h" for q, v in m.percentiles.items()))
print(f"chance that some task needs more than 10 retries at p = 20%: {1 - (1 - 0.2 ** 11) ** len(real):.1e}")
```

```text
p =   0%: P50 758 h, P80 791 h, P95 822 h
p =   5%: P50 783 h, P80 839 h, P95 929 h
p =  10%: P50 815 h, P80 898 h, P95 1012 h
p =  15%: P50 852 h, P80 952 h, P95 1072 h
p =  20%: P50 906 h, P80 1022 h, P95 1183 h
```

```text
chance that some task needs more than 10 retries at p = 20%: 3.7e-07
```

Rework widens the distribution more than it shifts it. At a 10% chance of redoing any task, P50 grows by 57 h but P95 by 190 h, because a single repeat of a long critical task (framing or the permit) lands in the tail. Each additional 5 percentage points of rework probability adds between 48 h and 70 h to P80.

```{figure} ../_static/generated/cases_a/house_rework.png
:alt: Line chart of P50, P80 and P95 of the makespan against a per-attempt rework probability from 0% to 20%, with two crews. All three rise; P95 rises fastest, from about 820 to about 1,180 hours.
:width: 90%

Finish-time percentiles against the per-attempt rework probability (two crews, 2,000 runs per point).
```

## What the model shows

These points restate the results as planning guidance. They follow from this model's estimates and assumptions, not from data about real builds.

- **Quote a P80 date, not the CPM date.** The deterministic 668 h is met in 13.8% of simulated runs; 734.5 h (about 18.4 working weeks) is met in 80%.
- **Two crews capture most of the benefit.** The second crew cuts 120 h from the plan and 124 h from P80; each further crew saves 29 h or less at P80 while crew utilization falls to 37% and below.
- **Start the permit early or narrow its range.** The permit alone accounts for about two thirds of the variance in the finish time. Anything that reduces its uncertainty (a pre-application meeting, a complete submission) is worth more than tightening any other estimate.
- **Dispatch by critical path when crews are scarce.** With one crew, critical-path-first finishes 24 h earlier than first-come-first-served by using the drying wait for landscaping.
- **Treat rework as a tail risk.** A 10% rework rate moves P95 by 190 h, more than three times its effect on the median.
- **Use the slack.** Landscaping (112 h of slack) and windows & doors (16 h) can absorb interruptions or move to fit crew availability without delaying the finish.

## Limitations

- **Illustrative data.** The plan and its estimates are a textbook-style example, not measurements from real projects.
- **Independent durations.** Each task's duration is sampled on its own. In practice delays are correlated (weather, one slow subcontractor), which would widen the distribution beyond what is shown here.
- **One clock for everything.** Waits are counted in working hours like the tasks, although concrete keeps curing overnight and on weekends; a calendar model would shorten the curing wait in elapsed time.
- **Generic crews.** Every task can use any crew. Real trades (electricians, plumbers, the city's inspector) are not interchangeable, and the permit and inspection do not occupy the builder's crews at all.
- **Full-duration rework.** A failed attempt occupies its crew for the whole drawn duration and is then repeated in full. Partial rework would cost less.

## Reproducibility

The code blocks on this page run top to bottom, in order, with AryaGraph {{ version }}; the seeds are fixed (`seed=7` for every simulation, `seed=8` for the one noise check), so the printed results above are the actual output. The figures come from the same computations in the site's asset script, [`website/scripts/assets/cases_a.py`](https://github.com/ruzbahani/AryaGraph/blob/main/website/scripts/assets/cases_a.py). To draw the schedule or the plan yourself:

```python
two = ag.sim.simulate_schedule(house, workers=None, resources={"crew": 2})
two.gantt(title="House construction with two crews", time_label="working hours").save("gantt.svg")
mc.plot(subtitle="5,000 PERT runs").save("montecarlo.svg")
ag.draw(house, highlight_path=cp.path, title="House construction plan").save("plan.html")
```

Related pages: [Pipelines as DAGs](../tutorials/pipeline-dag.md), [DAG workflows](../user-guide/dags.md), [Simulation](../user-guide/simulation.md), [Critical path and parallelism in an ML pipeline](ml-pipeline.md).
