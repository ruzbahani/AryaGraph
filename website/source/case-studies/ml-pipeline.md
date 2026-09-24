# Critical path and parallelism in an ML pipeline

A machine-learning pipeline is a DAG of data work, training runs and deployment steps owned by different teams. This case study measures what sets the end-to-end time of AryaGraph's bundled churn-model pipeline, how much of it can run in parallel, how many workers it needs, whether the dispatching policy matters, and how many retries failing tasks should get.

## The question

1. Which chain of tasks sets the end-to-end time, and how much work can run in parallel?
2. How many workers (machines, or people) does the pipeline need?
3. Does it matter which ready task a free worker picks up first?
4. When tasks can fail, how many retries does a run need to finish reliably, and what do the retries cost?

## Data

{py:func}`aryagraph.generators.dags.ml_pipeline` returns a 16-task DAG that trains, compares and ships a churn-prediction model. Each task has a planned `duration` in hours, a three-point estimate (`min`, `mode`, `max`), a stage (`kind`) and an owning `team`; each arc names the `artifact` handed downstream. The pipeline is an illustrative example, not a measured production workload.

```python
from collections import Counter

import aryagraph as ag

dag = ag.gen.ml_pipeline()
print(dag)
print("stages:", dict(Counter(kind for _, kind in dag.nodes.data("kind"))))
tasks, hours = Counter(), Counter()
for _, d in dag.nodes.data():
    tasks[d["team"]] += 1
    hours[d["team"]] += d["duration"]
for team in tasks:
    print(f"{team:<15} {tasks[team]:>2} tasks {hours[team]:>6g} h")
work = sum(hours.values())
print("total work:", work, "h")
```

```text
<DAG 'ML pipeline': 16 nodes, 19 edges>
stages: {'ingest': 2, 'validate': 1, 'transform': 3, 'features': 1, 'train': 4, 'evaluate': 2, 'deploy': 3}
data-eng         5 tasks    7.5 h
ml               7 tasks  35.25 h
responsible-ai   1 tasks      3 h
mlops            3 tasks   26.5 h
total work: 72.25 h
```

The ML team owns almost half of the work (35.25 of 72.25 h). The MLOps team's 26.5 h is dominated by one task, a 24 h canary release.

## Method

| Question | Measure | AryaGraph function |
|---|---|---|
| What sets the end-to-end time? | critical path and slack (CPM) | {py:func}`~aryagraph.algorithms.dag.critical_path` |
| How parallel is the work? | longest-path levels, width (largest antichain), work / span | {py:meth}`~aryagraph.core.dag.DAG.generations`, {py:func}`~aryagraph.algorithms.dag.dag_width`, {py:func}`~aryagraph.algorithms.dag.maximum_antichain` |
| How many workers? | simulated makespan and utilization for 1 to 6 workers | {py:func}`~aryagraph.sim.simulate_schedule` |
| How uncertain is the finish? | PERT Monte Carlo: P50, P80, P95 and criticality index | {py:func}`~aryagraph.sim.monte_carlo_schedule` |
| Does the policy matter? | makespan under four dispatching policies on the same sampled durations | {py:func}`~aryagraph.sim.simulate_schedule` with `policy=` and `duration=` |
| How many retries? | share of runs that finish when each attempt fails with probability *p* | `failure_rate` and `max_retries` |

## Results

### The critical path: 58.75 hours through 12 of 16 tasks

```python
cp = ag.alg.critical_path(dag)
print(f"critical path: {cp.length:g} h, {len(cp.critical)} of {len(dag)} tasks")
print(" → ".join(cp.path))
for n in dag.topological_order():
    if cp.slack[n] > 0:
        print(f"  {n:<26} {dag.nodes[n]['duration']:>5g} h, slack {cp.slack[n]:>5g} h")
```

```text
critical path: 58.75 h, 12 of 16 tasks
ingest events → validate schema → clean & deduplicate → join labels → train/test split → feature engineering → train gradient boosting → tune gradient boosting → evaluate models → fairness audit → deploy to staging → canary release
  ingest labels                  1 h, slack   4.5 h
  train logistic regression      1 h, slack    15 h
  train neural network          10 h, slack     6 h
  package model                1.5 h, slack   1.5 h
```

Two tasks account for 36 of the 58.75 critical hours: tuning the gradient-boosting model (12 h) and the canary release (24 h). The neural network is the longest training run (10 h) but not critical: training plus tuning the booster takes 16 h, so the network has 6 h of slack. The fairness audit (3 h) is critical because it takes longer than packaging the model (1.5 h), the other input to the staging deployment.

```{figure} ../_static/generated/cases_a/ml_pipeline.png
:alt: Layered diagram of the 16-task ML pipeline colored by team, from ingest events at the top to canary release at the bottom, with the critical path highlighted.
:width: 75%

The pipeline colored by team, with the critical path highlighted. Dimmed tasks have slack.
```

<iframe class="ag-embed" src="../_static/generated/cases_a/ml_pipeline.html" height="710" loading="lazy" title="Interactive ML pipeline DAG"></iframe>
<p class="ag-embed-note">Interactive: hover a task for its team, stage and estimates; drag to pan, scroll to zoom. <a href="../_static/generated/cases_a/ml_pipeline.html">Open full screen</a></p>

### How much can run in parallel

Longest-path levels group tasks into stages that could start together; the *width* is the largest set of tasks of which no two depend on each other.

```python
levels = dag.generations()
print("depth:", dag.depth(), "levels")
for i, level in enumerate(levels):
    print(f"  level {i:>2}: {', '.join(level)}")
print("width:", ag.alg.dag_width(dag), sorted(ag.alg.maximum_antichain(dag)))
print(f"average parallelism (work / critical path): {work / cp.length:.2f}")
reduced = dag.transitive_reduction()
print("arcs implied by other paths:", [(u, v) for u, v in dag.edges if not reduced.has_edge(u, v)])
```

```text
depth: 12 levels
  level  0: ingest events, ingest labels
  level  1: validate schema
  level  2: clean & deduplicate
  level  3: join labels
  level  4: train/test split
  level  5: feature engineering
  level  6: train logistic regression, train gradient boosting, train neural network
  level  7: tune gradient boosting
  level  8: evaluate models
  level  9: fairness audit, package model
  level 10: deploy to staging
  level 11: canary release
width: 3 ['train logistic regression', 'train neural network', 'tune gradient boosting']
average parallelism (work / critical path): 1.23
```

```text
arcs implied by other paths: [('train/test split', 'evaluate models')]
```

The pipeline is deep and narrow: 12 levels, at most three tasks that can run at the same time (the three model branches), and an average parallelism of 1.23, meaning that even with unlimited workers the work keeps just over one worker busy on average. The one arc implied by other paths, `train/test split → evaluate models`, adds no ordering constraint, but it is not redundant as data: it carries the holdout split that the evaluation needs.

```{figure} ../_static/generated/cases_a/ml_parallelism.png
:alt: Step chart of the number of running tasks over 58.75 hours. It starts at 2, drops to 1, peaks at 3 briefly around hour 13, stays at 2 until about hour 23, briefly returns to 2 around hour 31, and is 1 for the last 24 hours.
:width: 90%

Tasks running at each moment when every task starts as soon as its inputs are ready. Three tasks overlap for one hour; the last 24 hours are the canary release alone.
```

### Workers: two are enough

With a fixed plan, the simulation confirms what the structure suggests:

```python
POLICIES = ("critical_path", "fifo", "longest_first", "shortest_first")
print("workers  makespan  utilization  same for all four policies")
for w in range(1, 7):
    runs = {p: ag.sim.simulate_schedule(dag, workers=w, policy=p) for p in POLICIES}
    res = runs["critical_path"]
    same = len({r.makespan for r in runs.values()}) == 1
    print(f"{w:>7} {res.makespan:>9g} {res.utilization['overall']:>12.0%}  {same}")
```

```text
workers  makespan  utilization  same for all four policies
      1     72.25         100%  True
      2     58.75          61%  True
      3     58.75          41%  True
      4     58.75          31%  True
      5     58.75          25%  True
      6     58.75          20%  True
```

One worker runs everything back to back (72.25 h, the total work). Two workers already reach the 58.75 h critical-path bound, and every further worker only lowers utilization. In the plan, all four policies give the same makespan at every pool size.

```{figure} ../_static/generated/cases_a/ml_gantt.png
:alt: Gantt chart of the two-worker schedule colored by team. Worker 1 runs the critical chain end to end, including the 24-hour canary release; worker 2 runs ingest labels, the neural network, logistic regression and packaging.
:width: 100%

The two-worker schedule, colored by team. Worker 2 is idle for most of the run: the pipeline offers little parallel work.
```

### Uncertain durations: P80 is 71 hours

The Monte Carlo samples every task's duration from a PERT distribution fitted to its three-point estimate and recomputes the critical path, 5,000 times:

```python
mc = ag.sim.monte_carlo_schedule(dag, runs=5000, seed=11)
print(mc)
print(f"chance of finishing within the plan's {mc.cpm_length:g} h: {mc.probability(mc.cpm_length):.1%}")
for n, share in sorted(mc.criticality.items(), key=lambda kv: -kv[1]):
    if 0 < share < 1:
        print(f"  {n:<26} {share:6.1%}")
print("not critical in any run:", [n for n, share in mc.criticality.items() if share == 0])
```

```text
<MonteCarloResult: 5000 runs, makespan 64.74 ± 7.55; P50=64.41, P80=71.36, P95=77.78>
chance of finishing within the plan's 58.75 h: 22.9%
  fairness audit              99.3%
  train gradient boosting     96.1%
  tune gradient boosting      96.1%
  train neural network         3.9%
  package model                0.7%
not critical in any run: ['ingest labels', 'train logistic regression']
```

The planned 58.75 h is met in 22.9% of runs; the median is 64.41 h and P80 is 71.36 h. The estimates are right-skewed (the canary release ranges from 12 h to 48 h around a mode of 24 h), so a plan built from most likely durations is optimistic. Nine tasks were critical in every run. The booster branch was critical in 96.1% of runs and the neural network in the remaining 3.9%: its 6 h of slack is large but can be used up.

```{figure} ../_static/generated/cases_a/ml_montecarlo.png
:alt: Histogram of 5,000 simulated makespans from about 45 to 90 hours, with reference lines at the plan (58.75), P50 (64.41), P80 (71.36) and P95 (77.78).
:width: 90%

Distribution of the end-to-end time over 5,000 PERT runs with unlimited workers.
```

```{figure} ../_static/generated/cases_a/ml_criticality.png
:alt: Bar chart of the criticality index. Nine tasks at 100%, fairness audit 99.3%, train and tune gradient boosting 96.1%, train neural network 3.9%, package model 0.7%; ingest labels and logistic regression have no bar (0%).
:width: 90%

Criticality index over 5,000 runs: the share of runs in which each task had zero total float.
```

Does the worker count matter once durations vary? To compare pool sizes fairly, the next block draws 2,000 sets of durations once and runs every configuration on the same sets (common random numbers). This matters because the simulator draws a task's duration when the task starts, so two configurations that start tasks in a different order would otherwise see different random numbers, and small differences between them would be noise. The helper reproduces the simulator's Beta-PERT distribution.

```python
import numpy as np


def pert_draws(dag, rng):
    """One PERT duration per task (the same Beta-PERT the simulator uses)."""
    out = {}
    for n, d in dag.nodes.data():
        lo, mode, hi = d["min"], d["mode"], d["max"]
        a = 1 + 4 * (mode - lo) / (hi - lo)
        b = 1 + 4 * (hi - mode) / (hi - lo)
        out[n] = lo + (hi - lo) * rng.beta(a, b)
    return out


rng = np.random.default_rng(3)
samples = [pert_draws(dag, rng) for _ in range(2000)]


def spans(g, workers, policy="critical_path"):
    """Makespan of every sampled set of durations."""
    return np.array([ag.sim.simulate_schedule(g, workers=workers, policy=policy, duration=d).makespan
                     for d in samples])


for w in range(1, 7):
    s = spans(dag, w)
    print(f"{w} workers: mean {s.mean():.2f} h, P50 {np.percentile(s, 50):.1f} h, P80 {np.percentile(s, 80):.1f} h")
```

```text
1 workers: mean 79.53 h, P50 79.1 h, P80 86.5 h
2 workers: mean 64.72 h, P50 64.3 h, P80 71.1 h
3 workers: mean 64.70 h, P50 64.2 h, P80 71.1 h
4 workers: mean 64.70 h, P50 64.2 h, P80 71.1 h
5 workers: mean 64.70 h, P50 64.2 h, P80 71.1 h
6 workers: mean 64.70 h, P50 64.2 h, P80 71.1 h
```

With varying durations, the second worker still captures the whole gain: the mean falls from 79.53 h with one worker to 64.72 h with two, and a third worker lowers it by 0.02 h more. From three workers on the results are identical, because the pipeline has at most three tasks ready at once (its width).

```{figure} ../_static/generated/cases_a/ml_workers.png
:alt: Line chart of makespan against 1 to 6 workers for the plan and for the P50 and P80 of 2,000 PERT samples. All lines drop from 1 to 2 workers and are flat afterwards.
:width: 90%

Makespan against workers: the plan and the P50 and P80 over the same 2,000 PERT samples at every pool size.
```

### Dispatching policies: little effect on one pipeline, more on a shared pool

A policy decides which ready task a free worker takes: `critical_path` prefers the task with the longest remaining chain, `fifo` the one that became ready first, `longest_first` and `shortest_first` go by the task's own duration. On the same 2,000 samples:

```python
for w in (1, 2, 3):
    by_policy = {p: spans(dag, w, p) for p in POLICIES}
    base = by_policy["critical_path"]
    print(f"{w} workers: mean " + ", ".join(f"{p} {s.mean():.2f}" for p, s in by_policy.items()))
    print(f"   critical_path shorter than fifo in {np.mean(base < by_policy['fifo'] - 1e-9):.1%} of runs, "
          f"longer in {np.mean(base > by_policy['fifo'] + 1e-9):.1%}")
```

```text
1 workers: mean critical_path 79.53, fifo 79.53, longest_first 79.53, shortest_first 79.53
   critical_path shorter than fifo in 0.0% of runs, longer in 0.0%
2 workers: mean critical_path 64.72, fifo 64.75, longest_first 64.72, shortest_first 64.75
   critical_path shorter than fifo in 3.5% of runs, longer in 0.0%
3 workers: mean critical_path 64.70, fifo 64.70, longest_first 64.70, shortest_first 64.70
   critical_path shorter than fifo in 0.0% of runs, longer in 0.0%
```

With one worker every policy runs the same total work back to back, and with three workers every ready task starts at once, so the policies cannot differ. With two workers they differ only when the three training runs become ready together. `critical_path` and `longest_first` start the booster and the neural network; `fifo` and `shortest_first` start the logistic regression first (it is first in graph order and the shortest), which delays the network by an hour. The network has 6 h of slack, so that hour mattered in 3.5% of the samples, and `critical_path` was not slower than `fifo` in any sample. On average the difference is 0.03 h.

A single pipeline gives a policy few choices. Many teams run several pipelines on one pool, though, and then the choices multiply. The next block composes three copies of the pipeline (one per region) into one 48-task DAG and schedules it on 2 to 8 workers. The table shows each policy's makespan next to a simple lower bound, the larger of the critical path and the total work divided by the workers:

```python
shared = ag.DAG(name="three regional pipelines")
for region in ("us", "eu", "apac"):
    shared = shared.compose(dag.relabel(lambda n, r=region: f"{r}: {n}"))
shared_work = sum(d["duration"] for _, d in shared.nodes.data())
print(shared, "work:", shared_work, "h")
print("workers  bound  " + "  ".join(f"{p:>14}" for p in POLICIES))
for w in range(2, 9):
    bound = max(cp.length, shared_work / w)
    makespans = [ag.sim.simulate_schedule(shared, workers=w, policy=p).makespan for p in POLICIES]
    print(f"{w:>7} {bound:>6.2f}  " + "  ".join(f"{m:>14g}" for m in makespans))
```

```text
<DAG 'three regional pipelines': 48 nodes, 57 edges> work: 216.75 h
workers  bound   critical_path            fifo   longest_first  shortest_first
      2 108.38             120           121.5          125.25           111.5
      3  72.25           72.25           77.75           72.25           72.25
      4  58.75           68.75           66.75           68.75           68.75
      5  58.75           62.75           64.75           68.75           65.75
      6  58.75           58.75           59.75           58.75           58.75
      7  58.75           58.75           59.75           58.75           58.75
      8  58.75           58.75           58.75           58.75           58.75
```

On the shared pool the policies differ by up to 13.75 h, and no policy gives the shortest makespan at every pool size. With two workers `shortest_first` comes within 3.1 h of the bound, while `critical_path` is 11.6 h above it. `critical_path` alone is shortest with five workers and ties for shortest with three, six, seven and eight; `fifo` alone is shortest with four workers but longest with three, six and seven. These are the known limits of greedy list scheduling: each policy is a heuristic, and finding the optimal schedule on a limited pool is NP-hard in general. With six or seven workers every policy except `fifo` reaches the bound; with eight, all four do.

```{figure} ../_static/generated/cases_a/ml_policies.png
:alt: Line chart of hours above the lower bound for four policies on 2 to 8 workers. The lines cross: shortest_first is lowest at 2 workers, fifo is highest at 3, critical_path is lowest at 5, and all reach zero by 8 workers.
:width: 90%

Three pipelines on one pool: each policy's makespan minus the lower bound max(critical path, work / workers). Lower is better; the lines cross.
```

### Retries under failures

Tasks in real pipelines fail (a flaky data source, a preempted training job). In the simulator each attempt fails with probability *p*; the failure shows at the end of the attempt, and the task runs again while it has retries left. If it runs out, it and every task downstream fail. With the same *p* for all 16 tasks, a run finishes only if every task succeeds within `r + 1` attempts, so the expected share of finished runs is (1 − *p*<sup>r+1</sup>)<sup>16</sup>, a direct check on the simulation:

```python
print("   p  retries  finished  expected  P80 of finished runs")
for p in (0.05, 0.10, 0.20):
    for r in (0, 1, 2, 3):
        finished = []
        for seed in range(1000):
            res = ag.sim.simulate_schedule(dag, workers=2, failure_rate=p, max_retries=r, seed=seed)
            if sum(t.status == "done" for t in res.tasks) == len(dag):
                finished.append(res.makespan)
        expected = (1 - p ** (r + 1)) ** len(dag)
        print(f"{p:4.0%} {r:>8} {len(finished) / 1000:>9.1%} {expected:>9.1%} {np.percentile(finished, 80):>12.2f} h")
```

```text
   p  retries  finished  expected  P80 of finished runs
  5%        0     44.8%     44.0%        58.75 h
  5%        1     95.3%     96.1%        62.75 h
  5%        2     99.7%     99.8%        63.00 h
  5%        3    100.0%    100.0%        63.00 h
 10%        0     18.3%     18.5%        58.75 h
 10%        1     86.9%     85.1%        70.15 h
 10%        2     98.7%     98.4%        70.75 h
 10%        3     99.8%     99.8%        70.75 h
 20%        0      3.1%      2.8%        58.75 h
 20%        1     54.2%     52.0%        75.95 h
 20%        2     89.0%     87.9%        83.75 h
 20%        3     97.2%     97.5%        85.20 h
```

The simulated shares agree with the formula to within 2.2 percentage points, consistent with the sampling error of 1,000 runs (about 1.6 percentage points for shares near 50%). Without retries, even a 5% failure rate per attempt lets fewer than half of the runs finish, because 16 tasks all have to succeed. One retry lifts that to 95.3%, and two retries reach 99.7% at *p* = 5% and 98.7% at *p* = 10%. Retries cost time: the 80th percentile of finished runs rises from 58.75 h to 70.75 h at *p* = 10% with two retries, because a failed attempt is repeated in full: a single repeat of the canary release adds 24 h.

```{figure} ../_static/generated/cases_a/ml_retries.png
:alt: Line chart of the share of runs that finish every task against 0 to 3 retries, for failure rates of 5%, 10% and 20%. All three curves rise steeply from 0 to 1 retry; at 3 retries they are between 97% and 100%.
:width: 90%

Share of runs that finish every task against the retry budget, for three per-attempt failure rates (two workers, 1,000 runs per point).
```

## What the model shows

These points restate the results as guidance. They follow from this pipeline's estimates, not from measurements of a real system.

- **The end-to-end time is set by a single chain.** 12 of 16 tasks are critical, and two of them, tuning the booster (12 h) and the canary release (24 h), make up 36 of the 58.75 h. Shortening either shortens the pipeline hour for hour; speeding up the neural network does nothing until it becomes the longer branch.
- **Two workers are enough.** The pipeline's width is 3 and its average parallelism 1.23. A second worker saves 13.5 h in the plan; more workers change nothing in the plan and save 0.02 h on average when durations vary.
- **Budget for P80, not the plan.** The planned 58.75 h is met in 22.9% of runs; 71.36 h is met in 80%.
- **Pick the policy for the pool, not for the pipeline.** On one pipeline the four policies are within 0.03 h of each other on average; on a pool shared by three pipelines they differ by up to 13.75 h and the winner depends on the pool size, so it pays to simulate your own mix.
- **Allow two retries.** With 16 tasks, zero retries turns a small per-task failure rate into frequent run failures; two retries keep at least 98.7% of runs finishing for failure rates up to 10%.

## Limitations

- **Illustrative data.** Durations and estimates are an example, not measurements.
- **Independent durations and failures.** Real failures cluster (an outage fails several tasks at once) and durations correlate (a larger dataset slows every step); both would widen the distributions shown here.
- **Interchangeable workers.** Any worker can run any task. Real pipelines have GPU jobs, team ownership and approval steps that only certain workers can handle; `resources=` in {py:func}`~aryagraph.sim.simulate_schedule` can model such limits.
- **Full-length retries.** A failed attempt occupies its worker for the whole drawn duration and then starts from scratch. Checkpointing would make retries cheaper.
- **The canary release as a task.** The 24 h canary release occupies a worker in this model, although in practice it is mostly waiting for traffic. Modeling workers as a resource and giving the canary release zero demand, as the [construction case study](construction-schedule.md) does for its waits, would free that worker.

## Reproducibility

The code blocks on this page run top to bottom, in order, with AryaGraph {{ version }}. Every random process has a fixed seed: `seed=11` for the Monte Carlo, `np.random.default_rng(3)` for the shared samples, and seeds 0 to 999 for the failure runs. The figures come from the same computations in the site's asset script, [`website/scripts/assets/cases_a.py`](https://github.com/ruzbahani/AryaGraph/blob/main/website/scripts/assets/cases_a.py). To draw the pipeline and its schedule yourself:

```python
ag.draw(dag, node_color="team", highlight_path=cp.path, title="ML pipeline").save("pipeline.html")
two = ag.sim.simulate_schedule(dag, workers=2)
two.gantt(lanes="worker", color_by="team", time_label="hours").save("gantt.svg")
mc.plot(subtitle="5,000 PERT runs").save("montecarlo.svg")
```

Related pages: [Pipelines as DAGs](../tutorials/pipeline-dag.md), [DAG workflows](../user-guide/dags.md), [Simulation](../user-guide/simulation.md), [Scheduling a house build under uncertainty](construction-schedule.md).
