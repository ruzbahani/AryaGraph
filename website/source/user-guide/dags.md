# DAG workflows

Directed acyclic graphs model pipelines, builds, project plans and curricula: anything where some steps must finish before others can start. This page follows one workflow end to end, from building and validating a DAG to analyzing its structure, simulating its execution under uncertainty and drawing it.

## Building a DAG

{py:class}`~aryagraph.core.dag.DAG` is a directed graph that stays acyclic. Nodes and edges take attributes like any other AryaGraph graph; durations live on the nodes, because each node is a task.

```python
import aryagraph as ag

dag = ag.DAG(name="report")
dag.add_node("extract", duration=2)
dag.add_node("clean", duration=3)
dag.add_node("train", duration=5)
dag.add_node("report", duration=1)
dag.add_edges([("extract", "clean"), ("clean", "train"), ("clean", "report")])
print(dag)
print(dag.topological_order())
```

```text
<DAG 'report': 4 nodes, 3 edges>
['extract', 'clean', 'train', 'report']
```

Every edge that would close a cycle is rejected with a {py:class}`~aryagraph.core.exceptions.CycleError` that names the cycle, and the graph is left unchanged:

```python
try:
    dag.add_edge("report", "extract")
except ag.CycleError as err:
    print(err)
    print(err.cycle)

try:
    dag.add_edges([("train", "deploy"), ("deploy", "extract")])
except ag.CycleError as err:
    print(err.cycle)
print("deploy" in dag, dag.num_edges)
```

```text
edge 'report' → 'extract' would close the cycle 'report' → 'extract' → 'clean' → 'report'
['report', 'extract', 'clean', 'report']
['extract', 'clean', 'train', 'deploy', 'extract']
False 3
```

`add_edges` is atomic: it validates the whole batch in one linear-time pass and rolls back every edge and new node if any cycle appears, which is why `"deploy"` is absent after the failed call. Loading a large DAG in one `add_edges` call is therefore both safe and fast. Its message reads "these edges would create the cycle …" followed by the four arcs. For a batch, the reported cycle starts at the first of its nodes in graph order, so the same batch gives the same message in every run.

Graphs read from files or built as a {py:class}`~aryagraph.core.graph.DiGraph` can be checked and converted:

```python
flow = ag.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
print(ag.alg.is_dag(flow), ag.alg.find_cycle(flow))
flow.remove_edge("c", "a")
print(type(flow.to_dag()).__name__)
```

```text
False ['a', 'b', 'c', 'a']
DAG
```

Most DAG algorithms accept any acyclic `DiGraph` too; a cyclic input raises `CycleError` with the offending loop.

## Order and structure

The rest of this page uses the bundled machine-learning pipeline: 16 tasks from data ingestion to a canary release, each with a planned `duration` in hours, a three-point estimate (`min`, `mode`, `max`) and a `team`.

```python
ml = ag.gen.ml_pipeline()
print(ml)
print("sources:", ml.sources())
print("sinks:  ", ml.sinks())
print("depth:  ", ml.depth())
for level, tasks in enumerate(ml.generations()):
    print(level, tasks)
```

```text
<DAG 'ML pipeline': 16 nodes, 19 edges>
sources: ['ingest events', 'ingest labels']
sinks:   ['canary release']
depth:   12
0 ['ingest events', 'ingest labels']
1 ['validate schema']
2 ['clean & deduplicate']
3 ['join labels']
4 ['train/test split']
5 ['feature engineering']
6 ['train logistic regression', 'train gradient boosting', 'train neural network']
7 ['tune gradient boosting']
8 ['evaluate models']
9 ['fairness audit', 'package model']
10 ['deploy to staging']
11 ['canary release']
```

{py:meth}`~aryagraph.core.dag.DAG.levels` gives each node its longest-path level (sources at 0, every edge climbs at least one level) and {py:meth}`~aryagraph.core.dag.DAG.generations` groups nodes by level: the stages a pipeline could run in parallel if nothing else limited it. `depth()` is the number of levels.

| Question | Call |
|---|---|
| a valid execution order | `dag.topological_order()` (Kahn's algorithm, insertion order breaks ties) |
| an order that prefers some tasks | {py:func}`ag.alg.topological_sort(dag, key=...) <aryagraph.algorithms.dag.topological_sort>` (smallest key first among ready tasks) |
| every valid order | {py:func}`ag.alg.all_topological_sorts(dag) <aryagraph.algorithms.dag.all_topological_sorts>` (a lazy generator; there can be up to *n*! orders) |
| what a task depends on, what depends on it | `dag.ancestors(node)`, `dag.descendants(node)` |
| the nearest shared prerequisites | {py:func}`ag.alg.lowest_common_ancestors(dag, a, b) <aryagraph.algorithms.dag.lowest_common_ancestors>` |

```python
print(sorted(ml.descendants("feature engineering")))
print(ag.alg.lowest_common_ancestors(ml, "fairness audit", "package model"))
small = ag.DAG([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("a", "e")])
print(sum(1 for _ in ag.alg.all_topological_sorts(small)))
```

```text
['canary release', 'deploy to staging', 'evaluate models', 'fairness audit', 'package model', 'train gradient boosting', 'train logistic regression', 'train neural network', 'tune gradient boosting']
{'evaluate models'}
8
```

## Longest path and critical path

Two different questions hide behind "longest path":

* {py:meth}`dag.longest_path() <aryagraph.core.dag.DAG.longest_path>` and {py:func}`~aryagraph.algorithms.dag.dag_longest_path` find the path with the most edges, or the largest sum of an **edge** attribute when you pass `weight=`.
* {py:func}`~aryagraph.algorithms.dag.critical_path` runs the critical-path method (CPM) with durations on the **nodes**, which is what task graphs usually carry.

The distinction matters: `weight="duration"` looks for an edge attribute, finds none on this pipeline and counts every edge as 1. Use `critical_path` for task durations.

```python
cp = ag.alg.critical_path(ml)
print(cp)
print(cp.length, len(cp.critical))
```

```text
CriticalPath(length=58.75, path=['ingest events' → 'validate schema' → 'clean & deduplicate' → 'join labels' → 'train/test split' → 'feature engineering' → 'train gradient boosting' → 'tune gradient boosting' → …], critical=12 of 16 activities)
58.75 12
```

A forward pass computes each task's earliest start and finish; a backward pass from the makespan computes the latest start and finish; the difference is the *slack*, how long a task can slip without delaying the end. Tasks with zero slack are *critical*.

| Field | Content |
|---|---|
| `path` | one critical chain from a source to a sink |
| `length` (alias `makespan`) | the earliest time everything can be finished |
| `earliest_start`, `earliest_finish` | per-node {py:class}`~aryagraph.core.results.NodeMap` from the forward pass |
| `latest_start`, `latest_finish` | per-node values from the backward pass |
| `slack` | latest start minus earliest start |
| `critical` | the set of every zero-slack task (possibly several chains) |

```python
print(f"{'task':26} {'dur':>5} {'ES':>6} {'EF':>6} {'LS':>6} {'LF':>6} {'slack':>6}")
for task in ml:
    print(f"{task:26} {ml.nodes[task]['duration']:5g} {cp.earliest_start[task]:6g} "
          f"{cp.earliest_finish[task]:6g} {cp.latest_start[task]:6g} {cp.latest_finish[task]:6g} {cp.slack[task]:6g}")
```

```text
task                         dur     ES     EF     LS     LF  slack
ingest events                  2      0      2      0      2      0
ingest labels                  1      0      1    4.5    5.5    4.5
validate schema              0.5      2    2.5      2    2.5      0
clean & deduplicate            3    2.5    5.5    2.5    5.5      0
join labels                    1    5.5    6.5    5.5    6.5      0
train/test split            0.25    6.5   6.75    6.5   6.75      0
feature engineering            6   6.75  12.75   6.75  12.75      0
train logistic regression      1  12.75  13.75  27.75  28.75     15
train gradient boosting        4  12.75  16.75  12.75  16.75      0
tune gradient boosting        12  16.75  28.75  16.75  28.75      0
train neural network          10  12.75  22.75  18.75  28.75      6
evaluate models                2  28.75  30.75  28.75  30.75      0
fairness audit                 3  30.75  33.75  30.75  33.75      0
package model                1.5  30.75  32.25  32.25  33.75    1.5
deploy to staging              1  33.75  34.75  33.75  34.75      0
canary release                24  34.75  58.75  34.75  58.75      0
```

The pipeline needs 58.75 hours end to end. Training the logistic regression can slip by 15 hours and the neural network by 6 without changing that, while every hour lost on the gradient-boosting branch or the 24-hour canary release moves the end date.

```{figure} ../_static/generated/guide_sim_data/cpm_gantt.png
:alt: Gantt chart of the CPM schedule with one row per task; critical tasks in blue form a continuous chain from 0 to 58.75 hours.
:width: 100%

The CPM schedule as a Gantt chart, built with {py:func}`aryagraph.charts.gantt` from a list of `{"node", "worker", "start", "end"}` records holding each task's earliest start and finish, with `critical=cp.critical`.
```

```{figure} ../_static/generated/guide_sim_data/pipeline_cpm.png
:alt: Layered drawing of the machine-learning pipeline with the twelve critical tasks highlighted and the four others dimmed.
:width: 60%

`ag.draw(ml, highlight_path=cp.path)` brings the critical chain forward and dims the four tasks with slack.
```

```{note}
`critical_path` reads durations from node attributes only. Arc attributes such as the `lag` on {py:func}`~aryagraph.generators.dags.project_plan` (72 hours of concrete curing after the foundation, 24 hours of drying after drywall) are not included. To account for a mandatory wait, model it as a task of its own:
```

```python
plan = ag.gen.project_plan()
waits = plan.copy()
for u, v, d in list(plan.edges.data()):
    if d["lag"]:
        wait = f"wait after {u}"
        waits.remove_edge(u, v)
        waits.add_node(wait, duration=d["lag"])
        waits.add_edges([(u, wait), (wait, v)])
print(ag.alg.critical_path(plan).length, ag.alg.critical_path(waits).length)
```

```text
572.0 668.0
```

Both waits lie on the critical path of the house-construction plan, so the plan grows from 572 to 668 working hours.

## Transitive reduction and closure

An edge `u → v` is *redundant* when another path already leads from *u* to *v*: it adds no constraint. {py:func}`~aryagraph.algorithms.dag.transitive_reduction` removes every such edge and keeps all nodes and attributes; {py:func}`~aryagraph.algorithms.dag.transitive_closure` goes the other way and adds an edge for every path.

```python
build = ag.gen.software_build()
reduced = ag.alg.transitive_reduction(build)
print(build.num_edges, reduced.num_edges)
print([(u, v) for u, v in build.edges if not reduced.has_edge(u, v)])
print(ag.alg.transitive_closure(build).num_edges)
etl = ag.gen.data_warehouse_etl()
print(etl.num_edges - ag.alg.transitive_reduction(etl).num_edges)
```

```text
33 32
[('compile core', 'link libcore')]
110
10
```

In the software build, `compile core → link libcore` is implied by `compile core → compile net → link libcore`. The data-warehouse ETL has 10 redundant edges: 3 fact tables list a staging table that one of their dimensions already depends on, and 7 aggregate or export edges from fact tables are implied by the quality gate or by another aggregate. Critical paths and schedules are unchanged by the reduction, because it keeps every precedence; drawings of the reduced graph have fewer edges to route.

## Width and antichains

The *width* of a DAG is the size of its largest *antichain*: a set of tasks no two of which depend on each other, directly or indirectly. It bounds how many tasks can ever run at the same time, so it is the number of workers beyond which extra workers cannot help. By Dilworth's theorem it also equals the fewest chains that cover every task.

```python
print(ag.alg.dag_width(build))
print(sorted(ag.alg.maximum_antichain(build)))
for chain in ag.alg.minimum_chain_partition(build):
    print(chain)
```

```text
8
['build docs', 'bundle web ui', 'compile api', 'compile cli', 'compile net', 'compile storage', 'lint', 'type check']
['fetch dependencies', 'generate protobuf stubs', 'compile net', 'link libcore', 'unit tests']
['compile utils', 'compile core', 'compile storage', 'link server', 'sign artifacts']
['compile api', 'integration tests']
['compile cli', 'link cli', 'publish release']
['bundle web ui', 'container image']
['lint']
['type check']
['build docs']
```

The build has 20 targets over 9 levels, but up to 8 of them are mutually independent. The ML pipeline, which is mostly a chain, has a width of 3, for example the logistic regression, the neural network and the gradient-boosting tuning.

## Simulating execution

CPM assumes unlimited workers and exact durations. {py:func}`~aryagraph.sim.simulate_schedule` runs the DAG as a discrete-event simulation instead: a pool of workers takes ready tasks (all predecessors done), a *policy* chooses which ready task goes first, and durations, failures and resources can all be random or limited.

| Parameter | Meaning |
|---|---|
| `duration` | node attribute name (default `"duration"`; a missing value counts as 1), a number, a `{node: value}` mapping or `f(node, attrs)` |
| `workers` | pool size; `None` for unlimited (every ready task starts at once) |
| `policy` | `"critical_path"` (longest remaining path first, the default), `"fifo"`, `"longest_first"`, `"shortest_first"`, `"random"`, or a key function (smallest first) |
| `failure_rate`, `max_retries` | per-attempt failure probability (number, attribute name or mapping) and extra attempts allowed |
| `jitter` | random durations: `("uniform", 0.2)`, `("triangular", "min", "mode", "max")`, `("pert", "min", "mode", "max")` or `f(node, nominal, rng)` |
| `resources` | `{name: capacity}`; each task needs the amount in its node attribute of that name |
| `seed` | makes every random draw reproducible |

### Workers and policies

With unlimited workers the makespan equals the critical-path length; with one worker it is the sum of all durations. The house-construction plan shows the range in between:

```python
for workers in (None, 1, 2, 3):
    run = ag.sim.simulate_schedule(plan, workers=workers)
    print(f"workers={workers!s:4}  makespan={run.makespan:5g} h  utilization={run.utilization['overall']:.1%}")
```

```text
workers=None  makespan=  572 h  utilization=27.0%
workers=1     makespan=  772 h  utilization=100.0%
workers=2     makespan=  628 h  utilization=61.5%
workers=3     makespan=  604 h  utilization=42.6%
```

The dispatcher is work-conserving: a free worker takes a ready task whenever one exists, and the policy only decides which. That choice matters when many tasks are ready at once:

```python
courses = ag.gen.course_prerequisites()
for name, g in [("etl", etl), ("courses", courses)]:
    for workers in (2, 3):
        spans = {policy: ag.sim.simulate_schedule(g, workers=workers, policy=policy, seed=0).makespan
                 for policy in ("critical_path", "fifo", "longest_first", "shortest_first", "random")}
        print(name, workers, {p: round(s, 2) for p, s in spans.items()})
```

```text
etl 2 {'critical_path': 6.4, 'fifo': 6.5, 'longest_first': 6.55, 'shortest_first': 7.0, 'random': 6.6}
etl 3 {'critical_path': 5.0, 'fifo': 5.35, 'longest_first': 5.6, 'shortest_first': 5.75, 'random': 5.5}
courses 2 {'critical_path': 1755.0, 'fifo': 1755.0, 'longest_first': 1845.0, 'shortest_first': 1710.0, 'random': 1755.0}
courses 3 {'critical_path': 1170.0, 'fifo': 1215.0, 'longest_first': 1305.0, 'shortest_first': 1260.0, 'random': 1305.0}
```

Critical-path-first (the HLFET list-scheduling heuristic) gives the shortest makespan in three of these four cases and reaches the CPM bound of 5 hours on the ETL with 3 workers, but on the course plan with 2 workers shortest-first does better. List scheduling is a heuristic, so compare policies on your own graphs.

### Reading a schedule

The result is a {py:class}`~aryagraph.sim.scheduling.ScheduleResult`: a {py:class}`~aryagraph.sim.base.SimulationResult` whose frames hold each task's state (`pending`, `ready`, `running`, `done`, `failed`) at every event time, plus the task log.

```python
run = ag.sim.simulate_schedule(ml, workers=2, failure_rate=0.15, max_retries=1, seed=4)
print(run)
print(run.tasks[:3])
print([r for r in run.tasks if r.status == "failed" or r.attempt > 1])
print(run.critical_path == cp.path, round(run.idle_time, 2), run.utilization)
```

```text
<ScheduleResult: 16 tasks, workers=2, makespan=61.75, utilization=60.9%, 16 done, 0 failed>
[TaskRun(node='ingest events', worker=0, start=0.0, end=2.0, attempt=1, status='done'), TaskRun(node='ingest labels', worker=1, start=0.0, end=1.0, attempt=1, status='done'), TaskRun(node='validate schema', worker=0, start=2.0, end=2.5, attempt=1, status='done')]
[TaskRun(node='clean & deduplicate', worker=0, start=2.5, end=5.5, attempt=1, status='failed'), TaskRun(node='clean & deduplicate', worker=0, start=5.5, end=8.5, attempt=2, status='done')]
True 48.25 {0: 1.0, 1: 0.21862348178137653, 'overall': 0.6093117408906883}
```

| Member | Content |
|---|---|
| `tasks` | every attempt as a {py:class}`~aryagraph.sim.scheduling.TaskRun` (`node`, `worker`, `start`, `end`, `attempt`, `status`), in dispatch order |
| `makespan` | when the last task finished or failed |
| `utilization` | busy share of the makespan per worker id, plus `"overall"` |
| `idle_time` | total idle worker time |
| `critical_path` | the chain of attempts that determined the makespan, traced back from the last task to finish |
| `blocked` | tasks that did not run because a predecessor failed for good |
| `runs_of(node)` | every attempt of one task |

In this run, `clean & deduplicate` failed once and succeeded on its retry, which cost 3 hours: the makespan is 61.75 hours instead of 58.75. A failed attempt holds its worker for its whole duration, and the failure only shows at its end.

```{figure} ../_static/generated/guide_sim_data/gantt.png
:alt: Gantt chart with two worker rows; worker 1 runs the critical chain including one failed attempt shown in red, worker 2 runs the four side tasks.
:width: 100%

`run.gantt(lanes="worker")` for the run above. With `lanes="task"` (the default) there is one row per task instead; `color_by="team"` colors by a node attribute.
```

When a task runs out of retries, it is marked failed and so is everything downstream, at the same instant:

```python
doomed = ag.sim.simulate_schedule(ml, workers=2, failure_rate=0.1, max_retries=0, seed=0)
print(doomed)
print(doomed.blocked[:4], len(doomed.blocked))
```

```text
<ScheduleResult: 16 tasks, workers=2, makespan=2.5, utilization=70.0%, 2 done, 14 failed>
['clean & deduplicate', 'join labels', 'train/test split', 'feature engineering'] 13
```

Here `validate schema` failed on its only attempt at t = 2.5. Its 13 descendants are blocked, so the run ends with 2 tasks done and 14 failed.

### Random durations and resources

`jitter` draws one duration per attempt. `("pert", "min", "mode", "max")` uses the three-point estimates stored on the nodes (a Beta-PERT distribution), `("triangular", ...)` a triangular one, `("uniform", 0.2)` scales the nominal duration by a factor between 0.8 and 1.2, and a function gives full control:

```python
pert = ag.sim.simulate_schedule(ml, workers=2, jitter=("pert", "min", "mode", "max"), seed=1)
lognormal = ag.sim.simulate_schedule(
    ml, workers=2, jitter=lambda node, nominal, rng: nominal * rng.lognormal(0.0, 0.3), seed=1
)
print(round(pert.makespan, 2), round(lognormal.makespan, 2))
```

```text
69.14 63.72
```

Resources model shared capacity other than workers, such as GPUs, licenses or a database that tolerates one migration at a time. Each task needs the amount given by its node attribute of the resource's name (0 when absent):

```python
gpu = ag.gen.ml_pipeline()
for task, attrs in gpu.nodes.data():
    if attrs["kind"] == "train":
        attrs["gpu"] = 1
limited = ag.sim.simulate_schedule(gpu, workers=4, resources={"gpu": 1})
free = ag.sim.simulate_schedule(gpu, workers=4)
print(free.makespan, limited.makespan)
print([(r.node, r.start, r.end) for r in limited.tasks if gpu.nodes[r.node]["kind"] == "train"])
```

```text
58.75 69.75
[('train gradient boosting', 12.75, 16.75), ('tune gradient boosting', 16.75, 28.75), ('train neural network', 28.75, 38.75), ('train logistic regression', 38.75, 39.75)]
```

With a single GPU the four training tasks run one after another, and the pipeline takes 11 hours longer even though 4 workers are available. When the preferred task does not fit the free resources, a lower-priority task that fits may start first.

## Monte Carlo schedules

A single random run is one possible future. {py:func}`~aryagraph.sim.monte_carlo_schedule` repeats the simulation and summarizes the distribution of the makespan. By default (`jitter="auto"`) it uses PERT durations whenever every task has `min`, `mode` and `max` attributes. With unlimited workers, no resources and no failures, the runs reduce to CPM on random durations and are vectorized; otherwise each run is a full `simulate_schedule`.

```python
mc = ag.sim.monte_carlo_schedule(ml, runs=2000, seed=1)
print(mc)
print(mc.cpm_length, round(mc.probability(mc.cpm_length), 3), round(mc.probability(70), 3))
print({task: round(c, 3) for task, c in mc.criticality.items() if c < 1})
```

```text
<MonteCarloResult: 2000 runs, makespan 64.76 ± 7.64; P50=64.5, P80=71.31, P95=78.04>
58.75 0.232 0.757
{'ingest labels': 0.0, 'train logistic regression': 0.0, 'train gradient boosting': 0.96, 'tune gradient boosting': 0.96, 'train neural network': 0.04, 'fairness audit': 0.991, 'package model': 0.009}
```

The deterministic plan of 58.75 hours is met in only 23% of the simulated futures. Every task's estimate is skewed: the pessimistic value lies further above the most likely one than the optimistic value lies below it (the canary release, for example, takes 12 to 48 hours around a most likely 24), so random durations run long more often than short. A commitment at the 80th percentile, about 71 hours, is met in 80% of the runs. The *criticality index* is the share of runs in which a task was critical: the gradient-boosting branch is critical in 96% of runs and the neural network in the remaining 4%, while the logistic regression and the label ingestion were critical in none.

```{figure} ../_static/generated/guide_sim_data/montecarlo.png
:alt: Histogram of 2,000 simulated pipeline durations with vertical markers at the 50th, 80th and 95th percentiles.
:width: 100%

`mc.plot()`: the makespan distribution with P50, P80 and P95 markers.
```

| `MonteCarloResult` member | Content |
|---|---|
| `makespans` | every run's makespan |
| `mean`, `std`, `stderr` | sample mean, standard deviation and standard error of the mean |
| `percentiles`, `percentile(q)` | P50, P80 and P95 (configurable with `percentiles=`), or any percentile on demand |
| `probability(deadline)` | share of runs finishing by the deadline |
| `criticality` | share of runs in which each task was critical |
| `cpm_length` | critical-path length of the nominal durations |

Pass `workers=`, `failure_rate=` or `resources=` to include those effects; each run then simulates the dispatcher, and criticality counts membership in each run's realized critical chain.

## Drawing DAGs

{py:func}`ag.draw <aryagraph.render.draw>` picks the hierarchical (Sugiyama) layout for any DAG up to 2,000 nodes: layers from network-simplex ranking, crossing reduction, Brandes–Köpf coordinates and routed long edges. Labels sit inside boxes sized to their text.

```python
fig = ag.draw(
    courses,
    layout_options={"orientation": "LR"},   # also "TB" (default), "BT", "RL"
    edge_style="flow",                      # or "orthogonal", "curved", "straight"
    node_color="department",
    title="Course prerequisites, left to right",
)
fig.save("courses_lr.svg")
print(fig)
```

```text
<Figure 638×624: 22 nodes, 31 edges, hierarchical layout, theme 'light'>
```

```{figure} ../_static/generated/guide_sim_data/courses_lr.png
:alt: The 22 courses of the degree plan in five columns, prerequisites on the left, joined by smooth flow edges, with boxes colored by department (CS, MATH, STAT).
:width: 90%

The course-prerequisite DAG laid out left to right with `edge_style="flow"`, colored by the `department` attribute. A left-to-right layout suits DAGs with few levels: drawn the same way, the 12-level ML pipeline comes out about five times wider than tall (2054×392), which is why the critical-path figure above draws it top to bottom.
```

Useful options for DAG drawings:

* `layout_options=` passes engine options to {py:func}`~aryagraph.layout.hierarchical`: `orientation`, `rank_sep` and `node_sep` (spacing in pixels), `layering` (`"network_simplex"`, `"longest_path"` or `"coffman_graham"` with `max_width`), and `ranks={node: layer}` to pin layers yourself.
* `highlight_path=cp.path` brings the critical chain forward and dims the rest.
* `edge_style="flow"` draws S-curves along the hierarchy; `"orthogonal"` routes right-angle edges with rounded corners.
* Draw `ag.alg.transitive_reduction(dag)` instead of the DAG when redundant edges clutter the picture.

For a complete worked example, see the [pipeline tutorial](../tutorials/pipeline-dag.md) and the [construction schedule case study](../case-studies/construction-schedule.md).
