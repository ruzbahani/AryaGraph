# Pipelines as DAGs

In this tutorial you model a data and machine-learning pipeline as a directed acyclic graph (DAG): tasks are nodes, and an arrow from one task to another means the second needs the output of the first. You validate the model, find its critical path, simulate it on 2 and 4 workers, draw a Gantt chart and test a what-if scenario.

**Goal:** know how long a pipeline takes, why, and which changes would make it faster.

**Prerequisites:**

- [Your first network](first-network.md), or familiarity with nodes, edges and attributes in AryaGraph.
- AryaGraph installed. PNG export of the charts needs `cairosvg` or a Chrome-family browser; this page saves SVG, which needs nothing extra.

## Step 1: Build a small pipeline by hand

Start with a nightly reporting job of eight tasks. Each task gets a `duration` attribute in hours; the numbers are illustrative estimates. Then add the dependencies:

```python
import aryagraph as ag

job = ag.DAG(name="Nightly sales report")
job.add_node("download sales", duration=0.5)
job.add_node("download inventory", duration=0.25)
job.add_node("clean sales", duration=1.0)
job.add_node("join tables", duration=0.5)
job.add_node("compute KPIs", duration=0.75)
job.add_node("forecast demand", duration=2.0)
job.add_node("render dashboard", duration=0.25)
job.add_node("email summary", duration=0.1)
job.add_edges([
    ("download sales", "clean sales"),
    ("clean sales", "join tables"),
    ("download inventory", "join tables"),
    ("join tables", "compute KPIs"),
    ("join tables", "forecast demand"),
    ("compute KPIs", "render dashboard"),
    ("forecast demand", "render dashboard"),
    ("join tables", "render dashboard"),
    ("render dashboard", "email summary"),
])
print(job)
print(job.topological_order())
```

```text
<DAG 'Nightly sales report': 8 nodes, 9 edges>
['download sales', 'download inventory', 'clean sales', 'join tables', 'compute KPIs', 'forecast demand', 'render dashboard', 'email summary']
```

A {py:class}`~aryagraph.core.dag.DAG` is a directed graph that stays acyclic. A *topological order* lists the tasks so that every arrow points forward: it is an order in which a single worker could run them. Ties follow insertion order, so the result is the same on every run.

## Step 2: Let the DAG reject cycles

A cycle in a pipeline means a task waits, directly or indirectly, for its own output, so the pipeline could not finish. A `DAG` refuses any edge that would close a cycle and names the cycle in the error:

```python
try:
    job.add_edge("email summary", "download sales")
except ag.CycleError as err:
    print(err)
    print(err.cycle)
print(job.num_edges)
```

```text
edge 'email summary' → 'download sales' would close the cycle 'email summary' → 'download sales' → 'clean sales' → 'join tables' → 'render dashboard' → 'email summary'
['email summary', 'download sales', 'clean sales', 'join tables', 'render dashboard', 'email summary']
9
```

The rejected edge was not added: the graph still has 9 edges. `add_edges` is atomic too: if any edge in a batch would create a cycle, none of the batch is added.

Dependencies often arrive from elsewhere (a config file, a scheduler export) as a plain {py:class}`~aryagraph.core.graph.DiGraph`. Check those before you trust them:

```python
imported = ag.DiGraph([("extract", "transform"), ("transform", "load"), ("load", "extract")])
print(ag.alg.is_dag(imported))
print(ag.alg.find_cycle(imported))
```

```text
False
['extract', 'transform', 'load', 'extract']
```

`ag.DAG(imported)` would raise the same `CycleError`, so converting is also a validation step.

## Step 3: Validate the model

An acyclic graph can still be a wrong model. Three quick checks catch common mistakes: the entry and exit tasks are the ones you expect, every task has a duration, and no dependency is redundant.

```python
print(job.sources(), job.sinks())
print([task for task, hours in job.nodes.data("duration") if hours is None])

reduced = job.transitive_reduction()
print(sorted(set(job.edges) - set(reduced.edges)))
```

```text
['download sales', 'download inventory'] ['email summary']
[]
[('join tables', 'render dashboard')]
```

The two downloads are the only entry points and `email summary` is the only exit, as intended, and no task lacks a duration. {py:func}`~aryagraph.algorithms.dag.transitive_reduction` keeps only the edges that no other path implies. It drops `join tables → render dashboard`: the dashboard already waits for `compute KPIs` and `forecast demand`, which both wait for `join tables`. A redundant edge does not change any result, but it clutters drawings and can hide a mistake in the dependency list, so remove it:

```python
job.remove_edge("join tables", "render dashboard")
print(job.num_edges)
```

```text
8
```

## Step 4: Find the critical path

The *critical path* is the chain of dependent tasks with the largest total duration. No schedule can finish sooner, however many workers you add. {py:meth}`DAG.critical_path <aryagraph.core.dag.DAG.critical_path>` runs the critical-path method (CPM) on the `duration` attribute:

```python
cp_job = job.critical_path()
print(cp_job.length)
print(" > ".join(cp_job.path))
print({task: hours for task, hours in cp_job.slack.items() if hours > 0})
```

```text
4.35
download sales > clean sales > join tables > forecast demand > render dashboard > email summary
{'download inventory': 1.25, 'compute KPIs': 1.25}
```

The report takes 4.35 h, set by the sales branch and the 2-hour forecast. *Slack* is how long a task can slip without delaying the end: the inventory download and the KPI computation each have 1.25 h. Tasks with zero slack are *critical*; the result also lists them in `cp_job.critical`.

```python
fig = ag.draw(job, highlight_path=cp_job.path, title="Nightly sales report",
              subtitle=f"Critical path: {cp_job.length:g} h")
fig.save("nightly-report.svg")
```

```{figure} ../_static/generated/tutorials_a/pipeline_job.png
:alt: Layered drawing of the eight-task nightly report, top to bottom, with the six critical tasks highlighted and the inventory download and KPI computation dimmed.
:width: 420px

The nightly report job after removing the redundant edge. The critical path is highlighted; the two tasks with slack are dimmed.
```

## Step 5: Load the ML pipeline

AryaGraph ships several example DAGs with realistic attributes. {py:func}`ag.gen.ml_pipeline <aryagraph.generators.dags.ml_pipeline>` is a 16-task pipeline that trains, compares and ships a churn-prediction model. Every task has a planned `duration` in hours, a three-point estimate (`min`, `mode`, `max`) and the `team` that owns it; every edge names the `artifact` passed downstream:

```python
dag = ag.gen.ml_pipeline()
print(dag)
print(dag.nodes["tune gradient boosting"])
print(dag.edges["feature engineering", "train neural network"])
```

```text
<DAG 'ML pipeline': 16 nodes, 19 edges>
{'duration': 12.0, 'min': 8.0, 'mode': 12.0, 'max': 24.0, 'kind': 'train', 'team': 'ml'}
{'artifact': 'feature matrix'}
```

`generations()` groups tasks into stages: each stage contains tasks whose longest chain of predecessors has the same length, so no task in a stage depends on another task of the same stage.

```python
for stage, tasks in enumerate(dag.generations()):
    print(stage, tasks)
print("depth:", dag.depth(), "width:", ag.alg.dag_width(dag))
```

```text
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
depth: 12 width: 3
```

The pipeline is mostly a chain: depth 12 means its longest chain of dependent tasks has 12 tasks. Its *width*, from {py:func}`~aryagraph.algorithms.dag.dag_width`, is 3: the largest set of tasks with no path between any two of them has three members (for example the three model trainings). Two tasks that run at the same time cannot depend on each other, so no schedule of this pipeline can keep more than three workers busy at once.

## Step 6: Critical path of the ML pipeline

```python
cp = ag.alg.critical_path(dag)
print(f"{cp.length:g} h")
print(" > ".join(cp.path))
for task, hours in cp.slack.items():
    if hours > 0:
        print(f"{task:<27}{hours:>5g} h of slack")
```

```text
58.75 h
ingest events > validate schema > clean & deduplicate > join labels > train/test split > feature engineering > train gradient boosting > tune gradient boosting > evaluate models > fairness audit > deploy to staging > canary release
ingest labels                4.5 h of slack
train logistic regression     15 h of slack
train neural network           6 h of slack
package model                1.5 h of slack
```

Twelve of the 16 tasks are critical, and the 24-hour canary release alone is about 41% of the 58.75 h. The other four tasks have slack: the logistic regression could take 15 h longer, and the neural network 6 h longer, without delaying the release.

Draw the pipeline with the critical path highlighted. For a DAG, `ag.draw` uses a layered layout by default, with every arrow pointing down:

```python
fig = ag.draw(
    dag,
    node_color="team",
    highlight_path=cp.path,
    title="ML pipeline",
    subtitle=f"Critical path highlighted · {cp.length:g} h end to end",
)
fig.save("ml-pipeline.html")
```

```{figure} ../_static/generated/tutorials_a/pipeline_critical.png
:alt: Layered drawing of the 16-task ML pipeline, top to bottom, colored by team; the 12 critical tasks are highlighted and the logistic regression, neural network, label ingestion and packaging tasks are dimmed.
:width: 560px

The ML pipeline with its critical path highlighted. Tasks off the path are dimmed; the legend shows the owning team.
```

<iframe class="ag-embed" src="../_static/generated/tutorials_a/pipeline_critical.html" height="1020" loading="lazy" title="Interactive drawing of the ML pipeline with its critical path"></iframe>
<p class="ag-embed-note">Interactive: hover a task to see its duration, estimates and team, or search for a task by name. <a href="../_static/generated/tutorials_a/pipeline_critical.html">Open full screen</a></p>

## Step 7: Simulate on 2 and 4 workers

The critical path assumes unlimited workers. {py:func}`ag.sim.simulate_schedule <aryagraph.sim.simulate_schedule>` runs the pipeline on a fixed pool: whenever a worker is free, it starts a ready task, and by default it picks the ready task with the longest remaining chain first (`policy="critical_path"`):

```python
total = sum(hours for _, hours in dag.nodes.data("duration"))
print(f"total work {total:g} h, critical path {cp.length:g} h")
for workers in (1, 2, 4):
    run = ag.sim.simulate_schedule(dag, workers=workers)
    busy = run.utilization["overall"]
    print(f"{workers} worker(s): makespan {run.makespan:g} h, utilization {busy:.0%}")
```

```text
total work 72.25 h, critical path 58.75 h
1 worker(s): makespan 72.25 h, utilization 100%
2 worker(s): makespan 58.75 h, utilization 61%
4 worker(s): makespan 58.75 h, utilization 31%
```

One worker needs 72.25 h, the sum of all durations. Two workers bring the *makespan* (the time until the last task finishes) down to 58.75 h, the critical-path length. With fixed durations the critical path is a lower bound for any number of workers, so four workers change only the utilization, which halves from 61% to 31%.

Two bounds predict this before any simulation: the makespan is at least the critical-path length (58.75 h) and at least the total work divided by the number of workers (72.25 / 2 ≈ 36.1 h for two workers). Here the critical path is the larger bound, so the pipeline is limited by its dependencies, not by its workers.

## Step 8: Draw a Gantt chart

A {py:class}`~aryagraph.sim.scheduling.ScheduleResult` records every task run (`run.tasks`) and draws itself as a Gantt chart. With `lanes="worker"` there is one row per worker:

```python
two = ag.sim.simulate_schedule(dag, workers=2)
two.gantt(
    lanes="worker",
    title="ML pipeline on 2 workers",
    subtitle=f"Makespan {two.makespan:g} h · utilization {two.utilization['overall']:.0%}",
    time_label="hours",
).save("gantt-2-workers.svg")

four = ag.sim.simulate_schedule(dag, workers=4)
four.gantt(
    lanes="worker",
    title="ML pipeline on 4 workers",
    subtitle=f"Makespan {four.makespan:g} h · utilization {four.utilization['overall']:.0%}",
    time_label="hours",
).save("gantt-4-workers.svg")
print(sorted({r.worker for r in four.tasks}))
```

```text
[0, 1, 2]
```

```{figure} ../_static/generated/tutorials_a/pipeline_gantt_2.png
:alt: Gantt chart with two worker rows over 58.75 hours; worker 1 runs the critical chain ending in a 24-hour canary release, worker 2 runs a few short tasks and the neural-network training.
:width: 100%

Two workers. Blue bars are critical-path tasks; gray bars are the others. Bars wide enough for their name are labeled.
```

```{figure} ../_static/generated/tutorials_a/pipeline_gantt_4.png
:alt: Gantt chart for four workers with only three rows; the schedule and the 58.75-hour makespan are the same as with two workers.
:width: 100%

Four workers. The makespan does not change, and in this run the fourth worker receives no task, so it has no row.
```

With two workers, the first worker runs the whole critical chain back to back while the second picks up the side tasks: `ingest labels`, the neural network, the logistic regression and `package model`. With four workers, only three ever get a task (`run.tasks` numbers workers from 0; the chart labels them from 1), as the width of 3 predicted.

For a project view, use one row per task and color the bars by an attribute:

```python
chart = two.gantt(lanes="task", color_by="team", title="ML pipeline by task",
                  time_label="hours")
chart.save("gantt-tasks.svg")
```

```{figure} ../_static/generated/tutorials_a/pipeline_gantt_tasks.png
:alt: Gantt chart with one row per task, ordered by start time and colored by team.
:width: 100%

The same 2-worker schedule, one row per task, colored by team.
```

## Step 9: Ask what-if questions

The critical path tells you where speed-ups pay off. Suppose the ML team could shorten `tune gradient boosting`, currently 12 h. Change the duration on a copy of the DAG and recompute:

```python
for hours in (12, 9, 6, 3, 0):
    variant = dag.copy()
    variant.nodes["tune gradient boosting"]["duration"] = hours
    result = ag.alg.critical_path(variant)
    on_path = "tune gradient boosting" in result.critical
    print(f"tuning {hours:>2} h: pipeline {result.length:g} h, tuning critical: {on_path}")
```

```text
tuning 12 h: pipeline 58.75 h, tuning critical: True
tuning  9 h: pipeline 55.75 h, tuning critical: True
tuning  6 h: pipeline 52.75 h, tuning critical: True
tuning  3 h: pipeline 52.75 h, tuning critical: False
tuning  0 h: pipeline 52.75 h, tuning critical: False
```

Each hour cut from tuning shortens the pipeline by an hour, down to 6 h of tuning and a 52.75 h pipeline. Below that, tuning is no longer critical: the neural-network branch, which had 6 h of slack, now sets the pace, and further savings on tuning do not shorten the pipeline. At 6 h the two branches tie and both are critical.

The line chart below shows the same calculation for every duration from 0 to 16 h. It is drawn with {py:func}`ag.charts.line_chart <aryagraph.charts.line_chart>`:

```python
hours = list(range(17))
lengths = []
for h in hours:
    variant = dag.copy()
    variant.nodes["tune gradient boosting"]["duration"] = h
    lengths.append(ag.alg.critical_path(variant).length)

ag.charts.line_chart(
    hours,
    {"pipeline length": lengths},
    title="What if tuning took longer or shorter?",
    subtitle="Critical-path length of the ML pipeline",
    x_label="hours spent tuning gradient boosting",
    y_label="hours",
    y_min=45,
    markers=[(12, "current plan")],
).save("what-if-tuning.svg")
```

```{figure} ../_static/generated/tutorials_a/pipeline_whatif.png
:alt: Line chart of pipeline length against tuning time; flat at 52.75 hours up to 6 hours of tuning, then rising one hour per hour to 62.75 hours at 16 hours, with a marker at the current plan of 12 hours.
:width: 100%

Below 6 h of tuning, the neural-network branch sets the pace and further tuning savings do not shorten the pipeline. The y-axis starts at 45 h to make the change of slope visible.
```

The same pattern answers other questions. Shorten the critical `canary release` from 24 h to 12 h and the pipeline drops to 46.75 h. Lengthen `train logistic regression` by less than its 15 h of slack and nothing changes. To plan with uncertain durations instead of fixed ones, pass the three-point estimates to {py:func}`ag.sim.monte_carlo_schedule <aryagraph.sim.monte_carlo_schedule>`; the [DAG workflows](../user-guide/dags.md#monte-carlo-schedules) guide shows how.

## Recap

You have:

- built a {py:class}`~aryagraph.core.dag.DAG` by hand, seen it reject a cycle, and checked an imported `DiGraph` with `is_dag` and `find_cycle`;
- validated a model with `sources()`, `sinks()`, a missing-duration check and `transitive_reduction()`;
- computed critical paths and slack with {py:func}`~aryagraph.algorithms.dag.critical_path`;
- simulated the ML pipeline on 1, 2 and 4 workers with `simulate_schedule` and drawn Gantt charts by worker and by task;
- tested what-if scenarios on copies of the DAG and plotted one with `ag.charts.line_chart`.

## Next steps

- Case studies: [Critical path and parallelism in an ML pipeline](../case-studies/ml-pipeline.md) and [Scheduling a house build under uncertainty](../case-studies/construction-schedule.md).
- User guide: [DAG workflows](../user-guide/dags.md), [Simulation](../user-guide/simulation.md) and [Charts](../user-guide/charts.md).
- [Publication figures](publication-figures.md): prepare a pipeline drawing or Gantt chart for a paper.
