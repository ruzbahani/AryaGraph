# =============================================================================
#
#      _                     ____                 _
#     / \   _ __ _   _  __ _ / ___|_ __ __ _ _ __ | |__
#    / _ \ | '__| | | |/ _` | |  _| '__/ _` | '_ \| '_ \
#   / ___ \| |  | |_| | (_| | |_| | | | (_| | |_) | | | |
#  /_/   \_\_|   \__, |\__,_|\____|_|  \__,_| .__/|_| |_|
#               |___/                     |_|
#
#          Graph & DAG visualization, analysis and simulation.
#
# -----------------------------------------------------------------------------
#  Copyright (c) 2026 Ali Mohammadi Ruzbahani
#  SPDX-License-Identifier: MIT
#
#  https://ruzbahani.com/aryagraph
# =============================================================================

"""Hand-crafted example DAGs with realistic names and attributes.

Every task node carries

* ``duration``: the planned time in **hours** (equal to ``mode``);
* ``min`` / ``mode`` / ``max``: optimistic, most likely and pessimistic
  hours, a triangular estimate for PERT and Monte-Carlo scheduling;
* ``kind`` and ``team``: categorical attributes for colouring and grouping.

Each function builds a fresh :class:`~aryagraph.DAG`, so callers may mutate the
result freely.

>>> plan = project_plan()
>>> plan.critical_path()          # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Sequence

from ..core.dag import DAG

Row = tuple  # (node, mode, min, max, kind, team[, extra_attrs])


def _build(name: str, description: str, tasks: Sequence[Row], deps: Sequence[tuple]) -> DAG:
    """Assemble a DAG from task rows and ``(u, v[, attrs])`` dependencies, checking the data."""
    dag = DAG(name=name, description=description)
    for node, mode, low, high, kind, team, *extra in tasks:
        if not low <= mode <= high:
            raise AssertionError(f"{name}: bad estimate for {node!r}: {low} / {mode} / {high}")
        attrs: dict[str, Any] = dict(duration=mode, min=low, mode=mode, max=high, kind=kind, team=team)
        if extra:
            attrs.update(extra[0])
        dag.add_node(node, **attrs)
    for dep in deps:
        for endpoint in dep[:2]:
            if endpoint not in dag:
                raise AssertionError(f"{name}: dependency on undeclared task {endpoint!r}")
    dag.add_edges(deps)
    return dag


def ml_pipeline() -> DAG:
    """Machine-learning pipeline, 16 tasks: ingest → validate → features → train variants → evaluate → deploy.

    Teams: ``data-eng``, ``ml``, ``responsible-ai``, ``mlops``. Arcs carry
    ``artifact``, the data product handed downstream.
    """
    tasks = [
        ("ingest events", 2.0, 1.5, 4.0, "ingest", "data-eng"),
        ("ingest labels", 1.0, 0.5, 2.0, "ingest", "data-eng"),
        ("validate schema", 0.5, 0.25, 1.0, "validate", "data-eng"),
        ("clean & deduplicate", 3.0, 2.0, 6.0, "transform", "data-eng"),
        ("join labels", 1.0, 0.5, 2.0, "transform", "data-eng"),
        ("train/test split", 0.25, 0.1, 0.5, "transform", "ml"),
        ("feature engineering", 6.0, 4.0, 12.0, "features", "ml"),
        ("train logistic regression", 1.0, 0.5, 2.0, "train", "ml"),
        ("train gradient boosting", 4.0, 3.0, 8.0, "train", "ml"),
        ("tune gradient boosting", 12.0, 8.0, 24.0, "train", "ml"),
        ("train neural network", 10.0, 6.0, 20.0, "train", "ml"),
        ("evaluate models", 2.0, 1.0, 4.0, "evaluate", "ml"),
        ("fairness audit", 3.0, 2.0, 6.0, "evaluate", "responsible-ai"),
        ("package model", 1.5, 1.0, 3.0, "deploy", "mlops"),
        ("deploy to staging", 1.0, 0.5, 2.0, "deploy", "mlops"),
        ("canary release", 24.0, 12.0, 48.0, "deploy", "mlops"),
    ]
    deps = [
        ("ingest events", "validate schema", {"artifact": "raw events"}),
        ("validate schema", "clean & deduplicate", {"artifact": "validated events"}),
        ("clean & deduplicate", "join labels", {"artifact": "clean events"}),
        ("ingest labels", "join labels", {"artifact": "labels"}),
        ("join labels", "train/test split", {"artifact": "labeled dataset"}),
        ("train/test split", "feature engineering", {"artifact": "training split"}),
        ("feature engineering", "train logistic regression", {"artifact": "feature matrix"}),
        ("feature engineering", "train gradient boosting", {"artifact": "feature matrix"}),
        ("feature engineering", "train neural network", {"artifact": "feature matrix"}),
        ("train gradient boosting", "tune gradient boosting", {"artifact": "baseline booster"}),
        ("train logistic regression", "evaluate models", {"artifact": "logistic model"}),
        ("tune gradient boosting", "evaluate models", {"artifact": "tuned booster"}),
        ("train neural network", "evaluate models", {"artifact": "network weights"}),
        ("train/test split", "evaluate models", {"artifact": "holdout split"}),
        ("evaluate models", "fairness audit", {"artifact": "best model"}),
        ("evaluate models", "package model", {"artifact": "best model"}),
        ("fairness audit", "deploy to staging", {"artifact": "audit report"}),
        ("package model", "deploy to staging", {"artifact": "model image"}),
        ("deploy to staging", "canary release", {"artifact": "staging endpoint"}),
    ]
    return _build("ML pipeline", "Train, compare and ship a churn-prediction model.", tasks, deps)


def software_build() -> DAG:
    """Build graph of a client/server product, 20 targets from dependency fetch to release.

    Durations are in hours (a compile step takes a few minutes). Arcs carry
    ``artifact``, the file or report the downstream target consumes.
    """
    tasks = [
        ("fetch dependencies", 0.10, 0.05, 0.30, "setup", "platform"),
        ("generate protobuf stubs", 0.05, 0.03, 0.10, "codegen", "platform"),
        ("compile utils", 0.08, 0.05, 0.15, "compile", "backend"),
        ("compile core", 0.25, 0.15, 0.40, "compile", "backend"),
        ("compile net", 0.12, 0.08, 0.20, "compile", "backend"),
        ("compile storage", 0.15, 0.10, 0.25, "compile", "backend"),
        ("compile api", 0.20, 0.12, 0.35, "compile", "backend"),
        ("compile cli", 0.06, 0.04, 0.10, "compile", "devex"),
        ("bundle web ui", 0.20, 0.12, 0.40, "bundle", "frontend"),
        ("link libcore", 0.05, 0.03, 0.08, "link", "backend"),
        ("link server", 0.06, 0.04, 0.10, "link", "backend"),
        ("link cli", 0.03, 0.02, 0.05, "link", "devex"),
        ("lint", 0.05, 0.03, 0.10, "check", "devex"),
        ("type check", 0.10, 0.06, 0.20, "check", "devex"),
        ("unit tests", 0.30, 0.20, 0.60, "test", "qa"),
        ("integration tests", 0.75, 0.50, 1.50, "test", "qa"),
        ("build docs", 0.15, 0.10, 0.30, "docs", "devex"),
        ("container image", 0.20, 0.10, 0.40, "package", "platform"),
        ("sign artifacts", 0.03, 0.02, 0.05, "release", "platform"),
        ("publish release", 0.10, 0.05, 0.25, "release", "platform"),
    ]
    deps = [
        ("fetch dependencies", "generate protobuf stubs", {"artifact": "protoc plugins"}),
        ("fetch dependencies", "compile utils", {"artifact": "third-party headers"}),
        ("fetch dependencies", "bundle web ui", {"artifact": "node_modules"}),
        ("fetch dependencies", "lint", {"artifact": "linters"}),
        ("fetch dependencies", "type check", {"artifact": "type stubs"}),
        ("fetch dependencies", "build docs", {"artifact": "doc toolchain"}),
        ("generate protobuf stubs", "compile net", {"artifact": "net.pb.h"}),
        ("generate protobuf stubs", "compile api", {"artifact": "api.pb.h"}),
        ("compile utils", "compile core", {"artifact": "libutils.a"}),
        ("compile core", "compile net", {"artifact": "core headers"}),
        ("compile core", "compile storage", {"artifact": "core headers"}),
        ("compile core", "compile api", {"artifact": "core headers"}),
        ("compile core", "compile cli", {"artifact": "core headers"}),
        ("compile core", "link libcore", {"artifact": "core.o"}),
        ("compile net", "link libcore", {"artifact": "net.o"}),
        ("compile storage", "link libcore", {"artifact": "storage.o"}),
        ("link libcore", "link server", {"artifact": "libcore.so"}),
        ("link libcore", "link cli", {"artifact": "libcore.so"}),
        ("link libcore", "unit tests", {"artifact": "libcore.so"}),
        ("compile api", "link server", {"artifact": "api.o"}),
        ("compile cli", "link cli", {"artifact": "cli.o"}),
        ("link server", "integration tests", {"artifact": "server binary"}),
        ("bundle web ui", "integration tests", {"artifact": "ui bundle"}),
        ("link server", "container image", {"artifact": "server binary"}),
        ("bundle web ui", "container image", {"artifact": "ui bundle"}),
        ("unit tests", "container image", {"artifact": "test report"}),
        ("lint", "container image", {"artifact": "lint report"}),
        ("type check", "container image", {"artifact": "type report"}),
        ("integration tests", "sign artifacts", {"artifact": "test report"}),
        ("container image", "sign artifacts", {"artifact": "image digest"}),
        ("link cli", "sign artifacts", {"artifact": "cli binary"}),
        ("sign artifacts", "publish release", {"artifact": "signed artifacts"}),
        ("build docs", "publish release", {"artifact": "docs site"}),
    ]
    return _build("software build", "Targets of a client/server build, from fetch to release.", tasks, deps)


def project_plan() -> DAG:
    """Construction of a single-family house in 18 tasks, a classic CPM/PERT example.

    Durations are working hours (40 h = one week). Arcs carry ``lag``, a
    mandatory wait in hours after the predecessor finishes (concrete curing,
    drywall mud drying); it is 0 elsewhere.
    """
    tasks = [
        ("site survey", 16.0, 12.0, 24.0, "planning", "surveyor"),
        ("architectural design", 80.0, 60.0, 120.0, "planning", "architect"),
        ("building permit", 120.0, 80.0, 240.0, "permit", "city"),
        ("excavation", 24.0, 16.0, 40.0, "sitework", "civil"),
        ("foundation", 40.0, 32.0, 64.0, "structure", "civil"),
        ("framing", 120.0, 96.0, 160.0, "structure", "carpentry"),
        ("roofing", 40.0, 32.0, 64.0, "envelope", "roofing"),
        ("windows & doors", 24.0, 16.0, 32.0, "envelope", "carpentry"),
        ("plumbing rough-in", 40.0, 32.0, 56.0, "systems", "plumbing"),
        ("electrical rough-in", 40.0, 32.0, 60.0, "systems", "electrical"),
        ("HVAC install", 32.0, 24.0, 48.0, "systems", "hvac"),
        ("insulation", 16.0, 12.0, 24.0, "interior", "finishing"),
        ("drywall", 48.0, 40.0, 72.0, "interior", "finishing"),
        ("interior painting", 32.0, 24.0, 48.0, "interior", "finishing"),
        ("flooring", 32.0, 24.0, 48.0, "interior", "finishing"),
        ("cabinets & fixtures", 24.0, 16.0, 40.0, "interior", "carpentry"),
        ("landscaping", 40.0, 24.0, 64.0, "exterior", "landscaping"),
        ("final inspection", 4.0, 2.0, 8.0, "inspection", "city"),
    ]
    pairs = [
        ("site survey", "architectural design"),
        ("architectural design", "building permit"),
        ("building permit", "excavation"),
        ("excavation", "foundation"),
        ("foundation", "framing"),
        ("framing", "roofing"),
        ("framing", "windows & doors"),
        ("framing", "plumbing rough-in"),
        ("framing", "electrical rough-in"),
        ("framing", "HVAC install"),
        ("roofing", "insulation"),
        ("windows & doors", "insulation"),
        ("plumbing rough-in", "insulation"),
        ("electrical rough-in", "insulation"),
        ("HVAC install", "insulation"),
        ("insulation", "drywall"),
        ("drywall", "interior painting"),
        ("interior painting", "flooring"),
        ("interior painting", "cabinets & fixtures"),
        ("roofing", "landscaping"),
        ("flooring", "final inspection"),
        ("cabinets & fixtures", "final inspection"),
        ("landscaping", "final inspection"),
    ]
    lags = {("foundation", "framing"): 72.0, ("drywall", "interior painting"): 24.0}
    deps = [(u, v, {"lag": lags.get((u, v), 0.0)}) for u, v in pairs]
    return _build("house construction", "Building a single-family house, survey to final inspection.", tasks, deps)


def data_warehouse_etl() -> DAG:
    """Nightly data-warehouse ETL, 25 tasks with heavy fan-out (extracts) and fan-in (facts, aggregates).

    Extract → stage per source system, conformed dimensions, fact tables, a
    data-quality gate, aggregates and publishing. Every task carries
    ``table``, the table it writes.
    """
    raw = [
        ("crm", "crm_accounts", 0.5, 0.3, 1.0, 0.3, 0.2, 0.5),
        ("erp", "erp_orders", 0.75, 0.5, 1.5, 0.5, 0.3, 0.8),
        ("web_logs", "web_events", 1.0, 0.6, 2.5, 0.8, 0.5, 1.5),
        ("payments", "payments", 0.4, 0.25, 0.8, 0.3, 0.2, 0.5),
        ("marketing", "campaigns", 0.3, 0.2, 0.6, 0.2, 0.1, 0.4),
        ("support", "tickets", 0.25, 0.15, 0.5, 0.2, 0.1, 0.4),
    ]
    tasks: list[Row] = []
    deps: list[tuple] = []
    for src, table, *est in raw:
        tasks.append((f"extract_{src}", est[0], est[1], est[2], "extract", "ingestion", {"table": f"raw.{table}"}))
        tasks.append((f"stage_{src}", est[3], est[4], est[5], "stage", "ingestion", {"table": f"staging.{table}"}))
        deps.append((f"extract_{src}", f"stage_{src}"))

    def model(node: str, est: tuple, kind: str, team: str, inputs: list[str]) -> None:
        schema = "dw" if kind in ("dimension", "fact") else "mart" if kind == "aggregate" else "ops"
        tasks.append((node, *est, kind, team, {"table": f"{schema}.{node}"}))
        deps.extend((u, node) for u in inputs)

    model("dim_customer", (0.6, 0.4, 1.0), "dimension", "modeling", ["stage_crm", "stage_erp", "stage_support"])
    model("dim_product", (0.3, 0.2, 0.5), "dimension", "modeling", ["stage_erp"])
    model("dim_date", (0.05, 0.03, 0.1), "dimension", "modeling", [])
    model("dim_channel", (0.2, 0.1, 0.3), "dimension", "modeling", ["stage_marketing", "stage_web_logs"])
    model(
        "fact_orders",
        (1.0, 0.7, 2.0),
        "fact",
        "modeling",
        ["stage_erp", "stage_payments", "dim_customer", "dim_product", "dim_date"],
    )
    model(
        "fact_sessions",
        (1.5, 1.0, 3.0),
        "fact",
        "modeling",
        ["stage_web_logs", "dim_customer", "dim_channel", "dim_date"],
    )
    model("fact_tickets", (0.4, 0.25, 0.8), "fact", "modeling", ["stage_support", "dim_customer", "dim_date"])
    model(
        "data_quality_checks",
        (0.5, 0.3, 1.0),
        "quality",
        "analytics-eng",
        ["fact_orders", "fact_sessions", "fact_tickets"],
    )
    model("agg_daily_revenue", (0.3, 0.2, 0.6), "aggregate", "analytics", ["fact_orders", "data_quality_checks"])
    model(
        "agg_customer_ltv",
        (0.6, 0.4, 1.2),
        "aggregate",
        "analytics",
        ["fact_orders", "fact_sessions", "fact_tickets", "data_quality_checks"],
    )
    model(
        "agg_marketing_funnel",
        (0.5, 0.3, 1.0),
        "aggregate",
        "analytics",
        ["fact_sessions", "fact_orders", "data_quality_checks"],
    )
    model(
        "refresh_dashboards",
        (0.25, 0.1, 0.5),
        "publish",
        "analytics",
        ["agg_daily_revenue", "agg_marketing_funnel", "agg_customer_ltv"],
    )
    model("export_ml_features", (0.4, 0.2, 0.8), "publish", "ml-platform", ["agg_customer_ltv", "fact_sessions"])
    return _build("data warehouse ETL", "Nightly load of a sales & engagement data warehouse.", tasks, deps)


_COURSES = [
    ("MATH101", "Calculus I", 4, "math"),
    ("MATH102", "Calculus II", 4, "math"),
    ("MATH150", "Discrete Mathematics", 3, "math"),
    ("MATH210", "Linear Algebra", 3, "math"),
    ("STAT220", "Probability & Statistics", 3, "math"),
    ("CS101", "Introduction to Programming", 4, "programming"),
    ("CS102", "Data Structures", 4, "programming"),
    ("CS201", "Algorithms", 4, "theory"),
    ("CS210", "Computer Organization", 3, "systems"),
    ("CS220", "Systems Programming", 4, "systems"),
    ("CS250", "Software Engineering", 3, "programming"),
    ("CS301", "Operating Systems", 4, "systems"),
    ("CS310", "Computer Networks", 3, "systems"),
    ("CS320", "Databases", 3, "data"),
    ("CS330", "Programming Languages", 3, "theory"),
    ("CS340", "Compilers", 4, "systems"),
    ("CS350", "Theory of Computation", 3, "theory"),
    ("CS360", "Artificial Intelligence", 3, "ai"),
    ("CS370", "Machine Learning", 4, "ai"),
    ("CS380", "Computer Graphics", 3, "graphics"),
    ("CS410", "Distributed Systems", 3, "systems"),
    ("CS490", "Capstone Project", 4, "project"),
]

_PREREQUISITES = [
    ("MATH101", "MATH102", "required"),
    ("MATH101", "MATH210", "required"),
    ("MATH102", "STAT220", "required"),
    ("CS101", "CS102", "required"),
    ("CS101", "CS210", "required"),
    ("CS102", "CS201", "required"),
    ("MATH150", "CS201", "required"),
    ("CS102", "CS220", "required"),
    ("CS210", "CS220", "required"),
    ("CS102", "CS250", "required"),
    ("CS220", "CS301", "required"),
    ("CS220", "CS310", "required"),
    ("STAT220", "CS310", "recommended"),
    ("CS102", "CS320", "required"),
    ("MATH150", "CS320", "recommended"),
    ("CS201", "CS330", "required"),
    ("CS330", "CS340", "required"),
    ("CS210", "CS340", "required"),
    ("MATH150", "CS350", "required"),
    ("CS201", "CS350", "required"),
    ("CS201", "CS360", "required"),
    ("STAT220", "CS360", "required"),
    ("CS201", "CS370", "required"),
    ("MATH210", "CS370", "required"),
    ("STAT220", "CS370", "required"),
    ("CS201", "CS380", "required"),
    ("MATH210", "CS380", "required"),
    ("CS301", "CS410", "required"),
    ("CS310", "CS410", "required"),
    ("CS250", "CS490", "required"),
    ("CS320", "CS490", "required"),
]


def course_prerequisites() -> DAG:
    """Prerequisite structure of a computer-science degree, 22 courses.

    Nodes are course codes with ``title``, ``credits``, ``level`` (100–400),
    ``department``, ``kind`` (subject area) and ``team`` (= department).
    ``duration`` is the semester workload in hours (45 h per credit, the
    Carnegie unit), with ``min``/``max`` at 80 % / 150 % of it. Arcs carry
    ``requirement``: ``"required"`` or ``"recommended"``.
    """
    tasks = []
    for code, title, credits, area in _COURSES:
        department = code.rstrip("0123456789")
        hours = 45.0 * credits
        extra = {"title": title, "credits": credits, "level": int(code[len(department)]) * 100, "department": department}
        tasks.append((code, hours, 0.8 * hours, 1.5 * hours, area, department, extra))
    deps = [(u, v, {"requirement": req}) for u, v, req in _PREREQUISITES]
    return _build("course prerequisites", "Prerequisites of an undergraduate computer-science curriculum.", tasks, deps)


__all__ = ["ml_pipeline", "software_build", "project_plan", "data_warehouse_etl", "course_prerequisites"]
