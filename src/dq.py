"""
Data quality checks, run between layers.

Real pipelines use a framework for this (dbt tests, Great Expectations,
Soda). We hand-roll a tiny version so the concept is visible: a check is
just a query that should return zero bad rows, and a failing check should
stop the pipeline rather than silently pass bad data downstream.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


class DataQualityError(Exception):
    pass


# --- Recording results -------------------------------------------------------
# Every check reports through passed()/warn()/fail() instead of just printing,
# so a run leaves a record behind: ops.pipeline_runs (one row per run) and
# ops.dq_results (one row per check). That's what the dashboard's traffic
# lights read. `ops` is operational metadata about the pipeline itself, not
# part of the data product, so it deliberately sits outside bronze/silver/gold.

CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id VARCHAR, started_at TIMESTAMP, finished_at TIMESTAMP, status VARCHAR,
    n_pass INTEGER, n_warn INTEGER, n_fail INTEGER, error VARCHAR, triggered_by VARCHAR
);
"""
CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS ops.dq_results (
    run_id VARCHAR, checked_at TIMESTAMP, step VARCHAR, check_name VARCHAR,
    target VARCHAR, status VARCHAR, detail VARCHAR
);
"""

_run = {"run_id": None, "started_at": None, "step": None, "results": []}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def start_run(con: duckdb.DuckDBPyConnection, started_at: datetime) -> None:
    # Created up front (not at the end) so the catalog step can document them
    # and the lineage check sees them in the same run.
    con.execute("CREATE SCHEMA IF NOT EXISTS ops")
    con.execute(CREATE_RUNS)
    con.execute(CREATE_RESULTS)
    _run.update(run_id=started_at.strftime("%Y%m%dT%H%M%S"), started_at=started_at, step=None, results=[])


def set_step(step: str) -> None:
    """Labels the checks that follow (bronze, silver, ...) for grouping on the dashboard."""
    _run["step"] = step


def _record(check: str, target: str, status: str, detail: str) -> None:
    _run["results"].append(
        {"run_id": _run["run_id"], "checked_at": _now(), "step": _run["step"], "check_name": check,
         "target": target, "status": status, "detail": detail}
    )
    print(f"  dq [{status}]: {detail}")


def passed(check: str, target: str, detail: str) -> None:
    _record(check, target, "pass", detail)


def warn(check: str, target: str, detail: str) -> None:
    """Something to look at, but not wrong enough to stop the run."""
    _record(check, target, "warn", detail)


def fail(check: str, target: str, detail: str) -> None:
    _record(check, target, "fail", detail)
    raise DataQualityError(detail)


def finish_run(con: duckdb.DuckDBPyConnection, error: str | None = None) -> str:
    """Saves the run and all its check results. Called from a `finally`, so a
    failed run is recorded too - that's the run you most need to see."""
    results = _run["results"]
    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("pass", "warn", "fail")}
    status = "failed" if error else ("warning" if counts["warn"] else "success")
    con.execute("CREATE SCHEMA IF NOT EXISTS ops")
    con.execute(CREATE_RUNS)
    con.execute(CREATE_RESULTS)
    run_df = pd.DataFrame([{
        "run_id": _run["run_id"], "started_at": _run["started_at"], "finished_at": _now(), "status": status,
        "n_pass": counts["pass"], "n_warn": counts["warn"], "n_fail": counts["fail"], "error": error,
        "triggered_by": os.environ.get("GITHUB_EVENT_NAME", "local"),
    }])
    con.execute("INSERT INTO ops.pipeline_runs SELECT * FROM run_df")
    if results:
        results_df = pd.DataFrame(results)
        con.execute("INSERT INTO ops.dq_results SELECT * FROM results_df")
    return status


def check_not_empty(con: duckdb.DuckDBPyConnection, table: str) -> None:
    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if count == 0:
        fail("check_not_empty", table, f"{table} is empty.")
    passed("check_not_empty", table, f"{table} has {count} rows")


def check_no_nulls(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> None:
    for column in columns:
        bad = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL").fetchone()[0]
        if bad > 0:
            fail("check_no_nulls", table, f"{table}.{column} has {bad} NULL values.")
    passed("check_no_nulls", table, f"{table} has no NULLs in {columns}")


def check_temperature_range(con: duckdb.DuckDBPyConnection, table: str, column: str) -> None:
    bad = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} < -90 OR {column} > 60"
    ).fetchone()[0]
    if bad > 0:
        fail("check_temperature_range", table, f"{table}.{column} has {bad} physically implausible values.")
    passed("check_temperature_range", table, f"{table}.{column} within plausible range")


def check_non_negative(con: duckdb.DuckDBPyConnection, table: str, column: str) -> None:
    bad = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} < 0").fetchone()[0]
    if bad > 0:
        fail("check_non_negative", table, f"{table}.{column} has {bad} negative values (e.g. precipitation, wind speed can't be negative).")
    passed("check_non_negative", table, f"{table}.{column} has no negative values")


def check_range(con: duckdb.DuckDBPyConnection, table: str, column: str, low: float, high: float) -> None:
    """NULLs pass: for LLM-extracted fields, 'not stated' is a valid answer."""
    bad = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} < ? OR {column} > ?", [low, high]).fetchone()[0]
    if bad > 0:
        fail("check_range", table, f"{table}.{column} has {bad} values outside [{low}, {high}].")
    passed("check_range", table, f"{table}.{column} within [{low}, {high}]")


def check_row_count_per_group(
    con: duckdb.DuckDBPyConnection,
    table: str,
    group_column: str,
    expected_groups: list[str],
    min_count: int = 1,
    where_sql: str | None = None,
    where_params: list | None = None,
) -> None:
    """
    Catches partial failures that check_not_empty misses: a table can be
    non-empty overall while one specific group (e.g. one city) silently
    contributed zero rows - for instance if that city's fetch quietly
    failed while the others succeeded.
    """
    query = f"SELECT {group_column}, COUNT(*) FROM {table}"
    if where_sql:
        query += f" WHERE {where_sql}"
    query += f" GROUP BY {group_column}"
    counts = dict(con.execute(query, where_params or []).fetchall())

    missing = [g for g in expected_groups if counts.get(g, 0) < min_count]
    if missing:
        scope = f" (where {where_sql})" if where_sql else ""
        fail("check_row_count_per_group", table, 
            f"{table} has fewer than {min_count} row(s) for {group_column} = {missing}{scope}."
        )
    passed("check_row_count_per_group", table, f"{table} has >= {min_count} row(s) for every {group_column} in {expected_groups}")


def check_freshness(
    con: duckdb.DuckDBPyConnection,
    table: str,
    timestamp_column: str,
    max_age_hours: float,
) -> None:
    """
    Flags delayed/stale pipeline runs: fails if the newest timestamp in the
    table is older than max_age_hours. Note this only catches staleness
    *within* a run that actually executed - if the scheduler itself never
    fires (e.g. the daily GitHub Actions job stops running), no check
    inside the pipeline runs either. That failure mode needs an outside
    observer: GitHub emails the repo owner on a failed workflow run by
    default, and the dashboard shows a "last updated" freshness indicator
    for a human to notice.
    """
    newest = con.execute(f"SELECT MAX({timestamp_column}) FROM {table}").fetchone()[0]
    if newest is None:
        fail("check_freshness", table, f"{table}.{timestamp_column} has no rows to check freshness on.")

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    age_hours = (now_utc - newest).total_seconds() / 3600
    if age_hours > max_age_hours:
        fail("check_freshness", table, 
            f"{table}.{timestamp_column} is stale: newest value is {age_hours:.1f}h old "
            f"(limit {max_age_hours}h). The scheduled run may not have executed."
        )
    detail = f"{table}.{timestamp_column} is {age_hours:.1f}h old (limit {max_age_hours}h)"
    if age_hours > max_age_hours / 2:
        warn("check_freshness", table, detail + " - past half its allowed age")
    else:
        passed("check_freshness", table, detail)


def check_documented(con: duckdb.DuckDBPyConnection, schema: str) -> None:
    """Governance check: every table/view and column in the schema must have
    a description. Catches a new gold column added without updating
    semantic_layer.yml, which would otherwise leave the agent guessing."""
    undocumented_tables = [
        row[0] for row in con.execute(
            """
            SELECT table_name FROM duckdb_tables() WHERE database_name = current_database() AND schema_name = ? AND (comment IS NULL OR comment = '')
            UNION ALL
            SELECT view_name FROM duckdb_views() WHERE database_name = current_database() AND schema_name = ? AND (comment IS NULL OR comment = '')
            """,
            [schema, schema],
        ).fetchall()
    ]
    undocumented_columns = [
        f"{row[0]}.{row[1]}" for row in con.execute(
            "SELECT table_name, column_name FROM duckdb_columns() "
            "WHERE database_name = current_database() AND schema_name = ? AND (comment IS NULL OR comment = '')",
            [schema],
        ).fetchall()
    ]
    missing = undocumented_tables + undocumented_columns
    if missing:
        fail("check_documented", schema, f"Undocumented in {schema} (add to semantic_layer.yml): {missing}")
    passed("check_documented", schema, f"every {schema} table and column is documented")


TABLE_REFERENCE = re.compile(r"\b(?:bronze|silver|gold)\.[a-z_]+\b")

WAREHOUSE_OBJECTS_QUERY = """
SELECT schema_name, table_name, NULL FROM duckdb_tables() WHERE database_name = current_database()
UNION ALL
SELECT schema_name, view_name, sql FROM duckdb_views() WHERE database_name = current_database() AND NOT internal
"""


def check_lineage(con: duckdb.DuckDBPyConnection, lineage: dict, repo_root: Path) -> None:
    """Governance check: the declared lineage (semantic_layer.yml) must match
    reality, checked three ways - the warehouse's actual tables/views, the
    tables each view's SQL actually reads, and the tables each code file in
    this repo actually references. Declared lineage that nobody verifies
    quietly rots; this makes a stale declaration fail the pipeline instead."""
    problems = []

    for node, spec in lineage.items():
        problems += [f"{node} lists unknown upstream {up}" for up in spec.get("upstream", []) if up not in lineage]

    objects = con.execute(WAREHOUSE_OBJECTS_QUERY).fetchall()
    actual = {f"{schema}.{name}" for schema, name, _ in objects}
    declared = {node for node, spec in lineage.items() if spec["type"] in ("table", "view")}
    problems += [f"{t} exists in the warehouse but isn't in the lineage" for t in sorted(actual - declared)]
    problems += [f"{t} is in the lineage but not in the warehouse" for t in sorted(declared - actual)]

    for schema, name, view_sql in objects:
        if view_sql is None:
            continue
        view = f"{schema}.{name}"
        reads = set(TABLE_REFERENCE.findall(view_sql)) - {view}
        undeclared = reads - set(lineage.get(view, {}).get("upstream", []))
        problems += [f"view {view} reads {t}, which isn't declared as its upstream" for t in sorted(undeclared)]

    nodes_by_file: dict = {}
    for node, spec in lineage.items():
        if spec.get("built_by"):
            nodes_by_file.setdefault(spec["built_by"], set()).add(node)
    for path, nodes in nodes_by_file.items():
        allowed = nodes | {up for node in nodes for up in lineage[node].get("upstream", [])}
        # Text scanning is a heuristic: only count names that are real tables/
        # views, so a comment mentioning a column like bronze.fetched_at isn't
        # mistaken for a table read.
        referenced = set(TABLE_REFERENCE.findall((repo_root / path).read_text())) & (actual | declared)
        problems += [f"{path} references {t}, which its lineage doesn't declare" for t in sorted(referenced - allowed)]

    if problems:
        fail("check_lineage", "lineage", "Lineage out of date (fix semantic_layer.yml):\n  " + "\n  ".join(problems))
    passed("check_lineage", "lineage", f"lineage matches the warehouse, view SQL, and code ({len(lineage)} nodes)")


def check_evidence_verified(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """LLM grounding: rows whose supporting quote wasn't found in the source are
    kept but unscored - worth a look (yellow), not a reason to stop (red)."""
    unverified = con.execute(f"SELECT COUNT(*) FROM {table} WHERE NOT evidence_verified").fetchone()[0]
    if unverified:
        warn("check_evidence_verified", table, f"{unverified} extraction(s) in {table} have unverified evidence (kept, not scored)")
    else:
        passed("check_evidence_verified", table, f"every extraction in {table} has verified evidence")
