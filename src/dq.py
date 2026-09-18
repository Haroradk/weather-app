"""
Data quality checks, run between layers.

Real pipelines use a framework for this (dbt tests, Great Expectations,
Soda). We hand-roll a tiny version so the concept is visible: a check is
just a query that should return zero bad rows, and a failing check should
stop the pipeline rather than silently pass bad data downstream.
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb


class DataQualityError(Exception):
    pass


def check_not_empty(con: duckdb.DuckDBPyConnection, table: str) -> None:
    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if count == 0:
        raise DataQualityError(f"{table} is empty.")
    print(f"  dq: {table} has {count} rows")


def check_no_nulls(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> None:
    for column in columns:
        bad = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL").fetchone()[0]
        if bad > 0:
            raise DataQualityError(f"{table}.{column} has {bad} NULL values.")
    print(f"  dq: {table} has no NULLs in {columns}")


def check_temperature_range(con: duckdb.DuckDBPyConnection, table: str, column: str) -> None:
    bad = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} < -90 OR {column} > 60"
    ).fetchone()[0]
    if bad > 0:
        raise DataQualityError(f"{table}.{column} has {bad} physically implausible values.")
    print(f"  dq: {table}.{column} within plausible range")


def check_non_negative(con: duckdb.DuckDBPyConnection, table: str, column: str) -> None:
    bad = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} < 0").fetchone()[0]
    if bad > 0:
        raise DataQualityError(f"{table}.{column} has {bad} negative values (e.g. precipitation, wind speed can't be negative).")
    print(f"  dq: {table}.{column} has no negative values")


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
        raise DataQualityError(
            f"{table} has fewer than {min_count} row(s) for {group_column} = {missing}{scope}."
        )
    print(f"  dq: {table} has >= {min_count} row(s) for every {group_column} in {expected_groups}")


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
        raise DataQualityError(f"{table}.{timestamp_column} has no rows to check freshness on.")

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    age_hours = (now_utc - newest).total_seconds() / 3600
    if age_hours > max_age_hours:
        raise DataQualityError(
            f"{table}.{timestamp_column} is stale: newest value is {age_hours:.1f}h old "
            f"(limit {max_age_hours}h). The scheduled run may not have executed."
        )
    print(f"  dq: {table}.{timestamp_column} is fresh ({age_hours:.1f}h old, limit {max_age_hours}h)")
