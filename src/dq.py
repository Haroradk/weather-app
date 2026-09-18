"""
Data quality checks, run between layers.

Real pipelines use a framework for this (dbt tests, Great Expectations,
Soda). We hand-roll a tiny version so the concept is visible: a check is
just a query that should return zero bad rows, and a failing check should
stop the pipeline rather than silently pass bad data downstream.
"""

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
