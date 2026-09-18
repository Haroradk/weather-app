"""
Silver layer: parsed, typed, deduplicated records.

Medallion principle: silver is where raw source shape gets turned into
a clean, queryable schema. Bad or malformed records get dropped or fixed
here — but decisions are still row-level, not yet business logic.

We re-derive the whole silver table from bronze on every run instead of
tracking "what's new". That's a deliberate simplification for learning:
it keeps the pipeline idempotent (re-running it never duplicates or
corrupts data) without needing watermark/incremental-load bookkeeping yet.
Real pipelines add that once full-refresh gets too slow.

Because the same hour can appear in multiple bronze fetches (forecast
today, then re-fetched as history tomorrow), we dedupe by keeping the
most recently fetched value for each (city, observation_time) pair.
"""

import json
from datetime import datetime, timezone

import duckdb
import pandas as pd

from config import get_connection

CREATE_SCHEMA = "CREATE SCHEMA IF NOT EXISTS silver;"


def parse_bronze_row(city: str, fetched_at, raw_json: str):
    """Turn one bronze row's raw JSON into a list of hourly observation tuples."""
    payload = json.loads(raw_json)
    hourly = payload["hourly"]

    times = hourly["time"]
    temps = hourly["temperature_2m"]
    precip = hourly["precipitation"]
    wind = hourly["wind_speed_10m"]
    humidity = hourly["relative_humidity_2m"]

    rows = []
    for i, observation_time in enumerate(times):
        rows.append(
            (
                city,
                fetched_at,
                observation_time,  # ISO string, e.g. "2026-09-18T00:00"
                temps[i],
                precip[i],
                wind[i],
                humidity[i],
            )
        )
    return rows


def run(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(CREATE_SCHEMA)

    bronze_rows = con.execute(
        "SELECT city, fetched_at, raw_json FROM bronze.raw_weather_observations"
    ).fetchall()

    # dedupe key: (city, observation_time) -> keep the row from the latest fetch
    latest: dict[tuple, tuple] = {}
    for city, fetched_at, raw_json in bronze_rows:
        for row in parse_bronze_row(city, fetched_at, raw_json):
            key = (row[0], row[2])  # (city, observation_time)
            existing = latest.get(key)
            if existing is None or row[1] > existing[1]:  # newer fetched_at wins
                latest[key] = row

    loaded_at = datetime.now(timezone.utc)
    final_rows = [
        (city, observation_time, temp_c, precip_mm, wind_kmh, humidity_pct, loaded_at)
        for (city, _fetched_at, observation_time, temp_c, precip_mm, wind_kmh, humidity_pct) in latest.values()
    ]

    # Loaded as one bulk statement via a DataFrame rather than one INSERT per
    # row: against a remote warehouse (MotherDuck), a per-row executemany
    # means one network round trip per row - fine locally, painfully slow
    # once the connection isn't in-process anymore.
    final_df = pd.DataFrame(
        final_rows,
        columns=["city", "observation_time", "temperature_c", "precipitation_mm", "wind_speed_kmh", "humidity_pct", "loaded_at"],
    )
    final_df["observation_time"] = pd.to_datetime(final_df["observation_time"])
    # Stored naive-UTC (not TIMESTAMPTZ), consistent with bronze.fetched_at,
    # so downstream comparisons (e.g. dq.check_freshness) don't have to
    # reason about timezone conversion on top of staleness math.
    final_df["loaded_at"] = pd.to_datetime(final_df["loaded_at"]).dt.tz_convert("UTC").dt.tz_localize(None)

    con.execute("CREATE OR REPLACE TABLE silver.weather_hourly AS SELECT * FROM final_df")

    return len(final_rows)


if __name__ == "__main__":
    con = get_connection()
    loaded = run(con)
    print(f"Silver: {loaded} deduplicated hourly rows.")
