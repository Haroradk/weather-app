"""
Bronze layer: land raw API responses exactly as received.

Medallion principle: bronze is a append-only, unopinionated copy of the
source. No parsing, no type casting, no filtering. If the source changes
its schema tomorrow, bronze still succeeds — only silver would need to
change. This is what makes bronze replayable: if we ever find a bug in
silver's transformation logic, we can fix it and re-run silver against
everything already sitting in bronze, no re-fetching needed.
"""

import json
from datetime import datetime, timezone

import duckdb
import requests

from config import API_URL, CITIES, FORECAST_DAYS, HOURLY_FIELDS, PAST_DAYS, get_connection

CREATE_SCHEMA = "CREATE SCHEMA IF NOT EXISTS bronze;"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS bronze.raw_weather_observations (
    city VARCHAR,
    fetched_at TIMESTAMP,
    source_url VARCHAR,
    raw_json VARCHAR
);
"""


def fetch_city_weather(city: dict) -> dict:
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "hourly": HOURLY_FIELDS,
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "timezone": "UTC",
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response


def run(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(CREATE_SCHEMA)
    con.execute(CREATE_TABLE)

    rows_landed = 0
    for city in CITIES:
        response = fetch_city_weather(city)
        # Naive-but-UTC on purpose: binding a tz-aware datetime into a plain
        # TIMESTAMP column lets duckdb convert it through the *local system*
        # timezone before storing, not UTC - stripping tzinfo ourselves after
        # converting to UTC avoids that ambiguity regardless of what machine
        # (or CI runner) this runs on.
        fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
        con.execute(
            "INSERT INTO bronze.raw_weather_observations VALUES (?, ?, ?, ?)",
            [city["name"], fetched_at, response.url, response.text],
        )
        rows_landed += 1
        print(f"  bronze: landed {city['name']} ({len(response.text)} bytes)")

    return rows_landed


if __name__ == "__main__":
    con = get_connection()
    landed = run(con)
    print(f"Bronze: landed {landed} raw responses.")
