"""Shared configuration for the pipeline."""

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

# Local secrets/config for dev; in CI these come from real environment variables instead.
load_dotenv(Path(__file__).parent / ".env")

CITIES = [
    {"name": "Copenhagen", "latitude": 55.6761, "longitude": 12.5683},
    {"name": "London", "latitude": 51.5074, "longitude": -0.1278},
    {"name": "New York", "latitude": 40.7128, "longitude": -74.0060},
]

# Resolved relative to this file, not the process's cwd, so the pipeline and
# the dashboard find the same database regardless of where they're launched from.
DB_PATH = str(Path(__file__).parent / "data" / "weather.duckdb")

MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN")
MOTHERDUCK_DATABASE = os.environ.get("MOTHERDUCK_DATABASE", "weather")

# Open-Meteo is free and requires no API key: https://open-meteo.com/en/docs
API_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m"
PAST_DAYS = 2
FORECAST_DAYS = 7


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """MotherDuck when MOTHERDUCK_TOKEN is set (cloud), else the local file."""
    if MOTHERDUCK_TOKEN:
        con = duckdb.connect("md:", config={"motherduck_token": MOTHERDUCK_TOKEN})
        con.execute(f"CREATE DATABASE IF NOT EXISTS {MOTHERDUCK_DATABASE}")
        con.execute(f"USE {MOTHERDUCK_DATABASE}")
    else:
        con = duckdb.connect(DB_PATH, read_only=read_only)

    # Without this, duckdb binds/converts naive datetimes through the
    # session's TimeZone setting (which defaults to the OS's local zone) even
    # for plain TIMESTAMP columns - silently shifting every timestamp we
    # store by the local UTC offset. Every timestamp in this project is
    # naive-but-UTC by convention, so the session timezone must be UTC too.
    con.execute("SET TimeZone = 'UTC'")
    return con
