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


# National Weather Service text products - free, no key, but asks for an
# identifying User-Agent: https://www.weather.gov/documentation/services-web-api
NWS_API_URL = "https://api.weather.gov"
NWS_FORECAST_OFFICE = "OKX"  # NWS New York, NY office - covers New York City
NWS_USER_AGENT = "weather-etl-pipeline learning project (github.com/Haroradk/weather-app)"

# Gemini is used by the pipeline for two things: extracting structured fields
# from forecasters' text, and embeddings for semantic search. Free-tier
# limits are counted per model, so a separate lite model here keeps the
# pipeline from eating the weather-agent's daily quota.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_EXTRACTION_MODEL = "gemini-3.1-flash-lite"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768


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
