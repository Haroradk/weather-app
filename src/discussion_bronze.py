"""
Bronze for unstructured text: NWS Area Forecast Discussions, landed raw.

Same principle as bronze.py - an unopinionated, append-only copy of the
source - but for free text instead of numbers. Two differences:
- The NWS API only keeps about 7 days of products, so bronze here is also
  the *archive*: once a discussion ages out of the API, this is the only
  copy. That's a real reason bronze stays append-only and is never rebuilt.
- Each product has a stable ID, so re-runs skip products already landed
  instead of storing duplicates.
"""

from datetime import datetime, timezone

import duckdb
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from config import NWS_API_URL, NWS_FORECAST_OFFICE, NWS_USER_AGENT, get_connection

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS bronze.raw_forecast_discussions (
    product_id VARCHAR,
    office VARCHAR,
    issued_at TIMESTAMP,
    fetched_at TIMESTAMP,
    source_url VARCHAR,
    raw_json VARCHAR
);
"""

_session = requests.Session()
_session.headers.update({"User-Agent": NWS_USER_AGENT, "Accept": "application/ld+json"})
_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])),
)


def _to_naive_utc(iso_timestamp: str) -> datetime:
    return datetime.fromisoformat(iso_timestamp).astimezone(timezone.utc).replace(tzinfo=None)


def run(con: duckdb.DuckDBPyConnection) -> int:
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    con.execute(CREATE_TABLE)
    already_landed = {row[0] for row in con.execute("SELECT product_id FROM bronze.raw_forecast_discussions").fetchall()}

    listing = _session.get(f"{NWS_API_URL}/products/types/AFD/locations/{NWS_FORECAST_OFFICE}", timeout=30)
    listing.raise_for_status()
    new_products = [p for p in listing.json()["@graph"] if p["id"] not in already_landed]

    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for product in new_products:
        url = f"{NWS_API_URL}/products/{product['id']}"
        response = _session.get(url, timeout=30)
        response.raise_for_status()
        rows.append(
            (product["id"], NWS_FORECAST_OFFICE, _to_naive_utc(product["issuanceTime"]), fetched_at, url, response.text)
        )

    if rows:
        # One bulk insert, not executemany: against MotherDuck every
        # statement is a network round trip (same lesson as silver.py).
        new_df = pd.DataFrame(rows, columns=["product_id", "office", "issued_at", "fetched_at", "source_url", "raw_json"])
        con.execute("INSERT INTO bronze.raw_forecast_discussions SELECT * FROM new_df")
    return len(rows)


if __name__ == "__main__":
    print(f"Bronze discussions: {run(get_connection())} new product(s) landed.")
