"""
Orchestrator: bronze -> dq -> silver -> dq -> gold -> dq.

This is what a scheduler (cron, GitHub Actions, Airflow) would call once
a day. Each layer is idempotent, so re-running this after a failure never
corrupts data - it just redoes the work.
"""

import duckdb

from config import DB_PATH
from src import bronze, dq, gold, silver


def main() -> None:
    con = duckdb.connect(DB_PATH)

    print("Bronze: fetching raw weather data...")
    landed = bronze.run(con)
    dq.check_not_empty(con, "bronze.raw_weather_observations")
    print(f"Bronze done: {landed} responses landed.\n")

    print("Silver: parsing and deduplicating...")
    loaded = silver.run(con)
    dq.check_not_empty(con, "silver.weather_hourly")
    dq.check_no_nulls(con, "silver.weather_hourly", ["city", "observation_time", "temperature_c"])
    dq.check_temperature_range(con, "silver.weather_hourly", "temperature_c")
    print(f"Silver done: {loaded} hourly rows.\n")

    print("Gold: building daily summary...")
    summarized = gold.run(con)
    dq.check_not_empty(con, "gold.weather_daily_summary")
    print(f"Gold done: {summarized} daily rows.\n")

    print("Pipeline complete.")
    con.close()


if __name__ == "__main__":
    main()
