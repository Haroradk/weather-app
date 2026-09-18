"""
Gold layer: business-ready aggregates.

Medallion principle: gold answers a question someone actually has
("how hot was it each day in each city?") rather than mirroring the
source shape. Gold tables are typically small, heavily aggregated, and
what dashboards/BI tools query directly — nobody should point a
dashboard at bronze or silver.
"""

import duckdb

from config import DB_PATH

CREATE_SCHEMA = "CREATE SCHEMA IF NOT EXISTS gold;"

BUILD_DAILY_SUMMARY = """
CREATE OR REPLACE TABLE gold.weather_daily_summary AS
SELECT
    city,
    CAST(observation_time AS DATE) AS date,
    ROUND(MIN(temperature_c), 1) AS temp_min_c,
    ROUND(MAX(temperature_c), 1) AS temp_max_c,
    ROUND(AVG(temperature_c), 1) AS temp_avg_c,
    ROUND(SUM(precipitation_mm), 1) AS precipitation_sum_mm,
    ROUND(MAX(wind_speed_kmh), 1) AS wind_speed_max_kmh,
    now() AS updated_at
FROM silver.weather_hourly
GROUP BY city, CAST(observation_time AS DATE)
ORDER BY city, date;
"""


def run(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(CREATE_SCHEMA)
    con.execute(BUILD_DAILY_SUMMARY)
    return con.execute("SELECT COUNT(*) FROM gold.weather_daily_summary").fetchone()[0]


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    row_count = run(con)
    print(f"Gold: {row_count} daily summary rows.")
