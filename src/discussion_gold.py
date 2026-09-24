"""
Gold for the text branch: where unstructured and structured data meet.

One row per target date (New York only - the NWS covers the US): what the
human forecasters wrote, what our own ML model predicted, and what actually
happened. A view rather than a table, so it's always current as new
actuals settle and new backtest points land.

Known mismatches, documented rather than hidden:
- Our daily aggregates are UTC days; forecasters mean New York local days
  (4-5 hours apart). Fine for highs and daily rain totals, rougher at the edges.
- Our wind is the max *sustained* hourly speed; forecasters quote *gusts*,
  which run higher. The two wind columns are not like-for-like.
"""

import duckdb

from config import get_connection

BUILD_VIEW = """
CREATE OR REPLACE VIEW gold.forecaster_vs_model_vs_actual AS
WITH model AS (
    SELECT target_date, predicted_temp_max_c, predicted_temp_min_c, predicted_precipitation_sum_mm, is_backtest,
           -- a real live prediction wins over a retroactive backtest point for the same day
           ROW_NUMBER() OVER (PARTITION BY target_date ORDER BY is_backtest) AS preference
    FROM gold.weather_forecast
    WHERE city = 'New York'
)
SELECT
    'New York' AS city,
    e.target_date,
    e.issued_at AS discussion_issued_at,
    e.rain_expected AS forecaster_rain_expected,
    ROUND((e.high_temp_f - 32) * 5 / 9, 1) AS forecaster_temp_max_c,
    ROUND((e.low_temp_f - 32) * 5 / 9, 1) AS forecaster_temp_min_c,
    ROUND(e.max_wind_gust_mph * 1.609, 1) AS forecaster_wind_gust_max_kmh,
    e.hazards AS forecaster_hazards,
    e.summary AS forecaster_summary,
    e.evidence_verified,
    m.predicted_temp_max_c AS model_temp_max_c,
    m.predicted_temp_min_c AS model_temp_min_c,
    m.predicted_precipitation_sum_mm AS model_precipitation_sum_mm,
    m.is_backtest AS model_is_backtest,
    g.temp_max_c AS actual_temp_max_c,
    g.temp_min_c AS actual_temp_min_c,
    g.precipitation_sum_mm AS actual_precipitation_sum_mm,
    g.wind_speed_max_kmh AS actual_wind_speed_max_kmh,
    -- Same "rainy day" threshold as the rainy_days metric (> 1 mm). Not scored:
    -- 'possible' (a hedge, not a yes/no call) and extractions whose evidence
    -- quote couldn't be found in the source text (untrusted).
    CASE
        WHEN g.date IS NULL OR e.rain_expected = 'possible' OR NOT e.evidence_verified THEN NULL
        ELSE (e.rain_expected = 'likely') = (g.precipitation_sum_mm > 1)
    END AS forecaster_rain_correct,
    CASE
        WHEN g.date IS NULL OR m.predicted_precipitation_sum_mm IS NULL THEN NULL
        ELSE (m.predicted_precipitation_sum_mm > 1) = (g.precipitation_sum_mm > 1)
    END AS model_rain_correct
FROM silver.forecast_discussion_extractions e
LEFT JOIN model m ON m.target_date = e.target_date AND m.preference = 1
LEFT JOIN gold.weather_daily_summary g
    ON g.city = 'New York' AND g.date = e.target_date AND g.date < CURRENT_DATE
"""


def run(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(BUILD_VIEW)
    return con.execute("SELECT COUNT(*) FROM gold.forecaster_vs_model_vs_actual").fetchone()[0]


if __name__ == "__main__":
    print(f"Gold discussions: {run(get_connection())} rows.")
