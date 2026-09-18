"""
Orchestrator: bronze -> dq -> silver -> dq -> gold -> dq.

This is what a scheduler (cron, GitHub Actions, Airflow) would call once
a day. Each layer is idempotent, so re-running this after a failure never
corrupts data - it just redoes the work.
"""

from datetime import datetime, timezone

from config import CITIES, get_connection
from src import bronze, dq, forecast, gold, silver

CITY_NAMES = [city["name"] for city in CITIES]

# Pipeline runs daily; 30h gives slack for scheduler jitter (GitHub Actions
# cron can slip by tens of minutes under load) without masking a real gap.
MAX_STALENESS_HOURS = 30


def main() -> None:
    con = get_connection()
    # Naive-but-UTC, matching how bronze.fetched_at is stored (see bronze.py).
    run_started_at = datetime.now(timezone.utc).replace(tzinfo=None)

    print("Bronze: fetching raw weather data...")
    landed = bronze.run(con)
    dq.check_not_empty(con, "bronze.raw_weather_observations")
    # Scoped to *this run*: catches one city silently landing zero rows
    # even though the bronze table overall is non-empty from past runs.
    dq.check_row_count_per_group(
        con, "bronze.raw_weather_observations", "city", CITY_NAMES,
        where_sql="fetched_at >= ?", where_params=[run_started_at],
    )
    dq.check_freshness(con, "bronze.raw_weather_observations", "fetched_at", MAX_STALENESS_HOURS)
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
    # Every city should have today's aggregate - catches a city missing
    # from today's forecast window even though gold overall isn't empty.
    dq.check_row_count_per_group(
        con, "gold.weather_daily_summary", "city", CITY_NAMES,
        where_sql="date = CURRENT_DATE",
    )
    print(f"Gold done: {summarized} daily rows.\n")

    print("Forecast: training and predicting tomorrow's weather...")
    predicted = forecast.run(con)
    if predicted:
        dq.check_not_empty(con, "gold.weather_forecast")
        dq.check_no_nulls(con, "gold.weather_forecast", [
            "city", "target_date", "predicted_temp_min_c", "predicted_temp_max_c",
            "predicted_temp_avg_c", "predicted_precipitation_sum_mm", "predicted_wind_speed_max_kmh",
        ])
        dq.check_temperature_range(con, "gold.weather_forecast", "predicted_temp_min_c")
        dq.check_temperature_range(con, "gold.weather_forecast", "predicted_temp_max_c")
        dq.check_temperature_range(con, "gold.weather_forecast", "predicted_temp_avg_c")
        dq.check_non_negative(con, "gold.weather_forecast", "predicted_precipitation_sum_mm")
        dq.check_non_negative(con, "gold.weather_forecast", "predicted_wind_speed_max_kmh")
    else:
        # Not a failure: every city just needs more accumulated settled
        # history than exists yet (see forecast.MIN_TRAINING_ROWS).
        print("  forecast: no predictions yet - not enough settled history for any city")
    print(f"Forecast done: {predicted} prediction(s).\n")

    print("Pipeline complete.")
    con.close()


if __name__ == "__main__":
    main()
