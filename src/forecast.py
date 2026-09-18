"""
A simple next-day temperature forecast, trained fresh on every run.

Not one of medallion's traditional three layers - this is what a lot of
real stacks add on top of gold once the basics work: a model consuming
gold data, with its own output landing back as a table other consumers
(the dashboard) can query exactly like any other gold table.

Deliberately simple: one scikit-learn LinearRegression per city, using
yesterday's average temperature and day-of-year (as sin/cos, since day 365
and day 1 are seasonally adjacent, not far apart) as the only two features.
This is meant to be a baseline worth beating, not a state-of-the-art
forecaster - and Open-Meteo's own forecast, already sitting in gold, is a
natural thing to eventually compare it against.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import get_connection

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS gold.temperature_forecast (
    city VARCHAR,
    target_date DATE,
    predicted_temp_avg_c DOUBLE,
    model_type VARCHAR,
    training_rows INTEGER,
    trained_at TIMESTAMP
);
"""

FEATURE_COLUMNS = ["lag_1_temp_c", "doy_sin", "doy_cos"]
MODEL_TYPE = "linear_regression_v1"

# Below this many training rows a fit is more coincidence than signal.
# Each daily run adds one more settled day, so cities/setups with too little
# history today will train fine on their own within a couple of months.
MIN_TRAINING_ROWS = 10


def _seasonal_features(day_of_year: int) -> tuple:
    angle = 2 * np.pi * day_of_year / 365.25
    return np.sin(angle), np.cos(angle)


def _build_features(city_history: pd.DataFrame) -> pd.DataFrame:
    city_history = city_history.sort_values("date").reset_index(drop=True)
    city_history["lag_1_temp_c"] = city_history["temp_avg_c"].shift(1)
    doy_sin, doy_cos = zip(*(_seasonal_features(d.timetuple().tm_yday) for d in city_history["date"]))
    city_history["doy_sin"] = doy_sin
    city_history["doy_cos"] = doy_cos
    return city_history


def _train_and_predict_one_city(city_history: pd.DataFrame) -> dict | None:
    city_history = _build_features(city_history)
    training_data = city_history.dropna(subset=FEATURE_COLUMNS + ["temp_avg_c"])

    if len(training_data) < MIN_TRAINING_ROWS:
        return None

    model = LinearRegression()
    model.fit(training_data[FEATURE_COLUMNS], training_data["temp_avg_c"])

    latest = city_history.iloc[-1]
    target_date = latest["date"] + timedelta(days=1)
    doy_sin, doy_cos = _seasonal_features(target_date.timetuple().tm_yday)
    next_day_features = pd.DataFrame(
        [{"lag_1_temp_c": latest["temp_avg_c"], "doy_sin": doy_sin, "doy_cos": doy_cos}]
    )
    predicted_temp_c = float(model.predict(next_day_features[FEATURE_COLUMNS])[0])

    return {
        "target_date": target_date,
        "predicted_temp_avg_c": round(predicted_temp_c, 1),
        "training_rows": len(training_data),
    }


def run(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(CREATE_TABLE)

    # Only settled days: forecast_days rows are Open-Meteo's own projection,
    # not a realized outcome yet, so training on them would just teach the
    # model to imitate Open-Meteo instead of learning from what happened.
    history = con.execute(
        "SELECT city, date, temp_avg_c FROM gold.weather_daily_summary WHERE date <= CURRENT_DATE"
    ).df()

    trained_at = datetime.now(timezone.utc).replace(tzinfo=None)
    forecasts = []
    for city in sorted(history["city"].unique()):
        city_history = history[history["city"] == city]
        result = _train_and_predict_one_city(city_history)
        if result is None:
            print(f"  forecast: skipping {city} - only {len(city_history)} settled days, need {MIN_TRAINING_ROWS}")
            continue
        forecasts.append({"city": city, "model_type": MODEL_TYPE, "trained_at": trained_at, **result})
        print(
            f"  forecast: {city} -> {result['target_date'].date()}: "
            f"{result['predicted_temp_avg_c']}C (trained on {result['training_rows']} days)"
        )

    if not forecasts:
        return 0

    forecast_df = pd.DataFrame(forecasts)[
        ["city", "target_date", "predicted_temp_avg_c", "model_type", "training_rows", "trained_at"]
    ]
    # Scoped delete, not a full rebuild: today's run only replaces today's
    # prediction-for-tomorrow. Every earlier day's forecast stays in the
    # table untouched, so predicted-vs-actual can be compared later once
    # those target dates become settled history themselves.
    con.execute(
        "DELETE FROM gold.temperature_forecast WHERE target_date IN (SELECT DISTINCT target_date FROM forecast_df)"
    )
    con.execute("INSERT INTO gold.temperature_forecast SELECT * FROM forecast_df")
    return len(forecast_df)


if __name__ == "__main__":
    con = get_connection()
    n = run(con)
    print(f"Forecast: {n} prediction(s).")
