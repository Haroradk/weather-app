"""
Next-day weather predictions, trained fresh on every run - plus a
walk-forward backtest to evaluate the approach against history.

Not one of medallion's traditional three layers - this is what a lot of
real stacks add on top of gold once the basics work: a model consuming
gold data, with its own output landing back as a table other consumers
(the dashboard) can query exactly like any other gold table.

Deliberately simple: one scikit-learn LinearRegression per (city, metric)
pair, using yesterday's value of that same metric and day-of-year (as
sin/cos, since day 365 and day 1 are seasonally adjacent, not far apart)
as its only two features. This is meant as a baseline worth beating, not
a state-of-the-art forecaster.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import get_connection

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS gold.weather_forecast (
    city VARCHAR,
    target_date DATE,
    predicted_temp_min_c DOUBLE,
    predicted_temp_max_c DOUBLE,
    predicted_temp_avg_c DOUBLE,
    predicted_precipitation_sum_mm DOUBLE,
    predicted_wind_speed_max_kmh DOUBLE,
    model_type VARCHAR,
    training_rows INTEGER,
    is_backtest BOOLEAN,
    trained_at TIMESTAMP
);
"""

# Each metric is predicted from its own lag-1 value + seasonality - no
# cross-metric features (e.g. wind predicting temperature). Simple to
# reason about; a real project would likely test richer feature sets.
TARGET_COLUMNS = ["temp_min_c", "temp_max_c", "temp_avg_c", "precipitation_sum_mm", "wind_speed_max_kmh"]
NON_NEGATIVE_COLUMNS = {"precipitation_sum_mm", "wind_speed_max_kmh"}
MODEL_TYPE = "linear_regression_v1"

# Below this many training rows a fit is more coincidence than signal.
# Each daily run adds one more settled day, so any city/setup with too
# little history today will train fine on its own within a couple of months.
MIN_TRAINING_ROWS = 10

HISTORY_QUERY = """
SELECT city, date, temp_min_c, temp_max_c, temp_avg_c, precipitation_sum_mm, wind_speed_max_kmh
FROM gold.weather_daily_summary
WHERE date <= CURRENT_DATE
ORDER BY date
"""


def _seasonal_features(day_of_year: int) -> tuple:
    angle = 2 * np.pi * day_of_year / 365.25
    return np.sin(angle), np.cos(angle)


def _build_features(city_history: pd.DataFrame) -> pd.DataFrame:
    city_history = city_history.sort_values("date").reset_index(drop=True)
    for metric in TARGET_COLUMNS:
        city_history[f"lag_1_{metric}"] = city_history[metric].shift(1)
    doy_sin, doy_cos = zip(*(_seasonal_features(d.timetuple().tm_yday) for d in city_history["date"]))
    city_history["doy_sin"] = doy_sin
    city_history["doy_cos"] = doy_cos
    return city_history


def _fit_metric_model(training_slice: pd.DataFrame, metric: str) -> tuple:
    """Fits one metric's model on rows *already restricted* to the desired
    training window (all settled history for a live prediction, or only
    days before the target date for a backtest point)."""
    feature_cols = [f"lag_1_{metric}", "doy_sin", "doy_cos"]
    training_data = training_slice.dropna(subset=feature_cols + [metric])
    if len(training_data) < MIN_TRAINING_ROWS:
        return None, len(training_data)
    model = LinearRegression().fit(training_data[feature_cols], training_data[metric])
    return model, len(training_data)


def _predict_metric(model: LinearRegression, metric: str, lag_1_value: float, doy_sin: float, doy_cos: float) -> float:
    feature_cols = [f"lag_1_{metric}", "doy_sin", "doy_cos"]
    features = pd.DataFrame([{feature_cols[0]: lag_1_value, "doy_sin": doy_sin, "doy_cos": doy_cos}])
    predicted = float(model.predict(features[feature_cols])[0])
    if metric in NON_NEGATIVE_COLUMNS:
        predicted = max(0.0, predicted)
    return round(predicted, 1)


def _predict_next_day_all_metrics(city_history: pd.DataFrame) -> dict | None:
    """Live prediction: train each metric's model on all settled history,
    predict tomorrow."""
    latest = city_history.iloc[-1]
    target_date = latest["date"] + timedelta(days=1)
    doy_sin, doy_cos = _seasonal_features(target_date.timetuple().tm_yday)

    predictions = {}
    training_rows_used = None
    for metric in TARGET_COLUMNS:
        model, n_rows = _fit_metric_model(city_history, metric)
        if model is None:
            return None
        predictions[metric] = _predict_metric(model, metric, latest[metric], doy_sin, doy_cos)
        training_rows_used = n_rows

    return {"target_date": target_date, "training_rows": training_rows_used, **predictions}


def _backtest_one_day_all_metrics(city_history: pd.DataFrame, day_index: int) -> dict | None:
    """Walk-forward backtest point: train each metric's model on only the
    days *before* day_index, then predict day_index and return alongside
    what actually happened - the same information a live run would have
    had, replayed against real history instead of waiting for it."""
    target_row = city_history.iloc[day_index]
    if pd.isna(target_row["doy_sin"]):
        return None  # first row has no lag - nothing to predict from

    training_slice = city_history.iloc[:day_index]
    predictions = {}
    training_rows_used = None
    for metric in TARGET_COLUMNS:
        model, n_rows = _fit_metric_model(training_slice, metric)
        if model is None:
            return None
        predictions[metric] = _predict_metric(
            model, metric, target_row[f"lag_1_{metric}"], target_row["doy_sin"], target_row["doy_cos"]
        )
        training_rows_used = n_rows

    return {"target_date": target_row["date"], "training_rows": training_rows_used, **predictions}


def run(con: duckdb.DuckDBPyConnection) -> int:
    """Live prediction: one row per city, predicting tomorrow."""
    con.execute(CREATE_TABLE)
    history = con.execute(HISTORY_QUERY).df()

    trained_at = datetime.now(timezone.utc).replace(tzinfo=None)
    forecasts = []
    for city in sorted(history["city"].unique()):
        city_history = _build_features(history[history["city"] == city])
        result = _predict_next_day_all_metrics(city_history)
        if result is None:
            print(f"  forecast: skipping {city} - only {len(city_history)} settled days, need {MIN_TRAINING_ROWS}")
            continue
        forecasts.append(_to_row(city, result, trained_at, is_backtest=False))
        print(
            f"  forecast: {city} -> {result['target_date'].date()}: "
            f"avg {result['temp_avg_c']}C, {result['precipitation_sum_mm']}mm rain, "
            f"{result['wind_speed_max_kmh']}km/h wind (trained on {result['training_rows']} days)"
        )

    if not forecasts:
        return 0

    forecast_df = pd.DataFrame(forecasts)[ROW_COLUMNS]
    # Scoped delete, not a full rebuild: today's run only replaces today's
    # prediction-for-tomorrow. Every earlier day's forecast stays in the
    # table untouched, so predicted-vs-actual can be compared later once
    # those target dates become settled history themselves.
    con.execute(
        "DELETE FROM gold.weather_forecast WHERE target_date IN (SELECT DISTINCT target_date FROM forecast_df)"
    )
    con.execute("INSERT INTO gold.weather_forecast SELECT * FROM forecast_df")
    return len(forecast_df)


def run_backtest(con: duckdb.DuckDBPyConnection) -> int:
    """
    Walk-forward backtest: for each settled day (after enough history
    precedes it), pretend we're back on the day before it, train on
    exactly what was known then, and predict it - the same thing run()
    does live, replayed retroactively against ~60 days of real history
    instead of waiting ~60 daily runs for that many evaluation points to
    accumulate. Never overwrites a real live prediction that already
    exists for a given (city, target_date).
    """
    con.execute(CREATE_TABLE)
    history = con.execute(HISTORY_QUERY).df()
    existing = set(
        map(tuple, con.execute("SELECT city, target_date FROM gold.weather_forecast").fetchall())
    )

    trained_at = datetime.now(timezone.utc).replace(tzinfo=None)
    backtest_rows = []
    for city in sorted(history["city"].unique()):
        city_history = _build_features(history[history["city"] == city])
        for day_index in range(MIN_TRAINING_ROWS, len(city_history)):
            result = _backtest_one_day_all_metrics(city_history, day_index)
            if result is None:
                continue
            target_date = result["target_date"]
            if (city, target_date.date() if hasattr(target_date, "date") else target_date) in existing:
                continue
            backtest_rows.append(_to_row(city, result, trained_at, is_backtest=True))

        n_for_city = sum(1 for r in backtest_rows if r["city"] == city)
        print(f"  backtest: {city} -> {n_for_city} new historical evaluation point(s)")

    if not backtest_rows:
        return 0

    backtest_df = pd.DataFrame(backtest_rows)[ROW_COLUMNS]
    con.execute("INSERT INTO gold.weather_forecast SELECT * FROM backtest_df")
    return len(backtest_df)


ROW_COLUMNS = [
    "city", "target_date",
    "predicted_temp_min_c", "predicted_temp_max_c", "predicted_temp_avg_c",
    "predicted_precipitation_sum_mm", "predicted_wind_speed_max_kmh",
    "model_type", "training_rows", "is_backtest", "trained_at",
]


def _to_row(city: str, result: dict, trained_at, is_backtest: bool) -> dict:
    return {
        "city": city,
        "target_date": result["target_date"],
        "predicted_temp_min_c": result["temp_min_c"],
        "predicted_temp_max_c": result["temp_max_c"],
        "predicted_temp_avg_c": result["temp_avg_c"],
        "predicted_precipitation_sum_mm": result["precipitation_sum_mm"],
        "predicted_wind_speed_max_kmh": result["wind_speed_max_kmh"],
        "model_type": MODEL_TYPE,
        "training_rows": result["training_rows"],
        "is_backtest": is_backtest,
        "trained_at": trained_at,
    }


if __name__ == "__main__":
    con = get_connection()
    n = run(con)
    print(f"Forecast: {n} prediction(s).")
