"""
Forecast model experiments, tracked in MLflow.

Compares the production per-city LinearRegression baseline against pooled
HistGradientBoostingRegressor variants (one model per metric, trained on
all three cities together with city as a feature). Every variant is logged
as an MLflow run - parameters, per-metric MAE, and the raw error table - so
runs can be compared side by side in the MLflow UI instead of in terminal
output that's gone after the next run.

Read-only against the warehouse: reuses the backtest points already in
gold.weather_forecast (run scripts/backtest.py first if that's empty) and
writes nothing back. Setup and usage:
    pip install -r requirements-ml.txt
    python scripts/compare_forecast_models.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # then open http://localhost:5000
"""

import sys
import tempfile
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import get_connection
from src import forecast

EXPERIMENT_NAME = "next-day-forecast"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (REPO_ROOT / "mlruns").as_uri()

FEATURE_SETS = {
    "basic": ["lag_1_{m}", "doy_sin", "doy_cos"],
    "rich": ["lag_1_{m}", "lag_2_{m}", "roll7_{m}", "doy_sin", "doy_cos"],
}
MAX_DEPTHS = [2, 3, 5]
LEARNING_RATE = 0.1


def _pooled_history(con) -> tuple:
    """Same per-city feature build as forecast.py, plus the 'rich' features,
    concatenated into one dataframe with a one-hot city column. Rich features
    are computed per city *before* pooling, and shifted so each row only sees
    days before it - no leakage from the day being predicted."""
    history = con.execute(forecast.HISTORY_QUERY).df()
    per_city = []
    for city in sorted(history["city"].unique()):
        city_df = forecast._build_features(history[history["city"] == city])
        for m in forecast.TARGET_COLUMNS:
            city_df[f"lag_2_{m}"] = city_df[m].shift(2)
            city_df[f"roll7_{m}"] = city_df[m].shift(1).rolling(7, min_periods=7).mean()
        per_city.append(city_df)
    pooled = pd.concat(per_city, ignore_index=True)
    dummies = pd.get_dummies(pooled["city"], prefix="city")
    return pd.concat([pooled, dummies], axis=1), list(dummies.columns)


def _existing_backtest_points(con) -> pd.DataFrame:
    return con.execute(
        """
        SELECT city, target_date,
               predicted_temp_min_c, predicted_temp_max_c, predicted_temp_avg_c,
               predicted_precipitation_sum_mm, predicted_wind_speed_max_kmh
        FROM gold.weather_forecast
        WHERE is_backtest AND model_type = 'linear_regression_v1'
        """
    ).df()


def _actuals(con) -> pd.DataFrame:
    return con.execute(
        "SELECT city, date, temp_min_c, temp_max_c, temp_avg_c, precipitation_sum_mm, wind_speed_max_kmh "
        "FROM gold.weather_daily_summary"
    ).df()


def _gbm_predictions_for_date(pooled, city_cols, target_date, cities_needed, feature_set, max_depth) -> dict:
    """Trains one model per metric on all pooled rows strictly before
    target_date, then predicts the requested cities on target_date."""
    train = pooled[pooled["date"] < target_date]
    predict_rows = pooled[(pooled["date"] == target_date) & (pooled["city"].isin(cities_needed))]
    if predict_rows.empty:
        return {}

    predictions: dict = {}
    for metric in forecast.TARGET_COLUMNS:
        feature_cols = [f.format(m=metric) for f in FEATURE_SETS[feature_set]] + city_cols
        # Only lag_1 and the target must be present - HistGradientBoosting
        # handles NaN in the other features natively (e.g. roll7 in week one).
        train_slice = train.dropna(subset=[f"lag_1_{metric}", metric])
        if len(train_slice) < forecast.MIN_TRAINING_ROWS:
            continue
        model = HistGradientBoostingRegressor(
            max_depth=max_depth, learning_rate=LEARNING_RATE, random_state=0
        ).fit(train_slice[feature_cols], train_slice[metric])
        preds = model.predict(predict_rows[feature_cols])
        if metric in forecast.NON_NEGATIVE_COLUMNS:
            preds = preds.clip(min=0)
        for city, pred in zip(predict_rows["city"], preds):
            predictions.setdefault(city, {})[metric] = round(float(pred), 1)
    return predictions


def _evaluate_gbm(pooled, city_cols, baseline, actuals, feature_set, max_depth) -> pd.DataFrame:
    """One row per (city, target_date, metric) with both models' absolute
    errors - on exactly the same points, so the comparison is fair."""
    actual_lookup = actuals.set_index(["city", "date"])
    rows = []
    for target_date, group in baseline.groupby("target_date"):
        gbm_preds = _gbm_predictions_for_date(pooled, city_cols, target_date, set(group["city"]), feature_set, max_depth)
        for point in group.itertuples():
            if point.city not in gbm_preds or (point.city, target_date) not in actual_lookup.index:
                continue
            actual_row = actual_lookup.loc[(point.city, target_date)]
            for metric in forecast.TARGET_COLUMNS:
                if metric not in gbm_preds[point.city]:
                    continue
                rows.append(
                    {
                        "city": point.city,
                        "target_date": target_date,
                        "metric": metric,
                        "linear_error": abs(getattr(point, f"predicted_{metric}") - actual_row[metric]),
                        "gbm_error": abs(gbm_preds[point.city][metric] - actual_row[metric]),
                    }
                )
    return pd.DataFrame(rows)


def _log_errors_artifact(errors_df: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "errors.csv"
        errors_df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def run_experiments() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT_NAME)

    con = get_connection()
    pooled, city_cols = _pooled_history(con)
    baseline = _existing_backtest_points(con)
    actuals = _actuals(con)
    if baseline.empty:
        print("No existing backtest points found - run scripts/backtest.py first.")
        return

    summary_rows = []
    baseline_logged = False
    for feature_set, max_depth in product(FEATURE_SETS, MAX_DEPTHS):
        errors_df = _evaluate_gbm(pooled, city_cols, baseline, actuals, feature_set, max_depth)
        mae = errors_df.groupby("metric")[["linear_error", "gbm_error"]].mean()

        # The baseline's predictions are fixed (already in the warehouse), so
        # it's logged once, on the same evaluation points as the GBM runs.
        if not baseline_logged:
            with mlflow.start_run(run_name="linear_per_city"):
                mlflow.log_params({"model": "LinearRegression", "pooling": "per_city", "feature_set": "basic"})
                mlflow.log_metrics({f"mae_{m}": v for m, v in mae["linear_error"].items()})
                mlflow.log_metric("n_points", len(errors_df) // len(forecast.TARGET_COLUMNS))
            summary_rows.append({"run": "linear_per_city", **mae["linear_error"].round(2).to_dict()})
            baseline_logged = True

        run_name = f"gbm_pooled_{feature_set}_depth{max_depth}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model": "HistGradientBoostingRegressor",
                    "pooling": "all_cities",
                    "feature_set": feature_set,
                    "features": ", ".join(FEATURE_SETS[feature_set]),
                    "max_depth": max_depth,
                    "learning_rate": LEARNING_RATE,
                }
            )
            mlflow.log_metrics({f"mae_{m}": v for m, v in mae["gbm_error"].items()})
            mlflow.log_metric("n_points", len(errors_df) // len(forecast.TARGET_COLUMNS))
            _log_errors_artifact(errors_df)
        summary_rows.append({"run": run_name, **mae["gbm_error"].round(2).to_dict()})
        print(f"  logged {run_name}")

    print(pd.DataFrame(summary_rows).set_index("run").to_string())
    print(f"\nView and compare runs: mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    run_experiments()
