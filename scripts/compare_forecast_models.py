"""
Compares the production per-city LinearRegression baseline against a pooled
HistGradientBoostingRegressor (one model per metric, trained on all three
cities together with city as a feature, instead of one model per city).

Read-only: reuses the backtest points already sitting in
gold.weather_forecast (run scripts/backtest.py first if that table is
empty) rather than writing anything new. Run with:
`python scripts/compare_forecast_models.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import get_connection
from src import forecast

CITY_DUMMY_PREFIX = "city"


def _pooled_history(con) -> tuple:
    """Same per-city feature build as forecast.py, concatenated into one
    dataframe with a one-hot city column so a single model per metric can
    see all three cities' history at once."""
    history = con.execute(forecast.HISTORY_QUERY).df()
    per_city = [forecast._build_features(history[history["city"] == city]) for city in sorted(history["city"].unique())]
    pooled = pd.concat(per_city, ignore_index=True)
    dummies = pd.get_dummies(pooled["city"], prefix=CITY_DUMMY_PREFIX)
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


def _gbm_predictions_for_date(pooled: pd.DataFrame, city_cols: list, target_date, cities_needed: set) -> dict:
    """Trains one HistGradientBoostingRegressor per metric on all pooled
    rows strictly before target_date (every city's history, not just one),
    then predicts the requested cities' values on target_date."""
    train = pooled[pooled["date"] < target_date]
    predict_rows = pooled[(pooled["date"] == target_date) & (pooled["city"].isin(cities_needed))]
    if predict_rows.empty:
        return {}

    predictions: dict = {}
    for metric in forecast.TARGET_COLUMNS:
        feature_cols = [f"lag_1_{metric}", "doy_sin", "doy_cos"] + city_cols
        train_slice = train.dropna(subset=feature_cols + [metric])
        if len(train_slice) < forecast.MIN_TRAINING_ROWS:
            continue
        model = HistGradientBoostingRegressor(max_depth=3, random_state=0).fit(
            train_slice[feature_cols], train_slice[metric]
        )
        preds = model.predict(predict_rows[feature_cols])
        if metric in forecast.NON_NEGATIVE_COLUMNS:
            preds = preds.clip(min=0)
        for city, pred in zip(predict_rows["city"], preds):
            predictions.setdefault(city, {})[metric] = round(float(pred), 1)
    return predictions


def run_comparison() -> None:
    con = get_connection()
    pooled, city_cols = _pooled_history(con)
    baseline = _existing_backtest_points(con)
    actuals = _actuals(con)

    if baseline.empty:
        print("No existing backtest points found - run scripts/backtest.py first.")
        return

    rows = []
    for target_date, group in baseline.groupby("target_date"):
        gbm_preds = _gbm_predictions_for_date(pooled, city_cols, target_date, set(group["city"]))
        for _, point in group.iterrows():
            city = point["city"]
            if city not in gbm_preds:
                continue
            actual_match = actuals[(actuals["city"] == city) & (actuals["date"] == target_date)]
            if actual_match.empty:
                continue
            actual_row = actual_match.iloc[0]
            for metric in forecast.TARGET_COLUMNS:
                if metric not in gbm_preds[city]:
                    continue
                rows.append(
                    {
                        "metric": metric,
                        "linear_error": abs(point[f"predicted_{metric}"] - actual_row[metric]),
                        "gbm_error": abs(gbm_preds[city][metric] - actual_row[metric]),
                    }
                )

    if not rows:
        print("No overlapping evaluation points - nothing to compare.")
        return

    errors_df = pd.DataFrame(rows)
    summary = errors_df.groupby("metric")[["linear_error", "gbm_error"]].mean().rename(
        columns={"linear_error": "linear_regression_mae", "gbm_error": "pooled_gbm_mae"}
    )
    summary["n_points"] = errors_df.groupby("metric").size()
    summary["gbm_better"] = summary["pooled_gbm_mae"] < summary["linear_regression_mae"]
    print(summary.round(2).to_string())


if __name__ == "__main__":
    run_comparison()
