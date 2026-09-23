"""
Catalog + semantic layer step: applies semantic_layer.yml to the warehouse.

Runs after gold/forecast on every pipeline run, because gold tables are
rebuilt with CREATE OR REPLACE and that wipes any COMMENTs on them. Three
things land in MotherDuck, so every consumer reads one shared definition:
- table/column descriptions as DuckDB COMMENTs (visible via duckdb_tables()
  / duckdb_columns(), in MotherDuck's UI, and to the weather-agent)
- gold.forecast_evaluation, the predicted-vs-actual join as a view
- gold.metric_definitions, the business metrics as a queryable table
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import yaml

from config import get_connection

SEMANTIC_LAYER_PATH = Path(__file__).parent.parent / "semantic_layer.yml"

FORECAST_METRICS = ["temp_min_c", "temp_max_c", "temp_avg_c", "precipitation_sum_mm", "wind_speed_max_kmh"]

BUILD_FORECAST_EVALUATION = f"""
CREATE OR REPLACE VIEW gold.forecast_evaluation AS
SELECT f.city, f.target_date, f.is_backtest, f.model_type,
       {", ".join(f"f.predicted_{m}, g.{m} AS actual_{m}" for m in FORECAST_METRICS)}
FROM gold.weather_forecast f
JOIN gold.weather_daily_summary g ON f.city = g.city AND f.target_date = g.date
WHERE f.target_date < CURRENT_DATE
"""


def load_semantic_layer() -> dict:
    with open(SEMANTIC_LAYER_PATH) as f:
        return yaml.safe_load(f)


def _sql_string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _object_kind(con: duckdb.DuckDBPyConnection, qualified_name: str) -> str:
    schema, name = qualified_name.split(".")
    is_view = con.execute(
        "SELECT COUNT(*) FROM duckdb_views() "
        "WHERE database_name = current_database() AND schema_name = ? AND view_name = ?",
        [schema, name],
    ).fetchone()[0]
    return "VIEW" if is_view else "TABLE"


def run(con: duckdb.DuckDBPyConnection) -> int:
    layer = load_semantic_layer()

    con.execute(BUILD_FORECAST_EVALUATION)

    metrics_df = pd.DataFrame(
        [
            {
                "name": m["name"],
                "label": m["label"],
                "description": m["description"],
                "table_name": m["table"],
                "expression": m["expression"],
                "filter": m["filter"],
                "unit": m["unit"],
            }
            for m in layer["metrics"]
        ]
    )
    con.execute("CREATE OR REPLACE TABLE gold.metric_definitions AS SELECT * FROM metrics_df")

    n_comments = 0
    for table, spec in layer["tables"].items():
        con.execute(f"COMMENT ON {_object_kind(con, table)} {table} IS {_sql_string(spec['description'])}")
        n_comments += 1
        for column, description in spec["columns"].items():
            con.execute(f"COMMENT ON COLUMN {table}.{column} IS {_sql_string(description)}")
            n_comments += 1
    return n_comments


if __name__ == "__main__":
    print(f"Catalog: {run(get_connection())} descriptions applied.")
