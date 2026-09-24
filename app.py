"""
Frontend: a small Streamlit dashboard over the gold/silver layers.

Deliberately reads only from gold (and silver, for the detail view) -
this is the "nobody points a dashboard at bronze" rule from the README
in practice. Run with: streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit Community Cloud's secrets manager exposes values via st.secrets,
# not as real OS environment variables - unlike GitHub Actions, which does
# inject secrets as env vars. Bridging it into os.environ here means
# config.py only ever has to know about os.environ, and the exact same
# get_connection() works locally (.env), in CI, and on Streamlit Cloud.
# Must happen before `import config`, since it reads os.environ at import time.
try:
    for key in ("MOTHERDUCK_TOKEN", "MOTHERDUCK_DATABASE"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # no secrets.toml locally - that's fine, .env covers local dev

import config

st.set_page_config(page_title="Weather ETL", page_icon="\U0001F326", layout="wide")

# scrollZoom: drag-to-zoom is on by default in Plotly, but scroll-wheel/pinch
# zoom isn't unless enabled explicitly. displaylogo=False just hides the
# Plotly wordmark from the toolbar - every chart gets zoom, pan, box-zoom,
# and a reset-axes button via the toolbar regardless.
PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}


def line_chart(df: pd.DataFrame, x: str, y, **kwargs):
    fig = px.line(df, x=x, y=y, markers=True, **kwargs)
    fig.update_layout(legend_title_text="", hovermode="x unified", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def grouped_bar_chart(df: pd.DataFrame, x: str, y: str, color: str, **kwargs):
    # barmode="group": one bar per city side by side, not stacked into one bar.
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", **kwargs)
    fig.update_layout(legend_title_text="", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


@st.cache_resource
def get_dashboard_connection():
    # read_only=True: the dashboard should never be able to write to the warehouse.
    # (MotherDuck connections ignore this and stay read-write regardless.)
    return config.get_connection(read_only=True)


con = get_dashboard_connection()

st.title("Weather ETL Pipeline")
st.caption("Bronze -> Silver -> Gold, served straight out of DuckDB.")

LIGHTS = {"success": "🟢", "pass": "🟢", "warning": "🟡", "warn": "🟡", "failed": "🔴", "fail": "🔴"}

st.subheader("Pipeline health")
has_run_log = con.execute(
    "SELECT COUNT(*) FROM duckdb_tables() WHERE database_name = current_database() "
    "AND schema_name = 'ops' AND table_name = 'pipeline_runs'"
).fetchone()[0]
runs_df = (
    con.execute("SELECT * FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 14").df() if has_run_log else pd.DataFrame()
)
if runs_df.empty:
    st.info("No recorded pipeline runs yet - the next run will record its checks here.")
else:
    latest = runs_df.iloc[0]
    duration = (latest["finished_at"] - latest["started_at"]).total_seconds()
    st.markdown(
        f"### {LIGHTS.get(latest['status'], '⚪')} Latest run: **{latest['status']}**\n"
        f"{latest['started_at']:%Y-%m-%d %H:%M} UTC · {duration:.0f}s · triggered by {latest['triggered_by']} · "
        f"{latest['n_pass']} passed, {latest['n_warn']} warned, {latest['n_fail']} failed"
    )
    if latest["error"]:
        st.error(f"Stopped by: {latest['error']}")
    st.caption(
        "Orchestration (GitHub Actions) answers *did the job run*; these checks answer *is the data "
        "right*. 🟢 passed · 🟡 worth a look but didn't stop the run (a skipped source, data past half "
        "its freshness limit, unverified LLM evidence) · 🔴 failed and stopped the run."
    )
    checks_df = con.execute(
        "SELECT step, check_name, target, status, detail FROM ops.dq_results WHERE run_id = ? ORDER BY checked_at",
        [latest["run_id"]],
    ).df()
    checks_df.insert(0, "", checks_df["status"].map(LIGHTS))
    with st.expander(f"All {len(checks_df)} checks in the latest run", expanded=bool(latest["n_warn"] or latest["n_fail"])):
        st.dataframe(checks_df.drop(columns="status"), use_container_width=True, hide_index=True)
    with st.expander("Recent runs"):
        history_df = runs_df[["started_at", "status", "n_pass", "n_warn", "n_fail", "triggered_by", "error"]].copy()
        history_df.insert(0, "", history_df["status"].map(LIGHTS))
        st.dataframe(history_df, use_container_width=True, hide_index=True)

NODE_COLORS = {"source": "#9e9e9e", "service": "#b39ddb", "file": "#cfd8dc", "consumer": "#90caf9"}
LAYER_COLORS = {"bronze": "#cd9b6a", "silver": "#c0c4c8", "gold": "#e6c35c", "ops": "#a5d6a7"}

with st.expander("Data lineage: how everything on this page is built"):
    st.caption(
        "Declared in `semantic_layer.yml`, published to `gold.lineage_edges`, and verified "
        "against the real warehouse, view SQL, and code on every pipeline run."
    )
    edges = con.execute("SELECT upstream, downstream, upstream_type, downstream_type FROM gold.lineage_edges").fetchall()
    node_types = {}
    for upstream, downstream, upstream_type, downstream_type in edges:
        node_types[upstream] = upstream_type
        node_types[downstream] = downstream_type
    dot = ['digraph { rankdir=LR; bgcolor="transparent";',
           'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#555555"];',
           'edge [color="#888888"];']
    for node, node_type in node_types.items():
        color = LAYER_COLORS.get(node.split(".")[0], NODE_COLORS.get(node_type, "#ffffff"))
        dot.append(f'"{node}" [fillcolor="{color}"];')
    dot += [f'"{upstream}" -> "{downstream}";' for upstream, downstream, _, _ in edges]
    dot.append("}")
    st.graphviz_chart("\n".join(dot), use_container_width=True)

cities = [row[0] for row in con.execute("SELECT DISTINCT city FROM gold.weather_daily_summary ORDER BY city").fetchall()]
selected_cities = st.multiselect("Cities", cities, default=cities)

if not selected_cities:
    st.info("Pick at least one city.")
    st.stop()

placeholders = ", ".join(["?"] * len(selected_cities))

gold_df = con.execute(
    f"""
    SELECT city, date, temp_min_c, temp_max_c, temp_avg_c, precipitation_sum_mm, wind_speed_max_kmh
    FROM gold.weather_daily_summary
    WHERE city IN ({placeholders})
    ORDER BY date
    """,
    selected_cities,
).df()

st.subheader("Key metrics")
st.caption(
    "Every number here comes from a definition in `semantic_layer.yml` (published to "
    "`gold.metric_definitions`) - the same definitions the weather-agent reads. Hover a "
    "column header for what it means."
)
metric_defs = con.execute(
    "SELECT name, label, description, table_name, expression, filter, unit FROM gold.metric_definitions"
).df()
metric_values = pd.DataFrame(index=selected_cities)
column_config = {}
for m in metric_defs.itertuples():
    header = f"{m.label} ({m.unit})"
    rows = con.execute(
        f"SELECT city, {m.expression} AS value FROM {m.table_name} "
        f"WHERE ({m.filter}) AND city IN ({placeholders}) GROUP BY city",
        selected_cities,
    ).fetchall()
    metric_values[header] = pd.Series(dict(rows))
    column_config[header] = st.column_config.NumberColumn(header, help=m.description)
st.dataframe(metric_values, use_container_width=True, column_config=column_config)

st.subheader("Daily average temperature")
line_chart(gold_df, x="date", y="temp_avg_c", color="city")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Daily precipitation (mm)")
    grouped_bar_chart(gold_df, x="date", y="precipitation_sum_mm", color="city")
with col2:
    st.subheader("Daily max wind speed (km/h)")
    grouped_bar_chart(gold_df, x="date", y="wind_speed_max_kmh", color="city")

st.subheader("Gold: daily summary table")
st.dataframe(gold_df, use_container_width=True)

st.subheader("Next-day weather forecast")
st.caption(
    "One scikit-learn linear regression per city per metric, trained fresh each run on "
    "yesterday's value of that same metric + day-of-year seasonality. A baseline to beat, "
    "not a state-of-the-art forecaster."
)

forecast_df = con.execute(
    f"""
    SELECT city, target_date, predicted_temp_min_c, predicted_temp_max_c, predicted_temp_avg_c,
           predicted_precipitation_sum_mm, predicted_wind_speed_max_kmh, training_rows
    FROM gold.weather_forecast
    WHERE city IN ({placeholders}) AND NOT is_backtest
    ORDER BY target_date DESC
    """,
    selected_cities,
).df()

if forecast_df.empty:
    st.info("No live forecast yet - every city needs more accumulated settled history first.")
else:
    latest_forecast = forecast_df.sort_values("target_date").groupby("city").tail(1)
    st.dataframe(latest_forecast, use_container_width=True)

METRICS = {
    "temp_min_c": "Min temperature (°C)",
    "temp_max_c": "Max temperature (°C)",
    "temp_avg_c": "Avg temperature (°C)",
    "precipitation_sum_mm": "Precipitation (mm)",
    "wind_speed_max_kmh": "Max wind speed (km/h)",
}

st.subheader("Forecast accuracy: predicted vs. actual")
st.caption(
    "Includes both live daily predictions and a walk-forward backtest (run "
    "`python scripts/backtest.py` to (re)fill in historical evaluation points from "
    "existing settled history, without touching real live predictions)."
)

evaluated_df = con.execute(
    f"SELECT * FROM gold.forecast_evaluation WHERE city IN ({placeholders}) ORDER BY target_date",
    selected_cities,
).df()

if evaluated_df.empty:
    st.info("No evaluated predictions yet - run `python scripts/backtest.py`, or check back after tomorrow's run.")
else:
    n_backtest = int(evaluated_df["is_backtest"].sum())
    st.caption(f"{len(evaluated_df)} evaluated predictions ({n_backtest} backtested, {len(evaluated_df) - n_backtest} live).")

    eval_city = st.selectbox("City", selected_cities, key="eval_city")
    city_eval_df = evaluated_df[evaluated_df["city"] == eval_city].set_index("target_date")

    tabs = st.tabs(list(METRICS.values()))
    for tab, (metric, label) in zip(tabs, METRICS.items()):
        with tab:
            pair_df = city_eval_df[[f"predicted_{metric}", f"actual_{metric}"]].dropna().reset_index()
            if pair_df.empty:
                st.caption("No evaluated predictions for this metric yet.")
            else:
                mae = (pair_df[f"predicted_{metric}"] - pair_df[f"actual_{metric}"]).abs().mean()
                st.caption(f"Mean absolute error: {mae:.2f}")
                line_chart(pair_df, x="target_date", y=[f"predicted_{metric}", f"actual_{metric}"])

st.subheader("Forecasters vs. our model vs. reality (New York)")
st.caption(
    "Unstructured data meeting structured data. National Weather Service forecasters write a "
    "free-text Area Forecast Discussion several times a day; the pipeline lands it raw, splits it "
    "into sections, and has an LLM extract the next day's forecast into typed fields - then this "
    "compares it with our ML model and with what actually happened. 'possible' rain isn't scored, "
    "and neither is any extraction whose supporting quote couldn't be found in the source text."
)
comparison_df = con.execute(
    """
    SELECT target_date, forecaster_rain_expected, model_precipitation_sum_mm, actual_precipitation_sum_mm,
           forecaster_rain_correct, model_rain_correct, forecaster_summary, evidence_verified
    FROM gold.forecaster_vs_model_vs_actual
    ORDER BY target_date DESC
    """
).df()
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

with st.expander("Silver: raw hourly readings"):
    hourly_df = con.execute(
        f"""
        SELECT city, observation_time, temperature_c, precipitation_mm, wind_speed_kmh, humidity_pct
        FROM silver.weather_hourly
        WHERE city IN ({placeholders})
        ORDER BY observation_time
        """,
        selected_cities,
    ).df()
    st.dataframe(hourly_df, use_container_width=True)

with st.expander("Bronze: fetch log"):
    fetch_log_df = con.execute(
        f"""
        SELECT city, fetched_at, source_url, LENGTH(raw_json) AS response_bytes
        FROM bronze.raw_weather_observations
        WHERE city IN ({placeholders})
        ORDER BY fetched_at DESC
        """,
        selected_cities,
    ).df()
    st.dataframe(fetch_log_df, use_container_width=True)
