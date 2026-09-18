"""
Frontend: a small Streamlit dashboard over the gold/silver layers.

Deliberately reads only from gold (and silver, for the detail view) -
this is the "nobody points a dashboard at bronze" rule from the README
in practice. Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="Weather ETL", page_icon="\U0001F326", layout="wide")


@st.cache_resource
def get_dashboard_connection():
    # read_only=True: the dashboard should never be able to write to the warehouse.
    # (MotherDuck connections ignore this and stay read-write regardless.)
    return config.get_connection(read_only=True)


con = get_dashboard_connection()

st.title("Weather ETL Pipeline")
st.caption("Bronze -> Silver -> Gold, served straight out of DuckDB.")

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

st.subheader("Daily average temperature")
temp_pivot = gold_df.pivot(index="date", columns="city", values="temp_avg_c")
st.line_chart(temp_pivot)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Daily precipitation (mm)")
    precip_pivot = gold_df.pivot(index="date", columns="city", values="precipitation_sum_mm")
    st.bar_chart(precip_pivot)
with col2:
    st.subheader("Daily max wind speed (km/h)")
    wind_pivot = gold_df.pivot(index="date", columns="city", values="wind_speed_max_kmh")
    st.bar_chart(wind_pivot)

st.subheader("Gold: daily summary table")
st.dataframe(gold_df, use_container_width=True)

st.subheader("Next-day temperature forecast")
st.caption(
    "One scikit-learn linear regression per city, trained fresh each run on yesterday's "
    "temperature + day-of-year seasonality. A baseline to beat, not a state-of-the-art forecaster."
)

forecast_df = con.execute(
    f"""
    SELECT city, target_date, predicted_temp_avg_c, training_rows, trained_at
    FROM gold.temperature_forecast
    WHERE city IN ({placeholders})
    ORDER BY target_date DESC
    """,
    selected_cities,
).df()

if forecast_df.empty:
    st.info("No forecast yet - every city needs more accumulated settled history first.")
else:
    latest_forecast = forecast_df.sort_values("target_date").groupby("city").tail(1)
    cols = st.columns(len(latest_forecast))
    for col, (_, row) in zip(cols, latest_forecast.iterrows()):
        target_date_label = pd.Timestamp(row["target_date"]).strftime("%Y-%m-%d")
        col.metric(f"{row['city']} - {target_date_label}", f"{row['predicted_temp_avg_c']} °C")

    # Only predictions whose target_date has since become a settled actual
    # can be scored - a forecast for tomorrow has no outcome yet to compare against.
    evaluated_df = con.execute(
        f"""
        SELECT f.city, f.target_date, f.predicted_temp_avg_c, g.temp_avg_c AS actual_temp_avg_c,
               ROUND(f.predicted_temp_avg_c - g.temp_avg_c, 1) AS error_c
        FROM gold.temperature_forecast f
        JOIN gold.weather_daily_summary g ON f.city = g.city AND f.target_date = g.date
        WHERE f.target_date <= CURRENT_DATE AND f.city IN ({placeholders})
        ORDER BY f.target_date DESC
        """,
        selected_cities,
    ).df()

    if evaluated_df.empty:
        st.caption("No predictions have reached their target date yet - check back after tomorrow's run.")
    else:
        st.caption(f"Predicted vs. actual, mean absolute error: {evaluated_df['error_c'].abs().mean():.1f} °C")
        st.dataframe(evaluated_df, use_container_width=True)

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
