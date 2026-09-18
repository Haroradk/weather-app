"""
Frontend: a small Streamlit dashboard over the gold/silver layers.

Deliberately reads only from gold (and silver, for the detail view) -
this is the "nobody points a dashboard at bronze" rule from the README
in practice. Run with: streamlit run app.py
"""

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
