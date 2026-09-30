"""
Frontend: a Streamlit dashboard over the gold/silver layers.

Deliberately reads only from gold (and silver/bronze, for the raw-data view) -
the "nobody points a dashboard at bronze" rule from the README, with raw
layers shown for inspection only. Every table read here is declared as the
dashboard's upstream in semantic_layer.yml; dq.check_lineage fails the
pipeline if this file starts reading anything else.
Run with: streamlit run app.py
"""

import os
import re

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
from src import catalog, catalog_search

st.set_page_config(page_title="Weather data platform", page_icon="\U0001F326", layout="wide")
# Streamlit has no size option for tab labels, so a little CSS bumps them up.
st.markdown("<style>button[data-baseweb='tab'] p {font-size: 1.1rem;}</style>", unsafe_allow_html=True)
# Keeps the catalog's detail panel in view while scrolling through search
# results, like the side panel of a real data catalog.
st.markdown(
    "<style>[data-testid='stColumn']:has(.st-key-catalog_detail) {position: sticky; top: 4rem; align-self: flex-start; "
    "max-height: calc(100vh - 5rem); overflow-y: auto;}</style>",
    unsafe_allow_html=True,
)

# scrollZoom: drag-to-zoom is on by default in Plotly, but scroll-wheel/pinch
# zoom isn't unless enabled explicitly. displaylogo=False just hides the
# Plotly wordmark from the toolbar.
PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}
# Immeo palette; bordeaux is a brand colour reserved for charts.
CITY_COLORS = {"Copenhagen": "#00412D", "London": "#9BCDA0", "New York": "#4B1932"}
LIGHTS = {"success": "🟢", "pass": "🟢", "warning": "🟡", "warn": "🟡", "failed": "🔴", "fail": "🔴"}


def _style(fig, height=340):
    fig.update_layout(legend_title_text="", margin=dict(t=10, b=10, l=10, r=10), height=height,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def line_chart(df: pd.DataFrame, x: str, y, **kwargs):
    fig = px.line(df, x=x, y=y, markers=True, **kwargs)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(_style(fig), use_container_width=True, config=PLOTLY_CONFIG)


def grouped_bar_chart(df: pd.DataFrame, x: str, y: str, color: str, **kwargs):
    # barmode="group": one bar per city side by side, not stacked into one bar.
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", **kwargs)
    st.plotly_chart(_style(fig), use_container_width=True, config=PLOTLY_CONFIG)


@st.cache_resource
def get_dashboard_connection():
    # read_only=True: the dashboard should never be able to write to the warehouse.
    # (MotherDuck connections ignore this and stay read-write regardless.)
    return config.get_connection(read_only=True)


con = get_dashboard_connection()

# ---------------------------------------------------------------- sidebar
cities = [row[0] for row in con.execute("SELECT DISTINCT city FROM gold.weather_daily_summary ORDER BY city").fetchall()]
with st.sidebar:
    st.header("Filters")
    selected_cities = st.multiselect("Cities", cities, default=cities)
    st.divider()
    st.caption(
        "Weather data from Open-Meteo and forecaster texts from the US National Weather Service, "
        "run through bronze, silver and gold in MotherDuck every morning around 05:17 UTC."
    )

if not selected_cities:
    st.info("Pick at least one city in the sidebar.")
    st.stop()
placeholders = ", ".join(["?"] * len(selected_cities))

# ---------------------------------------------------------------- header + status strip
st.title("Weather data platform")

has_run_log = con.execute(
    "SELECT COUNT(*) FROM duckdb_tables() WHERE database_name = current_database() "
    "AND schema_name = 'ops' AND table_name = 'pipeline_runs'"
).fetchone()[0]
runs_df = con.execute("SELECT * FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 14").df() if has_run_log else pd.DataFrame()
latest_settled = con.execute("SELECT MAX(date) FROM gold.weather_daily_summary WHERE date <= CURRENT_DATE").fetchone()[0]

with st.container(border=True):
    if runs_df.empty:
        st.markdown(f"⚪ No recorded pipeline runs yet · data up to **{latest_settled:%d %b %Y}**")
    else:
        latest = runs_df.iloc[0]
        st.markdown(
            f"{LIGHTS.get(latest['status'], '⚪')} Latest run **{latest['status']}** on "
            f"{latest['started_at']:%d %b %H:%M} UTC · {latest['n_pass']} checks passed, "
            f"{latest['n_warn']} warned, {latest['n_fail']} failed · data up to **{latest_settled:%d %b %Y}**"
        )
        if latest["error"]:
            st.error(f"The latest run stopped: {latest['error']}")

# ---------------------------------------------------------------- catalog search helpers
KIND_ICONS = {
    "Table": ":material/table_chart:", "View": ":material/view_agenda:", "Metric": ":material/functions:",
    "Source": ":material/cloud_download:", "Service": ":material/smart_toy:", "Consumer": ":material/dashboard:",
    "File": ":material/description:",
}
LAYER_BADGES = {"gold": "green", "silver": "gray", "bronze": "orange", "ops": "blue", "semantic layer": "violet", "outside the warehouse": "gray"}
CATALOG_SUGGESTIONS = ["rain", "wind gust", "forecaster text", "data quality", "embedding"]
MAX_CATALOG_RESULTS = 8


@st.cache_data(ttl=600)
def load_catalog():
    """Everything searchable, from the published catalog - see src/catalog_search.py."""
    layer = catalog.load_semantic_layer()
    return catalog_search.build_index(con, layer), layer.get("glossary", {})


@st.cache_data(ttl=600)
def row_count(table_id: str) -> int:
    # table_id comes from the warehouse's own object list, never from what the user typed.
    return con.execute(f"SELECT COUNT(*) FROM {table_id}").fetchone()[0]


@st.cache_data(ttl=600)
def preview_rows(table_id: str, column_names: tuple) -> pd.DataFrame:
    select_list = ", ".join('"' + c + '"' for c in column_names)
    df = con.execute(f"SELECT {select_list} FROM {table_id} LIMIT 5").df()
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(lambda v: v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
    return df


def use_catalog_suggestion() -> None:
    st.session_state.catalog_query = st.session_state.catalog_suggestion or ""
    st.session_state.catalog_suggestion = None
    clear_catalog_selection()


def clear_catalog_selection() -> None:
    st.session_state.pop("catalog_selected", None)


def select_catalog_asset(asset_id: str) -> None:
    st.session_state.catalog_selected = asset_id


def asset_badges(asset: dict) -> str:
    badges = f":{LAYER_BADGES.get(asset['layer'], 'gray')}-badge[{asset['layer']}] :gray-badge[{asset['kind']}]"
    if not asset["description"] and asset["kind"] in ("Table", "View"):
        badges += " :red-badge[undocumented]"
    return badges


def snippet(text: str, limit: int = 180) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def highlight(text: str, words: list) -> str:
    """Bold every word that starts with a searched word (or a glossary synonym of one)."""
    if not words:
        return text
    pattern = "|".join(re.escape(w) for w in words)
    return re.sub(rf"(?<![A-Za-z0-9_])((?:{pattern})[A-Za-z0-9]*)", r"**\1**", text, flags=re.IGNORECASE)


def _asset_buttons(label: str, asset_ids: list, assets: dict, key: str, on_select) -> None:
    if not asset_ids:
        return
    st.markdown(f"**{label}**")
    columns = st.columns(2)
    for i, asset_id in enumerate(asset_ids):
        name = assets[asset_id]["name"] if asset_id in assets else asset_id
        columns[i % 2].button(name, key=f"{key}_{asset_id}", on_click=on_select, args=(asset_id,),
                              disabled=asset_id not in assets, width="stretch")


def render_catalog_asset(asset: dict, matched: list, on_select=select_catalog_asset, key_prefix: str = "") -> None:
    """One asset's catalog page. on_select is what its lineage buttons do; key_prefix
    keeps widget keys apart when the same asset is also open in the catalog tab."""
    assets, _ = load_catalog()
    with st.container(border=True):
        st.markdown(f"#### {KIND_ICONS[asset['kind']]} {asset['name']}")
        st.markdown(asset_badges(asset))
        st.write(asset["description"] or "No description yet - add one in semantic_layer.yml.")

        if asset["kind"] in ("Table", "View"):
            rows_tile, columns_tile = st.columns(2)
            rows_tile.metric("Rows", f"{row_count(asset['id']):,}", border=True)
            columns_tile.metric("Columns", len(asset["columns"]), border=True)
        if asset.get("built_by"):
            st.markdown(f"**Built by** `{asset['built_by']}`")
        if asset.get("metric"):
            metric = asset["metric"]
            st.markdown(f"**Unit** {metric['unit']}")
            st.code(f"SELECT {metric['expression']}\nFROM {metric['table']}\nWHERE {metric['filter']}", language="sql")

        _asset_buttons("Comes from", asset["upstream"], assets, f"{key_prefix}up_{asset['id']}", on_select)
        _asset_buttons("Feeds into", asset["downstream"], assets, f"{key_prefix}down_{asset['id']}", on_select)
        _asset_buttons("Metrics computed from it", asset.get("metrics", []), assets, f"{key_prefix}metric_{asset['id']}", on_select)

        if asset["columns"]:
            st.markdown("**Columns**")
            columns_df = pd.DataFrame(asset["columns"])
            columns_df.insert(0, "match", columns_df["name"].map(lambda c: "\u2714" if c in matched else ""))
            st.dataframe(
                columns_df, hide_index=True, width="stretch",
                column_config={"match": st.column_config.TextColumn("", width="small"), "name": "Column", "type": "Type",
                               "description": st.column_config.TextColumn("Description", width="large")},
            )
            # Embeddings and raw API payloads are thousands of characters of noise in a preview.
            previewable = tuple(c["name"] for c in asset["columns"] if c["name"] != "raw_json" and not re.search(r"\[\d+\]", c["type"]))
            with st.expander("Preview 5 rows"):
                st.dataframe(preview_rows(asset["id"], previewable), hide_index=True, width="stretch")


def select_dialog_asset(asset_id: str) -> None:
    st.session_state.dialog_asset = asset_id


@st.dialog("Data catalog", width="large")
def catalog_dialog() -> None:
    """A metric tile's catalog page as a pop-up. Streamlit can't switch tabs from
    code, so this brings the catalog to the tile instead. Lineage buttons inside
    it only rerun the dialog, so you can walk from the metric to its tables."""
    assets, _ = load_catalog()
    asset_id = st.session_state.get("dialog_asset")
    if asset_id in assets:
        render_catalog_asset(assets[asset_id], [], on_select=select_dialog_asset, key_prefix="dialog_")
    st.caption("The full searchable catalog is on the Data & governance tab.")


def catalog_link(column, metric_name: str, label: str = "In catalog") -> None:
    if column.button(label, key=f"catalog_link_{metric_name}", icon=":material/menu_book:", type="tertiary",
                     help="Open this metric's definition, SQL and lineage in the data catalog"):
        st.session_state.dialog_asset = f"metric:{metric_name}"
        catalog_dialog()


tab_weather, tab_ml, tab_pipeline, tab_data = st.tabs(
    ["\U0001F326 **Weather**", "\U0001F4C8 **Forecasts & ML**", "\u2699\ufe0f **Pipeline**", "\U0001F5C2 **Data & governance**"]
)

metric_defs = con.execute(
    "SELECT name, label, description, table_name, expression, filter, unit FROM gold.metric_definitions"
).df()


def metric_values(names: list) -> dict:
    """{metric name: {city: value}}, computed from the semantic layer's own definitions."""
    out = {}
    for m in metric_defs[metric_defs["name"].isin(names)].itertuples():
        rows = con.execute(
            f"SELECT city, {m.expression} AS value FROM {m.table_name} "
            f"WHERE ({m.filter}) AND city IN ({placeholders}) GROUP BY city",
            selected_cities,
        ).fetchall()
        out[m.name] = dict(rows)
    return out


def metric_tile(column, name: str, value, fmt: str = "{:g}"):
    m = metric_defs.set_index("name").loc[name]
    shown = "–" if value is None or pd.isna(value) else f"{fmt.format(value)} {m.unit}"
    column.metric(m.label, shown, help=m.description, border=True)


# ---------------------------------------------------------------- Weather
with tab_weather:
    gold_df = con.execute(
        f"""
        SELECT city, date, temp_min_c, temp_max_c, temp_avg_c, precipitation_sum_mm, wind_speed_max_kmh
        FROM gold.weather_daily_summary
        WHERE city IN ({placeholders}) AND date <= CURRENT_DATE
        ORDER BY date
        """,
        selected_cities,
    ).df()
    first_day, last_day = gold_df["date"].min(), gold_df["date"].max()
    st.caption(
        f"Settled history {first_day:%d %b} to {last_day:%d %b %Y}. Every number comes from a definition in "
        "the semantic layer (gold.metric_definitions), the same ones the weather agent uses. Hover a tile for its definition, or open it in the data catalog."
    )
    weather_metrics = ["avg_temperature", "rainy_days", "total_precipitation", "max_wind_speed"]
    values = metric_values(weather_metrics)
    for col, name in zip(st.columns(len(weather_metrics)), weather_metrics):
        catalog_link(col, name, f"{metric_defs.set_index('name').loc[name].label} in catalog")
    for city in selected_cities:
        st.markdown(f"**{city}**")
        for col, name in zip(st.columns(len(weather_metrics)), weather_metrics):
            metric_tile(col, name, values[name].get(city))

    st.subheader("Daily average temperature (°C)")
    line_chart(gold_df, x="date", y="temp_avg_c", color="city", color_discrete_map=CITY_COLORS,
               labels={"date": "", "temp_avg_c": "°C"})
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily precipitation (mm)")
        grouped_bar_chart(gold_df, x="date", y="precipitation_sum_mm", color="city", color_discrete_map=CITY_COLORS,
                          labels={"date": "", "precipitation_sum_mm": "mm"})
    with col2:
        st.subheader("Daily max wind speed (km/h)")
        grouped_bar_chart(gold_df, x="date", y="wind_speed_max_kmh", color="city", color_discrete_map=CITY_COLORS,
                          labels={"date": "", "wind_speed_max_kmh": "km/h"})

    with st.expander("Daily summary table"):
        st.dataframe(
            gold_df.sort_values(["date", "city"], ascending=[False, True]),
            use_container_width=True, hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Date", format="D MMM YYYY"),
                "city": "City",
                "temp_min_c": st.column_config.NumberColumn("Min °C", format="%.1f"),
                "temp_max_c": st.column_config.NumberColumn("Max °C", format="%.1f"),
                "temp_avg_c": st.column_config.NumberColumn("Avg °C", format="%.1f"),
                "precipitation_sum_mm": st.column_config.NumberColumn("Rain mm", format="%.1f"),
                "wind_speed_max_kmh": st.column_config.NumberColumn("Max wind km/h", format="%.1f"),
            },
        )

# ---------------------------------------------------------------- Forecasts & ML
with tab_ml:
    st.subheader("Tomorrow, according to our model")
    st.caption(
        "One scikit-learn linear regression per city and metric, retrained every morning on yesterday's "
        "value plus the season. A baseline to learn from, not a serious forecaster."
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
        latest_forecast = forecast_df.sort_values("target_date").groupby("city").tail(1).set_index("city")
        for city in selected_cities:
            if city not in latest_forecast.index:
                continue
            f = latest_forecast.loc[city]
            st.markdown(f"**{city}**, {f['target_date']:%A %d %b}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Low / high", f"{f['predicted_temp_min_c']:.1f} / {f['predicted_temp_max_c']:.1f} °C", border=True)
            c2.metric("Average", f"{f['predicted_temp_avg_c']:.1f} °C", border=True)
            c3.metric("Rain", f"{f['predicted_precipitation_sum_mm']:.1f} mm", border=True)
            c4.metric("Max wind", f"{f['predicted_wind_speed_max_kmh']:.1f} km/h", border=True,
                      help=f"Trained on {int(f['training_rows'])} settled days.")

    st.divider()
    st.subheader("How accurate has it been?")
    evaluated_df = con.execute(
        f"SELECT * FROM gold.forecast_evaluation WHERE city IN ({placeholders}) ORDER BY target_date",
        selected_cities,
    ).df()
    if evaluated_df.empty:
        st.info("No evaluated predictions yet - run `python scripts/backtest.py`, or check back after tomorrow's run.")
    else:
        n_backtest = int(evaluated_df["is_backtest"].sum())
        st.caption(
            f"{len(evaluated_df)} predictions compared with what actually happened: {n_backtest} from a walk-forward "
            f"backtest (trained only on days before each test day) and {len(evaluated_df) - n_backtest} live."
        )
        METRICS = {
            "temp_avg_c": "Avg temperature (°C)",
            "temp_min_c": "Min temperature (°C)",
            "temp_max_c": "Max temperature (°C)",
            "precipitation_sum_mm": "Precipitation (mm)",
            "wind_speed_max_kmh": "Max wind speed (km/h)",
        }
        c_city, c_metric = st.columns(2)
        eval_city = c_city.selectbox("City", selected_cities, key="eval_city")
        metric = c_metric.selectbox("Metric", list(METRICS), format_func=METRICS.get, key="eval_metric")
        pair_df = (
            evaluated_df[evaluated_df["city"] == eval_city][["target_date", f"predicted_{metric}", f"actual_{metric}"]]
            .dropna()
            .rename(columns={f"predicted_{metric}": "Predicted", f"actual_{metric}": "Actual"})
        )
        if pair_df.empty:
            st.caption("No evaluated predictions for this metric yet.")
        else:
            mae = (pair_df["Predicted"] - pair_df["Actual"]).abs().mean()
            st.metric("Mean absolute error", f"{mae:.2f}", border=True,
                      help="Average distance between prediction and reality, in the metric's own unit. Lower is better.")
            line_chart(pair_df, x="target_date", y=["Predicted", "Actual"],
                       color_discrete_sequence=["#9BCDA0", "#00412D"], labels={"target_date": "", "value": METRICS[metric]})

    st.divider()
    st.subheader("Forecasters vs. our model vs. reality (New York)")
    st.caption(
        "National Weather Service forecasters write a free-text discussion several times a day. The pipeline "
        "keeps it raw in bronze, and Gemini extracts the next day's rain call into fields. 'Possible' isn't scored, "
        "and neither is an extraction whose supporting quote couldn't be found in the source text."
    )
    hit = metric_values(["forecaster_rain_hit_rate", "model_rain_hit_rate"])
    h1, h2, _ = st.columns([1, 1, 2])
    metric_tile(h1, "forecaster_rain_hit_rate", hit["forecaster_rain_hit_rate"].get("New York"), "{:.0%}")
    metric_tile(h2, "model_rain_hit_rate", hit["model_rain_hit_rate"].get("New York"), "{:.0%}")
    catalog_link(h1, "forecaster_rain_hit_rate")
    catalog_link(h2, "model_rain_hit_rate")
    comparison_df = con.execute(
        """
        SELECT target_date, forecaster_rain_expected, model_precipitation_sum_mm, actual_precipitation_sum_mm,
               forecaster_rain_correct, model_rain_correct, forecaster_summary
        FROM gold.forecaster_vs_model_vs_actual
        ORDER BY target_date DESC
        """
    ).df()
    st.dataframe(
        comparison_df, use_container_width=True, hide_index=True,
        column_config={
            "target_date": st.column_config.DateColumn("Day", format="ddd D MMM"),
            "forecaster_rain_expected": "Forecasters said",
            "model_precipitation_sum_mm": st.column_config.NumberColumn("Model rain mm", format="%.1f"),
            "actual_precipitation_sum_mm": st.column_config.NumberColumn("Actual rain mm", format="%.1f"),
            "forecaster_rain_correct": st.column_config.CheckboxColumn("Forecasters right"),
            "model_rain_correct": st.column_config.CheckboxColumn("Model right"),
            "forecaster_summary": st.column_config.TextColumn("What the forecasters wrote (LLM summary)", width="large"),
        },
    )

# ---------------------------------------------------------------- Pipeline
with tab_pipeline:
    if runs_df.empty:
        st.info("No recorded pipeline runs yet - the next run will record its checks here.")
    else:
        latest = runs_df.iloc[0]
        duration = (latest["finished_at"] - latest["started_at"]).total_seconds()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest run", f"{LIGHTS.get(latest['status'], '⚪')} {latest['status']}", border=True)
        c2.metric("Checks passed", f"{latest['n_pass']}", border=True)
        c3.metric("Warnings", f"{latest['n_warn']}", border=True)
        c4.metric("Duration", f"{duration:.0f} s", border=True, help=f"Triggered by {latest['triggered_by']}.")
        st.caption(
            "Orchestration (GitHub Actions) answers *did the job run*. These checks answer *is the data right*. "
            "🟢 passed · 🟡 worth a look, but didn't stop the run (a skipped source, data past half its freshness "
            "limit, unverified LLM evidence) · 🔴 failed and stopped the run."
        )
        checks_df = con.execute(
            "SELECT status, step, check_name, target, detail FROM ops.dq_results WHERE run_id = ? ORDER BY checked_at",
            [latest["run_id"]],
        ).df()
        checks_df["status"] = checks_df["status"].map(LIGHTS)
        st.subheader(f"All {len(checks_df)} checks in the latest run")
        st.dataframe(
            checks_df, use_container_width=True, hide_index=True,
            column_config={"status": st.column_config.TextColumn("", width="small"), "step": "Step",
                           "check_name": "Check", "target": "Table", "detail": st.column_config.TextColumn("Result", width="large")},
        )
        st.subheader("Recent runs")
        history_df = runs_df[["status", "started_at", "n_pass", "n_warn", "n_fail", "triggered_by", "error"]].copy()
        history_df["status"] = history_df["status"].map(LIGHTS) + " " + runs_df["status"]
        st.dataframe(
            history_df, use_container_width=True, hide_index=True,
            column_config={"status": "Status", "started_at": st.column_config.DatetimeColumn("Started (UTC)", format="D MMM HH:mm"),
                           "n_pass": "Passed", "n_warn": "Warned", "n_fail": "Failed", "triggered_by": "Triggered by",
                           "error": "Error"},
        )

# ---------------------------------------------------------------- Data & governance
with tab_data:
    st.subheader("Find data")
    st.caption(
        "Search every table, column, metric and source in the catalog, bronze to gold. Everyday words work too: "
        "the business glossary in semantic_layer.yml maps e.g. \u201crain\u201d to \u201cprecipitation\u201d."
    )
    assets, glossary = load_catalog()
    query = st.text_input(
        "Search the catalog", key="catalog_query", on_change=clear_catalog_selection, icon=":material/search:",
        placeholder="e.g. rain, wind gust, forecaster text, data quality", label_visibility="collapsed",
    )
    if not query:
        st.pills("Try", CATALOG_SUGGESTIONS, key="catalog_suggestion", on_change=use_catalog_suggestion)

    all_results = catalog_search.search(assets, query, glossary)
    kind_counts = pd.Series([a["kind"] for a, _, _ in all_results]).value_counts().to_dict() if all_results else {}
    layer_counts = pd.Series([a["layer"] for a, _, _ in all_results]).value_counts().to_dict() if all_results else {}
    facet_left, facet_right = st.columns(2)
    kinds = facet_left.pills(
        "Type", [k for k in KIND_ICONS if k in kind_counts], selection_mode="multi", key="catalog_kinds",
        format_func=lambda k: f"{k} ({kind_counts.get(k, 0)})", on_change=clear_catalog_selection,
    )
    layers = facet_right.pills(
        "Layer", [l for l in LAYER_BADGES if l in layer_counts], selection_mode="multi", key="catalog_layers",
        format_func=lambda l: f"{l} ({layer_counts.get(l, 0)})", on_change=clear_catalog_selection,
    )
    results = [r for r in all_results if (not kinds or r[0]["kind"] in kinds) and (not layers or r[0]["layer"] in layers)]
    matched_columns = {asset["id"]: columns for asset, _, columns in results}
    words = catalog_search.matched_words(query, glossary)

    results_column, detail_column = st.columns([2, 3], gap="large")
    with results_column:
        if not results:
            st.info("Nothing matches. Try a broader word, or clear the filters.")
        else:
            st.caption(f"{len(results)} result{'s' if len(results) != 1 else ''}" + (f" for \u201c{query}\u201d" if query else ", gold first"))
        for asset, _, columns in results[:MAX_CATALOG_RESULTS]:
            with st.container(border=True):
                st.markdown(f"{KIND_ICONS[asset['kind']]} **{asset['name']}**")
                st.markdown(asset_badges(asset))
                if asset["description"]:
                    st.caption(highlight(snippet(asset["description"]), words))
                if columns:
                    st.caption("Matching columns: " + ", ".join(f"`{c}`" for c in columns[:4]) + (" ..." if len(columns) > 4 else ""))
                st.button("Open", key=f"open_{asset['id']}", on_click=select_catalog_asset, args=(asset["id"],),
                          icon=":material/arrow_forward:", type="tertiary")
        if len(results) > MAX_CATALOG_RESULTS:
            st.caption(f"+{len(results) - MAX_CATALOG_RESULTS} more - narrow the search or use the filters.")

    selected_id = st.session_state.get("catalog_selected") or (results[0][0]["id"] if results else None)
    with detail_column.container(key="catalog_detail"):
        if selected_id in assets:
            render_catalog_asset(assets[selected_id], matched_columns.get(selected_id, []))

    st.divider()
    st.subheader("Lineage: how everything on this page is built")
    st.caption(
        "Declared in semantic_layer.yml, published to gold.lineage_edges, and checked on every run against "
        "the real warehouse, the SQL inside each view, and the tables each code file uses."
    )
    NODE_COLORS = {"source": "#9e9e9e", "service": "#b39ddb", "file": "#cfd8dc", "consumer": "#AFCDFF"}
    LAYER_COLORS = {"bronze": "#DCD2C8", "silver": "#D7F5C3", "gold": "#9BCDA0", "ops": "#F5F0F0"}
    edges = con.execute("SELECT upstream, downstream, upstream_type, downstream_type FROM gold.lineage_edges").fetchall()
    node_types = {}
    for upstream, downstream, upstream_type, downstream_type in edges:
        node_types[upstream] = upstream_type
        node_types[downstream] = downstream_type
    dot = ['digraph { rankdir=LR; bgcolor="transparent";',
           'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#00412D"];',
           'edge [color="#888888"];']
    for node, node_type in node_types.items():
        color = LAYER_COLORS.get(node.split(".")[0], NODE_COLORS.get(node_type, "#ffffff"))
        dot.append(f'"{node}" [fillcolor="{color}"];')
    dot += [f'"{upstream}" -> "{downstream}";' for upstream, downstream, _, _ in edges]
    dot.append("}")
    st.graphviz_chart("\n".join(dot), use_container_width=True)

    st.subheader("Metric definitions")
    st.caption("The semantic layer: each metric is defined once, and the dashboard and the agent both compute it from here.")
    st.dataframe(
        metric_defs[["label", "description", "unit", "table_name", "expression", "filter"]],
        use_container_width=True, hide_index=True,
        column_config={"label": "Metric", "description": st.column_config.TextColumn("Meaning", width="large"),
                       "unit": "Unit", "table_name": "Computed from", "expression": "SQL", "filter": "Always filtered by"},
    )

    st.subheader("Raw layers")
    st.caption("For inspection only. Nothing on the other tabs reads these directly.")
    with st.expander("Silver: hourly readings"):
        hourly_df = con.execute(
            f"""
            SELECT city, observation_time, temperature_c, precipitation_mm, wind_speed_kmh, humidity_pct
            FROM silver.weather_hourly
            WHERE city IN ({placeholders})
            ORDER BY observation_time DESC
            """,
            selected_cities,
        ).df()
        st.dataframe(hourly_df, use_container_width=True, hide_index=True)
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
        st.dataframe(fetch_log_df, use_container_width=True, hide_index=True)
