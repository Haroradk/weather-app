# Weather ETL Pipeline (Medallion Architecture)

A hands-on ETL pipeline that pulls weather data from the free [Open-Meteo](https://open-meteo.com) API
and moves it through bronze → silver → gold layers, running on a daily schedule against
[MotherDuck](https://motherduck.com) (hosted DuckDB) via GitHub Actions, with a Streamlit dashboard
on top.

## Why this design

**Medallion architecture** splits a pipeline into three layers, each with a different job:

| Layer | Job | Table |
|---|---|---|
| Bronze | Land raw source data unchanged, append-only | `bronze.raw_weather_observations` |
| Silver | Parse, type, clean, deduplicate | `silver.weather_hourly` |
| Gold | Aggregate into what people/dashboards actually query | `gold.weather_daily_summary` |

The point of splitting it up: if silver's parsing logic has a bug, you fix it and re-run silver
against bronze — you never have to re-fetch from the source. Bronze is your safety net.

On top of gold sits one more table that isn't a traditional medallion layer:
`gold.temperature_forecast`, a next-day temperature prediction from a small model trained fresh
every run (see [Forecasting](#forecasting) below).

## Project layout

```
config.py               city list, API settings, get_connection() (local file or MotherDuck)
src/bronze.py            fetches raw API responses, lands them as-is
src/silver.py             parses bronze JSON into typed, deduplicated rows
src/gold.py              aggregates silver into daily summaries
src/forecast.py           trains a next-day temperature model, predicts tomorrow
src/dq.py                tiny hand-rolled data quality checks
run_pipeline.py           orchestrates bronze -> dq -> silver -> dq -> gold -> dq -> forecast -> dq
scripts/backfill_history.py  one-off: seed real historical days so forecast.py has enough to train on
app.py                   Streamlit dashboard over gold/silver/bronze + the forecast
.github/workflows/pipeline.yml  daily cron (+ manual trigger) that runs run_pipeline.py in CI
data/weather.duckdb        local DB file, used only when MOTHERDUCK_TOKEN isn't set (gitignored)
```

## Running it

```bash
source .venv/bin/activate
python run_pipeline.py
```

Safe to run repeatedly: bronze always appends (that's intentional — it's a historical log
of every fetch), but silver and gold are rebuilt idempotently, so row counts there don't
balloon on re-runs.

To poke at the data directly:

```bash
python -c "
import duckdb
con = duckdb.connect('data/weather.duckdb')
print(con.execute('SELECT * FROM gold.weather_daily_summary ORDER BY city, date').fetchall())
"
```

Or use the DuckDB CLI (`brew install duckdb`) for a proper SQL shell against `data/weather.duckdb`.

### Dashboard

```bash
source .venv/bin/activate
streamlit run app.py
```

Opens at http://localhost:8501. Reads gold (charts + table), with silver and bronze
available in collapsed expanders further down the page.

## Cloud setup (done)

- **GitHub** — code lives at [Haroradk/weather-app](https://github.com/Haroradk/weather-app).
- **MotherDuck** — `config.get_connection()` targets it whenever `MOTHERDUCK_TOKEN` is set (falls
  back to the local file otherwise), so the same code runs locally and in CI. Requires a `.env`
  with `MOTHERDUCK_TOKEN` (and optionally `MOTHERDUCK_DATABASE`, default `weather`) — see `.env`,
  which is gitignored and never committed.
- **GitHub Actions** (`.github/workflows/pipeline.yml`) — runs `run_pipeline.py` daily at 06:00 UTC
  against MotherDuck, using a `MOTHERDUCK_TOKEN` repository secret. Retries the whole run up to 3
  times (30s apart) before giving up, then opens a GitHub issue with a link to the failed run.
  Also has a manual "Run workflow" button in the Actions tab for on-demand runs.
- **Streamlit Community Cloud** — deploy `app.py` there (sign in with GitHub, add the same
  `MOTHERDUCK_TOKEN` as a secret in the app's settings) for a dashboard on a public URL, instead
  of only `streamlit run app.py` locally.

## Forecasting

`src/forecast.py` trains one scikit-learn `LinearRegression` per city on every run, using
yesterday's average temperature and day-of-year (as sin/cos, so day 365 and day 1 read as
seasonally adjacent) as its only two features, then predicts tomorrow's average temperature.
It's meant as a baseline worth beating, not a serious forecaster.

Training only uses **settled** days (`date <= CURRENT_DATE`) from `gold.weather_daily_summary` -
never the forecast-window days Open-Meteo itself projects, since those aren't a realized outcome
yet and training on them would just teach the model to imitate Open-Meteo instead of learning
from what actually happened. Each city needs at least `MIN_TRAINING_ROWS` (10) settled days
before it predicts anything; below that it's skipped, not failed.

Predictions accumulate in `gold.temperature_forecast` (one row per city per target date) rather
than being overwritten every run, so once a target date's actual temperature becomes settled
history, the dashboard's "Next-day temperature forecast" section can show predicted-vs-actual
and a running mean absolute error.

**Cold start:** a model trained on 2-3 days of history is nearly meaningless, and the regular
daily run only adds one settled day at a time. `scripts/backfill_history.py` is a one-off script
that fetches `past_days=60` in a single request to seed real history immediately instead of
waiting ~60 days for it to accumulate. Run it once by hand, then `python run_pipeline.py` to fold
it into silver/gold. (We tried `past_days=90` first - Open-Meteo returns nulls for the oldest ~19
days at that range, caught by `dq.check_no_nulls`; `60` was verified clean.)

## Other deliberate simplifications (learn these next)

- **Full refresh, not incremental.** Silver/gold rebuild from scratch every run. Real pipelines
  track a watermark (e.g. "only process bronze rows newer than X") once full-refresh gets slow.
- **Hand-rolled DQ checks.** `src/dq.py` is a toy version of what dbt tests or Great Expectations
  do for real. Worth trying once you outgrow this.
- **One feature set, one model type.** `src/forecast.py` doesn't try alternative features or
  models, or track which version predicted what - a real ML pipeline versions both.
