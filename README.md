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
`gold.weather_forecast`, next-day predictions (min/max/avg temp, precipitation, wind) from
small models trained fresh every run (see [Forecasting](#forecasting) below).

## Project layout

```
config.py               city list, API settings, get_connection() (local file or MotherDuck)
src/bronze.py            fetches raw API responses, lands them as-is
src/silver.py             parses bronze JSON into typed, deduplicated rows
src/gold.py              aggregates silver into daily summaries
src/forecast.py           trains next-day weather models, predicts tomorrow, walk-forward backtest
src/dq.py                tiny hand-rolled data quality checks
run_pipeline.py           orchestrates bronze -> dq -> silver -> dq -> gold -> dq -> forecast -> dq
scripts/backfill_history.py  one-off: seed real historical days so forecast.py has enough to train on
scripts/backtest.py       one-off/occasional: fill in historical forecast accuracy retroactively
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
  of only `streamlit run app.py` locally. Deployed at
  [haroradk-weatherapp.streamlit.app](https://haroradk-weatherapp.streamlit.app).
  (Needed a public repo and the Streamlit GitHub App actually installed on the account, not just
  logged in via OAuth - two easy things to trip on.)

## Forecasting

`src/forecast.py` trains one scikit-learn `LinearRegression` per (city, metric) pair on every
run - five metrics: min/max/avg temperature, precipitation, max wind speed - using yesterday's
value of that same metric plus day-of-year (as sin/cos, so day 365 and day 1 read as seasonally
adjacent) as its only two features, then predicts tomorrow's value. It's meant as a baseline
worth beating, not a serious forecaster - and in practice it clearly is a rough one for
precipitation specifically, which is far spikier than temperature and much less well explained
by "yesterday's value" (see the backtest MAE per metric in the dashboard).

Training only uses **settled** days (`date <= CURRENT_DATE`) from `gold.weather_daily_summary` -
never the forecast-window days Open-Meteo itself projects, since those aren't a realized outcome
yet and training on them would just teach the model to imitate Open-Meteo instead of learning
from what actually happened. Each city needs at least `MIN_TRAINING_ROWS` (10) settled days
before it predicts anything; below that it's skipped, not failed. Predicted precipitation/wind
are clipped at 0 (a linear model can otherwise happily predict negative rainfall).

Predictions accumulate in `gold.weather_forecast` (one row per city per target date, an
`is_backtest` flag distinguishing how it was produced) rather than being overwritten every run,
so once a target date's actual weather becomes settled history, the dashboard's "Forecast
accuracy" section can show predicted-vs-actual per metric with a running mean absolute error.

**Backtest:** waiting for live predicted-vs-actual pairs to accumulate one day at a time would
take weeks. `scripts/backtest.py` instead walks forward through existing settled history: for
each day (once enough days precede it), it trains only on what was known *before* that day and
predicts it - the same thing a live run does, replayed retroactively - then records
predicted-vs-actual for a day that already happened. It never overwrites a real live prediction
that already exists for a given (city, target_date). Run it any time you want fresh evaluation
history, e.g. after backfilling more real days.

**Cold start:** a model trained on 2-3 days of history is nearly meaningless, and the regular
daily run only adds one settled day at a time. `scripts/backfill_history.py` is a one-off script
that fetches `past_days=60` in a single request to seed real history immediately instead of
waiting ~60 days for it to accumulate. Run it once by hand, then `python run_pipeline.py` to fold
it into silver/gold, then `python scripts/backtest.py` to generate evaluation history against it.
(We tried `past_days=90` first - Open-Meteo returns nulls for the oldest ~19 days at that range,
caught by `dq.check_no_nulls`; `60` was verified clean.)

## Other deliberate simplifications (learn these next)

- **Full refresh, not incremental.** Silver/gold rebuild from scratch every run. Real pipelines
  track a watermark (e.g. "only process bronze rows newer than X") once full-refresh gets slow.
- **Hand-rolled DQ checks.** `src/dq.py` is a toy version of what dbt tests or Great Expectations
  do for real. Worth trying once you outgrow this.
- **One feature set, one model type, per metric.** `src/forecast.py` doesn't try alternative
  features (e.g. cross-metric signals, more lags) or model types, or track which version
  predicted what - a real ML pipeline versions both, and would likely reach for something other
  than lag-1 for precipitation specifically.
