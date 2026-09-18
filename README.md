# Weather ETL Pipeline (Medallion Architecture)

A hands-on ETL pipeline that pulls weather data from the free [Open-Meteo](https://open-meteo.com) API
and moves it through bronze → silver → gold layers in a local [DuckDB](https://duckdb.org) database.

## Why this design

**Medallion architecture** splits a pipeline into three layers, each with a different job:

| Layer | Job | Table |
|---|---|---|
| Bronze | Land raw source data unchanged, append-only | `bronze.raw_weather_observations` |
| Silver | Parse, type, clean, deduplicate | `silver.weather_hourly` |
| Gold | Aggregate into what people/dashboards actually query | `gold.weather_daily_summary` |

The point of splitting it up: if silver's parsing logic has a bug, you fix it and re-run silver
against bronze — you never have to re-fetch from the source. Bronze is your safety net.

## Project layout

```
config.py          city list + API settings
src/bronze.py       fetches raw API responses, lands them as-is
src/silver.py        parses bronze JSON into typed, deduplicated rows
src/gold.py         aggregates silver into daily summaries
src/dq.py           tiny hand-rolled data quality checks
run_pipeline.py      orchestrates bronze -> dq -> silver -> dq -> gold -> dq
data/weather.duckdb   the database file (gitignored - it's local state, not code)
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

## Roadmap to cloud + a real frontend

1. **GitHub.** Push this repo so you get history and a place for CI/Actions to run from.
2. **MotherDuck** (hosted DuckDB, free tier). Swap the local `data/weather.duckdb` connection
   for a MotherDuck one — same SQL, same code, just a different connection string + token.
   This is what makes the data reachable from something other than your laptop.
3. **GitHub Actions**, scheduled (`on: schedule: cron: ...`), runs `run_pipeline.py` against
   MotherDuck daily. This is the pipeline "running in the cloud."
4. **Streamlit Community Cloud** (free, sign in with GitHub) deploys `app.py` pointed at
   MotherDuck instead of the local file. This is your frontend, live on a public URL.

## Other deliberate simplifications (learn these next, once the above is running)

- **Full refresh, not incremental.** Silver/gold rebuild from scratch every run. Real pipelines
  track a watermark (e.g. "only process bronze rows newer than X") once full-refresh gets slow.
- **Hand-rolled DQ checks.** `src/dq.py` is a toy version of what dbt tests or Great Expectations
  do for real. Worth trying once you outgrow this.
