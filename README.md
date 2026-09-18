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

## What's deliberately simplified (and what to learn next)

- **Full refresh, not incremental.** Silver/gold rebuild from scratch every run. Real pipelines
  track a watermark (e.g. "only process bronze rows newer than X") once full-refresh gets slow.
- **No orchestrator/scheduler yet.** Right now you run `run_pipeline.py` by hand. Next step:
  a daily cron job, or a GitHub Actions workflow on a schedule (`on: schedule: cron: ...`).
- **No version control yet.** Once you're happy with this, `git init` + push to GitHub, so
  you get history and can eventually run CI (e.g. run the pipeline + a smoke test on every push).
- **Local only.** DuckDB file lives on your laptop. Natural next step once comfortable:
  point the same SQL logic at a free-tier cloud warehouse (BigQuery sandbox, Snowflake trial,
  Motherduck for hosted DuckDB) or swap the orchestrator for something like Airflow/Dagster.
- **Hand-rolled DQ checks.** `src/dq.py` is a toy version of what dbt tests or Great Expectations
  do for real. Worth trying once you outgrow this.
