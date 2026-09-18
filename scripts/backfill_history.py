"""
One-off: seed ~90 days of real settled history per city.

Run this once by hand (`python scripts/backfill_history.py`), not as part
of the scheduled pipeline. The daily run keeps PAST_DAYS small (2) so bronze
doesn't re-land the same 90 days of JSON every single day forever - this
script borrows a bigger past_days just for a single backfill fetch, then
lets silver/gold pick it up exactly like any other bronze rows.

Why this exists: the forecast model (src/forecast.py) needs real historical
temperatures to train on. Waiting for that to accumulate one day at a time
from the regular daily run would take ~60 days; Open-Meteo will hand us the
same history right now in a single request.

past_days=60, not 90: tested empirically (see PR discussion / commit log) -
past_days=90 returns a clean response, but the oldest ~19 days inside it come
back as JSON `null` for every field, apparently past the edge of Open-Meteo's
real hourly coverage at that range. past_days=60 was verified to return zero
nulls. dq.check_no_nulls in the regular pipeline is what caught this the
first time - a good example of that gate doing its job.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_connection
from src import bronze

BACKFILL_PAST_DAYS = 60


def main() -> None:
    con = get_connection()
    landed = bronze.run(con, past_days=BACKFILL_PAST_DAYS, forecast_days=1)
    print(f"Backfilled {landed} responses ({BACKFILL_PAST_DAYS} days of history per city).")
    print("Run `python run_pipeline.py` next to fold this into silver/gold.")


if __name__ == "__main__":
    main()
