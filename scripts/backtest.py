"""
Run the walk-forward backtest once (or whenever you want to re-check
accuracy after adding more history): `python scripts/backtest.py`.

Not part of the scheduled daily pipeline - a live prediction and its
comparison-to-actual will accumulate naturally one day at a time anyway,
so re-running a full historical backtest every day would just redo the
same work. This is for filling in evaluation history retroactively and
for occasional re-checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_connection
from src import forecast


def main() -> None:
    con = get_connection()
    n = forecast.run_backtest(con)
    print(f"Backtest: {n} new historical evaluation point(s) added to gold.weather_forecast.")


if __name__ == "__main__":
    main()
