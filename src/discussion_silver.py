"""
Silver for unstructured text: bronze's raw discussions turned into
queryable data, in three steps with very different costs:

1. Pick one discussion per day - the latest one issued before our daily
   06:00 UTC run, i.e. what the forecasters were saying when our own model
   made its prediction. That keeps the human-vs-model comparison fair, and
   keeps the Gemini cost at one extraction per day. Bronze keeps all ~8.
2. Split it into its labelled sections (.KEY MESSAGES, .DISCUSSION, ...)
   with plain code. The text is semi-structured, so no LLM is needed here -
   use the cheapest tool that works.
3. Two Gemini steps: embed each section (for semantic search), and extract
   tomorrow's forecast into typed fields (for analysis).

Unlike silver.py, this is incremental rather than a full rebuild: rebuilding
would re-pay for every embedding and extraction on every run. Only days not
yet processed are touched, so a day that failed (quota, outage) is simply
picked up by the next run.
"""

import json
import re
from datetime import datetime, time, timedelta, timezone
from typing import List, Literal, Optional

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from config import EMBEDDING_DIMENSIONS, GEMINI_EMBEDDING_MODEL, GEMINI_EXTRACTION_MODEL, get_connection
from src import dq, llm

RUN_CUTOFF = time(6, 0)  # matches the daily GitHub Actions schedule (06:00 UTC)
MIN_SECTION_CHARS = 40

CREATE_SECTIONS = f"""
CREATE TABLE IF NOT EXISTS silver.forecast_discussion_sections (
    product_id VARCHAR,
    issued_at TIMESTAMP,
    run_date DATE,
    section_index INTEGER,
    section_name VARCHAR,
    content VARCHAR,
    embedding FLOAT[{EMBEDDING_DIMENSIONS}],
    embedding_model VARCHAR
);
"""

CREATE_EXTRACTIONS = """
CREATE TABLE IF NOT EXISTS silver.forecast_discussion_extractions (
    product_id VARCHAR,
    issued_at TIMESTAMP,
    target_date DATE,
    rain_expected VARCHAR,
    rain_amount_inches_max DOUBLE,
    high_temp_f DOUBLE,
    low_temp_f DOUBLE,
    max_wind_gust_mph DOUBLE,
    hazards VARCHAR[],
    forecaster_confidence VARCHAR,
    summary VARCHAR,
    evidence VARCHAR,
    evidence_verified BOOLEAN,
    extraction_model VARCHAR,
    extracted_at TIMESTAMP
);
"""

# ".KEY MESSAGES...", ".NEAR TERM /THROUGH TONIGHT/...", ".KEY MESSAGE 1..."
SECTION_HEADER = re.compile(r"^\.([A-Z][A-Z0-9 /,&-]*?)\.\.\.(.*)$")


class DiscussionForecast(BaseModel):
    """The shape Gemini must return. Optional + 'return null if not stated'
    is what stops the model filling gaps with plausible-sounding guesses."""

    rain_expected: Literal["none", "possible", "likely"] = Field(
        description="Does the text expect rain in New York City on the target date? 'possible' for slight/low chances."
    )
    rain_amount_inches_max: Optional[float] = Field(description="Upper end of any stated rainfall amount, in inches. Null if no amount is given.")
    high_temp_f: Optional[float] = Field(description="Stated high temperature for NYC on the target date, °F. Null if not stated.")
    low_temp_f: Optional[float] = Field(description="Stated low temperature for NYC on the target date, °F. Null if not stated.")
    max_wind_gust_mph: Optional[float] = Field(description="Highest wind gust stated for NYC or the coast on the target date, mph. Null if not stated.")
    hazards: List[str] = Field(description="Hazards the text mentions for the target date, e.g. 'coastal flooding', 'rip currents'. Empty if none.")
    forecaster_confidence: Optional[Literal["low", "medium", "high"]] = Field(
        description="How confident the forecasters say they are about the target date. Null if they don't say."
    )
    summary: str = Field(description="One plain-English sentence summarising the forecast for the target date.")
    evidence: str = Field(description="A short verbatim quote from the text that supports rain_expected.")


EXTRACTION_PROMPT = """\
Below is an Area Forecast Discussion written by National Weather Service
forecasters at the New York, NY office. Extract their forecast for
{target_day} {target_date} for New York City specifically.

Rules:
- Use only what the text says. If a value isn't stated for that date, return null - never guess.
- If the text distinguishes NYC from other areas (Long Island, the Hudson Valley...), use NYC.
- Keep the units the text uses: °F, inches, mph.
- `evidence` must be copied word for word from the text.

Discussion issued {issued_at} UTC:

{text}
"""


def split_sections(product_text: str) -> list:
    """Returns [(section_name, content), ...] in document order. Sections
    start at a '.NAME...' header line and end at the next header or '&&'."""
    sections, name, lines = [], None, []

    def close():
        content = " ".join(" ".join(lines).split())
        if name and len(content) >= MIN_SECTION_CHARS:
            sections.append((name, content))

    for line in product_text.splitlines():
        header = SECTION_HEADER.match(line.strip())
        if header:
            close()
            name, lines = header.group(1).strip(), [header.group(2)]
        elif line.strip() in ("&&", "$$"):
            close()
            name, lines = None, []
        elif name:
            lines.append(line)
    close()
    return sections


def pick_daily_discussions(bronze_df: pd.DataFrame) -> pd.DataFrame:
    """For each run date D: the latest discussion issued in the 24 hours
    before D 06:00 UTC. Returns columns run_date, product_id, issued_at, raw_json."""
    picks = []
    first_day = bronze_df["issued_at"].min().date() + timedelta(days=1)
    last_day = datetime.now(timezone.utc).date()
    day = first_day
    while day <= last_day:
        cutoff = datetime.combine(day, RUN_CUTOFF)
        window = bronze_df[(bronze_df["issued_at"] < cutoff) & (bronze_df["issued_at"] >= cutoff - timedelta(days=1))]
        if not window.empty:
            latest = window.sort_values("issued_at").iloc[-1]
            picks.append({"run_date": day, "product_id": latest["product_id"], "issued_at": latest["issued_at"], "raw_json": latest["raw_json"]})
        day += timedelta(days=1)
    return pd.DataFrame(picks)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def evidence_is_verbatim(evidence: str, source_text: str) -> bool:
    """Grounding check: the quote Gemini says supports its answer must
    really be in the source. A cheap, deterministic way to catch an
    extraction that was invented rather than read."""
    return bool(evidence.strip()) and _normalise(evidence) in _normalise(source_text)


def _process_day(pick, product_text: str) -> tuple:
    target_date = pick["run_date"] + timedelta(days=1)
    sections = split_sections(product_text)
    vectors = llm.embed([f"{name}: {content}" for name, content in sections])
    sections_df = pd.DataFrame(
        {
            "product_id": pick["product_id"],
            "issued_at": pick["issued_at"],
            "run_date": pick["run_date"],
            "section_index": range(len(sections)),
            "section_name": [name for name, _ in sections],
            "content": [content for _, content in sections],
            "embedding": vectors,
            "embedding_model": GEMINI_EMBEDDING_MODEL,
        }
    )

    forecast = llm.extract(
        EXTRACTION_PROMPT.format(
            target_day=target_date.strftime("%A"), target_date=target_date.isoformat(),
            issued_at=pick["issued_at"], text=product_text,
        ),
        DiscussionForecast,
    )
    extraction_df = pd.DataFrame(
        [
            {
                "product_id": pick["product_id"],
                "issued_at": pick["issued_at"],
                "target_date": target_date,
                **forecast.model_dump(),
                "evidence_verified": evidence_is_verbatim(forecast.evidence, product_text),
                "extraction_model": GEMINI_EXTRACTION_MODEL,
                "extracted_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
        ]
    )
    return sections_df, extraction_df


def run(con: duckdb.DuckDBPyConnection) -> int:
    """Processes every picked day that has no extraction yet. Returns the
    number of days processed. A day that fails is logged and skipped - the
    next run retries it - so a Gemini outage never blocks the weather data."""
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    con.execute(CREATE_SECTIONS)
    con.execute(CREATE_EXTRACTIONS)
    if not llm.is_available():
        dq.warn("gemini_available", "silver.forecast_discussion_extractions", "GEMINI_API_KEY not set - extraction and embeddings skipped")
        return 0

    bronze_df = con.execute("SELECT product_id, issued_at, raw_json FROM bronze.raw_forecast_discussions").df()
    if bronze_df.empty:
        return 0
    done = {row[0] for row in con.execute("SELECT product_id FROM silver.forecast_discussion_extractions").fetchall()}
    todo = pick_daily_discussions(bronze_df)
    todo = todo[~todo["product_id"].isin(done)]

    processed = 0
    for pick in todo.to_dict("records"):
        product_text = json.loads(pick["raw_json"])["productText"]
        try:
            sections_df, extraction_df = _process_day(pick, product_text)
        except Exception as e:
            dq.warn("discussion_extraction", "silver.forecast_discussion_extractions",
                    f"{pick['run_date']} skipped, retried next run ({type(e).__name__}: {str(e)[:120]})")
            continue
        # Delete-then-insert per product keeps a half-finished earlier
        # attempt from leaving duplicate sections behind.
        con.execute("DELETE FROM silver.forecast_discussion_sections WHERE product_id = ?", [pick["product_id"]])
        con.execute("INSERT INTO silver.forecast_discussion_sections SELECT * FROM sections_df")
        con.execute("INSERT INTO silver.forecast_discussion_extractions SELECT * FROM extraction_df")
        processed += 1
        print(
            f"  discussions: {pick['run_date']} -> forecast for {extraction_df.iloc[0]['target_date']}: "
            f"rain {extraction_df.iloc[0]['rain_expected']}, evidence verified: {extraction_df.iloc[0]['evidence_verified']}"
        )
    return processed


if __name__ == "__main__":
    print(f"Silver discussions: {run(get_connection())} day(s) processed.")
