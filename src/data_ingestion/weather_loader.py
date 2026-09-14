"""
Module 2 — data_ingestion/weather_loader.py

Responsibilities:
  1. Auto-discover field folders under data/raw/weather/<field>/*.csv
  2. Validate each raw CSV has the required schema (config.REQUIRED_RAW_COLUMNS)
  3. Parse dates, sort chronologically per field
  4. Detect and report duplicate dates and date gaps (missing days) -- these
     silently corrupt a 30-day rolling window downstream if not caught here
  5. Return one dict of {field_name: DataFrame} and/or one pooled DataFrame
     tagged with a 'field' column, ready for Module 3 (water_balance.py)

Nothing here computes Water_Balance/WB30/SPEI or touches labels -- this
module's only job is: read raw, validate, sort, report gaps.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field as dc_field

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class FieldLoadReport:
    """Per-field diagnostics, surfaced to the caller and to logs -- not
    swallowed silently. A field with unresolved date gaps should be a
    visible fact, not a number quietly baked into later statistics."""
    field_name: str
    source_path: Path
    n_rows_raw: int
    n_duplicate_dates: int
    n_date_gaps: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    missing_columns: list = dc_field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.n_duplicate_dates == 0 and self.n_date_gaps == 0 and not self.missing_columns


def discover_field_files(raw_weather_dir: Path = config.RAW_WEATHER_DIR) -> dict:
    """Find one CSV per field folder. If a folder has more than one CSV,
    the first (alphabetically) is used and a warning is logged."""
    raw_weather_dir = Path(raw_weather_dir)
    if not raw_weather_dir.exists():
        raise FileNotFoundError(f"Raw weather directory not found: {raw_weather_dir}")

    field_files = {}
    for sub in sorted(raw_weather_dir.iterdir()):
        if not sub.is_dir():
            continue
        csvs = sorted(sub.glob("*.csv"))
        if not csvs:
            logger.warning(f"Field folder '{sub.name}' has no CSV file -- skipping.")
            continue
        if len(csvs) > 1:
            logger.warning(
                f"Field folder '{sub.name}' has {len(csvs)} CSV files; "
                f"using '{csvs[0].name}'."
            )
        field_files[sub.name] = csvs[0]

    if not field_files:
        raise FileNotFoundError(f"No field CSVs discovered under {raw_weather_dir}")

    logger.info(f"Discovered {len(field_files)} field(s): {sorted(field_files.keys())}")
    return field_files


def _normalize_date_column(df: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """Rename whichever known date-column alias is present to
    config.DATE_COLUMN, so every downstream module can rely on one fixed
    name regardless of which export batch a field's CSV came from.
    Real-world case that motivated this: some fields export the date
    column as 'valid_time' instead of 'Date'."""
    if config.DATE_COLUMN in df.columns:
        return df  # already canonical

    for alias in config.DATE_COLUMN_ALIASES:
        if alias in df.columns:
            if alias != config.DATE_COLUMN:
                logger.info(f"[{field_name}] renaming date column '{alias}' -> "
                            f"'{config.DATE_COLUMN}'")
            return df.rename(columns={alias: config.DATE_COLUMN})

    return df  # no alias found; validation below will report it as missing


def _parse_dates_robustly(date_series: pd.Series, field_name: str) -> pd.Series:
    """Defensive date parsing for a column confirmed (from real data) to
    mix formats WITHIN THE SAME FILE -- e.g. '1979-01-01' (ISO) alongside
    '13-01-1979' (DD-MM-YYYY) in the same column. This is a real, observed
    case, not a hypothetical.

    IMPORTANT: pandas' format='mixed' + dayfirst=True is NOT safe here --
    verified failure: it silently reparsed '1979-01-02' (ISO) as
    1979-02-01, because dayfirst=True gets applied even to unambiguous
    ISO strings when the format is inferred per-row. That kind of silent
    date corruption is worse than a hard failure, so we do NOT use it.

    Instead: try each row against a fixed list of EXPLICIT, unambiguous
    formats. An explicit format string can only match a string laid out
    exactly that way -- it cannot misinterpret '1979-01-02' as
    day-first, because that string doesn't match the DD-MM-YYYY pattern
    (month 01 vs day 02 is fine, but the parser here matches by explicit
    position, not guesswork). Rows that don't match ANY known format
    raise, loudly, rather than being silently guessed."""
    candidate_formats = [
        "%Y-%m-%d",   # ISO: 1979-01-01
        "%d-%m-%Y",   # DD-MM-YYYY: 13-01-1979 (confirmed real format)
        "%Y/%m/%d",
        "%d/%m/%Y",
    ]

    remaining = date_series.astype(str)
    parsed = pd.Series(pd.NaT, index=date_series.index, dtype="datetime64[ns]")
    used_formats = []

    for fmt in candidate_formats:
        still_unparsed = parsed.isna()
        if not still_unparsed.any():
            break
        attempt = pd.to_datetime(
            remaining[still_unparsed], format=fmt, errors="coerce"
        )
        newly_parsed = attempt.notna()
        if newly_parsed.any():
            parsed.loc[still_unparsed[still_unparsed].index[newly_parsed.values]] = \
                attempt[newly_parsed]
            used_formats.append((fmt, int(newly_parsed.sum())))

    n_failed = parsed.isna().sum()
    if n_failed > 0:
        bad_examples = remaining[parsed.isna()].head(5).tolist()
        raise ValueError(
            f"[{field_name}] {n_failed} date value(s) matched none of the known "
            f"formats {candidate_formats}. Examples: {bad_examples}. "
            f"Add the correct format to candidate_formats in "
            f"_parse_dates_robustly() rather than guessing."
        )

    if len(used_formats) > 1:
        logger.info(
            f"[{field_name}] date column parsed using multiple formats: "
            f"{used_formats} (mixed-format source file confirmed)."
        )
    else:
        logger.info(f"[{field_name}] date column parsed via format '{used_formats[0][0]}'.")

    return parsed




def _validate_schema(df: pd.DataFrame, field_name: str) -> list:
    missing = [c for c in config.REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        logger.error(f"[{field_name}] missing required columns: {missing}")
    return missing


def load_single_field(csv_path: Path, field_name: str) -> tuple:
    """Load, validate, and chronologically sort a single field's CSV.
    Returns (DataFrame, FieldLoadReport). Raises if required columns are
    missing -- everything downstream assumes a validated schema."""
    df = pd.read_csv(csv_path)
    df = _normalize_date_column(df, field_name)

    missing = _validate_schema(df, field_name)
    if missing:
        raise ValueError(
            f"[{field_name}] cannot proceed -- missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df[config.DATE_COLUMN] = _parse_dates_robustly(df[config.DATE_COLUMN], field_name)
    n_rows_raw = len(df)

    n_duplicate_dates = int(df[config.DATE_COLUMN].duplicated().sum())
    if n_duplicate_dates:
        logger.warning(
            f"[{field_name}] {n_duplicate_dates} duplicate date(s) found -- "
            f"keeping first occurrence, dropping rest."
        )
        df = df.drop_duplicates(subset=[config.DATE_COLUMN], keep="first")

    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)

    date_min, date_max = df[config.DATE_COLUMN].min(), df[config.DATE_COLUMN].max()
    expected_days = (date_max - date_min).days + 1
    n_date_gaps = expected_days - len(df)
    if n_date_gaps > 0:
        logger.warning(
            f"[{field_name}] {n_date_gaps} missing day(s) between "
            f"{date_min.date()} and {date_max.date()} "
            f"({len(df)} rows present, {expected_days} expected)."
        )

    df["field"] = field_name

    report = FieldLoadReport(
        field_name=field_name,
        source_path=csv_path,
        n_rows_raw=n_rows_raw,
        n_duplicate_dates=n_duplicate_dates,
        n_date_gaps=max(n_date_gaps, 0),
        date_min=date_min,
        date_max=date_max,
        missing_columns=[],
    )
    return df, report


def load_all_fields(raw_weather_dir: Path = config.RAW_WEATHER_DIR) -> tuple:
    """Load every discovered field. Returns (field_frames, reports).
    A field that fails to load is logged and excluded rather than aborting
    the whole run -- one bad file shouldn't block the other fields."""
    field_files = discover_field_files(raw_weather_dir)
    field_frames, reports = {}, {}

    for field_name, csv_path in field_files.items():
        try:
            df, report = load_single_field(csv_path, field_name)
            field_frames[field_name] = df
            reports[field_name] = report
            status = "clean" if report.is_clean else "issues found (see log)"
            logger.info(
                f"[{field_name}] loaded {len(df)} rows "
                f"({report.date_min.date()} to {report.date_max.date()}) -- {status}"
            )
        except Exception as e:
            logger.error(f"[{field_name}] failed to load: {e}")

    return field_frames, reports


def pool_fields(field_frames: dict) -> pd.DataFrame:
    """Concatenate all per-field DataFrames, tagged by 'field', sorted by
    field name then date for reproducibility."""
    pooled = pd.concat(field_frames.values(), ignore_index=True)
    pooled = pooled.sort_values(["field", config.DATE_COLUMN]).reset_index(drop=True)
    return pooled


if __name__ == "__main__":
    frames, reports = load_all_fields()
    print(f"\nLoaded {len(frames)} field(s).\n")
    for name, r in reports.items():
        flag = "OK" if r.is_clean else "CHECK"
        print(f"  [{flag}] {name}: {r.n_rows_raw} rows, "
              f"{r.n_duplicate_dates} dup dates, {r.n_date_gaps} gap days, "
              f"range {r.date_min.date()}..{r.date_max.date()}")
    if frames:
        pooled = pool_fields(frames)
        print(f"\nPooled dataset: {len(pooled)} rows across {pooled['field'].nunique()} fields.")
