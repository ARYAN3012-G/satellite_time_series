"""
Module 3 -- preprocessing/water_balance.py

Responsibilities:
  1. Compute daily Water_Balance = Precipitation_Flux - ReferenceET_PenmanMonteith_FAO56
  2. Compute WB30: a 30-day rolling SUM of daily Water_Balance, per field
  3. Never let the rolling window cross a field boundary -- pooled data
     must be grouped by 'field' before rolling, or day 1 of field B would
     wrongly include trailing days from field A's tail.
  4. Report how many rows have an incomplete first-29-days rolling window
     (min_periods behavior) so the caller knows those early rows are
     NaN/undefined WB30, not silently zero.

Input: the validated, chronologically-sorted DataFrame(s) from Module 2
       (weather_loader.py) -- must already have 'Date', 'field', and the
       raw weather columns.
Output: same DataFrame(s) with 'Water_Balance' and 'WB30' columns added.

Water_Balance and WB30 are intermediate/derived columns -- config.py's
LEAKAGE_COLUMNS guard ensures they never leak into MODEL_INPUT_FEATURES
downstream, but they ARE needed here to compute SPEI in Module 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

PRECIP_COL = "Precipitation_Flux"
ET_COL = "ReferenceET_PenmanMonteith_FAO56"


def compute_daily_water_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'Water_Balance' = Precipitation_Flux - ReferenceET_PenmanMonteith_FAO56.
    Does not group by field -- this is a pure row-wise operation, safe to
    apply to a single field's frame or a pooled frame either way."""
    for col in (PRECIP_COL, ET_COL):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. "
                              f"Did this DataFrame come through Module 2's loader?")

    df = df.copy()
    df["Water_Balance"] = df[PRECIP_COL] - df[ET_COL]
    return df


def compute_wb30(df: pd.DataFrame, window_days: int = config.WB_ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """Adds 'WB30' = rolling SUM of Water_Balance over `window_days`,
    computed independently per field. Requires df to be sorted by Date
    within each field (guaranteed by Module 2's loader) and to have a
    'field' column.

    The first (window_days - 1) rows of each field will have WB30 = NaN
    (min_periods=window_days, deliberately -- a partial-window sum would
    be a systematically biased WB30 value, not a slightly-uncertain one).
    """
    if "Water_Balance" not in df.columns:
        raise ValueError("'Water_Balance' column missing -- call "
                          "compute_daily_water_balance() first.")
    if "field" not in df.columns:
        raise ValueError("'field' column missing -- this frame doesn't look "
                          "like it came from Module 2's loader.")

    df = df.sort_values(["field", config.DATE_COLUMN]).reset_index(drop=True)

    df["WB30"] = (
        df.groupby("field")["Water_Balance"]
        .transform(lambda s: s.rolling(window=window_days, min_periods=window_days).sum())
    )

    n_undefined = int(df["WB30"].isna().sum())
    if n_undefined:
        logger.info(
            f"{n_undefined} row(s) have undefined WB30 (first {window_days - 1} "
            f"day(s) of each field's series, as expected)."
        )

    return df


def add_water_balance_and_wb30(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: Module 2 output -> Module 2 output + Water_Balance + WB30."""
    df = compute_daily_water_balance(df)
    df = compute_wb30(df)
    return df


def run_water_balance_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper (alias) expected by spei.py's import:
    compute_daily_water_balance -> compute_wb30, in order.
    Functionally identical to add_water_balance_and_wb30(); kept as a
    separate name so spei.py's existing import doesn't need to change."""
    df = compute_daily_water_balance(df)
    df = compute_wb30(df)
    return df


if __name__ == "__main__":
    from data_ingestion.weather_loader import load_all_fields, pool_fields

    frames, reports = load_all_fields()
    if not frames:
        print("No fields loaded -- run Module 2 first / check errors above.")
        sys.exit(1)

    pooled = pool_fields(frames)
    pooled = add_water_balance_and_wb30(pooled)

    print(f"\nWater balance + WB30 computed for {pooled['field'].nunique()} field(s), "
          f"{len(pooled)} total rows.\n")
    for f in sorted(pooled["field"].unique()):
        sub = pooled[pooled["field"] == f]
        n_valid_wb30 = sub["WB30"].notna().sum()
        print(f"  {f}: {len(sub)} rows, {n_valid_wb30} with defined WB30 "
              f"(first valid: {sub.loc[sub['WB30'].notna(), config.DATE_COLUMN].min()})")

    sample_cols = [config.DATE_COLUMN, "field", PRECIP_COL, ET_COL, "Water_Balance", "WB30"]
    print("\nSample rows (first field, around day 30):")
    first_field = sorted(pooled["field"].unique())[0]
    print(pooled[pooled["field"] == first_field][sample_cols].iloc[27:33].to_string(index=False))
