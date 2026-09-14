"""
Module 4 -- preprocessing/spei.py

Converts WB30 (30-day rolling water balance, from Module 3) into SPEI
(Standardized Precipitation-Evapotranspiration Index) via a log-logistic
distribution fit SEPARATELY for each calendar month (1-12), per field.

Why per-month, not one global fit:
  Water balance is strongly seasonal (monsoon vs. dry season) and not
  normally distributed. A single global z-score would treat a "normal"
  January value as equally extreme to an equally-far-from-mean July
  value, even though the two months have entirely different climatology.
  Fitting one distribution per calendar month makes SPEI comparable
  ACROSS months and years -- "moderately dry for January" means the same
  standardized thing as "moderately dry for July" for that field.

Why per-field:
  Fields in different micro-climates have different baseline water
  balance distributions. Pooling all fields into one fit would let a
  wetter field's "normal" contaminate a drier field's baseline.

Method: fisk (log-logistic) distribution, fit via scipy.stats.fisk,
matching the standard McKee et al. (1993) / Vicente-Serrano et al. (2010)
SPEI methodology.

Input:  DataFrame with 'field', DATE_COLUMN, 'WB30' columns (from Module 3).
Output: same DataFrame with a 'SPEI' column added (NaN wherever WB30 is NaN).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _fit_log_logistic_and_transform(wb30_values: np.ndarray) -> tuple:
    """Fit a log-logistic (fisk) distribution to one (field, calendar-month)
    group's WB30 values, then transform each value to its standard-normal
    SPEI equivalent via the probability integral transform.

    Returns (spei_values, fit_params). Raises if fewer than 2 valid values
    are present (can't fit a distribution to 0-1 points)."""
    valid_mask = ~np.isnan(wb30_values)
    valid_values = wb30_values[valid_mask]

    if len(valid_values) < 2:
        raise ValueError(
            f"Cannot fit log-logistic distribution: only {len(valid_values)} "
            f"valid WB30 value(s) in this (field, month) group."
        )

    # fisk (log-logistic) is defined for positive support; WB30 can be
    # negative (water deficit), so shift by a safety margin before fitting,
    # then unshift is implicit since we only need the CDF for the transform.
    shift = -valid_values.min() + 1.0  # ensures all values > 0 before fitting
    shifted = valid_values + shift

    # Fit fisk: shape (c), location (fixed at 0 post-shift), scale
    c, loc, scale = stats.fisk.fit(shifted, floc=0)

    # Probability integral transform: CDF -> standard normal quantile
    cdf_values = stats.fisk.cdf(shifted, c, loc=loc, scale=scale)
    # Clip to avoid -inf/+inf at the extremes (cdf exactly 0 or 1)
    cdf_values = np.clip(cdf_values, 1e-6, 1 - 1e-6)
    spei_valid = stats.norm.ppf(cdf_values)

    spei_full = np.full_like(wb30_values, np.nan, dtype=float)
    spei_full[valid_mask] = spei_valid

    return spei_full, {"c": c, "loc": loc, "scale": scale, "shift": shift, "n": len(valid_values)}


def compute_spei(df: pd.DataFrame, min_years_warning: int = config.SPEI_MIN_YEARS_FOR_FIT) -> pd.DataFrame:
    """Add 'SPEI' column: per-(field, calendar-month) log-logistic fit of
    WB30, transformed to a standard-normal index.

    Rows with NaN WB30 (first 29 days of each field, per Module 3) get
    NaN SPEI. Processes each field independently, and within each field,
    each of the 12 calendar months independently."""
    required = ["field", config.DATE_COLUMN, "WB30"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"compute_spei: missing columns {missing}")

    df = df.copy()
    df["_month"] = df[config.DATE_COLUMN].dt.month
    df["SPEI"] = np.nan

    fit_summary = []

    for field_name, field_df in df.groupby("field"):
        n_years = (field_df[config.DATE_COLUMN].max() - field_df[config.DATE_COLUMN].min()).days / 365.25
        if n_years < min_years_warning:
            logger.warning(
                f"[{field_name}] only ~{n_years:.1f} years of history "
                f"(< {min_years_warning}) -- per-month SPEI fit may be unstable "
                f"for this field."
            )

        for month in range(1, 13):
            month_mask = (df["field"] == field_name) & (df["_month"] == month)
            wb30_values = df.loc[month_mask, "WB30"].values

            try:
                spei_values, params = _fit_log_logistic_and_transform(wb30_values)
                df.loc[month_mask, "SPEI"] = spei_values
                fit_summary.append({
                    "field": field_name, "month": month,
                    "n_fitted": params["n"], "shape_c": round(params["c"], 3),
                })
            except ValueError as e:
                logger.error(f"[{field_name}] month {month}: {e}")

    df = df.drop(columns=["_month"])

    n_total = len(df)
    n_spei_nan = df["SPEI"].isna().sum()
    n_wb30_nan = df["WB30"].isna().sum()
    logger.info(
        f"SPEI computed for {n_total} rows across {df['field'].nunique()} fields "
        f"x 12 calendar months. {n_spei_nan} NaN SPEI values "
        f"({n_wb30_nan} of those from NaN WB30, expected; "
        f"{n_spei_nan - n_wb30_nan} from failed fits, should be 0)."
    )

    return df, pd.DataFrame(fit_summary)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_ingestion"))
    from data_ingestion.weather_loader import load_all_fields, pool_fields
    from preprocessing.water_balance import run_water_balance_pipeline

    frames, reports = load_all_fields()
    if not frames:
        print("No fields loaded -- check Module 2 output first.")
        sys.exit(1)

    pooled = pool_fields(frames)
    pooled = run_water_balance_pipeline(pooled)
    pooled, fit_summary = compute_spei(pooled)

    print(f"\nSPEI computed for {pooled['field'].nunique()} fields, {len(pooled)} rows.")
    print(f"\nFit summary (first 15 field-month groups):")
    print(fit_summary.head(15).to_string(index=False))

    print(f"\nSPEI descriptive stats (should be roughly standard-normal: mean~0, std~1):")
    print(pooled["SPEI"].describe())

    print(f"\nSample (first field, days 25-40, showing WB30 -> SPEI):")
    first_field = pooled["field"].iloc[0]
    sample = pooled[pooled["field"] == first_field].iloc[25:40]
    print(sample[[config.DATE_COLUMN, "WB30", "SPEI"]].to_string(index=False))
