"""
generate_spei_labels.py
========================
Module 2 — Label Generation

Purpose
-------
Compute Water_Balance -> WB30 -> SPEI -> 3-class drought labels for each
cleaned location, and save a MASTER dataset (all intermediate columns
included) to data/labels/ for documentation and traceability.

This is NOT the training dataset — it deliberately keeps Water_Balance,
WB30, and SPEI alongside the final labels, so the label-generation
process is fully auditable. Module 3 is responsible for producing the
separate, leakage-free training CSV that excludes these columns.

SPEI methodology
-----------------
SPEI is computed by fitting a log-logistic distribution SEPARATELY for
each calendar month (not a single global z-score), following the
standard SPEI methodology (Vicente-Serrano et al. 2010). A plain z-score
assumes normally-distributed, non-seasonal data -- water balance is
neither. This per-month approach was independently validated earlier in
this project: two different fields' SPEI series both surfaced 2002 (a
documented regional drought year) as a top-5 driest year, discovered
independently rather than engineered.

3-class thresholds (McKee et al. 1993 convention, merged)
------------------------------------------------------------
    Healthy   : SPEI >= 0
    Moderate  : -1.5 <= SPEI < 0
    Severe    : SPEI < -1.5

Chosen over a 5-class scheme because the finer-grained Healthy/Mild
boundary was shown to be unstable and unreliable on data outside the
training distribution -- see project review notes.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_spei_labels")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"

SPEI_ACCUMULATION_DAYS = 30
MIN_SAMPLES_PER_MONTH = 10

DROUGHT_LABEL_MAP = {"Healthy": 0, "Moderate": 1, "Severe": 2}


class LabelGenerationError(ValueError):
    """Raised when labels cannot be generated due to missing/invalid input."""


def compute_water_balance(df: pd.DataFrame, window: int = SPEI_ACCUMULATION_DAYS) -> pd.DataFrame:
    """Compute daily Water_Balance (P - ET) and its rolling `window`-day sum (WB{window})."""
    required = ("Precipitation_Flux", "ReferenceET_PenmanMonteith_FAO56")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise LabelGenerationError(f"Missing column(s) for water balance: {missing}")

    out = df.copy()
    out["Water_Balance"] = out["Precipitation_Flux"] - out["ReferenceET_PenmanMonteith_FAO56"]
    wb_col = f"WB{window}"
    out[wb_col] = out["Water_Balance"].rolling(window=window, min_periods=window).sum()

    n_valid = out[wb_col].notna().sum()
    logger.info(f"Water balance computed: {n_valid}/{len(out)} rows have a valid {wb_col}")
    return out


def _fit_and_standardize_one_month(values: pd.Series) -> pd.Series:
    """Fit a log-logistic distribution to one calendar month's WB30 values
    and convert each to its standard-normal equivalent (SPEI)."""
    valid = values.dropna()
    if len(valid) < MIN_SAMPLES_PER_MONTH:
        return pd.Series(np.nan, index=values.index)

    params = stats.fisk.fit(valid)
    cdf_values = stats.fisk.cdf(values, *params)
    cdf_values = np.clip(cdf_values, 1e-6, 1 - 1e-6)
    return pd.Series(stats.norm.ppf(cdf_values), index=values.index)


def compute_spei(df: pd.DataFrame, wb_col: str = f"WB{SPEI_ACCUMULATION_DAYS}", date_col: str = "valid_time") -> pd.DataFrame:
    """Add an 'SPEI' column, fit per calendar month for seasonality."""
    missing = [c for c in (wb_col, date_col) if c not in df.columns]
    if missing:
        raise LabelGenerationError(f"Missing column(s) for SPEI: {missing}")

    out = df.copy()
    months = out[date_col].dt.month
    out["SPEI"] = np.nan
    for month in range(1, 13):
        mask = months == month
        out.loc[mask, "SPEI"] = _fit_and_standardize_one_month(out.loc[mask, wb_col]).values

    n_valid = out["SPEI"].notna().sum()
    logger.info(f"SPEI computed: {n_valid}/{len(out)} rows have a valid SPEI value")
    return out


def _three_class_label(spei: float) -> str | float:
    if pd.isna(spei):
        return np.nan
    if spei >= 0:
        return "Healthy"
    elif spei >= -1.5:
        return "Moderate"
    else:
        return "Severe"


def generate_labels(df: pd.DataFrame, spei_col: str = "SPEI") -> pd.DataFrame:
    """Add 'Drought_Label' (3-class) and 'Target' (0-2 integer) columns."""
    if spei_col not in df.columns:
        raise LabelGenerationError(f"Missing column for labeling: '{spei_col}'")

    out = df.copy()
    out["Drought_Label"] = out[spei_col].apply(_three_class_label)
    out["Target"] = out["Drought_Label"].map(DROUGHT_LABEL_MAP)

    n_labeled = out["Drought_Label"].notna().sum()
    logger.info(f"Labels generated: {n_labeled}/{len(out)} rows labeled")
    logger.info(f"3-class distribution: {out['Drought_Label'].value_counts().to_dict()}")
    return out


def generate_master_dataset(location: str) -> pd.DataFrame:
    """
    Run the full Module 2 chain for one location: load its Module-1
    cleaned CSV, compute Water_Balance -> WB30 -> SPEI -> labels, and
    save the MASTER dataset (all intermediate columns retained) to
    data/labels/{location}_master.csv.
    """
    location = location.lower().strip()
    clean_path = PROCESSED_DIR / f"{location}_clean.csv"
    if not clean_path.exists():
        raise LabelGenerationError(
            f"Cleaned data not found for '{location}': {clean_path}. Run clean_weather.py first."
        )

    df = pd.read_csv(clean_path, parse_dates=["valid_time"])
    df = compute_water_balance(df)
    df = compute_spei(df)
    df = generate_labels(df)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELS_DIR / f"{location}_master.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved master dataset -> {out_path}")
    return df


def generate_master_dataset_all_locations() -> dict[str, pd.DataFrame]:
    """Run Module 2 for every location that has a Module-1 cleaned CSV.
    Failures are logged and skipped, not fatal."""
    clean_files = sorted(PROCESSED_DIR.glob("*_clean.csv"))
    locations = [f.stem.replace("_clean", "") for f in clean_files]

    results = {}
    for location in locations:
        try:
            results[location] = generate_master_dataset(location)
        except LabelGenerationError as exc:
            logger.error(f"Skipping '{location}': {exc}")

    logger.info(f"Label generation complete: {len(results)}/{len(locations)} location(s) succeeded")
    return results


if __name__ == "__main__":
    all_results = generate_master_dataset_all_locations()
    print("\n--- Summary ---")
    for location, df in all_results.items():
        valid = df.dropna(subset=["Drought_Label"])
        counts = valid["Drought_Label"].value_counts().to_dict()
        print(f"{location:30s} rows={len(df):6d}  labeled={len(valid):6d}  distribution={counts}")
