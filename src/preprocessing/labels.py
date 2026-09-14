"""
Module 5 -- preprocessing/labels.py

Converts the SPEI column (from Module 4) into the final 3-class drought
label (Healthy / Moderate / Severe) via config.classify_spei() -- the
SAME function that will later be used at inference time (Module 9), so
the labeling rule can never drift between training and serving.

This module is also the first point in the pipeline that WRITES processed
output to disk: one CSV per field under data/processed/weather/, plus a
combined pooled file. Everything before this (Modules 2-4) only produced
in-memory DataFrames.

Input:  DataFrame with 'field', DATE_COLUMN, 'SPEI' columns (from Module 4).
Output: same DataFrame with 'drought_class' (string) and 'label' (int,
        via config.LABEL_MAP) columns added; also written to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'drought_class' (Healthy/Moderate/Severe string) and 'label'
    (0/1/2 int, via config.LABEL_MAP) columns, derived from SPEI.

    Rows with NaN SPEI (first 29 days of each field, from Module 3's
    rolling window) get NaN drought_class / label -- these rows cannot
    be used for training and must be dropped before sequence building
    (Module 6), not silently imputed."""
    if "SPEI" not in df.columns:
        raise ValueError("generate_labels: 'SPEI' column not found -- run Module 4 first.")

    df = df.copy()

    valid_mask = df["SPEI"].notna()
    df["drought_class"] = pd.NA
    df.loc[valid_mask, "drought_class"] = df.loc[valid_mask, "SPEI"].apply(config.classify_spei)

    df["label"] = df["drought_class"].map(config.LABEL_MAP)  # NaN stays NaN via map

    n_valid = valid_mask.sum()
    n_invalid = (~valid_mask).sum()
    logger.info(f"Labels generated for {n_valid} rows ({n_invalid} rows remain NaN "
                f"from undefined SPEI -- expected, will be dropped before sequencing).")

    return df


def report_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Log and return per-class counts and percentages, overall and per
    field. This is where the imbalance problem (Section 6/7 of the
    original review) becomes visible directly from your real data,
    rather than assumed from the 5-class run."""
    valid = df.dropna(subset=["drought_class"])
    counts = valid["drought_class"].value_counts().reindex(config.DROUGHT_CLASSES, fill_value=0)
    pct = (counts / len(valid) * 100).round(2)

    summary = pd.DataFrame({"count": counts, "pct": pct})
    logger.info(f"Overall class balance ({len(valid)} labeled rows):\n{summary.to_string()}")

    per_field = (
        valid.groupby(["field", "drought_class"]).size()
        .unstack(fill_value=0)
        .reindex(columns=config.DROUGHT_CLASSES, fill_value=0)
    )
    return summary, per_field


def save_processed(df: pd.DataFrame, processed_dir: Path = config.PROCESSED_WEATHER_DIR) -> None:
    """First disk-write point in the pipeline. Writes:
      - one '<field>_with_spei.csv' per field (matches the naming pattern
        already used in your project's _quarantine folder, for consistency)
      - one 'pooled_with_spei.csv' combined file
    Only rows with a valid label are written -- the leading NaN rows per
    field are dropped here, not carried forward silently."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    labeled = df.dropna(subset=["drought_class"]).copy()
    n_dropped = len(df) - len(labeled)
    logger.info(f"Dropping {n_dropped} unlabeled row(s) before saving "
                f"(leading NaN rows from the 30-day rolling window).")

    for field_name, field_df in labeled.groupby("field"):
        out_path = processed_dir / f"{field_name}_with_spei.csv"
        field_df.to_csv(out_path, index=False)
        logger.info(f"[{field_name}] saved {len(field_df)} rows -> {out_path}")

    pooled_path = processed_dir / "pooled_with_spei.csv"
    labeled.to_csv(pooled_path, index=False)
    logger.info(f"Pooled dataset saved: {len(labeled)} rows -> {pooled_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_ingestion"))
    from data_ingestion.weather_loader import load_all_fields, pool_fields
    from preprocessing.water_balance import run_water_balance_pipeline
    from preprocessing.spei import compute_spei

    frames, reports = load_all_fields()
    if not frames:
        print("No fields loaded -- check Module 2 output first.")
        sys.exit(1)

    pooled = pool_fields(frames)
    pooled = run_water_balance_pipeline(pooled)
    pooled, fit_summary = compute_spei(pooled)
    pooled = generate_labels(pooled)

    summary, per_field = report_class_balance(pooled)
    print(f"\nOverall class balance:\n{summary.to_string()}")
    print(f"\nPer-field class counts:\n{per_field.to_string()}")

    save_processed(pooled)
    print(f"\nProcessed files saved under: {config.PROCESSED_WEATHER_DIR}")
