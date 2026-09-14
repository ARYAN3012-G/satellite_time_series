"""
src/preprocessing/run_pipeline.py
==================================
Orchestrates the full preprocessing chain (Modules 2–5) for all 8 fields:
  load → water balance → SPEI → labels → save processed CSVs

Fixed to use the CURRENT src/ module API (weather_loader, water_balance,
spei, labels) rather than the stale v0 API that no longer exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from data_ingestion.weather_loader import load_all_fields, pool_fields
from preprocessing.water_balance import add_water_balance_and_wb30
from preprocessing.spei import compute_spei
from preprocessing.labels import generate_labels, save_processed, report_class_balance
from utils.logger import get_logger

logger = get_logger(__name__)


def run_preprocessing_pipeline(save: bool = True) -> dict:
    """Run the full Module 2→5 chain for ALL discovered fields.

    Returns: {field_name: labeled_DataFrame}
    Saves:   data/processed/weather/<field>_with_spei.csv
             data/processed/weather/pooled_with_spei.csv
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Module 2: Load and validate all fields
    logger.info("Module 2: Loading all field CSVs ...")
    field_frames, reports = load_all_fields()

    if not field_frames:
        logger.error("No fields loaded — aborting pipeline.")
        sys.exit(1)

    logger.info(f"Loaded {len(field_frames)} field(s).")
    for name, r in reports.items():
        flag = "OK" if r.is_clean else "ISSUES"
        logger.info(
            f"  [{flag}] {name}: {r.n_rows_raw} rows, "
            f"{r.n_duplicate_dates} dup dates, {r.n_date_gaps} gap days, "
            f"range {r.date_min.date()}..{r.date_max.date()}"
        )

    # Module 3: Water balance + WB30 (pooled, but computed per-field)
    logger.info("Module 3: Computing Water_Balance and WB30 ...")
    pooled = pool_fields(field_frames)
    pooled = add_water_balance_and_wb30(pooled)

    # Module 4: SPEI (per field × per calendar month)
    logger.info("Module 4: Computing SPEI ...")
    pooled, fit_summary = compute_spei(pooled)
    logger.info(f"SPEI fit summary:\n{fit_summary.head(15).to_string(index=False)}")

    # Module 5: Labels
    logger.info("Module 5: Generating drought labels ...")
    pooled = generate_labels(pooled)

    # Report class balance
    summary, per_field = report_class_balance(pooled)
    logger.info(f"\nOverall class balance:\n{summary.to_string()}")
    logger.info(f"\nPer-field class counts:\n{per_field.to_string()}")

    if save:
        logger.info("Saving processed files ...")
        save_processed(pooled)

    # Build per-field dict from the pooled labeled frame
    labeled_frames = {
        field: group.copy()
        for field, group in pooled.dropna(subset=["drought_class"]).groupby("field")
    }

    logger.info("=" * 60)
    logger.info(f"PREPROCESSING COMPLETE — {len(labeled_frames)} field(s) processed")
    logger.info("=" * 60)
    return labeled_frames


if __name__ == "__main__":
    result = run_preprocessing_pipeline(save=True)
    print(f"\nPreprocessing complete: {len(result)} fields ready.")
    for name, df in result.items():
        valid = df.dropna(subset=["drought_class"])
        counts = valid["drought_class"].value_counts().to_dict()
        print(
            f"  {name:30s} rows={len(df):6d}  "
            f"range={df[config.DATE_COLUMN].min().date()} → "
            f"{df[config.DATE_COLUMN].max().date()}  "
            f"classes={counts}"
        )
