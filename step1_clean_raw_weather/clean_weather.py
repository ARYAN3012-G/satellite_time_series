"""
clean_weather.py
=================
Module 1 — Data Preprocessing

Purpose
-------
Read raw per-field weather CSVs, validate schema, parse dates, clean
missing values, and save a cleaned version to data/processed/.

This is a standalone module — it does not depend on any prior project's
code, per the decision to start this pipeline fresh.

Design decisions
-----------------
- Missing values are DROPPED, not imputed. Fabricating weather values
  for a drought model risks silently corrupting the exact signal the
  model needs to learn honestly — an incomplete-but-honest dataset is
  preferable to a complete-but-partially-invented one.
- latitude/longitude are kept in the cleaned output as METADATA only.
  They are explicitly NOT treated as training features at this stage —
  that decision is deferred to Module 3, per project scope.
- Locations are auto-discovered from data/raw/ subfolders, not hardcoded,
  so adding a new field later requires no code changes here.
"""
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("clean_weather")

# ---------------------------------------------------------------------------
# Paths (relative to this file's project root, portable across machines)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Required raw schema (confirmed against real CDS agrometeorological export)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = (
    "valid_time",
    "Wind_Speed_10m_Mean_24h",
    "Temperature_Air_2m_Max_24h",
    "Temperature_Air_2m_Mean_24h",
    "Temperature_Air_2m_Min_24h",
    "Derived_Relative_Humidity_2m_Max_24h",
    "Derived_Relative_Humidity_2m_Min_24h",
    "Precipitation_Flux",
    "ReferenceET_PenmanMonteith_FAO56",
    "Solar_Radiation_Flux",
    "Vapour_Pressure_Mean_24h",
    "latitude",
    "longitude",
)


def _parse_valid_time(series: pd.Series) -> pd.Series:
    """
    Parse the 'valid_time' column robustly. Different fields' raw CSVs
    use different date conventions (confirmed: some use ISO YYYY-MM-DD,
    others use DD-MM-YYYY) -- pandas' automatic format inference can lock
    onto the wrong format from the first few rows and then fail on a
    later ambiguous date (e.g. day=13 misread as a month). This tries
    known formats explicitly, in order, before falling back to pandas'
    slower per-row mixed-format inference.
    """
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y"):
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    # last resort: infer format per-row (slower, but handles genuinely
    # mixed-format files rather than failing outright)
    logger.warning("Could not match a single consistent date format — falling back to mixed-format inference")
    return pd.to_datetime(series, format="mixed", dayfirst=True)


class WeatherCleaningError(ValueError):
    """Raised when a raw weather file cannot be cleaned due to a schema or data issue."""
    """Raised when a raw weather file cannot be cleaned due to a schema or data issue."""


def discover_raw_locations() -> list[str]:
    """Scan data/raw/ and return every location subfolder found."""
    if not RAW_DIR.exists():
        logger.warning(f"Raw data directory does not exist: {RAW_DIR}")
        return []
    locations = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))
    logger.info(f"Discovered {len(locations)} location(s): {locations}")
    return locations


def _find_raw_csv(location_folder: Path) -> Path:
    """Locate the CSV file inside a location folder, preferring the
    canonical {location}_weather.csv name but tolerating a single
    alternatively-named CSV."""
    canonical = location_folder / f"{location_folder.name}_weather.csv"
    if canonical.exists():
        return canonical

    csv_files = [f for f in location_folder.glob("*.csv") if not f.name.startswith(".")]
    if len(csv_files) == 1:
        logger.warning(f"'{location_folder.name}': using non-canonical file {csv_files[0].name}")
        return csv_files[0]
    if len(csv_files) == 0:
        raise WeatherCleaningError(f"No CSV file found in {location_folder}")
    raise WeatherCleaningError(
        f"Multiple CSV files found in {location_folder}, none canonically named: "
        f"{[f.name for f in csv_files]}"
    )


def clean_weather(location: str) -> pd.DataFrame:
    """
    Load, validate, and clean the raw weather CSV for one location.

    Parameters
    ----------
    location : str
        Location folder name under data/raw/.

    Returns
    -------
    Cleaned DataFrame: schema-validated, chronologically sorted, missing
    values dropped, with a 'location' column added.

    Raises
    ------
    WeatherCleaningError
        If the location folder/file is missing, required columns are
        absent, or duplicate dates are found.
    """
    location = location.lower().strip()
    folder = RAW_DIR / location
    if not folder.exists():
        raise WeatherCleaningError(
            f"Location folder not found: {folder}. Available: {discover_raw_locations()}"
        )

    csv_path = _find_raw_csv(folder)
    logger.info(f"Cleaning '{location}' from {csv_path.name}")

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise WeatherCleaningError(f"'{location}': missing required column(s): {missing_cols}")

    df["valid_time"] = _parse_valid_time(df["valid_time"])
    df = df.sort_values("valid_time").reset_index(drop=True)

    n_dupes = df["valid_time"].duplicated().sum()
    if n_dupes:
        raise WeatherCleaningError(f"'{location}': {n_dupes} duplicate date(s) found — resolve before cleaning")

    n_before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"'{location}': dropped {n_dropped} row(s) with missing values ({n_dropped/n_before:.1%})")

    df["location"] = location

    logger.info(
        f"'{location}' cleaned: {len(df)} rows, "
        f"{df['valid_time'].min().date()} -> {df['valid_time'].max().date()}"
    )
    return df


def clean_and_save(location: str) -> Path:
    """Clean one location's data and save to data/processed/{location}_clean.csv."""
    df = clean_weather(location)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{location}_clean.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved -> {out_path}")
    return out_path


def clean_all_locations() -> dict[str, pd.DataFrame]:
    """Clean every location found under data/raw/. Failures are logged
    and skipped, not fatal, so one bad field doesn't block the rest."""
    results = {}
    for location in discover_raw_locations():
        try:
            df = clean_weather(location)
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(PROCESSED_DIR / f"{location}_clean.csv", index=False)
            results[location] = df
        except WeatherCleaningError as exc:
            logger.error(f"Skipping '{location}': {exc}")

    logger.info(f"Cleaning complete: {len(results)}/{len(discover_raw_locations())} location(s) succeeded")
    return results


if __name__ == "__main__":
    all_results = clean_all_locations()
    print("\n--- Summary ---")
    for location, df in all_results.items():
        print(f"{location:30s} rows={len(df):6d}  range={df['valid_time'].min().date()} -> {df['valid_time'].max().date()}")
