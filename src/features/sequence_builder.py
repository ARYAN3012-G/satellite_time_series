"""
Module 6 -- features/sequence_builder.py

Builds 30-day sliding-window sequences for the LSTM:
  input:  30 consecutive days of RAW WEATHER ONLY (config.MODEL_INPUT_FEATURES)
  target: the drought label for the day AFTER the window ends

Strict forecast framing: the label for day t+1 is predicted using ONLY
weather from days [t-29 ... t]. Day t+1's own weather is never in the
input window -- that would be same-day leakage (using the future to
predict the future).

THE LEAKAGE BOUNDARY (structural, not just documented):
  This module reads features ONLY from config.MODEL_INPUT_FEATURES. It
  never touches Water_Balance, WB30, or SPEI as model input -- those are
  used only to look up the label, never placed into the feature array.
  A runtime assertion at import time (in config.py) already guarantees
  MODEL_INPUT_FEATURES and LEAKAGE_COLUMNS can't overlap; this module
  additionally asserts the built feature array's column count matches
  len(MODEL_INPUT_FEATURES) exactly, so a stray leakage column silently
  appended upstream would break loudly here, not corrupt training silently.

Input:  per-field labeled DataFrames from Module 5 (one CSV per field,
        or the pooled DataFrame), sorted chronologically, no gaps in the
        rows actually used (Module 2 already flagged any gaps).
Output: X (n_sequences, 30, 9) float32 array, y (n_sequences,) int array,
        plus a metadata DataFrame recording which field/date each
        sequence came from (needed later for chronological/LOFO splits).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _assert_no_leakage(columns_used: list) -> None:
    """Hard structural check: raise immediately if any leakage column
    ever ends up in the feature set, regardless of how it got there."""
    leaked = set(columns_used) & set(config.LEAKAGE_COLUMNS)
    if leaked:
        raise RuntimeError(
            f"LEAKAGE DETECTED: {leaked} present in sequence builder feature "
            f"columns. This must never happen -- aborting rather than "
            f"silently training on it."
        )


def build_sequences_for_field(
    field_df: pd.DataFrame,
    sequence_length: int = config.SEQUENCE_LENGTH_DAYS,
) -> tuple:
    """Build sliding-window sequences for ONE field's chronologically
    sorted, gap-free, labeled DataFrame.

    Returns (X, y, meta):
      X: (n_sequences, sequence_length, n_features) float32
      y: (n_sequences,) int, the label for the day AFTER each window
      meta: DataFrame with 'field' and 'target_date' per sequence, for
            later chronological/LOFO splitting.

    A sequence at position i uses rows [i, i+sequence_length) as input
    and row i+sequence_length's label as the target -- i.e. the window
    ends the day BEFORE the predicted day, never including it.
    """
    _assert_no_leakage(config.MODEL_INPUT_FEATURES)

    required_cols = config.MODEL_INPUT_FEATURES + ["label", config.DATE_COLUMN, "field"]
    missing = [c for c in required_cols if c not in field_df.columns]
    if missing:
        raise ValueError(f"build_sequences_for_field: missing columns {missing}")

    field_df = field_df.sort_values(config.DATE_COLUMN).reset_index(drop=True)

    # Verify no date gaps within the rows actually being sequenced --
    # a gap here would silently stitch together two non-consecutive
    # periods into one "30-day" window.
    date_diffs = field_df[config.DATE_COLUMN].diff().dropna()
    if not (date_diffs == pd.Timedelta(days=1)).all():
        n_gaps = (date_diffs != pd.Timedelta(days=1)).sum()
        raise ValueError(
            f"build_sequences_for_field: {n_gaps} non-consecutive-day gap(s) "
            f"found in field '{field_df['field'].iloc[0]}' after label "
            f"generation. Sequences must not be built across a gap -- "
            f"fix upstream (Module 2/5) rather than silently bridging it here."
        )

    feature_matrix = field_df[config.MODEL_INPUT_FEATURES].values.astype(np.float32)
    labels = field_df["label"].values
    dates = field_df[config.DATE_COLUMN].values
    field_name = field_df["field"].iloc[0]

    n_rows = len(field_df)
    n_sequences = n_rows - sequence_length  # need sequence_length input days + 1 target day

    if n_sequences <= 0:
        logger.warning(
            f"[{field_name}] only {n_rows} rows -- not enough for even one "
            f"{sequence_length}-day sequence + 1 target day. Skipping."
        )
        return (
            np.empty((0, sequence_length, len(config.MODEL_INPUT_FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DataFrame(columns=["field", "target_date"]),
        )

    X = np.empty((n_sequences, sequence_length, len(config.MODEL_INPUT_FEATURES)), dtype=np.float32)
    y = np.empty((n_sequences,), dtype=np.int64)
    prev_label = np.empty((n_sequences,), dtype=np.int64)  # label at last day of window (day t)
    target_dates = []

    for i in range(n_sequences):
        X[i] = feature_matrix[i: i + sequence_length]
        target_label = labels[i + sequence_length]
        if pd.isna(target_label):
            raise ValueError(
                f"[{field_name}] NaN label encountered at target index "
                f"{i + sequence_length} -- Module 5 output should already "
                f"be fully labeled."
            )
        y[i] = int(target_label)
        prev_label[i] = int(labels[i + sequence_length - 1])  # last day of window = known at inference
        target_dates.append(dates[i + sequence_length])

    meta = pd.DataFrame({"field": field_name, "target_date": target_dates})

    assert X.shape == (n_sequences, sequence_length, len(config.MODEL_INPUT_FEATURES))
    assert X.shape[2] == len(config.MODEL_INPUT_FEATURES), (
        "Feature dimension mismatch -- possible leakage column contamination."
    )

    logger.info(f"[{field_name}] built {n_sequences} sequences "
                f"({sequence_length}-day windows).")

    return X, y, prev_label, meta


def build_sequences_all_fields(labeled_frames: dict,
                                sequence_length: int = None) -> tuple:
    """Build sequences for every field, then concatenate into one pooled
    (X, y, prev_label, meta) tuple. labeled_frames: {field_name: DataFrame}.
    sequence_length: window size in days (overrides config if given)."""
    if sequence_length is None:
        sequence_length = config.SEQUENCE_LENGTH_DAYS

    X_list, y_list, pl_list, meta_list = [], [], [], []

    for field_name, field_df in labeled_frames.items():
        X, y, prev_label, meta = build_sequences_for_field(
            field_df, sequence_length=sequence_length)
        if len(y) > 0:
            X_list.append(X)
            y_list.append(y)
            pl_list.append(prev_label)
            meta_list.append(meta)

    if not X_list:
        raise RuntimeError("build_sequences_all_fields: no sequences built for any field.")

    X_pooled = np.concatenate(X_list, axis=0)
    y_pooled = np.concatenate(y_list, axis=0)
    pl_pooled = np.concatenate(pl_list, axis=0)
    meta_pooled = pd.concat(meta_list, ignore_index=True)

    logger.info(f"Pooled sequences: X={X_pooled.shape}, y={y_pooled.shape}, "
                f"{meta_pooled['field'].nunique()} fields.")

    assert X_pooled.shape[2] == len(config.MODEL_INPUT_FEATURES)

    return X_pooled, y_pooled, pl_pooled, meta_pooled


def save_sequences(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame,
                    out_path: Path = None, prev_label: np.ndarray = None) -> Path:
    """Save the pooled sequences as a single .npz."""
    if out_path is None:
        out_path = config.SEQUENCES_DIR / "pooled_sequences.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = dict(
        X=X, y=y,
        field=meta["field"].values,
        target_date=meta["target_date"].values.astype(str),
        feature_names=np.array(config.MODEL_INPUT_FEATURES),
        class_names=np.array(config.DROUGHT_CLASSES),
    )
    if prev_label is not None:
        save_kwargs["prev_label"] = prev_label

    np.savez_compressed(out_path, **save_kwargs)
    logger.info(f"Sequences saved -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    import glob

    labeled_frames = {}
    for csv_path in glob.glob(str(config.PROCESSED_WEATHER_DIR / "*_with_spei.csv")):
        field_name = Path(csv_path).stem.replace("_with_spei", "")
        if field_name == "pooled":
            continue  # skip the combined file -- not a real field
        df = pd.read_csv(csv_path, parse_dates=[config.DATE_COLUMN])
        labeled_frames[field_name] = df

    if not labeled_frames:
        print("No processed field files found -- run Module 5 first "
              f"(expected files under {config.PROCESSED_WEATHER_DIR}).")
        sys.exit(1)

    print(f"Loaded {len(labeled_frames)} processed field file(s).")

    X, y, meta = build_sequences_all_fields(labeled_frames)

    print(f"\nX shape: {X.shape}  (n_sequences, {config.SEQUENCE_LENGTH_DAYS} days, "
          f"{len(config.MODEL_INPUT_FEATURES)} features)")
    print(f"y shape: {y.shape}")
    print(f"\nFeatures used (in order): {config.MODEL_INPUT_FEATURES}")
    print(f"\nClass distribution in y:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {config.INVERSE_LABEL_MAP[u]}: {c} ({c/len(y)*100:.2f}%)")

    save_sequences(X, y, meta)
