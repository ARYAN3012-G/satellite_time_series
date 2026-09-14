"""
tests/test_sequence_builder.py -- automated tests for Module 6.
Run with: pytest tests/test_sequence_builder.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
from features.sequence_builder import (  # noqa: E402
    build_sequences_for_field,
    build_sequences_all_fields,
    save_sequences,
    _assert_no_leakage,
)


def _make_labeled_field_df(field_name="fieldA", n_days=50, start="2020-01-01", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    df = pd.DataFrame({config.DATE_COLUMN: dates, "field": field_name})
    for feat in config.MODEL_INPUT_FEATURES:
        df[feat] = rng.uniform(0, 10, n_days)
    df["label"] = rng.integers(0, 3, n_days)
    df["drought_class"] = df["label"].map(config.INVERSE_LABEL_MAP)
    # leakage columns present in the source file, as they would be from Module 5
    df["Water_Balance"] = rng.uniform(-5, 5, n_days)
    df["WB30"] = rng.uniform(-50, 50, n_days)
    df["SPEI"] = rng.normal(0, 1, n_days)
    return df


def test_leakage_assertion_passes_for_clean_feature_list():
    _assert_no_leakage(config.MODEL_INPUT_FEATURES)  # should not raise


def test_leakage_assertion_raises_if_leakage_column_present():
    with pytest.raises(RuntimeError, match="LEAKAGE DETECTED"):
        _assert_no_leakage(config.MODEL_INPUT_FEATURES + ["SPEI"])


def test_build_sequences_shape_is_correct():
    df = _make_labeled_field_df(n_days=50)  # 50 - 30 = 20 sequences
    X, y, meta = build_sequences_for_field(df, sequence_length=30)
    assert X.shape == (20, 30, len(config.MODEL_INPUT_FEATURES))
    assert y.shape == (20,)
    assert len(meta) == 20


def test_build_sequences_feature_count_matches_config_exactly():
    """Structural leakage check: output feature dimension must equal
    len(MODEL_INPUT_FEATURES) exactly, even though the source DataFrame
    ALSO contains Water_Balance/WB30/SPEI columns."""
    df = _make_labeled_field_df(n_days=40)
    X, y, meta = build_sequences_for_field(df, sequence_length=30)
    assert X.shape[2] == len(config.MODEL_INPUT_FEATURES)
    assert X.shape[2] == 9


def test_window_does_not_include_target_day_weather():
    """Strict forecast framing: sequence i's window is rows [i, i+30),
    and the target label comes from row i+30 -- day t+1's own weather
    must never appear inside its own input window."""
    df = _make_labeled_field_df(n_days=35)
    # Make feature values equal to the row index, so we can check exactly
    # which rows ended up in window 0's input.
    for feat in config.MODEL_INPUT_FEATURES:
        df[feat] = np.arange(len(df), dtype=float)

    X, y, meta = build_sequences_for_field(df, sequence_length=30)
    first_window = X[0, :, 0]  # first feature column of first window
    assert list(first_window) == list(range(30))  # rows 0..29
    # target for window 0 is row 30's label -- row 30's features must
    # NOT appear anywhere in the window
    assert 30 not in first_window


def test_target_label_is_day_after_window():
    df = _make_labeled_field_df(n_days=35)
    df["label"] = np.arange(len(df)) % 3  # deterministic labels by row index
    X, y, meta = build_sequences_for_field(df, sequence_length=30)
    # window 0 -> input rows 0-29, target = row 30's label
    assert y[0] == df["label"].iloc[30]


def test_date_gap_within_field_raises():
    df = _make_labeled_field_df(n_days=40)
    df = df.drop(df.index[20]).reset_index(drop=True)  # introduce a gap
    with pytest.raises(ValueError, match="non-consecutive-day gap"):
        build_sequences_for_field(df, sequence_length=30)


def test_insufficient_rows_returns_empty_not_error():
    df = _make_labeled_field_df(n_days=10)  # fewer than sequence_length
    X, y, meta = build_sequences_for_field(df, sequence_length=30)
    assert len(y) == 0
    assert X.shape == (0, 30, len(config.MODEL_INPUT_FEATURES))


def test_missing_required_column_raises():
    df = _make_labeled_field_df(n_days=40).drop(columns=["label"])
    with pytest.raises(ValueError, match="missing columns"):
        build_sequences_for_field(df, sequence_length=30)


def test_build_sequences_all_fields_pools_correctly():
    frames = {
        "fieldA": _make_labeled_field_df("fieldA", n_days=40, seed=1),
        "fieldB": _make_labeled_field_df("fieldB", n_days=40, seed=2),
    }
    X, y, meta = build_sequences_all_fields(frames)
    assert X.shape[0] == 10 + 10  # (40-30) sequences per field
    assert set(meta["field"].unique()) == {"fieldA", "fieldB"}


def test_sequences_do_not_cross_field_boundary_when_pooled():
    """Critical: pooling must not create a sequence that mixes rows from
    two different fields."""
    frames = {
        "fieldA": _make_labeled_field_df("fieldA", n_days=35, seed=1),
        "fieldB": _make_labeled_field_df("fieldB", n_days=35, seed=2),
    }
    X, y, meta = build_sequences_all_fields(frames)
    # Each field independently contributes (35-30)=5 sequences, never mixed
    assert (meta["field"] == "fieldA").sum() == 5
    assert (meta["field"] == "fieldB").sum() == 5


def test_save_sequences_writes_npz_with_expected_keys(tmp_path):
    df = _make_labeled_field_df(n_days=40)
    X, y, meta = build_sequences_for_field(df, sequence_length=30)
    out_path = tmp_path / "test_sequences.npz"
    save_sequences(X, y, meta, out_path=out_path)

    loaded = np.load(out_path, allow_pickle=True)
    assert set(loaded.keys()) >= {"X", "y", "field", "target_date", "feature_names", "class_names"}
    assert loaded["X"].shape == X.shape
    np.testing.assert_array_equal(loaded["y"], y)
    assert list(loaded["feature_names"]) == config.MODEL_INPUT_FEATURES
