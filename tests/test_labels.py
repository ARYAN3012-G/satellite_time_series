"""
tests/test_labels.py -- automated tests for Module 5.
Run with: pytest tests/test_labels.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
from preprocessing.labels import generate_labels, report_class_balance, save_processed  # noqa: E402


def _make_spei_df(field_name="fieldA", spei_values=None):
    if spei_values is None:
        spei_values = [1.5, 0.0, -0.5, -1.5, -2.0, np.nan]
    n = len(spei_values)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        config.DATE_COLUMN: dates,
        "field": field_name,
        "SPEI": spei_values,
    })


def test_generate_labels_adds_columns():
    df = _make_spei_df()
    out = generate_labels(df)
    assert "drought_class" in out.columns
    assert "label" in out.columns


def test_generate_labels_matches_config_thresholds():
    df = _make_spei_df(spei_values=[1.5, 0.0, -0.5, -1.5, -2.0])
    out = generate_labels(df)
    expected_classes = ["Healthy", "Healthy", "Moderate", "Moderate", "Severe"]
    assert out["drought_class"].tolist() == expected_classes


def test_generate_labels_nan_spei_stays_nan():
    df = _make_spei_df(spei_values=[1.0, np.nan, -1.0])
    out = generate_labels(df)
    assert pd.isna(out["drought_class"].iloc[1])
    assert pd.isna(out["label"].iloc[1])


def test_generate_labels_label_is_correct_integer():
    df = _make_spei_df(spei_values=[1.5, -0.5, -2.0])
    out = generate_labels(df)
    assert out["label"].tolist() == [
        config.LABEL_MAP["Healthy"],
        config.LABEL_MAP["Moderate"],
        config.LABEL_MAP["Severe"],
    ]


def test_generate_labels_missing_spei_column_raises():
    df = pd.DataFrame({"field": ["a"], config.DATE_COLUMN: [pd.Timestamp("2020-01-01")]})
    with pytest.raises(ValueError, match="'SPEI' column not found"):
        generate_labels(df)


def test_report_class_balance_counts_are_correct():
    df = _make_spei_df(spei_values=[1.0, 1.0, -0.5, -2.0])  # 2 Healthy, 1 Moderate, 1 Severe
    labeled = generate_labels(df)
    summary, per_field = report_class_balance(labeled)
    assert summary.loc["Healthy", "count"] == 2
    assert summary.loc["Moderate", "count"] == 1
    assert summary.loc["Severe", "count"] == 1


def test_report_class_balance_excludes_nan_rows():
    df = _make_spei_df(spei_values=[1.0, np.nan, np.nan])
    labeled = generate_labels(df)
    summary, _ = report_class_balance(labeled)
    assert summary["count"].sum() == 1  # only the one valid row counted


def test_save_processed_writes_per_field_and_pooled_files(tmp_path):
    df_a = _make_spei_df("fieldA", spei_values=[1.0, -0.5, -2.0])
    df_b = _make_spei_df("fieldB", spei_values=[1.0, -0.5, -2.0])
    combined = pd.concat([df_a, df_b], ignore_index=True)
    labeled = generate_labels(combined)

    save_processed(labeled, processed_dir=tmp_path)

    assert (tmp_path / "fieldA_with_spei.csv").exists()
    assert (tmp_path / "fieldB_with_spei.csv").exists()
    assert (tmp_path / "pooled_with_spei.csv").exists()


def test_save_processed_drops_unlabeled_rows(tmp_path):
    df = _make_spei_df(spei_values=[1.0, np.nan, -0.5])  # 1 NaN row
    labeled = generate_labels(df)

    save_processed(labeled, processed_dir=tmp_path)

    pooled = pd.read_csv(tmp_path / "pooled_with_spei.csv")
    assert len(pooled) == 2  # NaN row dropped
    assert pooled["drought_class"].isna().sum() == 0


def test_save_processed_roundtrip_preserves_label_values(tmp_path):
    df = _make_spei_df(spei_values=[1.5, -0.5, -2.0])
    labeled = generate_labels(df)
    save_processed(labeled, processed_dir=tmp_path)

    pooled = pd.read_csv(tmp_path / "pooled_with_spei.csv")
    assert pooled["drought_class"].tolist() == ["Healthy", "Moderate", "Severe"]
    assert pooled["label"].tolist() == [0, 1, 2]
