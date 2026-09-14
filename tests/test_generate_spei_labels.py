"""
Tests for Module_02_Label_Generation/generate_spei_labels.py

Run with: pytest tests/test_generate_spei_labels.py -v
(run from the DeepDroughtUAV project root)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Module_02_Label_Generation"))

from generate_spei_labels import (
    compute_water_balance,
    compute_spei,
    generate_labels,
    LabelGenerationError,
    DROUGHT_LABEL_MAP,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    return pd.DataFrame({
        "valid_time": dates,
        "Precipitation_Flux": [5.0] * 40,
        "ReferenceET_PenmanMonteith_FAO56": [2.0] * 40,
    })


def test_water_balance_formula(sample_df):
    out = compute_water_balance(sample_df)
    assert (out["Water_Balance"] == 3.0).all()


def test_water_balance_rolling_nan_before_window(sample_df):
    out = compute_water_balance(sample_df, window=30)
    assert out["WB30"].iloc[:29].isna().all()
    assert out["WB30"].iloc[29:].notna().all()


def test_water_balance_missing_column_raises(sample_df):
    bad = sample_df.drop(columns=["Precipitation_Flux"])
    with pytest.raises(LabelGenerationError, match="Precipitation_Flux"):
        compute_water_balance(bad)


def test_only_three_classes_exist():
    assert set(DROUGHT_LABEL_MAP.keys()) == {"Healthy", "Moderate", "Severe"}
    assert len(DROUGHT_LABEL_MAP) == 3


def test_label_boundaries():
    df = pd.DataFrame({"SPEI": [1.0, 0.0, -0.5, -1.49, -1.5, -1.51, -3.0, np.nan]})
    out = generate_labels(df)
    assert out.loc[0, "Drought_Label"] == "Healthy"
    assert out.loc[1, "Drought_Label"] == "Healthy"   # exactly 0.0
    assert out.loc[2, "Drought_Label"] == "Moderate"
    assert out.loc[3, "Drought_Label"] == "Moderate"
    assert out.loc[4, "Drought_Label"] == "Moderate"  # exactly -1.5
    assert out.loc[5, "Drought_Label"] == "Severe"
    assert out.loc[6, "Drought_Label"] == "Severe"
    assert pd.isna(out.loc[7, "Drought_Label"])


def test_target_matches_label_map():
    df = pd.DataFrame({"SPEI": [1.0, -0.5, -2.0]})
    out = generate_labels(df)
    for _, row in out.iterrows():
        assert row["Target"] == DROUGHT_LABEL_MAP[row["Drought_Label"]]


def test_missing_spei_column_raises():
    df = pd.DataFrame({"other": [1, 2, 3]})
    with pytest.raises(LabelGenerationError, match="SPEI"):
        generate_labels(df)


def test_spei_seasonality_respected():
    """Same raw WB30 value in a wide-variance month vs narrow-variance
    month should yield different SPEI — proves per-month fitting works,
    not a naive global z-score."""
    rng = np.random.default_rng(0)
    dates, wb_values = [], []
    for year in range(2000, 2010):
        for month, scale in [(1, 100), (7, 10)]:
            month_dates = pd.date_range(f"{year}-{month:02d}-01", periods=28, freq="D")
            dates.extend(month_dates)
            wb_values.extend(rng.normal(0, scale, 28))
    df = pd.DataFrame({"valid_time": pd.to_datetime(dates), "WB30": wb_values})

    jan_mask = (df["valid_time"].dt.year == 2005) & (df["valid_time"].dt.month == 1) & (df["valid_time"].dt.day == 1)
    jul_mask = (df["valid_time"].dt.year == 2005) & (df["valid_time"].dt.month == 7) & (df["valid_time"].dt.day == 1)
    df.loc[jan_mask, "WB30"] = 20.0
    df.loc[jul_mask, "WB30"] = 20.0

    out = compute_spei(df)
    jan_spei = out.loc[jan_mask, "SPEI"].iloc[0]
    jul_spei = out.loc[jul_mask, "SPEI"].iloc[0]
    assert abs(jan_spei) < abs(jul_spei)
