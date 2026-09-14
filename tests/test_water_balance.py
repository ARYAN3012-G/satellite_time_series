"""
tests/test_water_balance.py -- automated tests for Module 3.
Run with: pytest tests/test_water_balance.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
from preprocessing.water_balance import (  # noqa: E402
    compute_daily_water_balance,
    compute_wb30,
    add_water_balance_and_wb30,
)


def _make_field_df(field_name, n_days=40, start="2020-01-01", precip=2.0, et=5.0):
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({
        config.DATE_COLUMN: dates,
        "field": field_name,
        "Precipitation_Flux": precip,
        "ReferenceET_PenmanMonteith_FAO56": et,
    })


def test_daily_water_balance_is_precip_minus_et():
    df = _make_field_df("f1", precip=3.0, et=5.0)
    out = compute_daily_water_balance(df)
    assert (out["Water_Balance"] == -2.0).all()


def test_daily_water_balance_missing_column_raises():
    df = _make_field_df("f1").drop(columns=["Precipitation_Flux"])
    with pytest.raises(ValueError, match="Precipitation_Flux"):
        compute_daily_water_balance(df)


def test_wb30_first_29_rows_are_nan_per_field():
    df = _make_field_df("f1", n_days=40, precip=2.0, et=5.0)
    df = compute_daily_water_balance(df)
    out = compute_wb30(df, window_days=30)

    assert out["WB30"].iloc[:29].isna().all()
    assert out["WB30"].iloc[29:].notna().all()


def test_wb30_value_matches_manual_sum():
    df = _make_field_df("f1", n_days=35, precip=2.0, et=5.0)
    df = compute_daily_water_balance(df)
    out = compute_wb30(df, window_days=30)

    # constant daily water balance of -3.0 -> 30-day sum = -90.0
    expected = -3.0 * 30
    row30 = out.iloc[29]  # first row with a defined WB30 (0-indexed day 29 = 30th day)
    assert row30["WB30"] == pytest.approx(expected)


def test_wb30_does_not_cross_field_boundary():
    """The core correctness guarantee of this module: field B's early rows
    must NOT include any of field A's trailing Water_Balance values, even
    though they sit in adjacent rows of the same pooled/sorted DataFrame."""
    df_a = _make_field_df("fieldA", n_days=35, precip=0.0, et=10.0)   # very negative WB
    df_b = _make_field_df("fieldB", n_days=35, precip=10.0, et=0.0)   # very positive WB
    pooled = pd.concat([df_a, df_b], ignore_index=True)

    pooled = compute_daily_water_balance(pooled)
    out = compute_wb30(pooled, window_days=30)

    field_b_first_valid = out[(out["field"] == "fieldB") & out["WB30"].notna()].iloc[0]
    # If it wrongly leaked field A's negative water balance in, this would
    # be much less than the pure field-B expectation of +10*30 = 300.
    assert field_b_first_valid["WB30"] == pytest.approx(300.0)


def test_wb30_missing_field_column_raises():
    df = _make_field_df("f1").drop(columns=["field"])
    df = compute_daily_water_balance(df)
    with pytest.raises(ValueError, match="'field' column missing"):
        compute_wb30(df)


def test_wb30_requires_water_balance_first():
    df = _make_field_df("f1")  # no Water_Balance column yet
    with pytest.raises(ValueError, match="Water_Balance"):
        compute_wb30(df)


def test_add_water_balance_and_wb30_end_to_end():
    df_a = _make_field_df("fieldA", n_days=40)
    df_b = _make_field_df("fieldB", n_days=40)
    pooled = pd.concat([df_a, df_b], ignore_index=True)

    out = add_water_balance_and_wb30(pooled)

    assert "Water_Balance" in out.columns
    assert "WB30" in out.columns
    assert out.groupby("field")["WB30"].apply(lambda s: s.notna().sum() == 11).all()


def test_wb30_sorted_by_field_then_date_regardless_of_input_order():
    df_a = _make_field_df("fieldA", n_days=35)
    df_b = _make_field_df("fieldB", n_days=35)
    pooled = pd.concat([df_b, df_a], ignore_index=True)  # deliberately out of order
    pooled = pooled.sample(frac=1, random_state=2)        # and shuffled

    pooled = compute_daily_water_balance(pooled)
    out = compute_wb30(pooled, window_days=30)

    for f in ["fieldA", "fieldB"]:
        sub = out[out["field"] == f]
        assert sub[config.DATE_COLUMN].is_monotonic_increasing
