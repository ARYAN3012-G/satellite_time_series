"""
tests/test_spei.py -- automated tests for Module 4.
Run with: pytest tests/test_spei.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
from preprocessing.spei import compute_spei, _fit_log_logistic_and_transform  # noqa: E402


def _make_field_with_wb30(field_name="fieldA", n_years=15, seed=0):
    """Build a synthetic multi-year daily series with a WB30 column that
    has real seasonal variation, so per-month fitting has something
    meaningful to differentiate."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"2000-01-01", periods=365 * n_years, freq="D")
    month = dates.month
    # Seasonal signal: wetter (higher WB30) in months 6-9, drier otherwise
    seasonal_mean = np.where((month >= 6) & (month <= 9), 20.0, -30.0)
    wb30 = seasonal_mean + rng.normal(0, 15, len(dates))
    wb30[:29] = np.nan  # mimic Module 3's leading NaNs
    return pd.DataFrame({
        config.DATE_COLUMN: dates,
        "field": field_name,
        "WB30": wb30,
    })


def test_compute_spei_adds_column():
    df = _make_field_with_wb30()
    out, fit_summary = compute_spei(df)
    assert "SPEI" in out.columns


def test_spei_is_nan_where_wb30_is_nan():
    df = _make_field_with_wb30()
    out, _ = compute_spei(df)
    wb30_nan_mask = df["WB30"].isna()
    assert out.loc[wb30_nan_mask, "SPEI"].isna().all()


def test_spei_is_approximately_standard_normal():
    """The whole point of the log-logistic transform: output should be
    roughly mean 0, std 1 across a large enough sample."""
    df = _make_field_with_wb30(n_years=20)
    out, _ = compute_spei(df)
    valid_spei = out["SPEI"].dropna()
    assert abs(valid_spei.mean()) < 0.3
    assert 0.7 < valid_spei.std() < 1.3


def test_spei_fit_is_per_field_independent():
    """A wet field's distribution must not affect a dry field's SPEI."""
    df_wet = _make_field_with_wb30("wet_field", seed=1)
    df_wet["WB30"] = df_wet["WB30"] + 100  # shift much wetter
    df_dry = _make_field_with_wb30("dry_field", seed=2)

    combined = pd.concat([df_wet, df_dry], ignore_index=True)
    out, _ = compute_spei(combined)

    # Each field's SPEI should still be ~standard normal on its own terms,
    # despite wildly different absolute WB30 scales
    for f in ["wet_field", "dry_field"]:
        spei_f = out.loc[out["field"] == f, "SPEI"].dropna()
        assert abs(spei_f.mean()) < 0.3, f"{f} SPEI mean drifted: {spei_f.mean()}"


def test_spei_fit_is_per_month_not_global():
    """Verify the seasonal signal is actually being normalized away --
    i.e. wet-season and dry-season SPEI should have SIMILAR distributions
    after per-month fitting, even though raw WB30 differs hugely."""
    df = _make_field_with_wb30(n_years=20)
    out, _ = compute_spei(df)
    out["_month"] = out[config.DATE_COLUMN].dt.month

    wet_season_spei = out.loc[out["_month"].isin([6, 7, 8, 9]), "SPEI"].dropna()
    dry_season_spei = out.loc[~out["_month"].isin([6, 7, 8, 9]), "SPEI"].dropna()

    # After per-month standardization, both seasons' SPEI should center near 0
    assert abs(wet_season_spei.mean()) < 0.3
    assert abs(dry_season_spei.mean()) < 0.3


def test_missing_required_columns_raises():
    df = pd.DataFrame({"field": ["a"], config.DATE_COLUMN: [pd.Timestamp("2020-01-01")]})
    with pytest.raises(ValueError, match="missing columns"):
        compute_spei(df)


def test_fit_log_logistic_raises_on_insufficient_data():
    with pytest.raises(ValueError, match="only 1 valid"):
        _fit_log_logistic_and_transform(np.array([5.0, np.nan, np.nan]))


def test_fit_log_logistic_handles_negative_values():
    """WB30 is frequently negative (water deficit) -- the fit must handle
    this via the shift, not silently fail or produce NaN for valid input."""
    rng = np.random.default_rng(42)
    values = rng.normal(-50, 15, 200)  # all plausible negative WB30 values
    spei, params = _fit_log_logistic_and_transform(values)
    assert not np.isnan(spei).any()
    assert params["shift"] > 0  # confirms shift was applied


def test_classify_spei_matches_config_thresholds():
    """Cross-check: SPEI values, once computed, should classify sensibly
    via config.classify_spei (Module 5 will use this directly)."""
    assert config.classify_spei(1.0) == "Healthy"
    assert config.classify_spei(-0.5) == "Moderate"
    assert config.classify_spei(-2.0) == "Severe"


def test_fit_summary_covers_all_field_month_combinations():
    df = _make_field_with_wb30(n_years=15)
    _, fit_summary = compute_spei(df)
    assert len(fit_summary) == 12  # one field, 12 months
    assert set(fit_summary["month"]) == set(range(1, 13))
