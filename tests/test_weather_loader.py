"""
tests/test_weather_loader.py -- automated tests for Module 2.
Run with: pytest tests/test_weather_loader.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
from data_ingestion.weather_loader import (  # noqa: E402
    load_single_field,
    load_all_fields,
    pool_fields,
    discover_field_files,
)


def _make_valid_df(n_days=100, start="2020-01-01"):
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({
        config.DATE_COLUMN: dates,
        "Wind_Speed_10m_Mean_24h": 2.0,
        "Temperature_Air_2m_Max_24h": 35.0,
        "Temperature_Air_2m_Mean_24h": 28.0,
        "Temperature_Air_2m_Min_24h": 20.0,
        "Derived_Relative_Humidity_2m_Max_24h": 80.0,
        "Derived_Relative_Humidity_2m_Min_24h": 40.0,
        "Precipitation_Flux": 1.0,
        "ReferenceET_PenmanMonteith_FAO56": 5.0,
        "Solar_Radiation_Flux": 20.0,
    })


def test_load_single_field_clean(tmp_path):
    df = _make_valid_df()
    csv_path = tmp_path / "test_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, report = load_single_field(csv_path, "test_field")

    assert len(loaded) == 100
    assert report.is_clean
    assert report.n_duplicate_dates == 0
    assert report.n_date_gaps == 0
    assert (loaded[config.DATE_COLUMN].diff().dropna() == pd.Timedelta(days=1)).all()


def test_missing_required_column_raises(tmp_path):
    df = _make_valid_df().drop(columns=["Precipitation_Flux"])
    csv_path = tmp_path / "bad_weather.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_single_field(csv_path, "bad_field")


def test_duplicate_dates_are_dropped(tmp_path):
    df = _make_valid_df()
    dup_row = df.iloc[[10]]
    df = pd.concat([df, dup_row], ignore_index=True)
    csv_path = tmp_path / "dup_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, report = load_single_field(csv_path, "dup_field")

    assert report.n_duplicate_dates == 1
    assert loaded[config.DATE_COLUMN].duplicated().sum() == 0  # dropped in output


def test_date_gap_is_detected(tmp_path):
    df = _make_valid_df()
    df = df.drop(df.index[50]).reset_index(drop=True)  # remove one day -> gap
    csv_path = tmp_path / "gap_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, report = load_single_field(csv_path, "gap_field")

    assert report.n_date_gaps == 1
    assert not report.is_clean


def test_output_sorted_chronologically(tmp_path):
    df = _make_valid_df().sample(frac=1, random_state=1)  # shuffle rows
    csv_path = tmp_path / "shuffled_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, _ = load_single_field(csv_path, "shuffled_field")

    assert loaded[config.DATE_COLUMN].is_monotonic_increasing


def test_dayfirst_date_format_is_parsed_correctly(tmp_path):
    """Real-world failure case: '13-01-1979' does not match %m-%d-%Y since
    13 can't be a month. Must resolve via explicit DD-MM-YYYY matching."""
    df = _make_valid_df(n_days=20, start="1979-01-01")
    df[config.DATE_COLUMN] = df[config.DATE_COLUMN].dt.strftime("%d-%m-%Y")
    csv_path = tmp_path / "dmy_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, report = load_single_field(csv_path, "dmy_field")

    assert report.is_clean
    assert loaded[config.DATE_COLUMN].iloc[0] == pd.Timestamp("1979-01-01")
    assert loaded[config.DATE_COLUMN].iloc[12] == pd.Timestamp("1979-01-13")



def test_mixed_iso_and_dayfirst_dates_do_not_corrupt(tmp_path):
    """Regression test for a real bug found during development: mixing
    ISO ('1979-01-02') and DD-MM-YYYY ('13-01-1979') dates in the SAME
    column, then parsing with pandas format='mixed' + dayfirst=True,
    silently reparsed '1979-01-02' as 1979-02-01 -- a wrong date, not a
    crash. This must never happen again."""
    mixed_dates = ["1979-01-01", "1979-01-02", "13-01-1979", "1979-01-14", "1979-01-13"]
    n = len(mixed_dates)
    df = pd.DataFrame({
        config.DATE_COLUMN: mixed_dates,
        "Wind_Speed_10m_Mean_24h": 2.0,
        "Temperature_Air_2m_Max_24h": 35.0,
        "Temperature_Air_2m_Mean_24h": 28.0,
        "Temperature_Air_2m_Min_24h": 20.0,
        "Derived_Relative_Humidity_2m_Max_24h": 80.0,
        "Derived_Relative_Humidity_2m_Min_24h": 40.0,
        "Precipitation_Flux": 1.0,
        "ReferenceET_PenmanMonteith_FAO56": 5.0,
        "Solar_Radiation_Flux": 20.0,
    })
    csv_path = tmp_path / "mixed_format_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, report = load_single_field(csv_path, "mixed_field")
    parsed_dates = sorted(loaded[config.DATE_COLUMN].tolist())

    expected = sorted([
        pd.Timestamp("1979-01-01"),
        pd.Timestamp("1979-01-02"),  # must NOT become Feb 1
        pd.Timestamp("1979-01-13"),  # duplicate of 13-01-1979, kept as one
        pd.Timestamp("1979-01-14"),
    ])
    # Note: 1979-01-13 appears twice (once ISO, once DD-MM-YYYY) -> that's
    # a legitimate duplicate-date case the loader already handles.
    assert pd.Timestamp("1979-02-01") not in parsed_dates, (
        "DATE CORRUPTION: 1979-01-02 was misparsed as 1979-02-01"
    )
    assert pd.Timestamp("1979-01-02") in parsed_dates


def test_unparseable_date_raises_loudly(tmp_path):
    df = _make_valid_df(n_days=3)
    df[config.DATE_COLUMN] = df[config.DATE_COLUMN].dt.strftime("%Y-%m-%d")
    df.loc[1, config.DATE_COLUMN] = "not-a-real-date"  # inject after converting to str
    csv_path = tmp_path / "bad_date_weather.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="matched none of the known formats"):
        load_single_field(csv_path, "bad_date_field")


def test_iso_date_format_is_parsed_correctly(tmp_path):
    df = _make_valid_df(n_days=20, start="1979-01-01")
    df[config.DATE_COLUMN] = df[config.DATE_COLUMN].dt.strftime("%Y-%m-%d")
    csv_path = tmp_path / "iso_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, report = load_single_field(csv_path, "iso_field")

    assert report.is_clean
    assert loaded[config.DATE_COLUMN].iloc[12] == pd.Timestamp("1979-01-13")


def test_date_alias_is_renamed_to_canonical_column(tmp_path):
    """config.DATE_COLUMN is 'valid_time' (confirmed real export column
    name). If a future export batch instead uses 'Date' -- listed in
    config.DATE_COLUMN_ALIASES as a fallback -- it should still be
    recognized and renamed to the canonical 'valid_time'."""
    df = _make_valid_df().rename(columns={config.DATE_COLUMN: "Date"})
    csv_path = tmp_path / "date_alias_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, report = load_single_field(csv_path, "alias_field")

    assert config.DATE_COLUMN in loaded.columns  # renamed to 'valid_time'
    assert "Date" not in loaded.columns
    assert report.is_clean


def test_canonical_date_column_loads_without_renaming(tmp_path):
    """The common case: a field CSV already uses 'valid_time' directly,
    as confirmed in the real dataset. Should load with no renaming."""
    df = _make_valid_df()  # already uses config.DATE_COLUMN = 'valid_time'
    csv_path = tmp_path / "native_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, report = load_single_field(csv_path, "native_field")

    assert config.DATE_COLUMN in loaded.columns
    assert report.is_clean


def test_extra_columns_do_not_break_validation(tmp_path):
    """Fields carrying extra columns beyond the required 9 (e.g.
    Vapour_Pressure_Mean_24h, latitude, longitude) should load fine --
    only missing required columns should fail validation."""
    df = _make_valid_df()
    df["Vapour_Pressure_Mean_24h"] = 2.0
    df["latitude"] = 14.25
    df["longitude"] = 78.2
    csv_path = tmp_path / "extra_cols_weather.csv"
    df.to_csv(csv_path, index=False)

    loaded, report = load_single_field(csv_path, "extra_field")
    assert report.is_clean
    assert "Vapour_Pressure_Mean_24h" in loaded.columns


def test_discover_field_files(tmp_path):
    for name in ["fieldA", "fieldB"]:
        d = tmp_path / name
        d.mkdir()
        _make_valid_df(n_days=10).to_csv(d / f"{name}_weather.csv", index=False)

    files = discover_field_files(tmp_path)
    assert set(files.keys()) == {"fieldA", "fieldB"}


def test_pool_fields_tags_field_column(tmp_path):
    frames = {}
    for name in ["fieldA", "fieldB"]:
        frames[name] = _make_valid_df(n_days=5).assign(field=name)

    pooled = pool_fields(frames)
    assert set(pooled["field"].unique()) == {"fieldA", "fieldB"}
    assert len(pooled) == 10


def test_load_all_fields_skips_bad_field_but_keeps_good_ones(tmp_path):
    good_dir = tmp_path / "good_field"
    good_dir.mkdir()
    _make_valid_df(n_days=10).to_csv(good_dir / "good_field_weather.csv", index=False)

    bad_dir = tmp_path / "bad_field"
    bad_dir.mkdir()
    _make_valid_df(n_days=10).drop(columns=["Solar_Radiation_Flux"]).to_csv(
        bad_dir / "bad_field_weather.csv", index=False
    )

    frames, reports = load_all_fields(tmp_path)
    assert "good_field" in frames
    assert "bad_field" not in frames
