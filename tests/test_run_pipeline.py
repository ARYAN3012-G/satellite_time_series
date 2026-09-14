"""
Tests for src.preprocessing.run_pipeline

Run with: pytest tests/test_run_pipeline.py -v
"""
import pandas as pd
import pytest

from src.config import REQUIRED_RAW_COLUMNS


def _write_valid_csv(folder, location_name, n_rows=400):
    folder.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2000-01-01", periods=n_rows, freq="D")
    df = pd.DataFrame({col: range(n_rows) for col in REQUIRED_RAW_COLUMNS})
    df["valid_time"] = dates
    # give precipitation/ET some real variation so SPEI can be fit
    import numpy as np
    rng = np.random.default_rng(0)
    df["Precipitation_Flux"] = rng.gamma(2, 2, n_rows)
    df["ReferenceET_PenmanMonteith_FAO56"] = rng.normal(4, 0.5, n_rows)
    df.to_csv(folder / f"{location_name}_weather.csv", index=False)


@pytest.fixture
def fake_dirs(tmp_path, monkeypatch):
    """Isolate both raw and processed dirs to a temp location for every test."""
    import src.data_ingestion.weather_loader as wl
    import src.preprocessing.run_pipeline as rp
    import src.config as cfg

    fake_raw = tmp_path / "raw_weather"
    fake_processed = tmp_path / "processed_weather"
    fake_raw.mkdir()
    fake_processed.mkdir()

    monkeypatch.setattr(wl, "RAW_WEATHER_DIR", fake_raw)
    monkeypatch.setattr(cfg, "PROCESSED_WEATHER_DIR", fake_processed)
    monkeypatch.setattr(
        cfg, "processed_weather_path",
        lambda loc: fake_processed / f"{loc.lower()}_with_spei.csv"
    )
    monkeypatch.setattr(rp, "processed_weather_path", cfg.processed_weather_path)
    monkeypatch.setattr(rp, "ensure_directories_exist", lambda: None)

    return fake_raw, fake_processed


def test_run_pipeline_for_single_location_saves_file(fake_dirs):
    from src.preprocessing.run_pipeline import run_pipeline_for_location
    fake_raw, fake_processed = fake_dirs
    _write_valid_csv(fake_raw / "testfield", "testfield")

    df = run_pipeline_for_location("testfield", save=True)

    assert "SPEI" in df.columns
    assert "Drought_Label" in df.columns
    assert "Target_Binary" in df.columns
    assert (fake_processed / "testfield_with_spei.csv").exists()


def test_run_pipeline_all_locations_skips_bad_ones(fake_dirs):
    from src.preprocessing.run_pipeline import run_pipeline_all_locations
    fake_raw, fake_processed = fake_dirs

    _write_valid_csv(fake_raw / "good_field", "good_field")

    bad_folder = fake_raw / "bad_field"
    bad_folder.mkdir()
    pd.DataFrame({"valid_time": ["2020-01-01"]}).to_csv(
        bad_folder / "bad_field_weather.csv", index=False
    )

    results = run_pipeline_all_locations(save=True)

    assert "good_field" in results
    assert "bad_field" not in results
    assert (fake_processed / "good_field_with_spei.csv").exists()
    assert not (fake_processed / "bad_field_with_spei.csv").exists()
