"""
Tests for src.config

Run with:  pytest tests/test_config.py -v
"""
from pathlib import Path

from src.config import (
    PROJECT_ROOT,
    RAW_WEATHER_DIR,
    PROCESSED_WEATHER_DIR,
    MODELS_DIR,
    raw_weather_path,
    processed_weather_path,
    ensure_directories_exist,
    REQUIRED_RAW_COLUMNS,
    DROUGHT_LABEL_MAP,
    BINARY_DROUGHT_THRESHOLD,
)


def test_project_root_is_absolute_and_exists():
    assert PROJECT_ROOT.is_absolute()
    assert PROJECT_ROOT.exists()


def test_data_dirs_are_children_of_project_root():
    assert RAW_WEATHER_DIR.is_relative_to(PROJECT_ROOT)
    assert PROCESSED_WEATHER_DIR.is_relative_to(PROJECT_ROOT)
    assert MODELS_DIR.is_relative_to(PROJECT_ROOT)


def test_raw_weather_path_convention():
    path = raw_weather_path("Lepakshi")
    assert path == RAW_WEATHER_DIR / "lepakshi" / "lepakshi_weather.csv"
    # case-insensitive input should normalize to lowercase
    assert raw_weather_path("LEPAKSHI") == path


def test_processed_weather_path_convention():
    path = processed_weather_path("Kurnool")
    assert path == PROCESSED_WEATHER_DIR / "kurnool_with_spei.csv"


def test_ensure_directories_exist_is_idempotent(tmp_path, monkeypatch):
    # redirect to a temp location so this test never touches real project data
    import src.config as cfg
    monkeypatch.setattr(cfg, "RAW_WEATHER_DIR", tmp_path / "raw")
    monkeypatch.setattr(cfg, "PROCESSED_WEATHER_DIR", tmp_path / "processed")
    monkeypatch.setattr(cfg, "SEQUENCES_DIR", tmp_path / "processed" / "sequences")
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "LOGS_DIR", tmp_path / "logs")

    cfg.ensure_directories_exist()
    cfg.ensure_directories_exist()  # calling twice must not raise

    assert (tmp_path / "raw").exists()
    assert (tmp_path / "models").exists()


def test_required_raw_columns_non_empty():
    assert len(REQUIRED_RAW_COLUMNS) == 13
    assert "valid_time" in REQUIRED_RAW_COLUMNS


def test_drought_label_map_ordering():
    assert DROUGHT_LABEL_MAP["Healthy"] == 0
    assert DROUGHT_LABEL_MAP["Extreme"] == 4
    assert list(DROUGHT_LABEL_MAP.values()) == sorted(DROUGHT_LABEL_MAP.values())


def test_binary_threshold_matches_mckee_convention():
    assert BINARY_DROUGHT_THRESHOLD == -1.0
