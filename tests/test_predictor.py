"""
tests/test_predictor.py -- automated tests for Module 9.
Run with: pytest tests/test_predictor.py -v
Builds a tiny real (untrained-quality but structurally valid) model +
metadata in a tmp_path fixture, so tests don't depend on Module 8 having
been run first.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
from models.lstm_model import build_model  # noqa: E402
from inference.predictor import DroughtPredictor  # noqa: E402


@pytest.fixture
def tiny_model_and_metadata(tmp_path):
    """Build and save a real (randomly-initialized, untrained) model +
    metadata, mimicking Module 8's save format exactly."""
    model, device = build_model(device="cpu")
    model_path = tmp_path / "drought_lstm.pt"
    torch.save(model.state_dict(), model_path)

    metadata = {
        "architecture": {
            "input_size": len(config.MODEL_INPUT_FEATURES),
            "hidden_size": config.LSTM_HIDDEN_SIZE,
            "num_layers": config.LSTM_NUM_LAYERS,
            "num_classes": len(config.DROUGHT_CLASSES),
            "dropout": config.LSTM_DROPOUT,
        },
        "feature_names": config.MODEL_INPUT_FEATURES,
        "label_map": config.LABEL_MAP,
        "scaler": {
            "mean": [5.0] * len(config.MODEL_INPUT_FEATURES),
            "std": [2.0] * len(config.MODEL_INPUT_FEATURES),
        },
        "sequence_length": config.SEQUENCE_LENGTH_DAYS,
    }
    metadata_path = tmp_path / "drought_lstm_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return model_path, metadata_path


def _make_valid_window_df(n_days=30, start="2020-01-01", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    data = {config.DATE_COLUMN: dates}
    for feat in config.MODEL_INPUT_FEATURES:
        data[feat] = rng.uniform(0, 10, n_days)
    return pd.DataFrame(data)


def test_predictor_loads_successfully(tiny_model_and_metadata):
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    assert predictor.feature_names == config.MODEL_INPUT_FEATURES
    assert predictor.sequence_length == config.SEQUENCE_LENGTH_DAYS


def test_predict_returns_expected_keys(tiny_model_and_metadata):
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    df = _make_valid_window_df()
    result = predictor.predict(df)
    assert "predicted_class" in result
    assert "confidence" in result
    assert "class_probabilities" in result
    assert result["predicted_class"] in config.DROUGHT_CLASSES


def test_predict_probabilities_sum_to_one(tiny_model_and_metadata):
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    df = _make_valid_window_df()
    result = predictor.predict(df)
    total = sum(result["class_probabilities"].values())
    assert abs(total - 1.0) < 1e-4


def test_predict_confidence_matches_argmax_probability(tiny_model_and_metadata):
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    df = _make_valid_window_df()
    result = predictor.predict(df)
    max_prob = max(result["class_probabilities"].values())
    assert abs(result["confidence"] - max_prob) < 1e-6


def test_predict_wrong_row_count_raises(tiny_model_and_metadata):
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    df = _make_valid_window_df(n_days=25)  # wrong: needs exactly 30
    with pytest.raises(ValueError, match="exactly"):
        predictor.predict(df)


def test_predict_missing_feature_column_raises(tiny_model_and_metadata):
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    df = _make_valid_window_df().drop(columns=["Precipitation_Flux"])
    with pytest.raises(ValueError, match="missing required columns"):
        predictor.predict(df)


def test_predict_date_gap_raises(tiny_model_and_metadata):
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    df = _make_valid_window_df(n_days=31).drop(index=15).reset_index(drop=True)
    df = df.iloc[:30]  # still 30 rows, but with a gap where row 15 was removed
    with pytest.raises(ValueError, match="non-consecutive dates"):
        predictor.predict(df)


def test_predict_nan_in_features_raises(tiny_model_and_metadata):
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    df = _make_valid_window_df()
    df.loc[5, "Precipitation_Flux"] = np.nan
    with pytest.raises(ValueError, match="NaN values"):
        predictor.predict(df)


def test_predict_ignores_extra_columns(tiny_model_and_metadata):
    """Matches Module 2's tolerance: extra CDS columns (lat/long,
    Vapour_Pressure_*, etc.) present in the input must not break prediction."""
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    df = _make_valid_window_df()
    df["latitude"] = 14.3
    df["longitude"] = 78.2
    df["Vapour_Pressure_Mean_24h"] = 2.0
    result = predictor.predict(df)  # should not raise
    assert result["predicted_class"] in config.DROUGHT_CLASSES


def test_predict_batch_matches_single_predict(tiny_model_and_metadata):
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    df = _make_valid_window_df()

    single_result = predictor.predict(df)

    X = df[config.MODEL_INPUT_FEATURES].values.astype(np.float32)[np.newaxis, ...]
    batch_results = predictor.predict_batch(X)

    assert batch_results[0]["predicted_class"] == single_result["predicted_class"]
    assert abs(batch_results[0]["confidence"] - single_result["confidence"]) < 1e-5


def test_scaler_is_loaded_from_metadata_not_refit(tiny_model_and_metadata):
    """Critical correctness property: the scaler used at inference must
    come from saved metadata, never refit on the input being predicted."""
    model_path, metadata_path = tiny_model_and_metadata
    predictor = DroughtPredictor(model_path=model_path, metadata_path=metadata_path, device="cpu")
    assert predictor.scaler["mean"] == [5.0] * len(config.MODEL_INPUT_FEATURES)
    assert predictor.scaler["std"] == [2.0] * len(config.MODEL_INPUT_FEATURES)
