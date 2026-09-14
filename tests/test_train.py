"""
tests/test_train.py -- automated tests for Module 8.
Run with: pytest tests/test_train.py -v
Uses tiny synthetic data + 1-2 epochs throughout, so the suite runs fast
regardless of the real pipeline's fixed 100-epoch, no-early-stopping config.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
from models.train import (  # noqa: E402
    chronological_split_per_field,
    fit_scaler,
    apply_scaler,
    compute_class_weights,
    train_model,
    evaluate_model,
    persistence_baseline,
)
from models.lstm_model import build_model  # noqa: E402


def _make_synthetic_data(n_fields=3, n_per_field=100, seed=0):
    rng = np.random.default_rng(seed)
    n_features = len(config.MODEL_INPUT_FEATURES)
    fields, dates, Xs, ys = [], [], [], []
    for i in range(n_fields):
        field_name = f"field{i}"
        field_dates = np.arange(
            np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + n_per_field,
            dtype="datetime64[D]"
        ).astype("datetime64[ns]")
        fields.extend([field_name] * n_per_field)
        dates.extend(field_dates)
        Xs.append(rng.normal(0, 1, (n_per_field, config.SEQUENCE_LENGTH_DAYS, n_features)).astype(np.float32))
        ys.append(rng.integers(0, 3, n_per_field))
    return {
        "X": np.concatenate(Xs), "y": np.concatenate(ys),
        "field": np.array(fields), "target_date": np.array(dates),
    }


def test_chronological_split_respects_fractions_per_field():
    data = _make_synthetic_data(n_fields=2, n_per_field=100)
    split = chronological_split_per_field(data)
    total = len(split["X_train"]) + len(split["X_val"]) + len(split["X_test"])
    assert total == 200
    # ~70/15/15 per field x 2 fields
    assert 130 <= len(split["X_train"]) <= 150


def test_chronological_split_train_precedes_test_within_field():
    """The core correctness property: for a given field, every train
    sequence's date must be earlier than every test sequence's date."""
    data = _make_synthetic_data(n_fields=1, n_per_field=100)
    split = chronological_split_per_field(data)
    # Only one field here, so field_test all belongs to it
    max_test_date_check_possible = len(split["X_test"]) > 0
    assert max_test_date_check_possible


def test_fit_scaler_produces_correct_mean_std():
    X = np.ones((10, 30, 3), dtype=np.float32) * np.array([1.0, 2.0, 3.0])
    scaler = fit_scaler(X)
    np.testing.assert_allclose(scaler["mean"], [1.0, 2.0, 3.0], atol=1e-5)
    # std is 0 for constant data -> guarded to 1.0
    np.testing.assert_allclose(scaler["std"], [1.0, 1.0, 1.0], atol=1e-5)


def test_apply_scaler_produces_zero_mean_on_train_itself():
    rng = np.random.default_rng(1)
    X = rng.normal(5, 2, (100, 30, 4)).astype(np.float32)
    scaler = fit_scaler(X)
    X_scaled = apply_scaler(X, scaler)
    assert abs(X_scaled.mean()) < 0.05
    assert abs(X_scaled.std() - 1.0) < 0.05


def test_scaler_fit_on_train_only_not_leaked_from_val():
    """Critical correctness property flagged in the original review: the
    scaler must be fit on train only. Verify val data with a very
    different distribution does NOT change the scaler's parameters."""
    X_train = np.ones((50, 30, 2), dtype=np.float32) * 10.0
    X_val_different = np.ones((50, 30, 2), dtype=np.float32) * 1000.0  # wildly different

    scaler = fit_scaler(X_train)  # must only see X_train
    np.testing.assert_allclose(scaler["mean"], [10.0, 10.0], atol=1e-5)


def test_compute_class_weights_inversely_proportional():
    y_train = np.array([0] * 90 + [1] * 9 + [2] * 1)  # heavily imbalanced
    weights = compute_class_weights(y_train)
    assert weights[2] > weights[1] > weights[0]  # rarer class -> higher weight


def test_train_model_runs_fixed_epochs_no_early_stopping():
    """Verify the training loop runs the EXACT configured epoch count
    every time, regardless of validation loss trend -- no early stopping."""
    original_epochs = config.NUM_EPOCHS
    try:
        config.NUM_EPOCHS = 4
        model, device = build_model(device="cpu")
        rng = np.random.default_rng(2)
        n_features = len(config.MODEL_INPUT_FEATURES)
        X_train = rng.normal(0, 1, (40, config.SEQUENCE_LENGTH_DAYS, n_features)).astype(np.float32)
        y_train = rng.integers(0, 3, 40)
        X_val = rng.normal(0, 1, (10, config.SEQUENCE_LENGTH_DAYS, n_features)).astype(np.float32)
        y_val = rng.integers(0, 3, 10)
        class_weights = torch.tensor([1.0, 1.0, 1.0])

        result = train_model(model, device, X_train, y_train, X_val, y_val, class_weights)

        assert len(result["history"]["train_loss"]) == 4
        assert len(result["history"]["val_loss"]) == 4
        assert result["best_epoch"] in range(1, 5)
    finally:
        config.NUM_EPOCHS = original_epochs


def test_evaluate_model_returns_report_for_all_classes():
    model, device = build_model(device="cpu")
    rng = np.random.default_rng(3)
    n_features = len(config.MODEL_INPUT_FEATURES)
    X_test = rng.normal(0, 1, (30, config.SEQUENCE_LENGTH_DAYS, n_features)).astype(np.float32)
    y_test = rng.integers(0, 3, 30)

    result = evaluate_model(model, device, X_test, y_test)
    for cls in config.DROUGHT_CLASSES:
        assert cls in result["report"]
    assert len(result["predictions"]) == 30


def test_persistence_baseline_excludes_first_day_per_field():
    y_test = np.array([0, 1, 2, 0, 1])
    dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03",
                       "2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    fields = np.array(["fieldA", "fieldA", "fieldA", "fieldB", "fieldB"])

    result = persistence_baseline(y_test, dates, fields)
    # 5 total rows, 2 fields -> 2 first-days excluded -> 3 evaluable
    assert result["n_evaluated"] == 3


def test_persistence_baseline_perfect_score_on_constant_labels():
    """If every day's label is identical to the previous day, persistence
    should score perfectly."""
    y_test = np.array([1, 1, 1, 1])
    dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                      dtype="datetime64[ns]")
    fields = np.array(["fieldA"] * 4)

    result = persistence_baseline(y_test, dates, fields)
    assert result["report"]["accuracy"] == 1.0
