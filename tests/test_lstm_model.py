"""
tests/test_lstm_model.py -- automated tests for Module 7.
Run with: pytest tests/test_lstm_model.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
from models.lstm_model import DroughtLSTM, build_model  # noqa: E402


def test_model_output_shape():
    model, device = build_model(device="cpu")
    x = torch.randn(4, config.SEQUENCE_LENGTH_DAYS, len(config.MODEL_INPUT_FEATURES))
    logits = model(x)
    assert logits.shape == (4, len(config.DROUGHT_CLASSES))


def test_model_output_shape_batch_size_one():
    """Edge case: batch size of 1 shouldn't break LSTM's batching logic."""
    model, device = build_model(device="cpu")
    x = torch.randn(1, config.SEQUENCE_LENGTH_DAYS, len(config.MODEL_INPUT_FEATURES))
    logits = model(x)
    assert logits.shape == (1, len(config.DROUGHT_CLASSES))


def test_model_rejects_wrong_feature_dimension():
    model, device = build_model(device="cpu")
    x = torch.randn(4, config.SEQUENCE_LENGTH_DAYS, 5)  # wrong: should be 9
    with pytest.raises(ValueError, match="expected input shape"):
        model(x)


def test_model_rejects_wrong_ndim():
    model, device = build_model(device="cpu")
    x = torch.randn(4, len(config.MODEL_INPUT_FEATURES))  # missing seq_len dim
    with pytest.raises(ValueError, match="expected input shape"):
        model(x)


def test_parameter_count_matches_expected_range():
    """Matches the original spec's ~52,800 parameter target -- verifies
    architecture hasn't silently drifted (e.g. wrong hidden_size)."""
    model, device = build_model(device="cpu")
    n_params = model.count_parameters()
    assert 45_000 <= n_params <= 60_000, f"Parameter count {n_params} outside expected range"


def test_model_is_deterministic_with_fixed_seed():
    """Same seed -> same initial weights -> same output for same input."""
    model_a, _ = build_model(device="cpu")
    model_b, _ = build_model(device="cpu")
    x = torch.randn(2, config.SEQUENCE_LENGTH_DAYS, len(config.MODEL_INPUT_FEATURES))

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    torch.testing.assert_close(out_a, out_b)


def test_model_variable_sequence_length_still_works():
    """LSTMs can technically handle variable sequence length, even though
    we always feed exactly 30 days in practice -- confirms no hardcoded
    seq_len assumption snuck into the architecture."""
    model, device = build_model(device="cpu")
    x_short = torch.randn(2, 10, len(config.MODEL_INPUT_FEATURES))
    logits = model(x_short)
    assert logits.shape == (2, len(config.DROUGHT_CLASSES))


def test_dropout_is_disabled_in_eval_mode():
    """Two forward passes in eval() mode on the same input must be
    identical -- dropout should not be active."""
    model, device = build_model(device="cpu")
    model.eval()
    x = torch.randn(3, config.SEQUENCE_LENGTH_DAYS, len(config.MODEL_INPUT_FEATURES))
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2)


def test_model_moves_to_requested_device_cpu():
    model, device = build_model(device="cpu")
    assert device == "cpu"
    assert next(model.parameters()).device.type == "cpu"


def test_logits_are_finite():
    """Basic sanity: no NaN/Inf in a fresh model's output on random input."""
    model, device = build_model(device="cpu")
    x = torch.randn(4, config.SEQUENCE_LENGTH_DAYS, len(config.MODEL_INPUT_FEATURES))
    logits = model(x)
    assert torch.isfinite(logits).all()
