"""
src/inference/multi_predictor.py — Updated with correct model import paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import config
from utils.logger import get_logger
from models.lstm_model import DroughtLSTM
from models.ml_models import DroughtTransformer, flatten_sequences

logger = get_logger(__name__)

MODEL_TYPES = {
    "Random Forest": {"type": "sklearn",  "file": "random_forest.joblib",  "meta": "random_forest_metadata.json"},
    "XGBoost":       {"type": "sklearn",  "file": "xgboost.joblib",         "meta": "xgboost_metadata.json"},
    "LSTM":          {"type": "pytorch",  "file": "drought_lstm.pt",         "meta": "drought_lstm_metadata.json", "class": DroughtLSTM},
    "Transformer":   {"type": "pytorch",  "file": "transformer.pt",          "meta": "transformer_metadata.json",  "class": DroughtTransformer},
}


class MultiModelPredictor:
    def __init__(self, model_name: str = "Transformer", models_dir: Path = None, device: str = None):
        if models_dir is None:
            models_dir = config.MODELS_SAVED_DIR
        models_dir = Path(models_dir)

        if model_name not in MODEL_TYPES:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_TYPES.keys())}")

        self.model_name = model_name
        self.model_info = MODEL_TYPES[model_name]

        meta_path = models_dir / self.model_info["meta"]
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}. Run training first.")
        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["feature_names"]
        self.label_map = self.metadata["label_map"]
        self.inverse_label_map = {int(v): k for k, v in self.label_map.items()}
        self.scaler = self.metadata["scaler"]
        self.sequence_length = self.metadata["sequence_length"]

        model_path = models_dir / self.model_info["file"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Run training first.")

        if self.model_info["type"] == "sklearn":
            self.model = joblib.load(model_path)
            self.device = None

        elif self.model_info["type"] == "pytorch":
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            ModelClass = self.model_info["class"]
            arch = self.metadata.get("architecture", {})
            if ModelClass == DroughtLSTM:
                self.model = ModelClass(
                    input_size=arch.get("input_size", 9),
                    hidden_size=arch.get("hidden_size", 128),
                    num_layers=arch.get("num_layers", 2),
                    num_classes=arch.get("num_classes", 3),
                    dropout=arch.get("dropout", 0.3),
                )
            else:
                self.model = ModelClass(
                    input_size=arch.get("input_size", 9),
                    d_model=arch.get("d_model", 128),
                    nhead=arch.get("nhead", 4),
                    num_layers=arch.get("num_layers", 3),
                    dim_feedforward=arch.get("dim_feedforward", 256),
                    num_classes=arch.get("num_classes", 3),
                    seq_len=self.sequence_length,
                )
            self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            self.model.to(device)
            self.model.eval()

        logger.info(f"[{model_name}] Loaded from {model_path}")

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        mean = np.array(self.scaler["mean"], dtype=np.float32)
        std = np.array(self.scaler["std"], dtype=np.float32)
        return (X - mean) / std

    def predict(self, weather_df: pd.DataFrame) -> dict:
        missing = [c for c in self.feature_names if c not in weather_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if len(weather_df) != self.sequence_length:
            raise ValueError(f"Need exactly {self.sequence_length} rows, got {len(weather_df)}")

        X = weather_df[self.feature_names].values.astype(np.float32)
        X_scaled = self._apply_scaler(X)

        if self.model_info["type"] == "sklearn":
            X_flat = flatten_sequences(X_scaled[np.newaxis, ...])[0:1]
            probs = self.model.predict_proba(X_flat)[0]
            pred_idx = int(np.argmax(probs))
        else:
            X_tensor = torch.tensor(X_scaled).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        pred_class = self.inverse_label_map[pred_idx]
        return {
            "model_used": self.model_name,
            "predicted_class": pred_class,
            "confidence": float(probs[pred_idx]),
            "class_probabilities": {
                self.inverse_label_map[i]: float(probs[i]) for i in range(len(probs))
            },
        }

    @staticmethod
    def available_models() -> list:
        return list(MODEL_TYPES.keys())
