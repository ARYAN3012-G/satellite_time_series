"""
src/inference/predictor.py — Updated to use correct model import paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import config
from utils.logger import get_logger
from models.lstm_model import DroughtLSTM

logger = get_logger(__name__)


class DroughtPredictor:
    def __init__(self, model_path: Path = None, metadata_path: Path = None, device: str = None):
        if model_path is None:
            model_path = config.MODELS_SAVED_DIR / "drought_lstm.pt"
        if metadata_path is None:
            metadata_path = config.MODELS_SAVED_DIR / "drought_lstm_metadata.json"

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        arch = self.metadata["architecture"]
        self.feature_names = self.metadata["feature_names"]
        self.label_map = self.metadata["label_map"]
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.scaler = self.metadata["scaler"]
        self.sequence_length = self.metadata["sequence_length"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = DroughtLSTM(
            input_size=arch.get("input_size", 9),
            hidden_size=arch.get("hidden_size", 128),
            num_layers=arch.get("num_layers", 2),
            num_classes=arch.get("num_classes", 3),
            dropout=arch.get("dropout", 0.3),
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()
        logger.info(f"Predictor loaded: {model_path}")

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
        X_tensor = torch.tensor(X_scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = self.inverse_label_map[pred_idx]
        return {
            "predicted_class": pred_class,
            "confidence": float(probs[pred_idx]),
            "class_probabilities": {
                self.inverse_label_map[i]: float(probs[i]) for i in range(len(probs))
            },
        }
